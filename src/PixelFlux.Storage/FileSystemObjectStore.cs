using System.Runtime.CompilerServices;

namespace PixelFlux.Storage;

/// <summary>
/// An <see cref="IObjectStore"/> backed by a directory on disk.
/// </summary>
/// <remarks>
/// <para>
/// This is the default and most important implementation, because the headline deployment for
/// PixelFlux is "point every machine at the same OneDrive/Dropbox/Syncthing folder and they
/// find each other". The store therefore has to behave well under a sync client that is
/// watching the directory and uploading files as they appear.
/// </para>
/// <para>
/// Two consequences shape the code below:
/// </para>
/// <list type="number">
/// <item><description>
/// <b>Never let a partially-written file be visible.</b> A sync client will happily upload a
/// half-written JSON document to every other device. All writes go to a uniquely-named
/// temporary file in the same directory and are then renamed into place, since a same-volume
/// rename is atomic on both NTFS and POSIX.
/// </description></item>
/// <item><description>
/// <b>Never trust a key from outside.</b> Keys become paths, so a key containing <c>..</c>
/// would let a caller write outside the root. <see cref="ResolvePath"/> rejects those.
/// </description></item>
/// </list>
/// </remarks>
public sealed class FileSystemObjectStore : IObjectStore
{
    // Written alongside the target then renamed over it. The prefix is recognisable so that
    // a crash mid-write leaves an obviously-junk file rather than something that looks real,
    // and so listings can filter these out.
    private const string TempPrefix = ".pfxtmp-";

    private readonly string _root;

    /// <summary>
    /// Creates a store rooted at <paramref name="rootDirectory"/>, creating the directory if it
    /// does not already exist.
    /// </summary>
    /// <param name="rootDirectory">
    /// Absolute path to the directory that backs this store. Every key is a path relative to it.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="rootDirectory"/> is null or blank.</exception>
    /// <exception cref="ObjectStoreException">The directory could not be created.</exception>
    public FileSystemObjectStore(string rootDirectory)
    {
        if (string.IsNullOrWhiteSpace(rootDirectory))
        {
            throw new ArgumentException("A root directory is required.", nameof(rootDirectory));
        }

        _root = Path.GetFullPath(rootDirectory);

        try
        {
            Directory.CreateDirectory(_root);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new ObjectStoreException($"Could not create or access the storage folder '{_root}'.", ex);
        }
    }

    /// <summary>The absolute path of the directory backing this store.</summary>
    public string RootDirectory => _root;

    /// <inheritdoc />
    public string DisplayName => $"file:{_root}";

    /// <inheritdoc />
    public async IAsyncEnumerable<ObjectEntry> ListAsync(
        string prefix,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prefix);

        // A prefix is a string prefix on keys, not a directory. To avoid walking the entire
        // root for a prefix like "jobs/pending/", start the walk at the deepest directory the
        // prefix fully names and filter the remainder as a string.
        int lastSlash = prefix.LastIndexOf('/');
        string directoryPart = lastSlash < 0 ? string.Empty : prefix[..lastSlash];
        string searchRoot = directoryPart.Length == 0 ? _root : ResolvePath(directoryPart, allowDirectory: true);

        if (!Directory.Exists(searchRoot))
        {
            yield break;
        }

        IEnumerable<string> files;
        try
        {
            files = Directory.EnumerateFiles(searchRoot, "*", SearchOption.AllDirectories);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new ObjectStoreException($"Could not list '{prefix}' under '{_root}'.", ex);
        }

        foreach (string path in files)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string name = Path.GetFileName(path);
            if (name.StartsWith(TempPrefix, StringComparison.Ordinal))
            {
                continue; // in-flight write from this or another process
            }

            string key = ToKey(path);
            if (!key.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }

            ObjectEntry? entry = TryDescribe(path, key);
            if (entry is not null)
            {
                yield return entry.Value;
            }
        }

        await Task.CompletedTask.ConfigureAwait(false); // keeps the method async for the iterator contract
    }

    /// <inheritdoc />
    /// <remarks>
    /// Retries briefly on a sharing violation. Opening a file in this store races against three
    /// things that are always happening in the folders PixelFlux targets: the sync client
    /// uploading it, an antivirus scanning it, and this store's own
    /// <see cref="PublishAtomicallyAsync"/> swapping a new version in — <c>ReplaceFile</c> holds
    /// an exclusive handle on the destination for the instant of the swap. All three clear in
    /// milliseconds. Surfacing them as hard failures would make ingestion look flaky for
    /// reasons that have nothing to do with the caller.
    /// <para>
    /// A genuinely missing file is still reported immediately as <see langword="null"/> — it is
    /// distinguished from a locked one by exception type, so absence never costs a delay.
    /// </para>
    /// </remarks>
    public async Task<Stream?> OpenReadAsync(string key, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string path = ResolvePath(key);

        const int maxAttempts = 12;
        Exception? last = null;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            try
            {
                // FileShare.ReadWrite | Delete: never block a writer or a sync client on our
                // account, and tolerate the file being replaced underneath an open handle.
                return new FileStream(
                    path,
                    FileMode.Open,
                    FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete,
                    bufferSize: 64 * 1024,
                    FileOptions.Asynchronous | FileOptions.SequentialScan);
            }
            catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
            {
                return null;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                last = ex;
                await Task.Delay(5 * (attempt + 1), cancellationToken).ConfigureAwait(false);
            }
        }

        throw new ObjectStoreException(
            $"Could not read '{key}' from '{_root}': the file stayed locked across {maxAttempts} attempts.",
            last!);
    }

    /// <inheritdoc />
    public async Task WriteAsync(string key, Stream content, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(content);

        string path = ResolvePath(key);
        string directory = Path.GetDirectoryName(path)!;
        string temp = Path.Combine(directory, TempPrefix + Guid.NewGuid().ToString("n"));

        try
        {
            Directory.CreateDirectory(directory);

            await using (var destination = new FileStream(
                temp,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 64 * 1024,
                FileOptions.Asynchronous))
            {
                await content.CopyToAsync(destination, cancellationToken).ConfigureAwait(false);
                await destination.FlushAsync(cancellationToken).ConfigureAwait(false);
            }

            await PublishAtomicallyAsync(temp, path, key, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            TryDeleteTemp(temp);
            throw new ObjectStoreException($"Could not write '{key}' to '{_root}'.", ex);
        }
        catch (OperationCanceledException)
        {
            TryDeleteTemp(temp);
            throw;
        }
    }

    /// <inheritdoc />
    public async Task<bool> TryCreateAsync(
        string key,
        ReadOnlyMemory<byte> content,
        CancellationToken cancellationToken = default)
    {
        string path = ResolvePath(key);

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            // FileMode.CreateNew is the atomic test-and-set: the OS either creates the file or
            // fails because it already exists. Writing directly rather than via a temp+rename
            // is essential — a rename would clobber a rival's claim instead of losing to it.
            await using var destination = new FileStream(
                path,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 4096,
                FileOptions.Asynchronous);

            await destination.WriteAsync(content, cancellationToken).ConfigureAwait(false);
            await destination.FlushAsync(cancellationToken).ConfigureAwait(false);
            return true;
        }
        catch (IOException) when (File.Exists(path))
        {
            // Lost the race, or the file was already there. Either way: not ours.
            return false;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new ObjectStoreException($"Could not create '{key}' in '{_root}'.", ex);
        }
    }

    /// <inheritdoc />
    public Task DeleteAsync(string key, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string path = ResolvePath(key);

        try
        {
            File.Delete(path); // no-op when absent, which is the documented contract
        }
        catch (DirectoryNotFoundException)
        {
            // Also absent.
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new ObjectStoreException($"Could not delete '{key}' from '{_root}'.", ex);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task<ObjectEntry?> StatAsync(string key, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(TryDescribe(ResolvePath(key), key));
    }

    /// <summary>
    /// Moves a fully-written temporary file over the target key, keeping the swap atomic for
    /// readers and surviving the transient sharing failures Windows produces under contention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A plain <c>File.Move(overwrite: true)</c> is not sufficient on Windows. It maps to
    /// <c>MoveFileEx(MOVEFILE_REPLACE_EXISTING)</c>, which needs delete access to the
    /// destination and fails with <see cref="UnauthorizedAccessException"/> whenever another
    /// handle is open on it — even a reader that opened share-delete. That is not a rare edge
    /// case here: the whole point of this store is that it lives in a folder a sync client is
    /// actively scanning, an antivirus is sampling, and other threads are reading.
    /// </para>
    /// <para>
    /// So there are two mechanisms and a retry. <c>File.Replace</c> maps to the Win32
    /// <c>ReplaceFile</c>, which is designed for exactly this and tolerates open readers far
    /// better, but it requires the destination to already exist — so <c>Move</c> stays as the
    /// create case. Whichever applies, a brief backoff loop absorbs the genuinely transient
    /// sharing violations. The swap itself is still atomic in both paths: a reader gets the old
    /// object or the new one, never a mixture.
    /// </para>
    /// </remarks>
    private static async Task PublishAtomicallyAsync(
        string temp,
        string path,
        string key,
        CancellationToken cancellationToken)
    {
        const int maxAttempts = 12;
        Exception? last = null;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            try
            {
                if (File.Exists(path))
                {
                    // ignoreMetadataErrors: we do not care about carrying ACLs/attributes from
                    // the object being replaced; the new content is authoritative.
                    File.Replace(temp, path, destinationBackupFileName: null, ignoreMetadataErrors: true);
                }
                else
                {
                    File.Move(temp, path, overwrite: true);
                }

                return;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                last = ex;

                // Linear-ish backoff: a sharing violation from a scanner clears in milliseconds,
                // so this tops out around a third of a second rather than stalling ingestion.
                await Task.Delay(5 * (attempt + 1), cancellationToken).ConfigureAwait(false);
            }
        }

        throw new ObjectStoreException(
            $"Could not publish '{key}': the destination stayed locked across {maxAttempts} attempts.",
            last!);
    }

    /// <summary>
    /// Maps a key onto an absolute path inside the root, rejecting any key that would escape it.
    /// </summary>
    /// <remarks>
    /// Keys reach this store from shared storage written by other machines, so they are treated
    /// as untrusted input. The check is done on the resolved path rather than by scanning for
    /// <c>..</c> in the key, which catches encoded and mixed-separator variants too.
    /// </remarks>
    private string ResolvePath(string key, bool allowDirectory = false)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            throw new ArgumentException("An object key is required.", nameof(key));
        }

        if (key.StartsWith('/'))
        {
            throw new ArgumentException($"Object keys must be relative; got '{key}'.", nameof(key));
        }

        string combined = Path.GetFullPath(Path.Combine(_root, key.Replace('/', Path.DirectorySeparatorChar)));

        if (!combined.StartsWith(_root, StringComparison.OrdinalIgnoreCase) ||
            (combined.Length > _root.Length && combined[_root.Length] != Path.DirectorySeparatorChar))
        {
            throw new ArgumentException($"Object key '{key}' escapes the storage root.", nameof(key));
        }

        if (!allowDirectory && combined.Length == _root.Length)
        {
            throw new ArgumentException("Object key must name a file, not the storage root.", nameof(key));
        }

        return combined;
    }

    /// <summary>Converts an absolute path under the root back into a forward-slash key.</summary>
    private string ToKey(string absolutePath)
        => absolutePath[(_root.Length + 1)..].Replace(Path.DirectorySeparatorChar, '/');

    /// <summary>
    /// Reads file metadata, returning <see langword="null"/> when the file has gone.
    /// </summary>
    /// <remarks>
    /// The absent case is genuinely expected rather than exceptional: another device can delete
    /// a completed job between our listing it and our statting it, and a sync client can move
    /// files out from under an enumeration.
    /// </remarks>
    private static ObjectEntry? TryDescribe(string path, string key)
    {
        try
        {
            var info = new FileInfo(path);
            return info.Exists
                ? new ObjectEntry(key, info.Length, new DateTimeOffset(info.LastWriteTimeUtc, TimeSpan.Zero))
                : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Best-effort cleanup of an abandoned temporary file; failure here is not worth reporting.</summary>
    private static void TryDeleteTemp(string temp)
    {
        try
        {
            File.Delete(temp);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Leaving a .pfxtmp- file behind is harmless: listings skip them.
        }
    }
}
