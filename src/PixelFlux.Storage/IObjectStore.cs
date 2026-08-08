namespace PixelFlux.Storage;

/// <summary>
/// A flat key/value blob store shared by every PixelFlux device.
/// </summary>
/// <remarks>
/// <para>
/// PixelFlux has no server. Devices coordinate entirely by reading and writing objects in a
/// location they can all see — a synchronised folder (OneDrive, Dropbox, Syncthing, a network
/// share) or an S3-compatible bucket. This interface is that location, and nothing more.
/// </para>
/// <para>
/// The abstraction is deliberately shallow: six operations, no transactions, no listing
/// hierarchy, no metadata beyond what <see cref="ObjectEntry"/> carries. Everything above it
/// — job claiming, revision sync — is built from these primitives rather than from
/// store-specific features, so a filesystem and a bucket behave identically.
/// </para>
/// <para>
/// <b>Keys</b> are forward-slash separated, case-sensitive, and must not begin with a slash
/// (for example <c>jobs/pending/0f3a...json</c>). A filesystem store maps them onto
/// subdirectories; an S3 store uses them verbatim.
/// </para>
/// <para>
/// <b>Consistency assumptions.</b> Implementations are assumed read-after-write consistent for
/// a single key but say nothing about listing freshness — a newly written object may not appear
/// in <see cref="ListAsync"/> immediately, and on a sync-folder store it may take minutes to
/// reach another device. Callers must therefore treat every listing as possibly stale and must
/// never assume they have seen the complete set. This is why job claiming uses
/// <see cref="TryCreateAsync"/> (which is atomic per key) rather than "list, pick, then write".
/// </para>
/// <para>
/// Implementations are safe for concurrent use by multiple threads.
/// </para>
/// </remarks>
public interface IObjectStore
{
    /// <summary>
    /// A short, stable, human-readable description of where this store points, used in logs
    /// and in the Settings UI (for example <c>file:C:\Users\me\OneDrive\PixelFlux</c> or
    /// <c>s3:my-bucket/photos</c>). Must never contain credentials.
    /// </summary>
    string DisplayName { get; }

    /// <summary>
    /// Enumerates every object whose key starts with <paramref name="prefix"/>.
    /// </summary>
    /// <param name="prefix">
    /// Key prefix to match. Pass an empty string to enumerate the whole store. This is a
    /// literal string prefix, not a glob, and not a directory: the prefix <c>jobs/p</c>
    /// matches <c>jobs/pending/a.json</c>.
    /// </param>
    /// <param name="cancellationToken">Cancels the enumeration.</param>
    /// <returns>
    /// A lazily-produced stream of entries in unspecified order. Buckets are paged behind the
    /// scenes, so callers that only need the first few results should stop enumerating rather
    /// than materialising the sequence.
    /// </returns>
    IAsyncEnumerable<ObjectEntry> ListAsync(string prefix, CancellationToken cancellationToken = default);

    /// <summary>
    /// Opens an object for reading.
    /// </summary>
    /// <param name="key">The object key.</param>
    /// <param name="cancellationToken">Cancels the open.</param>
    /// <returns>
    /// A readable stream the caller owns and must dispose, or <see langword="null"/> if no
    /// object exists at <paramref name="key"/>. A missing object is an ordinary outcome in this
    /// system (another device may have completed and cleaned up a job), so it is signalled by
    /// <see langword="null"/> rather than by an exception.
    /// </returns>
    Task<Stream?> OpenReadAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Writes an object, replacing any existing object at the same key.
    /// </summary>
    /// <param name="key">The object key.</param>
    /// <param name="content">
    /// The bytes to store. The implementation reads this stream to its end from its current
    /// position; the caller retains ownership and must dispose it.
    /// </param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// The write is atomic with respect to readers: a concurrent <see cref="OpenReadAsync"/>
    /// sees either the whole previous object or the whole new one, never a partial file. On a
    /// filesystem store this is achieved by writing to a temporary name and renaming into place,
    /// which matters a great deal when the target is a folder a sync client is watching.
    /// </remarks>
    Task WriteAsync(string key, Stream content, CancellationToken cancellationToken = default);

    /// <summary>
    /// Atomically creates an object <em>only if</em> no object already exists at that key.
    /// </summary>
    /// <param name="key">The object key to claim.</param>
    /// <param name="content">The bytes to store. Kept small — this is a claim marker, not a payload.</param>
    /// <param name="cancellationToken">Cancels the attempt.</param>
    /// <returns>
    /// <see langword="true"/> if this caller created the object; <see langword="false"/> if it
    /// already existed and was left untouched.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This is the one primitive in the interface with a concurrency guarantee, and the entire
    /// distributed job system rests on it: two devices that race to claim the same image both
    /// call this, exactly one gets <see langword="true"/>, and the loser moves on.
    /// </para>
    /// <para>
    /// <b>Caveat for sync-folder stores.</b> The guarantee is only as strong as the underlying
    /// store. A local filesystem gives a true atomic create. A folder synchronised by OneDrive
    /// or Dropbox does not: two devices can each create the file locally and the sync client
    /// will resolve the collision afterwards, usually by keeping both under conflict-renamed
    /// filenames. Work is therefore required to be idempotent — a double-processed image must
    /// produce the same result, not a corrupted one. See
    /// <c>PixelFlux.Core.Jobs.JobQueue</c> for how that is handled.
    /// </para>
    /// </remarks>
    Task<bool> TryCreateAsync(string key, ReadOnlyMemory<byte> content, CancellationToken cancellationToken = default);

    /// <summary>
    /// Deletes an object. Deleting a key that does not exist succeeds silently, so callers can
    /// treat cleanup as idempotent.
    /// </summary>
    /// <param name="key">The object key.</param>
    /// <param name="cancellationToken">Cancels the delete.</param>
    Task DeleteAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Fetches an object's metadata without transferring its body.
    /// </summary>
    /// <param name="key">The object key.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The entry, or <see langword="null"/> if no object exists at that key.</returns>
    /// <remarks>
    /// Used by the job system to age out stale claims: the claim marker's
    /// <see cref="ObjectEntry.LastModified"/> is how another device decides a worker has died.
    /// </remarks>
    Task<ObjectEntry?> StatAsync(string key, CancellationToken cancellationToken = default);
}
