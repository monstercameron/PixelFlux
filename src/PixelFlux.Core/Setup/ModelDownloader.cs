using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace PixelFlux.Core.Setup;

/// <summary>How a download is getting on.</summary>
/// <param name="Model">Which model.</param>
/// <param name="File">The file being fetched, for the line of text under the bar.</param>
/// <param name="ReceivedBytes">Bytes of this model written so far, including finished files.</param>
/// <param name="TotalBytes">Bytes this model needs in total.</param>
public readonly record struct DownloadProgress(
    string Model,
    string File,
    long ReceivedBytes,
    long TotalBytes)
{
    /// <summary>How far along, from 0 to 1.</summary>
    public double Fraction => TotalBytes <= 0 ? 0 : Math.Clamp((double)ReceivedBytes / TotalBytes, 0, 1);
}

/// <summary>Raised when a download could not be completed.</summary>
public sealed class ModelDownloadException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What went wrong, in words a person can act on.</param>
    /// <param name="inner">The underlying failure, if there was one.</param>
    public ModelDownloadException(string message, Exception? inner = null)
        : base(message, inner)
    {
    }
}

/// <summary>
/// Fetches models into the local models directory.
/// </summary>
/// <remarks>
/// <para>
/// The only code in PixelFlux that opens a socket, and it does so only when somebody presses a
/// button. It deliberately lives outside the WebView: the interface's content security policy
/// forbids network access, and rather than punching a hole in it for setup, the download is plain
/// HTTP from the application process with the interface only watching.
/// </para>
/// <para>
/// <b>Written to a temporary name and moved into place.</b> A partially written model that has the
/// right filename is worse than no model: it loads, or half-loads, and every photograph fails in a
/// way that points at the wrong thing. Nothing appears under its real name until it is complete
/// and the right length.
/// </para>
/// </remarks>
public sealed class ModelDownloader
{
    private readonly HttpClient _http;
    private readonly ILogger _log;

    /// <summary>Creates a downloader.</summary>
    /// <param name="http">
    /// The client to fetch with. Supplied rather than created so a test can hand over a fake and
    /// so the caller owns its lifetime and any proxy configuration.
    /// </param>
    /// <param name="logger">Logger.</param>
    public ModelDownloader(HttpClient http, ILogger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(http);

        _http = http;
        _log = logger ?? NullLogger.Instance;
    }

    /// <summary>Downloads everything one model needs, skipping files already present.</summary>
    /// <param name="model">The model to install.</param>
    /// <param name="modelsRoot">Where models live.</param>
    /// <param name="progress">Called as bytes arrive.</param>
    /// <param name="cancellationToken">Stops the download; partial files are cleaned up.</param>
    /// <returns>How many bytes were actually fetched.</returns>
    /// <exception cref="ModelDownloadException">The download failed or arrived the wrong size.</exception>
    public async Task<long> InstallAsync(
        CatalogueModel model,
        string modelsRoot,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);

        long alreadyHave = 0;
        long fetched = 0;

        foreach (ModelFile file in model.Files)
        {
            string destination = Path.Combine(
                modelsRoot, file.RelativePath.Replace('/', Path.DirectorySeparatorChar));

            // Resuming a part-finished install is the normal case, not an edge case: the vision
            // model is 1.4 GB and people close laptops.
            if (File.Exists(destination) && new FileInfo(destination).Length == file.Bytes)
            {
                alreadyHave += file.Bytes;
                progress?.Report(new DownloadProgress(
                    model.Id, Path.GetFileName(destination), alreadyHave + fetched, model.Bytes));
                continue;
            }

            fetched += await FetchAsync(
                file,
                destination,
                model,
                alreadyHave + fetched,
                progress,
                cancellationToken).ConfigureAwait(false);
        }

        _log.LogInformation("Installed {Model} ({Bytes} bytes fetched).", model.Id, fetched);
        return fetched;
    }

    private async Task<long> FetchAsync(
        ModelFile file,
        string destination,
        CatalogueModel model,
        long completedBefore,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);

        string temporary = destination + ".part";
        string name = Path.GetFileName(destination);

        try
        {
            using HttpResponseMessage response = await _http
                .GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                throw new ModelDownloadException(
                    $"{name} could not be downloaded: the server answered {(int)response.StatusCode}.");
            }

            await using (Stream incoming = await response.Content
                             .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
            await using (var target = new FileStream(
                             temporary, FileMode.Create, FileAccess.Write, FileShare.None,
                             bufferSize: 128 * 1024, useAsync: true))
            {
                var buffer = new byte[128 * 1024];
                long written = 0;
                long lastReport = 0;

                while (true)
                {
                    int read = await incoming
                        .ReadAsync(buffer, cancellationToken).ConfigureAwait(false);

                    if (read == 0)
                    {
                        break;
                    }

                    await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken)
                        .ConfigureAwait(false);
                    written += read;

                    // Reported every megabyte rather than every buffer. At 128 KB a chunk a large
                    // model would raise ten thousand UI updates and spend more time rendering a
                    // progress bar than writing the file.
                    if (written - lastReport >= 1024 * 1024)
                    {
                        lastReport = written;
                        progress?.Report(new DownloadProgress(
                            model.Id, name, completedBefore + written, model.Bytes));
                    }
                }
            }

            long actual = new FileInfo(temporary).Length;

            if (actual != file.Bytes)
            {
                throw new ModelDownloadException(
                    $"{name} arrived incomplete — expected {file.Bytes:N0} bytes and got {actual:N0}.");
            }

            // Only now does it take the name the application looks for.
            File.Move(temporary, destination, overwrite: true);

            progress?.Report(new DownloadProgress(
                model.Id, name, completedBefore + file.Bytes, model.Bytes));

            return file.Bytes;
        }
        catch (Exception error) when (error is HttpRequestException or IOException or TaskCanceledException)
        {
            throw new ModelDownloadException(
                $"{name} could not be downloaded. Check the connection and try again.", error);
        }
        finally
        {
            // A .part left behind would be invisible to the catalogue and would still occupy the
            // disk, so it goes whether this succeeded, failed, or was cancelled.
            TryDelete(temporary);
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
            // Held open by a scanner. It is a stray temporary file, not a failure worth raising.
        }
    }
}
