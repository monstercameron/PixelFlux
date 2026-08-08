using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;

namespace PixelFlux.Ai.Semantic;

/// <summary>Progress from a description run.</summary>
/// <param name="Total">Photographs queued when the run started.</param>
/// <param name="Done">Photographs finished.</param>
/// <param name="Current">Name of the photograph being described.</param>
/// <param name="Latest">The description just written, for showing the work as it happens.</param>
public readonly record struct DescriptionProgress(int Total, int Done, string? Current, string? Latest);

/// <summary>
/// Works through the library writing a description of each photograph.
/// </summary>
/// <remarks>
/// <para>
/// The slowest thing in the application by a wide margin: seconds per photograph against
/// milliseconds for everything else. That shapes the design. It commits each description as it
/// is written, so stopping the run keeps the work; it visits only photographs with no
/// description, so it is resumable and a newly imported photograph costs one run rather than a
/// full pass; and it reports the text it just wrote, so somebody watching can tell early whether
/// the output is worth the wait.
/// </para>
/// <para>
/// Sequential, and deliberately using a fraction of the machine — see the thread count on the
/// describer. This is meant to run in the background while the computer is being used for
/// something else.
/// </para>
/// </remarks>
public sealed class DescriptionWorker
{
    private readonly PhotoStore _photos;
    private readonly IPhotoDescriber _describer;
    private readonly string _cacheRoot;
    private readonly ILogger<DescriptionWorker> _log;

    /// <summary>Creates a worker.</summary>
    /// <param name="photos">The photo index.</param>
    /// <param name="describer">The model.</param>
    /// <param name="cacheRoot">Derivative cache; proxies are read from here.</param>
    /// <param name="logger">Optional logger.</param>
    public DescriptionWorker(
        PhotoStore photos,
        IPhotoDescriber describer,
        string cacheRoot,
        ILogger<DescriptionWorker>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(photos);
        ArgumentNullException.ThrowIfNull(describer);

        _photos = photos;
        _describer = describer;
        _cacheRoot = cacheRoot;
        _log = logger ?? NullLogger<DescriptionWorker>.Instance;
    }

    /// <summary>Describes every photograph that has no description yet.</summary>
    /// <param name="limit">Most photographs to describe in this run.</param>
    /// <param name="progress">Optional progress sink.</param>
    /// <param name="cancellationToken">Stops the run. Finished photographs are kept.</param>
    /// <returns>How many photographs were described.</returns>
    public async Task<int> RunAsync(
        int limit = 100000,
        IProgress<DescriptionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!_describer.IsAvailable)
        {
            _log.LogInformation("No vision-language model installed; nothing to describe");
            return 0;
        }

        IReadOnlyList<long> pending = await _photos
            .UndescribedAsync(limit, cancellationToken).ConfigureAwait(false);

        int done = 0;

        foreach (long photoId in pending)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            PhotoRecord? photo = await _photos.GetAsync(photoId, cancellationToken).ConfigureAwait(false);
            if (photo is null)
            {
                continue;
            }

            progress?.Report(new DescriptionProgress(pending.Count, done, photo.FileName, null));

            try
            {
                // The proxy: the model scales its input down to a few hundred pixels anyway, so
                // decoding a 45-megapixel original would cost more than the description.
                string source = photo.ProxyKey is { } proxy
                    ? Path.Combine(_cacheRoot, proxy.Replace('/', Path.DirectorySeparatorChar))
                    : photo.OriginalPath;

                if (!File.Exists(source))
                {
                    source = photo.OriginalPath;
                }

                string? description = await _describer
                    .DescribeAsync(source, cancellationToken).ConfigureAwait(false);

                if (!string.IsNullOrWhiteSpace(description))
                {
                    await _photos.SetDescriptionAsync(
                        photo.Id, description, _describer.ModelVersion, cancellationToken)
                        .ConfigureAwait(false);

                    progress?.Report(new DescriptionProgress(
                        pending.Count, done + 1, photo.FileName, description));
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                // One unreadable photograph must not stop a run that takes half an hour. It
                // simply has no description, which is where it started.
                _log.LogWarning(ex, "Could not describe {File}", photo.FileName);
            }

            done++;
        }

        progress?.Report(new DescriptionProgress(pending.Count, done, null, null));
        return done;
    }
}
