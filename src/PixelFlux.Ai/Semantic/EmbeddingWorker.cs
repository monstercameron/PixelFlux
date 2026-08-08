using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using PixelFlux.Core.Search;

namespace PixelFlux.Ai.Semantic;

/// <summary>Progress from an embedding run.</summary>
/// <param name="Total">Photographs queued when the run started.</param>
/// <param name="Done">Photographs finished.</param>
/// <param name="Current">Name of the photograph being described.</param>
public readonly record struct EmbeddingProgress(int Total, int Done, string? Current);

/// <summary>
/// Describes every photograph in the library so it can be searched by meaning.
/// </summary>
/// <remarks>
/// <para>
/// Sequential and resumable, like the other sweeps: ONNX Runtime already spreads one inference
/// across several cores, and this runs on a machine somebody is using. Each vector is committed
/// as it is produced, so stopping the run keeps the work already done.
/// </para>
/// <para>
/// Only photographs that have no vector from the current model are visited. That makes the run
/// idempotent, makes a model upgrade a background migration rather than a stop-the-world
/// rebuild, and means a newly imported photograph costs one inference rather than a full pass.
/// </para>
/// </remarks>
public sealed class EmbeddingWorker
{
    private readonly PhotoStore _photos;
    private readonly VectorIndex _vectors;
    private readonly IImageTextEmbedder _embedder;
    private readonly string _cacheRoot;
    private readonly ILogger<EmbeddingWorker> _log;

    /// <summary>Creates a worker.</summary>
    /// <param name="photos">The photo index.</param>
    /// <param name="vectors">Where vectors are written.</param>
    /// <param name="embedder">The model.</param>
    /// <param name="cacheRoot">Derivative cache; proxies are read from here.</param>
    /// <param name="logger">Optional logger.</param>
    public EmbeddingWorker(
        PhotoStore photos,
        VectorIndex vectors,
        IImageTextEmbedder embedder,
        string cacheRoot,
        ILogger<EmbeddingWorker>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(photos);
        ArgumentNullException.ThrowIfNull(vectors);
        ArgumentNullException.ThrowIfNull(embedder);

        _photos = photos;
        _vectors = vectors;
        _embedder = embedder;
        _cacheRoot = cacheRoot;
        _log = logger ?? NullLogger<EmbeddingWorker>.Instance;
    }

    /// <summary>Describes every photograph that has no vector yet.</summary>
    /// <param name="limit">Most photographs to process in this run.</param>
    /// <param name="progress">Optional progress sink.</param>
    /// <param name="cancellationToken">Stops the run. Finished photographs are kept.</param>
    /// <returns>How many photographs were described.</returns>
    public async Task<int> RunAsync(
        int limit = 100000,
        IProgress<EmbeddingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!_embedder.IsAvailable)
        {
            _log.LogInformation("CLIP is not installed; nothing to embed");
            return 0;
        }

        IReadOnlyList<long> pending = await _vectors
            .PendingAsync(_embedder.ModelVersion, limit, cancellationToken).ConfigureAwait(false);

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

            progress?.Report(new EmbeddingProgress(pending.Count, done, photo.FileName));

            try
            {
                // The proxy, not the original: the encoder resizes to 224 square regardless, so
                // decoding a 45-megapixel file would cost several seconds to throw away.
                string source = photo.ProxyKey is { } proxy
                    ? Path.Combine(_cacheRoot, proxy.Replace('/', Path.DirectorySeparatorChar))
                    : photo.OriginalPath;

                if (!File.Exists(source))
                {
                    source = photo.OriginalPath;
                }

                if (await _embedder.EmbedImageAsync(source, cancellationToken).ConfigureAwait(false)
                    is { } vector)
                {
                    await _vectors.StoreAsync(photo.Id, _embedder.ModelVersion, vector, cancellationToken)
                        .ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                // One unreadable photograph must not stop the run. It simply has no vector, and
                // search by meaning will not find it — which is the same position it was in
                // before this ran.
                _log.LogWarning(ex, "Could not describe {File}", photo.FileName);
            }

            done++;
        }

        progress?.Report(new EmbeddingProgress(pending.Count, done, null));
        return done;
    }
}
