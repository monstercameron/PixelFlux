using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using PixelFlux.Core.Pipeline;

namespace PixelFlux.Ai.Pipeline;

/// <summary>
/// The stage that has a vision model look at a photograph and write about it.
/// </summary>
/// <remarks>
/// First in the queue and by far the most expensive — roughly sixteen seconds a photograph against
/// milliseconds for everything else — which is why it is also the stage the cache matters most for.
/// Its output is the only stage output another stage reads: <see cref="EmbedHandler"/> folds the
/// description into the search vector, so a photograph described is a photograph findable by words
/// that appear nowhere in its filename, its tags, or its detected objects.
/// </remarks>
public sealed class DescribeHandler : IStageHandler
{
    private readonly PhotoStore _photos;
    private readonly IPhotoDescriber _describer;
    private readonly string _cacheRoot;

    /// <summary>Creates the handler.</summary>
    /// <param name="photos">The photo index.</param>
    /// <param name="describer">The vision model.</param>
    /// <param name="cacheRoot">Derivative cache, where proxies live.</param>
    public DescribeHandler(PhotoStore photos, IPhotoDescriber describer, string cacheRoot)
    {
        ArgumentNullException.ThrowIfNull(photos);
        ArgumentNullException.ThrowIfNull(describer);

        _photos = photos;
        _describer = describer;
        _cacheRoot = cacheRoot;
    }

    /// <inheritdoc/>
    public PipelineStage Stage => PipelineStage.Describe;

    /// <inheritdoc/>
    public string? ModelVersion => _describer.IsAvailable ? _describer.ModelVersion : null;

    /// <inheritdoc/>
    public async Task ApplyAsync(long photoId, string payload, CancellationToken cancellationToken)
    {
        // The payload is the description itself. No envelope, because there is exactly one field
        // and wrapping a paragraph in JSON to store it in a text column would be ceremony.
        await _photos.SetDescriptionAsync(photoId, payload, _describer.ModelVersion,
            cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task<string?> ExecuteAsync(long photoId, CancellationToken cancellationToken)
    {
        PhotoRecord? photo = await _photos.GetAsync(photoId, cancellationToken)
            .ConfigureAwait(false);
        if (photo is null)
        {
            return null;
        }

        string source = StageSource.For(photo, _cacheRoot);
        string? description = await _describer.DescribeAsync(source, cancellationToken)
            .ConfigureAwait(false);

        if (string.IsNullOrWhiteSpace(description))
        {
            // A model that returned nothing is not the same as a model that failed. Returning null
            // marks the photograph skipped rather than burning a retry on an answer that will be
            // just as empty next time.
            return null;
        }

        await ApplyAsync(photoId, description, cancellationToken).ConfigureAwait(false);
        return description;
    }
}

/// <summary>Picks the image file a stage should actually read.</summary>
/// <remarks>
/// Every model here resizes its input to a few hundred pixels square before doing anything, so
/// handing one a 45-megapixel original spends more time in the JPEG decoder than in the network.
/// The proxy is already on disk, already upright, and already the right order of magnitude — and
/// falling back to the original when it is missing means a library whose derivative cache has been
/// cleared still analyses, just more slowly.
/// </remarks>
public static class StageSource
{
    /// <summary>The best available file to analyse for a photograph.</summary>
    /// <param name="photo">The photograph.</param>
    /// <param name="cacheRoot">Root of the derivative cache.</param>
    /// <returns>An absolute path, which may still not exist if the original has moved.</returns>
    public static string For(PhotoRecord photo, string cacheRoot)
    {
        ArgumentNullException.ThrowIfNull(photo);

        if (photo.ProxyKey is not { } proxy)
        {
            return photo.OriginalPath;
        }

        string candidate = Path.Combine(
            cacheRoot, proxy.Replace('/', Path.DirectorySeparatorChar));
        return File.Exists(candidate) ? candidate : photo.OriginalPath;
    }
}
