namespace PixelFlux.Ai.Segmentation;

/// <summary>One thing the model found in a photograph, with the pixels it occupies.</summary>
/// <param name="Label">Class name, for example <c>dog</c>.</param>
/// <param name="Confidence">Detector confidence, 0-1.</param>
/// <param name="X">Left edge of the bounding box as a fraction of image width.</param>
/// <param name="Y">Top edge as a fraction of image height.</param>
/// <param name="Width">Box width as a fraction of image width.</param>
/// <param name="Height">Box height as a fraction of image height.</param>
/// <param name="AreaFraction">
/// Share of the whole image this segment's mask actually covers, 0-1.
/// </param>
/// <param name="Mask">
/// The mask itself, one byte per pixel (0 or 255), laid out row-major at
/// <paramref name="MaskWidth"/> x <paramref name="MaskHeight"/> and covering the bounding box
/// only — not the whole image.
/// </param>
/// <param name="MaskWidth">Mask width in pixels.</param>
/// <param name="MaskHeight">Mask height in pixels.</param>
/// <remarks>
/// <para>
/// Coordinates are fractions rather than pixels so a segment stays valid against the thumbnail,
/// the proxy, and the original without a conversion at every use — the same reason
/// <c>DetectedObject</c> works that way.
/// </para>
/// <para>
/// <see cref="AreaFraction"/> is the mask's true coverage, not the box's. The distinction
/// matters for ranking: a giraffe and a hoop can have identically sized bounding boxes while one
/// fills it and the other is mostly air, and "prominent" should mean the one that actually
/// occupies the frame.
/// </para>
/// </remarks>
public sealed record PhotoSegment(
    string Label,
    double Confidence,
    double X,
    double Y,
    double Width,
    double Height,
    double AreaFraction,
    byte[] Mask,
    int MaskWidth,
    int MaskHeight)
{
    /// <summary>
    /// How prominent this segment is in the photograph, 0-1.
    /// </summary>
    /// <remarks>
    /// Blends how much of the frame it covers with how sure the model is, weighted towards
    /// area. A confident detection of something tiny in a corner is not what a person means by
    /// "what is in this photo"; a large, reasonably-confident subject is. The square root on
    /// area stops one enormous background segment — a wall, a field — from flattening
    /// everything else to zero.
    /// </remarks>
    public double Prominence => (Math.Sqrt(AreaFraction) * 0.7) + (Confidence * 0.3);
}

/// <summary>Everything a segmentation pass found in one photograph.</summary>
/// <param name="Segments">Segments, most prominent first.</param>
/// <param name="ModelVersion">Identifier of the model that produced them.</param>
/// <param name="ElapsedMs">Wall-clock inference time, for the scheduler and for diagnostics.</param>
public sealed record SegmentationResult(
    IReadOnlyList<PhotoSegment> Segments,
    string ModelVersion,
    long ElapsedMs)
{
    /// <summary>An empty result, for when no model is installed or nothing was found.</summary>
    /// <param name="modelVersion">Model identifier, or a marker such as <c>none</c>.</param>
    /// <returns>A result with no segments.</returns>
    public static SegmentationResult Empty(string modelVersion = "none")
        => new([], modelVersion, 0);
}

/// <summary>
/// Finds and outlines the prominent things in a photograph.
/// </summary>
/// <remarks>
/// An interface so that the model can be replaced without touching storage or the UI — which
/// matters more than usual here, because the only good open segmentation models carry awkward
/// licences and the choice may well change. Implementations must be safe to call concurrently.
/// </remarks>
public interface ISegmenter
{
    /// <summary>Whether a usable model is actually installed.</summary>
    /// <remarks>
    /// False is an ordinary state, not an error: a fresh install has no model until one is
    /// downloaded. Callers check this rather than catching an exception per photograph.
    /// </remarks>
    bool IsAvailable { get; }

    /// <summary>Identifier of the model in use, for recording against each photo's results.</summary>
    string ModelVersion { get; }

    /// <summary>Segments one image file.</summary>
    /// <param name="imagePath">Absolute path to the image to analyse.</param>
    /// <param name="cancellationToken">Cancels inference.</param>
    /// <returns>What was found, or an empty result when no model is installed.</returns>
    Task<SegmentationResult> SegmentAsync(string imagePath, CancellationToken cancellationToken = default);
}
