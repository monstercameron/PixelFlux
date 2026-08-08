using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;

namespace PixelFlux.Ai.Compute;

/// <summary>
/// Loads a photograph at roughly the size a model is about to shrink it to anyway.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this is the optimisation that matters.</b> Face detection over the test album costs about
/// 114 ms a photograph end to end. Of that, the network is under 10 ms — measured, and the same
/// sweep takes the same time whether it runs on the processor or on DirectML, because inference is
/// not where the time goes. Decoding a four-megapixel JPEG and scaling it to 640 square is.
/// </para>
/// <para>
/// JPEG is stored as 8x8 blocks of frequency coefficients, so a decoder can reconstruct at 1/2,
/// 1/4 or 1/8 scale by simply ignoring the high-frequency ones — a fraction of the work of
/// decoding fully and then resampling. Every model here immediately shrinks its input to 640 or
/// 224 square, so the full-size pixels are decoded, resized, and discarded. Asking the decoder for
/// the size we actually want skips that entirely.
/// </para>
/// <para>
/// It is a hint, not a promise. The decoder picks the smallest scale that is still at least the
/// requested size, so the result is never smaller than asked for and is often somewhat larger; the
/// caller resizes to the exact input as before. Formats without scaled decoding — PNG, WebP —
/// ignore it and load as they always did.
/// </para>
/// <para>
/// <b>It is not entirely free, and the cost is measured.</b> A full decode followed by a proper
/// resample keeps high-frequency detail that discarding coefficients does not, and face detection
/// notices: across the 132-photograph test library, full decode finds 246 faces in 82.0 s and this
/// finds 244 in 25.1 s. Two faces, 0.8%, for 3.3x. Both were marginal — the grouping is unchanged
/// at the same people — and for a sweep that runs unattended over a whole library that is the
/// right way round. It is written down here because it is a real trade and not an obvious one.
/// </para>
/// <para>
/// The measurement that found it was nearly missed. Comparing a 640 target against a 1280 target
/// gave the same 244 both times, which looked like proof that decoding had nothing to do with it —
/// but both are scaled the same way from the same file, so neither was a control. The control is
/// the original code path, and running it is what settled the question.
/// </para>
/// </remarks>
public static class AnalysisImage
{
    /// <summary>Loads an image at no less than the given square size, as cheaply as the format allows.</summary>
    /// <param name="path">The image file.</param>
    /// <param name="target">The size the caller is about to resize to.</param>
    /// <returns>The image, which the caller owns and must dispose.</returns>
    public static Image<Rgb24> Load(string path, int target)
    {
        var options = new DecoderOptions
        {
            // Square, because callers letterbox or centre-crop into a square. A decoder given a
            // square target on a landscape photograph picks the scale that keeps the short edge
            // above the target, which is what both of those need.
            TargetSize = new Size(target, target),
        };

        return Image.Load<Rgb24>(options, path);
    }
}
