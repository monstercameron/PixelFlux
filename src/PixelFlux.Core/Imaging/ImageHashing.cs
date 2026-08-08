using System.Globalization;
using System.Security.Cryptography;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Core.Imaging;

/// <summary>
/// The two hashes every photo carries, and the comparison between them.
/// </summary>
/// <remarks>
/// They answer different questions and a photo manager needs both.
/// <list type="bullet">
/// <item><description>
/// <b>Content hash</b> (SHA-256 of the bytes) answers <em>is this the same file?</em> It is the
/// photo's identity across devices, and it is what makes ingestion idempotent — re-scanning a
/// folder finds the same hashes and inserts nothing.
/// </description></item>
/// <item><description>
/// <b>Perceptual hash</b> (difference hash of the pixels) answers <em>does this look like
/// that?</em> It is what catches the re-export at 80% quality, the burst sequence, and the same
/// shot saved as both JPEG and PNG — none of which share a content hash.
/// </description></item>
/// </list>
/// </remarks>
public static class ImageHashing
{
    /// <summary>Edge length of the difference-hash sample grid. 8 yields a 64-bit hash.</summary>
    private const int HashSize = 8;

    /// <summary>
    /// Computes the lowercase hex SHA-256 of a file's bytes.
    /// </summary>
    /// <param name="path">Absolute path to the file.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>64 lowercase hex characters.</returns>
    /// <remarks>
    /// Streamed rather than buffered: a 60 MB raw file should not become 60 MB of managed heap
    /// during a folder scan that is already running several files in parallel.
    /// </remarks>
    public static async Task<string> ComputeContentHashAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        await using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete,
            bufferSize: 1 << 20, FileOptions.Asynchronous | FileOptions.SequentialScan);

        byte[] digest = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexStringLower(digest);
    }

    /// <summary>
    /// Computes the difference hash of an already-decoded image.
    /// </summary>
    /// <param name="image">The decoded image. Not modified.</param>
    /// <returns>16 lowercase hex characters encoding 64 bits.</returns>
    /// <remarks>
    /// <para>
    /// The algorithm: reduce to <c>(8+1) x 8</c> greyscale, then emit one bit per horizontal
    /// neighbour pair — 1 where the left pixel is brighter. Because every bit is a
    /// <em>comparison</em> rather than a value, the result is inherently immune to uniform
    /// brightness and contrast changes, which is exactly what separates "re-exported" from
    /// "different photo".
    /// </para>
    /// <para>
    /// Two properties are worth knowing before relying on it. It is <b>not</b> rotation
    /// invariant — a rotated copy hashes as an unrelated image, which is deliberate, since a
    /// rotated version is usually a distinct edit worth keeping. And it is sensitive to
    /// per-pixel noise in flat regions: on a smooth gradient, neighbouring pixels are nearly
    /// equal and small noise flips comparisons. Both behaviours are pinned by tests against the
    /// fixture album.
    /// </para>
    /// </remarks>
    public static string ComputePerceptualHash(Image image)
    {
        ArgumentNullException.ThrowIfNull(image);

        using Image<L8> sample = image.CloneAs<L8>();
        sample.Mutate(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new Size(HashSize + 1, HashSize),
            Mode = ResizeMode.Stretch,      // ignore aspect: the grid must be exactly 9x8
            Sampler = KnownResamplers.Lanczos3,
        }));

        ulong bits = 0;
        for (int y = 0; y < HashSize; y++)
        {
            for (int x = 0; x < HashSize; x++)
            {
                bits <<= 1;
                if (sample[x, y].PackedValue > sample[x + 1, y].PackedValue)
                {
                    bits |= 1;
                }
            }
        }

        return bits.ToString("x16", CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Counts differing bits between two perceptual hashes.
    /// </summary>
    /// <param name="a">First hash, 16 hex characters.</param>
    /// <param name="b">Second hash, 16 hex characters.</param>
    /// <returns>0 (identical) to 64 (fully inverted).</returns>
    /// <exception cref="FormatException">Either argument is not a 64-bit hex hash.</exception>
    public static int Distance(string a, string b)
        => System.Numerics.BitOperations.PopCount(
            ulong.Parse(a, NumberStyles.HexNumber, CultureInfo.InvariantCulture) ^
            ulong.Parse(b, NumberStyles.HexNumber, CultureInfo.InvariantCulture));

    /// <summary>
    /// Distance at or below which two images are treated as near-duplicates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measured, not guessed. Against the fixture album, using <em>this</em> implementation:
    /// a genuine three-frame burst spans 3-5 bits, and the closest pair of unrelated scenes
    /// sits at 8. Six is the only value with clearance on both sides.
    /// </para>
    /// <para>
    /// <b>The margin is narrow and that is worth knowing.</b> One bit above the widest burst,
    /// two below the nearest false positive. Two things squeeze it, and only one is a real
    /// limitation:
    /// </para>
    /// <list type="number">
    /// <item><description>
    /// The fixture scenes are synthetic and low-entropy — flat colour bands with little
    /// high-frequency detail. The pair that sits at 8 is a beach and a lake, both "sky band over
    /// water band", which at an 8x8 sample really are similar. Photographs carry far more
    /// structure, so the false-positive floor on a real library is higher than 8 and the true
    /// margin is wider than this measurement suggests.
    /// </description></item>
    /// <item><description>
    /// A 64-bit hash is simply not very discriminating. Moving to a 12x12 sample (144 bits)
    /// would separate these cases comfortably, at the cost of no longer fitting in a
    /// <see cref="ulong"/>. That is the upgrade path if false positives ever show up on a real
    /// library; the threshold and this comment should be re-measured together if it is taken.
    /// </description></item>
    /// </list>
    /// <para>
    /// Do not change this constant without re-running
    /// <c>PerceptualHashCalibrationTests.Threshold_SeparatesBurstsFromUnrelatedScenes</c>, which
    /// fails if the value stops separating the two populations.
    /// </para>
    /// </remarks>
    public const int NearDuplicateThreshold = 6;

    /// <summary>Whether two perceptual hashes are within <see cref="NearDuplicateThreshold"/>.</summary>
    /// <param name="a">First hash.</param>
    /// <param name="b">Second hash.</param>
    /// <returns><see langword="true"/> when the images are probably the same shot.</returns>
    public static bool AreNearDuplicates(string a, string b) => Distance(a, b) <= NearDuplicateThreshold;
}
