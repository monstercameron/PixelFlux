using System.Globalization;
using System.Text;
using PixelFlux.Core.Imaging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using Xunit.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// Calibration for <see cref="ImageHashing.NearDuplicateThreshold"/>, measured with the
/// production hash against the fixture album.
///
/// This exists because the threshold was very nearly set from the wrong measurements. The
/// Python fixture verifier reimplements the difference hash in order to sanity-check the
/// corpus, and the two implementations do not agree: PIL converts to greyscale with BT.601
/// luma coefficients while ImageSharp uses BT.709, so saturated scenes land several bits
/// apart. A threshold tuned against the Python numbers is tuned against a hash the application
/// never runs.
///
/// The rule this file enforces: the threshold must sit strictly between the widest burst and
/// the closest unrelated pair, as measured by the C# implementation, with real margin on both
/// sides.
/// </summary>
public sealed class PerceptualHashCalibrationTests
{
    private readonly ITestOutputHelper _output;

    public PerceptualHashCalibrationTests(ITestOutputHelper output) => _output = output;

    private static string AlbumPath
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "testdata", "album")))
            {
                dir = dir.Parent;
            }

            return dir is null
                ? throw new DirectoryNotFoundException("Could not locate testdata/album.")
                : Path.Combine(dir.FullName, "testdata", "album");
        }
    }

    private static Dictionary<string, string> HashAlbum()
    {
        var hashes = new Dictionary<string, string>(StringComparer.Ordinal);

        // Recursive: the album is organised into year folders, so a flat enumeration finds
        // nothing at all and every calibration test silently degrades to "no data".
        foreach (string file in Directory
                     .EnumerateFiles(AlbumPath, "*", SearchOption.AllDirectories)
                     .OrderBy(f => f, StringComparer.Ordinal))
        {
            if (file.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            try
            {
                using Image image = Image.Load(file);
                hashes[Path.GetFileName(file)] = ImageHashing.ComputePerceptualHash(image);
            }
            catch (Exception ex) when (ex is ImageFormatException or NotSupportedException or InvalidOperationException)
            {
                // The deliberately truncated file. Excluded from calibration, as it has no pixels.
                // ImageFormatException is the base of both UnknownImageFormat and
                // InvalidImageContent; catching only the leaves missed the actual type thrown
                // for a half-written JPEG.
            }
        }

        return hashes;
    }

    [Fact]
    public void Threshold_SeparatesBurstsFromUnrelatedScenes()
    {
        Dictionary<string, string> hashes = HashAlbum();

        string[] burst = hashes.Keys.Where(k => k.Contains("burst", StringComparison.Ordinal)).ToArray();
        // "Unrelated" must exclude every file that is deliberately the same photograph:
        // the burst frames, the byte-identical duplicates, and the re-encoded copies. The
        // re-encodes were missed in the first version and produced a closest-unrelated distance
        // of 0 — a PNG of photo 000 is not an unrelated photo, it IS photo 000.
        string[] unrelated = hashes.Keys
            .Where(k => !k.Contains("burst", StringComparison.Ordinal)
                     && !k.Contains("duplicate", StringComparison.Ordinal)
                     && !k.Contains("reencoded", StringComparison.Ordinal))
            .ToArray();

        Assert.Equal(3, burst.Length);

        int[] burstDistances = [.. from i in Enumerable.Range(0, burst.Length)
                                   from j in Enumerable.Range(i + 1, burst.Length - i - 1)
                                   select ImageHashing.Distance(hashes[burst[i]], hashes[burst[j]])];

        var unrelatedPairs = (from i in Enumerable.Range(0, unrelated.Length)
                              from j in Enumerable.Range(i + 1, unrelated.Length - i - 1)
                              let d = ImageHashing.Distance(hashes[unrelated[i]], hashes[unrelated[j]])
                              orderby d
                              select (Distance: d, A: unrelated[i], B: unrelated[j])).ToArray();

        int widestBurst = burstDistances.Max();
        int closestUnrelated = unrelatedPairs[0].Distance;

        var report = new StringBuilder()
            .AppendLine(CultureInfo.InvariantCulture, $"burst distances     : {string.Join(", ", burstDistances.Order())}")
            .AppendLine(CultureInfo.InvariantCulture, $"widest burst        : {widestBurst}")
            .AppendLine(CultureInfo.InvariantCulture, $"closest unrelated   : {closestUnrelated}")
            .AppendLine(CultureInfo.InvariantCulture, $"configured threshold: {ImageHashing.NearDuplicateThreshold}")
            .AppendLine("closest unrelated pairs:");
        foreach ((int d, string a, string b) in unrelatedPairs.Take(6))
        {
            report.AppendLine(CultureInfo.InvariantCulture, $"  {d,2}  {a} <-> {b}");
        }

        _output.WriteLine(report.ToString());

        // The threshold must catch every burst frame...
        Assert.True(
            ImageHashing.NearDuplicateThreshold > widestBurst,
            $"threshold {ImageHashing.NearDuplicateThreshold} would split a real burst (widest {widestBurst}).\n{report}");

        // ...and reject every unrelated pair.
        Assert.True(
            ImageHashing.NearDuplicateThreshold < closestUnrelated,
            $"threshold {ImageHashing.NearDuplicateThreshold} would merge unrelated photos "
            + $"(closest {closestUnrelated}).\n{report}");

        // The two populations must stay separable at all — if the widest burst ever reaches the
        // closest unrelated pair, no threshold works and the hash itself needs to get wider
        // (see the note on NearDuplicateThreshold about moving to a 144-bit sample).
        Assert.True(
            closestUnrelated - widestBurst >= 2,
            $"burst and unrelated populations have collapsed into each other "
            + $"(widest burst {widestBurst}, closest unrelated {closestUnrelated}); "
            + $"no threshold can separate them.\n{report}");
    }

    [Fact]
    public void Hash_IsStableAcrossReEncoding()
    {
        // The property that makes a perceptual hash worth having: a re-encode at lower quality
        // must not move it. If this drifts, duplicate detection silently stops working on
        // exactly the files people have most copies of.
        string source = Directory
            .EnumerateFiles(AlbumPath, "*.jpg", SearchOption.AllDirectories)
            .OrderBy(f => f, StringComparer.Ordinal).First();

        using Image original = Image.Load(source);
        string before = ImageHashing.ComputePerceptualHash(original);

        using var buffer = new MemoryStream();
        original.SaveAsJpeg(buffer, new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder { Quality = 40 });
        buffer.Position = 0;

        using Image reencoded = Image.Load(buffer);
        string after = ImageHashing.ComputePerceptualHash(reencoded);

        int drift = ImageHashing.Distance(before, after);
        _output.WriteLine($"quality 40 re-encode drift: {drift} bits");
        Assert.True(drift <= 2, $"re-encoding moved the hash by {drift} bits ({before} -> {after})");
    }

    [Theory]
    [InlineData(0.88f)]   // darker
    [InlineData(1.12f)]   // brighter
    public void ExposureAdjustedCopy_IsStillRecognisedAsTheSamePhoto(float factor)
    {
        // The textbook claim is that a difference hash is *immune* to exposure changes, because
        // every bit compares two neighbours rather than storing a value. Two earlier versions of
        // this test asserted that and both failed, for two different reasons — which is worth
        // recording, because the folklore is wrong in a way that matters:
        //
        //   * brightening clips. Once pixels saturate at 255 the transform stops being
        //     monotonic, two neighbours that differed both pin to the same value, and their
        //     comparison bit becomes arbitrary.
        //   * darkening quantises. Scaling to 8-bit integers collapses neighbours that differed
        //     by one into equality, flipping those bits too.
        //
        // Measured here: 3-4 bits in each direction on the fixtures. So the honest property is
        // robustness, not immunity — and the thing actually worth asserting is not an arbitrary
        // bit count but the product requirement: an exposure-adjusted copy must still be caught
        // as a near-duplicate. Tying the assertion to the threshold constant also means the two
        // can never drift apart silently.
        string source = Directory
            .EnumerateFiles(AlbumPath, "*.jpg", SearchOption.AllDirectories)
            .OrderBy(f => f, StringComparer.Ordinal).First();

        using Image<SixLabors.ImageSharp.PixelFormats.Rgba32> image =
            Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(source);
        string before = ImageHashing.ComputePerceptualHash(image);

        image.Mutate(ctx => ctx.Brightness(factor));
        string after = ImageHashing.ComputePerceptualHash(image);

        int drift = ImageHashing.Distance(before, after);
        _output.WriteLine(
            $"brightness x{factor}: drift {drift} bits, threshold {ImageHashing.NearDuplicateThreshold}");

        Assert.True(
            ImageHashing.AreNearDuplicates(before, after),
            $"an exposure-adjusted copy drifted {drift} bits, past the "
            + $"{ImageHashing.NearDuplicateThreshold}-bit near-duplicate threshold — "
            + "edited copies would stop being recognised as the same photo.");
    }
}
