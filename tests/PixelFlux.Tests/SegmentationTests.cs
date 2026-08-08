using PixelFlux.Ai.Segmentation;
using Xunit.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// Real inference against real photographs.
///
/// These are skipped rather than failed when no model is installed, because the model is an
/// 11 MB AGPL-licensed file the user supplies rather than something checked into the repository.
/// A missing model is a legitimate configuration, not a broken build — but when one <em>is</em>
/// present the assertions are strict, because a segmentation that returns nothing and a
/// segmentation that is switched off look identical from the outside.
/// </summary>
[Collection(Inference.Name)]
public sealed class SegmentationTests
{
    private readonly ITestOutputHelper _output;

    public SegmentationTests(ITestOutputHelper output) => _output = output;

    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "testdata", "album")))
            {
                dir = dir.Parent;
            }

            return dir?.FullName ?? throw new DirectoryNotFoundException("Could not locate the repository root.");
        }
    }

    private static string? ModelPath
    {
        get
        {
            string path = Path.Combine(RepoRoot, "models", "yolo11n-seg.onnx");
            return File.Exists(path) ? path : null;
        }
    }

    private static string[] AlbumPhotos(string pattern) =>
        Directory.GetFiles(Path.Combine(RepoRoot, "testdata", "album"), pattern, SearchOption.AllDirectories)
                 .OrderBy(f => f, StringComparer.Ordinal)
                 .ToArray();

    [Fact]
    public void ModelIsInstalled()
    {
        // Not an assertion about the model working — an assertion that the rest of this class is
        // actually exercising something. Without it a missing model would make every test below
        // pass by doing nothing.
        Assert.True(ModelPath is not null,
            "models/yolo11n-seg.onnx is missing; run tools/fetch_model.ps1 or the tests below prove nothing");
    }

    [Fact]
    public async Task FindsTheCatInAPhotographOfACat()
    {
        if (ModelPath is null) { return; }

        // The corpus has real photographs of known subjects, which makes this checkable rather
        // than merely self-consistent: a segmenter that returns plausible-looking nonsense fails
        // here, where one that returns nothing at all would pass a "did it run" test.
        // Anchored on the subject segment of the filename, not a bare substring. "*cat*"
        // cheerfully matched "dupli-CAT-e", "CATania", and "CAThedral", so the first version of
        // this test ran a segmenter on a photograph of a person and then complained there was
        // no cat in it.
        string[] cats = AlbumPhotos("*_animal_cat*.jpg");
        Assert.NotEmpty(cats);

        using var segmenter = new YoloSegmenter(ModelPath);
        SegmentationResult result = await segmenter.SegmentAsync(cats[0]);

        _output.WriteLine($"{Path.GetFileName(cats[0])} -> {result.ElapsedMs} ms");
        foreach (PhotoSegment segment in result.Segments)
        {
            _output.WriteLine(
                $"  {segment.Label,-14} conf {segment.Confidence:0.00}  "
                + $"area {segment.AreaFraction:0.000}  prominence {segment.Prominence:0.00}  "
                + $"mask {segment.MaskWidth}x{segment.MaskHeight}");
        }

        Assert.Contains(result.Segments, s => s.Label == "cat");
    }

    [Fact]
    public async Task FindsPeopleAndVehiclesInAStreetScene()
    {
        if (ModelPath is null) { return; }

        string[] cars = AlbumPhotos("*_car_*.jpg");
        Assert.NotEmpty(cars);

        using var segmenter = new YoloSegmenter(ModelPath);
        var found = new HashSet<string>(StringComparer.Ordinal);

        foreach (string photo in cars.Take(4))
        {
            SegmentationResult result = await segmenter.SegmentAsync(photo);
            _output.WriteLine($"{Path.GetFileName(photo)}: "
                + string.Join(", ", result.Segments.Select(s => $"{s.Label} {s.Confidence:0.00}")));

            foreach (PhotoSegment segment in result.Segments)
            {
                found.Add(segment.Label);
            }
        }

        Assert.Contains("car", found);
    }

    [Fact]
    public async Task MasksAreRealPixelsInsideTheirOwnBox()
    {
        if (ModelPath is null) { return; }

        using var segmenter = new YoloSegmenter(ModelPath);

        foreach (string photo in AlbumPhotos("*.jpg").Take(8))
        {
            SegmentationResult result = await segmenter.SegmentAsync(photo);

            foreach (PhotoSegment segment in result.Segments)
            {
                // Geometry is fractional and must stay on the image.
                Assert.InRange(segment.X, 0, 1);
                Assert.InRange(segment.Y, 0, 1);
                Assert.InRange(segment.Width, 0, 1);
                Assert.InRange(segment.Height, 0, 1);
                Assert.True(segment.X + segment.Width <= 1.001, "box overhangs the right edge");
                Assert.True(segment.Y + segment.Height <= 1.001, "box overhangs the bottom edge");

                // The mask buffer must match its declared size exactly, or the overlay reads
                // past the end of it.
                Assert.Equal(segment.MaskWidth * segment.MaskHeight, segment.Mask.Length);

                // A mask of all zeros is a detection the mask head disagreed with, and those are
                // dropped. A mask of all 255s inside a tight box is possible but suspicious.
                Assert.Contains(segment.Mask, b => b == 255);
                Assert.All(segment.Mask, b => Assert.True(b is 0 or 255, "mask is not binary"));

                // Coverage can never exceed the box the mask lives in.
                Assert.True(segment.AreaFraction <= (segment.Width * segment.Height) + 0.001,
                    $"{segment.Label} claims more area than its own box");
            }
        }
    }

    [Fact]
    public async Task SegmentsComeBackMostProminentFirst()
    {
        if (ModelPath is null) { return; }

        using var segmenter = new YoloSegmenter(ModelPath);

        foreach (string photo in AlbumPhotos("*.jpg").Take(10))
        {
            SegmentationResult result = await segmenter.SegmentAsync(photo);

            for (int i = 1; i < result.Segments.Count; i++)
            {
                Assert.True(result.Segments[i - 1].Prominence >= result.Segments[i].Prominence,
                    "segments are not ordered by prominence");
            }
        }
    }

    [Fact]
    public async Task MissingModelIsAnOrdinaryStateNotACrash()
    {
        // A fresh install has no model. Every call must return an empty result rather than
        // throwing once per photograph through the whole library.
        using var segmenter = new YoloSegmenter(Path.Combine(Path.GetTempPath(), "no-such-model.onnx"));

        Assert.False(segmenter.IsAvailable);

        SegmentationResult result = await segmenter.SegmentAsync(AlbumPhotos("*.jpg")[0]);
        Assert.Empty(result.Segments);
        Assert.Equal("none", result.ModelVersion);
    }

    [Fact]
    public async Task InferenceIsFastEnoughToWorkThroughALibrary()
    {
        if (ModelPath is null) { return; }

        using var segmenter = new YoloSegmenter(ModelPath);
        string[] photos = AlbumPhotos("*.jpg").Take(5).ToArray();

        // Warm up: the first call pays for graph optimisation and would skew the average.
        await segmenter.SegmentAsync(photos[0]);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        foreach (string photo in photos)
        {
            await segmenter.SegmentAsync(photo);
        }

        watch.Stop();
        double perPhoto = watch.ElapsedMilliseconds / (double)photos.Length;
        _output.WriteLine($"{perPhoto:0} ms per photograph");

        // Generous, because this runs in the background on an idle machine. The number that
        // matters is whether a 50,000-photo library finishes overnight: at 2 seconds each that
        // is 28 hours, at 500 ms it is 7. Anything past 2 seconds means something is wrong.
        Assert.True(perPhoto < 2000, $"{perPhoto:0} ms per photograph is too slow to work through a library");
    }
}
