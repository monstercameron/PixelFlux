using PixelFlux.Ai.Faces;
using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;
using SixLabors.ImageSharp;
using Xunit.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// Face storage and the sweep, against a real library built from the real corpus.
///
/// The detector has its own tests; these are about everything downstream of it — that faces
/// survive a round trip, that a re-run replaces rather than accumulates, that a photograph with
/// nobody in it is not examined for ever, and that the crops written to the cache are square,
/// present, and centred on something.
/// </summary>
[Collection(Inference.Name)]
public sealed class FaceStoreTests : IAsyncLifetime
{
    private readonly ITestOutputHelper _output;

    private string _workDir = string.Empty;
    private PhotoDatabase _database = null!;
    private PhotoStore _store = null!;
    private FaceStore _faces = null!;

    public FaceStoreTests(ITestOutputHelper output) => _output = output;

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

    private static string AlbumPath => Path.Combine(RepoRoot, "testdata", "album");

    private static string? ModelPath
    {
        get
        {
            string path = Path.Combine(RepoRoot, "models", "face_yunet_2023mar.onnx");
            return File.Exists(path) ? path : null;
        }
    }

    private string CacheRoot => Path.Combine(_workDir, "cache");

    public async Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-faces", Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(_workDir);

        _database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        _database.Migrate();

        _store = new PhotoStore(_database);
        _faces = new FaceStore(_database);

        var ingestor = new LibraryIngestor(_store, new DerivativeGenerator(CacheRoot));
        await ingestor.ImportAsync([AlbumPath]);
    }

    public Task DisposeAsync()
    {
        try
        {
            Directory.Delete(_workDir, recursive: true);
        }
        catch (IOException)
        {
            // A temp directory nobody deleted is not a failing test.
        }

        return Task.CompletedTask;
    }

    private async Task<PhotoRecord> AnyPhotoAsync()
        => (await _store.QueryAsync(new PhotoQuery { Limit = 1 }))[0];

    /// <summary>A unit vector pointing mostly along one axis, for similarity arithmetic.</summary>
    /// <remarks>
    /// Synthetic on purpose. These tests are about storage and ranking, not about whether the
    /// model can tell people apart — that is measured in FaceRecognitionTests against real
    /// photographs. Hand-made vectors let a test state exactly which pairs should match.
    /// </remarks>
    private static float[] Vector(int axis, int dims = 128, float lean = 1f)
    {
        var v = new float[dims];
        v[axis] = lean;
        v[(axis + 1) % dims] = 1 - lean;

        double norm = Math.Sqrt(v.Sum(x => (double)x * x));
        for (int i = 0; i < dims; i++)
        {
            v[i] = (float)(v[i] / norm);
        }

        return v;
    }

    private static PhotoFaceRecord Face(long photoId, double confidence, double x, double y, double size) =>
        new(0, photoId, confidence, x, y, size, size, size * size, 0,
            PhotoFaceRecord.FormatLandmarks([(x + 0.01, y + 0.01), (x + 0.02, y + 0.01),
                                             (x + 0.015, y + 0.02), (x + 0.012, y + 0.03),
                                             (x + 0.018, y + 0.03)]),
            null, "test-model");

    // ------------------------------------------------------------------------ storage

    [Fact]
    public async Task FacesSurviveARoundTrip()
    {
        PhotoRecord photo = await AnyPhotoAsync();

        var written = new PhotoFaceRecord(
            0, photo.Id, 0.91, 0.25, 0.3, 0.12, 0.16, 0.0192, -7.5,
            PhotoFaceRecord.FormatLandmarks([(0.28, 0.34), (0.33, 0.34), (0.305, 0.38), (0.29, 0.42), (0.32, 0.42)]),
            "face/ab/abc-00.jpg",
            "yunet");

        await _faces.ReplaceAsync(photo.Id, [written]);

        IReadOnlyList<PhotoFaceRecord> read = await _faces.GetAsync(photo.Id);
        PhotoFaceRecord got = Assert.Single(read);

        Assert.Equal(written.Confidence, got.Confidence, 6);
        Assert.Equal(written.X, got.X, 6);
        Assert.Equal(written.Height, got.Height, 6);
        Assert.Equal(written.RollDegrees, got.RollDegrees, 6);
        Assert.Equal(written.CropKey, got.CropKey);
        Assert.Equal(written.Model, got.Model);

        // The landmarks must come back as five usable points, not as text that happens to
        // round-trip. This is the assertion that catches a culture-dependent format.
        IReadOnlyList<(double X, double Y)> points = got.ParseLandmarks();
        Assert.Equal(5, points.Count);
        Assert.Equal(0.28, points[0].X, 5);
        Assert.Equal(0.42, points[4].Y, 5);
    }

    [Fact]
    public async Task LandmarksRoundTripUnderACommaDecimalCulture()
    {
        // A machine set to German writes "0,28" for 0.28 unless the formatting is pinned. The
        // text would still parse — into a completely different set of numbers.
        var previous = System.Globalization.CultureInfo.CurrentCulture;
        System.Globalization.CultureInfo.CurrentCulture = new System.Globalization.CultureInfo("de-DE");

        try
        {
            string text = PhotoFaceRecord.FormatLandmarks([(0.28, 0.34), (0.5178, 0.391)]);
            Assert.DoesNotContain(";", text, StringComparison.Ordinal);

            PhotoFaceRecord record = Face(1, 0.9, 0.1, 0.1, 0.1) with { Landmarks = text };
            IReadOnlyList<(double X, double Y)> points = record.ParseLandmarks();

            Assert.Equal(2, points.Count);
            Assert.Equal(0.5178, points[1].X, 5);
        }
        finally
        {
            System.Globalization.CultureInfo.CurrentCulture = previous;
        }
    }

    [Fact]
    public async Task MalformedLandmarksAreEmptyRatherThanFatal()
    {
        PhotoFaceRecord record = Face(1, 0.9, 0.1, 0.1, 0.1) with { Landmarks = "not,a,number" };
        Assert.Empty(record.ParseLandmarks());

        // Odd count is also unusable: points come in pairs.
        Assert.Empty((record with { Landmarks = "0.1,0.2,0.3" }).ParseLandmarks());

        await Task.CompletedTask;
    }

    [Fact]
    public async Task ReplacingDoesNotAccumulate()
    {
        PhotoRecord photo = await AnyPhotoAsync();

        await _faces.ReplaceAsync(photo.Id, [Face(photo.Id, 0.9, 0.1, 0.1, 0.1),
                                             Face(photo.Id, 0.8, 0.4, 0.1, 0.1)]);
        Assert.Equal(2, (await _faces.GetAsync(photo.Id)).Count);

        // A second pass finds one face where the first found two. The row count must follow the
        // second pass exactly — merging would leave a ghost of the first.
        await _faces.ReplaceAsync(photo.Id, [Face(photo.Id, 0.95, 0.1, 0.1, 0.1)]);
        PhotoFaceRecord only = Assert.Single(await _faces.GetAsync(photo.Id));
        Assert.Equal(0.95, only.Confidence, 6);

        await _faces.ReplaceAsync(photo.Id, []);
        Assert.Empty(await _faces.GetAsync(photo.Id));
    }

    // ------------------------------------------------------------------------ listing

    [Fact]
    public async Task ListingOrdersAndFilters()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 3 });

        await _faces.ReplaceAsync(photos[0].Id, [Face(photos[0].Id, 0.99, 0.1, 0.1, 0.04)]);  // certain, tiny
        await _faces.ReplaceAsync(photos[1].Id, [Face(photos[1].Id, 0.82, 0.1, 0.1, 0.40)]);  // unsure, large
        await _faces.ReplaceAsync(photos[2].Id, [Face(photos[2].Id, 0.70, 0.1, 0.1, 0.20)]);  // unsure, middling

        IReadOnlyList<FaceListing> byProminence = await _faces.ListAsync(FaceOrder.Prominence);
        Assert.Equal(3, byProminence.Count);

        // Prominence blends size and certainty, so the large-but-unsure face beats the tiny
        // certain one. Ordering by confidence alone would reverse them, which is the bug this
        // catches: a wall led by a speck the detector happened to be sure about.
        Assert.Equal(photos[1].Id, byProminence[0].Face.PhotoId);

        IReadOnlyList<FaceListing> byConfidence = await _faces.ListAsync(FaceOrder.Confidence);
        Assert.Equal(photos[0].Id, byConfidence[0].Face.PhotoId);

        IReadOnlyList<FaceListing> certain = await _faces.ListAsync(minimumConfidence: 0.9);
        Assert.Equal(photos[0].Id, Assert.Single(certain).Face.PhotoId);

        // The listing carries its photograph, so a card can name it without a second query.
        Assert.All(byProminence, l => Assert.False(string.IsNullOrEmpty(l.PhotoFileName)));
    }

    [Fact]
    public async Task PagingNeitherRepeatsNorDropsAFace()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 12 });

        // Every face identical, so nothing but the tiebreak can order them. Without a unique
        // tiebreak SQLite is free to return ties differently per call and paging silently
        // repeats some rows while skipping others.
        foreach (PhotoRecord photo in photos)
        {
            await _faces.ReplaceAsync(photo.Id, [Face(photo.Id, 0.9, 0.2, 0.2, 0.1)]);
        }

        var seen = new List<long>();
        for (int offset = 0; offset < photos.Count; offset += 5)
        {
            IReadOnlyList<FaceListing> page = await _faces.ListAsync(limit: 5, offset: offset);
            seen.AddRange(page.Select(p => p.Face.Id));
        }

        Assert.Equal(photos.Count, seen.Count);
        Assert.Equal(photos.Count, seen.Distinct().Count());
    }

    [Fact]
    public async Task CountsReportFacesAndPhotographsSeparately()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 2 });

        await _faces.ReplaceAsync(photos[0].Id, [Face(photos[0].Id, 0.9, 0.1, 0.1, 0.1),
                                                 Face(photos[0].Id, 0.9, 0.4, 0.1, 0.1),
                                                 Face(photos[0].Id, 0.9, 0.7, 0.1, 0.1)]);
        await _faces.ReplaceAsync(photos[1].Id, [Face(photos[1].Id, 0.9, 0.1, 0.1, 0.1)]);

        (int faces, int photographs) = await _faces.CountAsync();
        Assert.Equal(4, faces);
        Assert.Equal(2, photographs);
    }

    // ------------------------------------------------------------------- same person

    [Fact]
    public async Task EmbeddingsSurviveARoundTrip()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        float[] written = Vector(7);

        await _faces.ReplaceAsync(photo.Id, [
            Face(photo.Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = written, EmbedModel = "sface" }]);

        PhotoFaceRecord got = Assert.Single(await _faces.GetAsync(photo.Id));

        Assert.True(got.IsComparable);
        Assert.Equal("sface", got.EmbedModel);
        Assert.Equal(written.Length, got.Embedding!.Length);

        // Exactly, not approximately. The blob is a byte copy of the floats; anything less than
        // bit-identical means the packing is lossy and every similarity is slightly wrong.
        Assert.Equal(written, got.Embedding);
    }

    [Fact]
    public async Task AFaceWithNoVectorIsStoredAndMarkedUncomparable()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _faces.ReplaceAsync(photo.Id, [Face(photo.Id, 0.9, 0.1, 0.1, 0.2)]);

        PhotoFaceRecord got = Assert.Single(await _faces.GetAsync(photo.Id));

        Assert.False(got.IsComparable);
        Assert.Null(got.Embedding);
        Assert.Null(got.EmbedModel);

        // And it cannot be searched for, rather than matching everything or nothing at random.
        Assert.Empty(await _faces.FindSimilarAsync(got.Id, FaceGrouping.DefaultThreshold));
    }

    [Fact]
    public async Task SearchingAFaceReturnsItsMatchesRankedAndIncludesItself()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 4 });

        // Three faces near axis 3 — the same "person" at decreasing likeness — and one far away.
        await _faces.ReplaceAsync(photos[0].Id, [
            Face(photos[0].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = Vector(3, lean: 1.00f), EmbedModel = "sface" }]);
        await _faces.ReplaceAsync(photos[1].Id, [
            Face(photos[1].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = Vector(3, lean: 0.95f), EmbedModel = "sface" }]);
        await _faces.ReplaceAsync(photos[2].Id, [
            Face(photos[2].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = Vector(3, lean: 0.80f), EmbedModel = "sface" }]);
        await _faces.ReplaceAsync(photos[3].Id, [
            Face(photos[3].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = Vector(60), EmbedModel = "sface" }]);

        long queryId = (await _faces.GetAsync(photos[0].Id))[0].Id;
        IReadOnlyList<FaceMatch> matches = await _faces.FindSimilarAsync(queryId, 0.5);

        // The face searched for comes back, first, at exactly 1. Leaving it out would make the
        // result look as though it had lost the photograph the user clicked on.
        Assert.Equal(queryId, matches[0].Listing.Face.Id);
        Assert.Equal(1.0, matches[0].Similarity, 6);

        // Ranked by likeness, and the stranger excluded.
        Assert.Equal(3, matches.Count);
        Assert.True(matches[1].Similarity > matches[2].Similarity);
        Assert.DoesNotContain(matches, m => m.Listing.Face.PhotoId == photos[3].Id);

        // Each match carries its photograph, so a card can be drawn without another query.
        Assert.All(matches, m => Assert.False(string.IsNullOrEmpty(m.Listing.PhotoFileName)));
    }

    [Fact]
    public async Task ATighterThresholdReturnsFewerMatchesAndNeverMore()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 3 });
        float[][] vectors = [Vector(5, lean: 1.0f), Vector(5, lean: 0.9f), Vector(5, lean: 0.72f)];

        for (int i = 0; i < 3; i++)
        {
            await _faces.ReplaceAsync(photos[i].Id, [
                Face(photos[i].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = vectors[i], EmbedModel = "sface" }]);
        }

        long queryId = (await _faces.GetAsync(photos[0].Id))[0].Id;

        int loose = (await _faces.FindSimilarAsync(queryId, FaceGrouping.LooseThreshold)).Count;
        int normal = (await _faces.FindSimilarAsync(queryId, FaceGrouping.DefaultThreshold)).Count;
        int strict = (await _faces.FindSimilarAsync(queryId, FaceGrouping.StrictThreshold)).Count;

        Assert.True(loose >= normal && normal >= strict,
            $"loose {loose}, normal {normal}, strict {strict} — tightening should never widen the result");
    }

    [Fact]
    public async Task VectorsFromADifferentModelAreNotCompared()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 2 });

        // Identical vectors, different models. They must not match: two models put "the same
        // face" in different places, so a high dot product between them means nothing at all.
        await _faces.ReplaceAsync(photos[0].Id, [
            Face(photos[0].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = Vector(9), EmbedModel = "sface" }]);
        await _faces.ReplaceAsync(photos[1].Id, [
            Face(photos[1].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = Vector(9), EmbedModel = "other-model" }]);

        long queryId = (await _faces.GetAsync(photos[0].Id))[0].Id;
        FaceMatch only = Assert.Single(await _faces.FindSimilarAsync(queryId, 0.5));

        Assert.Equal(queryId, only.Listing.Face.Id);
    }

    [Fact]
    public async Task CoverageDistinguishesUnembeddedFromUndetected()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 2 });

        await _faces.ReplaceAsync(photos[0].Id, [
            Face(photos[0].Id, 0.9, 0.1, 0.1, 0.2) with { Embedding = Vector(1), EmbedModel = "sface" }]);
        await _faces.ReplaceAsync(photos[1].Id, [Face(photos[1].Id, 0.9, 0.1, 0.1, 0.2)]);

        (int embedded, int total) = await _faces.EmbeddingCoverageAsync();

        // The difference between these two numbers is the honest answer to "why can I not
        // search for this face?", which is a question the page has to be able to answer.
        Assert.Equal(1, embedded);
        Assert.Equal(2, total);
    }

    // ------------------------------------------------------------------------ sweeping

    [Fact]
    public async Task APhotographWithoutPeopleIsNotExaminedForever()
    {
        PhotoRecord photo = await AnyPhotoAsync();

        Assert.Contains(photo.Id, await _faces.PendingAsync("test-model"));

        // No faces found, but the photograph has been looked at. "No rows" and "not looked at"
        // are indistinguishable in the faces table, so the sweep keeps its own marker.
        await _faces.ReplaceAsync(photo.Id, []);
        await _faces.MarkSweptAsync(photo.Id, "test-model");

        Assert.DoesNotContain(photo.Id, await _faces.PendingAsync("test-model"));

        // A different detector has not seen it, and must sweep it again.
        Assert.Contains(photo.Id, await _faces.PendingAsync("better-model"));
    }

    [Fact]
    public async Task ResettingClearsBothTheFacesAndTheSweepMarkers()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _faces.ReplaceAsync(photo.Id, [Face(photo.Id, 0.9, 0.1, 0.1, 0.1)]);
        await _faces.MarkSweptAsync(photo.Id, "test-model");

        await _faces.ResetAsync();

        Assert.Empty(await _faces.GetAsync(photo.Id));
        Assert.Contains(photo.Id, await _faces.PendingAsync("test-model"));
    }

    // ------------------------------------------------------------------------ the worker

    [Fact]
    public async Task TheSweepFindsFacesWritesSquareCropsAndTagsThePhotographs()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);
        var worker = new FaceWorker(_store, _faces, detector, CacheRoot);

        (int examined, int found) = await worker.RunAsync();
        _output.WriteLine($"examined {examined}, found {found}");

        Assert.True(examined > 40, $"only {examined} photographs were examined");
        Assert.True(found >= 8, $"only {found} faces were found in a corpus with people in it");

        IReadOnlyList<FaceListing> listing = await _faces.ListAsync();
        Assert.NotEmpty(listing);

        foreach (FaceListing item in listing)
        {
            Assert.NotNull(item.Face.CropKey);

            string path = Path.Combine(CacheRoot, item.Face.CropKey!.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"{item.Face.CropKey} was recorded but not written");

            // Square, always: a wall of faces only reads as a grid if every cell matches.
            using Image crop = await Image.LoadAsync(path);
            Assert.Equal(crop.Width, crop.Height);
            Assert.True(crop.Width >= 64, $"{crop.Width}px is too small to recognise anybody");
        }

        // A photograph with people in it becomes findable by typing "person", without anyone
        // having to know this page exists.
        long withFaces = listing[0].Face.PhotoId;
        IReadOnlyList<PhotoTag> tags = await _store.GetTagsAsync(withFaces);
        Assert.Contains(tags, t => t.Tag == "person");
    }

    [Fact]
    public async Task TheSweepIsResumableAndDoesNotRedoFinishedWork()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);
        var worker = new FaceWorker(_store, _faces, detector, CacheRoot);

        (int first, int foundFirst) = await worker.RunAsync(limit: 5);
        Assert.Equal(5, first);

        (int second, _) = await worker.RunAsync();
        (int third, int foundThird) = await worker.RunAsync();

        // The second run picks up what the first left; the third has nothing left to do.
        Assert.True(second > 0);
        Assert.Equal(0, third);
        Assert.Equal(0, foundThird);
        _output.WriteLine($"{first} + {second} photographs, {foundFirst} faces in the first pass");
    }

    [Fact]
    public async Task InstallingRecognitionLaterResweepsTheLibrary()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);

        // First pass: detection only. Faces are found; none of them can be compared.
        var detectOnly = new FaceWorker(_store, _faces, detector, CacheRoot);
        await detectOnly.RunAsync(limit: 6);

        (int embedded, int total) = await _faces.EmbeddingCoverageAsync();
        Assert.Equal(0, embedded);

        // A recognition model appears. The library must not be considered finished: every
        // photograph swept by the detector alone still has faces with no vectors, and the only
        // symptom of getting this wrong is that "find this person" silently finds nobody.
        var withRecognition = new FaceWorker(
            _store, _faces, detector, CacheRoot, new StubRecognizer());

        Assert.NotEqual(detectOnly.SweepVersion, withRecognition.SweepVersion);

        await withRecognition.RunAsync(limit: 6);
        (int nowEmbedded, _) = await _faces.EmbeddingCoverageAsync();

        Assert.True(nowEmbedded > 0,
            $"{total} faces were found before recognition was installed and none were re-examined");
    }

    /// <summary>A recognizer that answers instantly, for tests about plumbing rather than models.</summary>
    private sealed class StubRecognizer : IFaceRecognizer
    {
        public bool IsAvailable => true;

        public string ModelVersion => "stub";

        public int Dimensions => 8;

        public float[] Embed(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> image,
                             DetectedFace face)
        {
            // Derived from the face's position so two calls on one face agree and two different
            // faces differ — enough for a test about whether vectors get written at all.
            var v = new float[8];
            v[0] = (float)face.X;
            v[1] = (float)face.Y;
            v[2] = 1f;

            double norm = Math.Sqrt(v.Sum(x => (double)x * x));
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = (float)(v[i] / norm);
            }

            return v;
        }
    }

    [Fact]
    public async Task NoModelIsAQuietNoOpRatherThanAFailure()
    {
        using var detector = new YuNetFaceDetector(Path.Combine(_workDir, "absent.onnx"));
        var worker = new FaceWorker(_store, _faces, detector, CacheRoot);

        (int examined, int found) = await worker.RunAsync();

        Assert.Equal(0, examined);
        Assert.Equal(0, found);
        Assert.Equal((0, 0), await _faces.CountAsync());
    }
}
