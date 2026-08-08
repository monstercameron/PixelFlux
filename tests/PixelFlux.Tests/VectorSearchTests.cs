using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;
using PixelFlux.Core.Search;
using Xunit.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// Search by meaning, against the real album.
///
/// The question this feature exists to answer is "red car" — a phrase matching no filename, no
/// tag, and no detector class in this corpus, where the right answer is nevertheless obvious to
/// anyone looking at the photographs. So that is the test, alongside the tokenizer it depends
/// on: text and images only land in the same space if the phrase is split into exactly the
/// tokens CLIP was trained on, and a wrong tokenizer fails silently with a confident vector for
/// a sentence nobody wrote.
/// </summary>
[Collection(Inference.Name)]
public sealed class VectorSearchTests : IAsyncLifetime
{
    private readonly ITestOutputHelper _output;

    private string _workDir = string.Empty;
    private PhotoStore _store = null!;
    private VectorIndex _vectors = null!;

    public VectorSearchTests(ITestOutputHelper output) => _output = output;

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

    private static string Model(string name) => Path.Combine(RepoRoot, "models", name);

    private static bool ClipInstalled =>
        File.Exists(Model("clip_vision_model.onnx")) && File.Exists(Model("clip_text_model.onnx"))
        && File.Exists(Model("clip_vocab.json")) && File.Exists(Model("clip_merges.txt"));

    private static ClipEmbedder OpenClip() => new(
        Model("clip_vision_model.onnx"), Model("clip_text_model.onnx"),
        Model("clip_vocab.json"), Model("clip_merges.txt"));

    public async Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-vector", Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(_workDir);

        var database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        database.Migrate();

        _store = new PhotoStore(database);
        _vectors = new VectorIndex(database);

        var ingestor = new LibraryIngestor(_store, new DerivativeGenerator(Path.Combine(_workDir, "cache")));
        await ingestor.ImportAsync([Path.Combine(RepoRoot, "testdata", "album")]);
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

    // ------------------------------------------------------------------------ tokenizer

    [Fact]
    public void TheTokenizerMatchesTheReferenceImplementation()
    {
        if (!ClipInstalled) { return; }

        var tokenizer = new ClipTokenizer(Model("clip_vocab.json"), Model("clip_merges.txt"));

        Assert.Equal(49408, tokenizer.VocabularySize);
        Assert.Equal(49406, tokenizer.StartId);
        Assert.Equal(49407, tokenizer.EndId);

        // The canonical CLIP example, and the ids the reference tokenizer produces for it. This
        // is the assertion that catches a wrong merge order or a missing word-end marker — every
        // one of which still returns plausible-looking ids.
        Assert.Equal(
            [49406, 320, 1125, 539, 320, 2368, 49407],
            tokenizer.Encode("a photo of a cat"));

        // Case and surrounding whitespace are normalised away, so these must be identical.
        Assert.Equal(tokenizer.Encode("a photo of a cat"), tokenizer.Encode("  A PHOTO  of a Cat "));
    }

    [Fact]
    public void TheTokenizerHandlesTextItWasNotDesignedFor()
    {
        if (!ClipInstalled) { return; }

        var tokenizer = new ClipTokenizer(Model("clip_vocab.json"), Model("clip_merges.txt"));

        // Every byte maps into the private alphabet, so nothing throws and nothing is dropped.
        foreach (string awkward in new[] { "café", "日本語", "🚗 red", "IMG_4021.JPG", "don't", "—" })
        {
            int[] ids = tokenizer.Encode(awkward);

            Assert.True(ids.Length >= 3, $"'{awkward}' produced nothing between the markers");
            Assert.Equal(tokenizer.StartId, ids[0]);
            Assert.Equal(tokenizer.EndId, ids[^1]);
            Assert.All(ids, id => Assert.InRange(id, 0, tokenizer.VocabularySize - 1));
        }

        // Empty input is a pair of markers, not a crash and not a null.
        Assert.Equal([tokenizer.StartId, tokenizer.EndId], tokenizer.Encode("   "));
    }

    [Fact]
    public void ALongPhraseIsTruncatedRatherThanRefused()
    {
        if (!ClipInstalled) { return; }

        var tokenizer = new ClipTokenizer(Model("clip_vocab.json"), Model("clip_merges.txt"));
        int[] ids = tokenizer.Encode(string.Join(' ', Enumerable.Repeat("photograph", 200)));

        // CLIP has 77 positions and no way to represent a 78th.
        Assert.True(ids.Length <= ClipTokenizer.MaximumTokens);
        Assert.Equal(tokenizer.EndId, ids[^1]);
    }

    // -------------------------------------------------------------------------- embedding

    [Fact]
    public async Task TextAndImagesLandInTheSameSpace()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        Assert.True(clip.IsAvailable);

        float[] text = (await clip.EmbedTextAsync("a photograph of a cat"))!;
        Assert.Equal(512, text.Length);
        Assert.Equal(1.0, Math.Sqrt(text.Sum(v => (double)v * v)), 4);

        string cat = Directory.GetFiles(Path.Combine(RepoRoot, "testdata", "album"),
            "*_animal_cat*.jpg", SearchOption.AllDirectories)[0];
        string door = Directory.GetFiles(Path.Combine(RepoRoot, "testdata", "album"),
            "*_door_*.jpg", SearchOption.AllDirectories)[0];

        float[] catVector = (await clip.EmbedImageAsync(cat))!;
        float[] doorVector = (await clip.EmbedImageAsync(door))!;

        double toCat = Dot(text, catVector);
        double toDoor = Dot(text, doorVector);

        _output.WriteLine($"'a photograph of a cat'  cat {toCat:0.000}  door {toDoor:0.000}");

        // The whole premise: a sentence is nearer the photograph it describes than one it does
        // not. If the two encoders were misaligned — wrong normalisation, wrong tokenizer — this
        // is where it shows, and nowhere else would.
        Assert.True(toCat > toDoor + 0.05,
            $"a cat photograph scored {toCat:0.000} against {toDoor:0.000} for a door");
    }

    [Fact]
    public async Task DescribingIsFastEnoughToSweepALibrary()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        string[] photos = Directory
            .GetFiles(Path.Combine(RepoRoot, "testdata", "album"), "*.jpg", SearchOption.AllDirectories)
            .Take(6).ToArray();

        await clip.EmbedImageAsync(photos[0]);   // warm the graph

        var watch = System.Diagnostics.Stopwatch.StartNew();
        foreach (string photo in photos)
        {
            await clip.EmbedImageAsync(photo);
        }

        watch.Stop();
        double per = watch.ElapsedMilliseconds / (double)photos.Length;
        _output.WriteLine($"{per:0} ms per photograph");

        Assert.True(per < 1500, $"{per:0} ms per photograph is too slow for a library sweep");
    }

    // ----------------------------------------------------------------------------- search

    /// <summary>Describes the album, then asks the question this feature was built for.</summary>
    [Fact]
    public async Task RedCarFindsTheRedCars()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        await DescribeAsync(clip);

        IReadOnlyList<VectorHit> hits = await _vectors.SearchAsync((await clip.EmbedTextAsync("red car"))!, 6);
        var names = new List<string>();

        foreach (VectorHit hit in hits)
        {
            PhotoRecord photo = (await _store.GetAsync(hit.PhotoId))!;
            names.Add(photo.FileName);
            _output.WriteLine($"  {hit.Similarity:0.000}  {photo.FileName}");
        }

        // Nothing in this corpus is named "red" and no detector class is a colour, so a
        // word-based search cannot answer this at all. The two red cars are 008 (a red Sunbeam
        // roadster) and 009 (a red Lamborghini); both must be above every non-car.
        Assert.Contains("008_car", names[0], StringComparison.Ordinal);
        Assert.Contains("009_car", names[1], StringComparison.Ordinal);

        // And the whole top of the ranking should be cars.
        Assert.True(names.Take(3).All(n => n.Contains("_car", StringComparison.Ordinal)),
            $"expected cars at the top, got: {string.Join(", ", names.Take(3))}");
    }

    [Fact]
    public async Task EverydayPhrasesFindTheRightPhotographs()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        await DescribeAsync(clip);

        // Each phrase, and a filename fragment the top hit must contain. None of these words
        // appear in the filenames they have to match by way of the word index — "the beach" is
        // in the filename, but "somebody smiling" and "a plate of food" are not.
        (string Phrase, string Expected)[] cases =
        [
            ("a cat", "_animal_cat"),
            ("a bicycle", "_bicycle_"),
            ("a wooden door", "_door_"),
            ("flowers", "_flowers_"),
            ("a sandy beach", "_landscape_"),
        ];

        foreach ((string phrase, string expected) in cases)
        {
            IReadOnlyList<VectorHit> hits = await _vectors.SearchAsync((await clip.EmbedTextAsync(phrase))!, 1);
            PhotoRecord top = (await _store.GetAsync(hits[0].PhotoId))!;

            _output.WriteLine($"{phrase,-18} {hits[0].Similarity:0.000}  {top.FileName}");
            Assert.Contains(expected, top.FileName, StringComparison.Ordinal);
        }
    }

    [Fact]
    public async Task DescribingIsResumableAndDoesNotRedoWork()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        var worker = new EmbeddingWorker(_store, _vectors, clip, Path.Combine(_workDir, "cache"));

        int first = await worker.RunAsync(limit: 4);
        Assert.Equal(4, first);

        (int described, int total) = await _vectors.CoverageAsync();
        Assert.Equal(4, described);
        Assert.True(total > 4);

        // A second run picks up where it left off rather than starting over.
        Assert.Equal(4, (await _vectors.PendingAsync(clip.ModelVersion, 4)).Count);
        Assert.Equal(4, await worker.RunAsync(limit: 4));
        Assert.Equal(8, (await _vectors.CoverageAsync()).Described);
    }

    [Fact]
    public async Task NoModelIsAQuietNoOp()
    {
        using var absent = new ClipEmbedder(
            Path.Combine(_workDir, "nope-vision.onnx"), Path.Combine(_workDir, "nope-text.onnx"),
            Path.Combine(_workDir, "nope.json"), Path.Combine(_workDir, "nope.txt"));

        Assert.False(absent.IsAvailable);
        Assert.Null(await absent.EmbedTextAsync("red car"));

        var worker = new EmbeddingWorker(_store, _vectors, absent, Path.Combine(_workDir, "cache"));
        Assert.Equal(0, await worker.RunAsync());
    }

    [Fact]
    public async Task SearchStillWorksWithNoVectorsAtAll()
    {
        // The engine takes an optional vector, and a library nobody has described must still
        // answer word searches exactly as it did before this feature existed.
        var engine = new SearchEngine(_store, _vectors);
        SearchResult result = await engine.SearchAsync(new PhotoQuery { Text = "cathedral" });

        Assert.False(result.UsedSemantic);
        Assert.NotEmpty(result.Hits);
    }

    // --------------------------------------------------------------- what the user sees

    /// <summary>
    /// The blended search, which is what the search box actually runs.
    /// </summary>
    /// <remarks>
    /// Raw embedding ranking looked good long before the product did. The blend is where it went
    /// wrong: spelling correction turned "red" into "redmi" and "cat" into "cathedral", and the
    /// semantic contribution was numerically too small to outvote either. Every case here is one
    /// that shipped broken and was found by looking at real output.
    /// </remarks>
    private async Task<SearchResult> FindAsync(ClipEmbedder clip, string phrase)
    {
        var bank = new List<ReadOnlyMemory<float>>();

        foreach (string reference in ReferencePhrases.All)
        {
            if (await clip.EmbedTextAsync(reference) is { } v)
            {
                bank.Add(v);
            }
        }

        await _vectors.CalibrateAsync(bank);

        var engine = new SearchEngine(_store, _vectors);
        return await engine.SearchAsync(
            new PhotoQuery { Text = phrase, Limit = 40 },
            await clip.EmbedQueryAsync(phrase));
    }

    [Fact]
    public async Task AShortWordIsNeverCorrectedIntoALongerOne()
    {
        // "cat" is a prefix of "cathedral" and "catania"; "red" of "redmi" and "redditch". The
        // corrector used to be the autocomplete function, so a search for a red car returned a
        // plate of food and a living room.
        IReadOnlyList<string> vocabulary =
            ["cathedral", "catania", "redmi", "redditch", "car", "cat", "cathederal"];

        Assert.Empty(FuzzyMatch.Correct("cat", vocabulary));
        Assert.Empty(FuzzyMatch.Correct("red", vocabulary));

        // A real typo in a long enough word is still corrected — that is the feature.
        Assert.Contains("cathedral",
            FuzzyMatch.Correct("cathedrel", vocabulary).Select(c => c.Term), StringComparer.Ordinal);
    }

    [Fact]
    public async Task TheSearchBoxAnswersTheQueriesItWasBuiltFor()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        await DescribeAsync(clip);

        (string Phrase, string Expected)[] cases =
        [
            ("red car", "_car_"),
            ("cat", "_animal_cat"),
            ("beach", "_landscape_"),
            ("a bicycle", "_bicycle_"),
        ];

        foreach ((string phrase, string expected) in cases)
        {
            SearchResult result = await FindAsync(clip, phrase);
            Assert.NotEmpty(result.Hits);

            string top = result.Hits[0].Photo.FileName;
            _output.WriteLine($"{phrase,-12} {result.Hits.Count,2} results, top: {top}");

            Assert.Contains(expected, top, StringComparison.Ordinal);
        }
    }

    [Fact]
    public async Task AQueryWithNothingToFindReturnsNothing()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        await DescribeAsync(clip);

        // There is no submarine in this library. Before the relevance floor, embedding search
        // returned the whole library ranked — every photograph has some similarity to every
        // phrase, so "top matches" with no cut-off means "everything, in an order".
        SearchResult result = await FindAsync(clip, "a submarine underwater");

        _output.WriteLine($"{result.Hits.Count} results");
        Assert.Empty(result.Hits);
    }

    [Fact]
    public async Task AFewCorrectAnswersAreNotTreatedAsAFailedQuery()
    {
        if (!ClipInstalled) { return; }

        using ClipEmbedder clip = OpenClip();
        await DescribeAsync(clip);

        // Only a handful of photographs match, and none of them by any word. That used to look
        // enough like a failed query to trigger spelling correction, which turned "hair" into
        // "chair" and buried the right answers under furniture.
        SearchResult result = await FindAsync(clip, "blonde hair");

        Assert.NotEmpty(result.Hits);
        Assert.Empty(result.Corrections);

        foreach (SearchHit hit in result.Hits)
        {
            _output.WriteLine($"  {hit.Score:0.000}  {hit.Photo.FileName}");
        }
    }

    private async Task DescribeAsync(ClipEmbedder clip)
    {
        var worker = new EmbeddingWorker(_store, _vectors, clip, Path.Combine(_workDir, "cache"));
        await worker.RunAsync();
    }

    private static double Dot(float[] a, float[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }
}
