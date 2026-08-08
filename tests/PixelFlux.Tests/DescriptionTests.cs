using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;
using Xunit.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// Written descriptions: storage, the sweep, and whether the model earns its thirteen seconds.
///
/// The expensive test is last and guarded, because it loads 1.4 GB and takes a quarter of a
/// minute per photograph. Everything above it runs against a stub, because the parts that break
/// quietly — a description that is written but never indexed, a sweep that redoes finished work
/// — have nothing to do with the model.
/// </summary>
[Collection(Inference.Name)]
public sealed class DescriptionTests : IAsyncLifetime
{
    private readonly ITestOutputHelper _output;

    private string _workDir = string.Empty;
    private PhotoStore _store = null!;

    public DescriptionTests(ITestOutputHelper output) => _output = output;

    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "testdata", "album")))
            {
                dir = dir.Parent;
            }

            return dir?.FullName ?? throw new DirectoryNotFoundException("No repository root.");
        }
    }

    private static string ModelDirectory => Path.Combine(RepoRoot, "models", "qwen3vl");

    private static bool ModelInstalled => File.Exists(Path.Combine(ModelDirectory, "genai_config.json"));

    public async Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-describe", Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(_workDir);

        var database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        database.Migrate();
        _store = new PhotoStore(database);

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

    /// <summary>A describer that answers instantly, for the tests that are not about the model.</summary>
    private sealed class Stub(string text) : IPhotoDescriber
    {
        public int Calls { get; private set; }

        public bool IsAvailable { get; init; } = true;

        public string ModelVersion => "stub";

        public Task<string?> DescribeAsync(string imagePath, CancellationToken cancellationToken = default)
        {
            Calls++;
            return Task.FromResult<string?>(text);
        }
    }

    // ------------------------------------------------------------------------- storage

    [Fact]
    public async Task ADescriptionBecomesSearchable()
    {
        PhotoRecord photo = (await _store.QueryAsync(new PhotoQuery { Limit = 1 }))[0];

        // A word that appears nowhere in the filename, the EXIF, or any tag. If the description
        // is stored but not reindexed, the photograph is unfindable and nothing else fails.
        await _store.SetDescriptionAsync(
            photo.Id,
            "A red and white striped deckchair stands on the grass beside a white marquee.",
            "stub");

        IReadOnlyList<PhotoRecord> found = await _store.QueryAsync(new PhotoQuery { Text = "deckchair" });
        Assert.Contains(found, p => p.Id == photo.Id);

        Assert.Contains(await _store.QueryAsync(new PhotoQuery { Text = "marquee" }),
            p => p.Id == photo.Id);
    }

    [Fact]
    public async Task CoverageCountsOnlyRealDescriptions()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 3 });

        await _store.SetDescriptionAsync(photos[0].Id, "A dog on a beach.", "stub");
        await _store.SetDescriptionAsync(photos[1].Id, "   ", "stub");
        await _store.SetDescriptionAsync(photos[2].Id, null, "stub");

        (int described, int total) = await _store.DescriptionCoverageAsync();

        // Whitespace is not a description. Counting it would make the sweep look finished while
        // leaving photographs unsearchable.
        Assert.Equal(1, described);
        Assert.True(total > 3);

        IReadOnlyList<long> pending = await _store.UndescribedAsync();
        Assert.DoesNotContain(photos[0].Id, pending);
        Assert.Contains(photos[1].Id, pending);
        Assert.Contains(photos[2].Id, pending);
    }

    // -------------------------------------------------------------------------- the sweep

    [Fact]
    public async Task TheSweepIsResumableAndDoesNotRedoWork()
    {
        var stub = new Stub("A photograph of something.");
        var worker = new DescriptionWorker(_store, stub, Path.Combine(_workDir, "cache"));

        Assert.Equal(4, await worker.RunAsync(limit: 4));
        Assert.Equal(4, stub.Calls);

        // Thirteen seconds a photograph is too expensive to repeat by accident.
        Assert.Equal(4, (await _store.DescriptionCoverageAsync()).Described);
        Assert.Equal(3, await worker.RunAsync(limit: 3));
        Assert.Equal(7, stub.Calls);
        Assert.Equal(7, (await _store.DescriptionCoverageAsync()).Described);
    }

    [Fact]
    public async Task TheSweepReportsWhatItWrote()
    {
        var seen = new List<string>();
        var worker = new DescriptionWorker(_store, new Stub("A dog on a beach."), Path.Combine(_workDir, "cache"));

        var progress = new Progress<DescriptionProgress>(p =>
        {
            if (p.Latest is { } text)
            {
                seen.Add(text);
            }
        });

        await worker.RunAsync(limit: 2, progress);

        // The text, not just a counter. On a run that takes half an hour this is what tells
        // somebody early whether it is worth leaving on.
        await Task.Delay(50);
        Assert.NotEmpty(seen);
    }

    [Fact]
    public async Task NoModelIsAQuietNoOp()
    {
        var worker = new DescriptionWorker(
            _store, new Stub("unused") { IsAvailable = false }, Path.Combine(_workDir, "cache"));

        Assert.Equal(0, await worker.RunAsync());
        Assert.Equal(0, (await _store.DescriptionCoverageAsync()).Described);
    }

    [Fact]
    public void AMissingModelDirectoryIsNotAnError()
    {
        using var describer = new QwenVisionDescriber(Path.Combine(_workDir, "no-such-model"));
        Assert.False(describer.IsAvailable);
    }

    // ---------------------------------------------------------------------- the real model

    /// <summary>
    /// The model itself, on a photograph whose contents are known.
    /// </summary>
    /// <remarks>
    /// One photograph, because each costs about thirteen seconds. It is the 1954 Sunbeam-Talbot
    /// at a car show, and the assertions are things a person can verify by looking at it: the
    /// car, its colour, and the registration plate. The plate is the interesting one — no other
    /// pass in the application can read text, and it is the detail that proves the model is
    /// looking at the picture rather than pattern-matching a category.
    /// </remarks>
    [Fact]
    public async Task TheModelReadsWhatIsActuallyInThePhotograph()
    {
        if (!ModelInstalled)
        {
            _output.WriteLine("model not installed");
            return;
        }

        string photo = Directory.GetFiles(
            Path.Combine(RepoRoot, "testdata", "album"),
            "*_car_1954*.jpg", SearchOption.AllDirectories)[0];

        using var describer = new QwenVisionDescriber(ModelDirectory);
        Assert.True(describer.IsAvailable);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        string? text = await describer.DescribeAsync(photo);
        watch.Stop();

        _output.WriteLine($"{watch.ElapsedMilliseconds} ms");
        _output.WriteLine(text ?? "(nothing)");

        Assert.False(string.IsNullOrWhiteSpace(text));

        Assert.Contains("red", text!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("car", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("UVT", text, StringComparison.OrdinalIgnoreCase);

        // Prose for an index, not a formatted answer. Headings and bullets are noise in a
        // full-text column, and the instruction asks for neither.
        Assert.DoesNotContain("**", text, StringComparison.Ordinal);
        Assert.DoesNotContain("<|", text, StringComparison.Ordinal);

        // Ends on a complete sentence: a stored half-clause helps nobody.
        Assert.EndsWith(".", text.Trim(), StringComparison.Ordinal);
    }
}
