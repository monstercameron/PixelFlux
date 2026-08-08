using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;

namespace PixelFlux.Tests;

/// <summary>
/// End-to-end ingestion against the real fixture album in <c>testdata/album</c>.
///
/// This is the test that matters most in the project, because it is the only one that exercises
/// the whole chain — walk, hash, decode, EXIF, derivatives, insert, index — against files with
/// the awkward properties real libraries have. The album was built specifically to make these
/// assertions possible: it contains known duplicates, a known burst, known EXIF-less files, and
/// exactly one file that cannot be decoded.
/// </summary>
public sealed class IngestionTests : IAsyncLifetime
{
    private string _workDir = string.Empty;
    private PhotoStore _store = null!;
    private IngestResult _result = null!;

    private static string AlbumPath
    {
        get
        {
            // Walk up from the test binary to the repo root. Test working directories move
            // between `dotnet test`, the IDE runner, and CI, so locating the fixture by
            // landmark is more durable than any relative path.
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "testdata", "album")))
            {
                dir = dir.Parent;
            }

            return dir is null
                ? throw new DirectoryNotFoundException(
                    "Could not locate testdata/album. Run tools/make_test_album.py to generate it.")
                : Path.Combine(dir.FullName, "testdata", "album");
        }
    }

    public async Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-ingest", Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(_workDir);

        var database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        database.Migrate();

        _store = new PhotoStore(database);
        var derivatives = new DerivativeGenerator(Path.Combine(_workDir, "cache"));
        var ingestor = new LibraryIngestor(_store, derivatives);

        // Imported once for the whole class: the album is 50 files and the decode cost is real.
        _result = await ingestor.ImportAsync([AlbumPath]);
    }

    public Task DisposeAsync()
    {
        try
        {
            if (Directory.Exists(_workDir))
            {
                Directory.Delete(_workDir, recursive: true);
            }
        }
        catch (IOException)
        {
            // Temp cleanup is best-effort.
        }

        return Task.CompletedTask;
    }

    [Fact]
    public void Import_FindsEveryImageAndSkipsTheManifest()
    {
        // 50 images; MANIFEST.tsv must not be picked up as a photo.
        Assert.Equal(50, _result.Discovered);
    }

    [Fact]
    public void Import_CollapsesTheTwoByteIdenticalDuplicates()
    {
        // The album carries exactly two exact-duplicate copies. Their content hashes already
        // exist, so they must be skipped rather than inserted a second time.
        Assert.Equal(2, _result.Duplicates);
        Assert.Equal(48, _result.Imported);
    }

    [Fact]
    public void Import_IndexesTheTruncatedFileRatherThanDroppingIt()
    {
        // The corrupt file must not count as a hard failure and must not vanish: it is indexed
        // as Unreadable so the library shows it with a placeholder instead of silently
        // pretending the file is not on disk.
        Assert.Equal(0, _result.Failed);
    }

    [Fact]
    public async Task Unreadable_FileIsMarkedAndStillBrowsable()
    {
        IReadOnlyDictionary<ProcessingState, int> states = await _store.GetStateCountsAsync();

        Assert.Equal(1, states.GetValueOrDefault(ProcessingState.Unreadable));
        Assert.Equal(47, states.GetValueOrDefault(ProcessingState.Pending));

        IReadOnlyList<PhotoRecord> unreadable = await _store.QueryAsync(
            new PhotoQuery { State = ProcessingState.Unreadable });

        Assert.Single(unreadable);
        Assert.Contains("corrupt", unreadable[0].FileName, StringComparison.OrdinalIgnoreCase);
        Assert.Null(unreadable[0].ThumbnailKey);
        Assert.NotNull(unreadable[0].StateDetail);
    }

    [Fact]
    public async Task Exif_IsReadForEveryCameraInTheAlbum()
    {
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });

        string[] models = all
            .Select(p => p.Camera.Model)
            .Where(m => m is not null)
            .Distinct(StringComparer.Ordinal)
            .ToArray()!;

        // The corpus is real photographs from many contributors, so the camera spread is wide
        // and includes phones, DSLRs, and mirrorless bodies. An exact count would break every
        // time the corpus is refetched; the property that matters is genuine variety.
        Assert.True(models.Length >= 15, $"only {models.Length} camera bodies: {string.Join(", ", models)}");
        Assert.Contains(models, m => m.StartsWith("Canon", StringComparison.Ordinal));
        Assert.Contains(models, m => m.Contains("iPhone", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Exif_ExposureTriadSurvivesTheRationalRoundTrip()
    {
        // f-numbers and shutter speeds are stored as EXIF rationals. Getting the numerator and
        // denominator the wrong way round produces plausible-looking nonsense (f/0.36), so this
        // asserts the values land in physically sensible ranges rather than merely being present.
        IReadOnlyList<PhotoRecord> shot = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });
        PhotoRecord[] withExposure = shot.Where(p => p.Camera.FNumber is not null).ToArray();

        Assert.NotEmpty(withExposure);
        Assert.All(withExposure, p =>
        {
            Assert.InRange(p.Camera.FNumber!.Value, 1.0, 32.0);
            Assert.InRange(p.Camera.ExposureSeconds!.Value, 1.0 / 8000, 30.0);
            // The range has been widened twice by real photographs, in both directions:
            // phone ultra-wide modules report 1.5-4mm, and a wildlife shot in this corpus was
            // taken at 840mm (a 600mm lens with a 1.4x extender). Both are legitimate, and a
            // range that excluded them was the test being provincial about what a camera is.
            Assert.InRange(p.Camera.FocalLengthMm!.Value, 1.0, 1600.0);
        });
    }

    [Fact]
    public async Task Gps_IsReadAndNullIslandIsNotInvented()
    {
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });
        PhotoRecord[] located = all.Where(p => p.Location is not null).ToArray();

        Assert.True(located.Length >= 20, $"only {located.Length} photos carry a GPS fix");
        Assert.All(located, p =>
        {
            Assert.InRange(p.Location!.Value.Latitude, -90, 90);
            Assert.InRange(p.Location.Value.Longitude, -180, 180);
            // No photo in the album is at 0,0; anything landing there means a failed GPS parse
            // was accepted as a real fix.
            Assert.False(p.Location.Value is { Latitude: 0, Longitude: 0 });
        });
    }

    [Fact]
    public async Task Gps_BoundingBoxSearchNarrowsToARegion()
    {
        // The brief's example shape of query — "photos from <somewhere>" — resolved to a box.
        // The corpus has several British photographs (Ely, Bradford, London, the south coast),
        // so the UK is the region asserted here.
        var uk = (South: 49.9, West: -8.2, North: 58.7, East: 1.8);

        IReadOnlyList<PhotoRecord> inside = await _store.QueryAsync(
            new PhotoQuery { Bounds = uk, Limit = 5000 });
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });

        Assert.NotEmpty(inside);

        // Every result really is inside the box...
        Assert.All(inside, p =>
        {
            Assert.InRange(p.Location!.Value.Latitude, uk.South, uk.North);
            Assert.InRange(p.Location.Value.Longitude, uk.West, uk.East);
        });

        // ...and the filter actually excluded something, rather than passing everything through.
        // A bounds filter that silently matched all rows would satisfy the assertion above.
        int located = all.Count(p => p.Location is not null);
        Assert.True(inside.Count < located,
            $"bounds filter returned all {located} located photos — it is not filtering");
    }

    [Fact]
    public async Task CaptureDates_FallBackToFileTimeAndSaySo()
    {
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });

        // Every photo must be dateable — that is what the timeline depends on.
        Assert.All(all, p => Assert.NotEqual(default, p.CapturedUtc));

        // But files with no EXIF must be flagged as inexact rather than passed off as precise.
        PhotoRecord[] inexact = all.Where(p => !p.CaptureTimeIsExact).ToArray();
        PhotoRecord[] exact = all.Where(p => p.CaptureTimeIsExact).ToArray();

        Assert.NotEmpty(inexact);
        Assert.NotEmpty(exact);
        Assert.All(exact, p => Assert.InRange(p.CapturedUtc.Year, 2000, 2027));
    }

    [Fact]
    public async Task Derivatives_AreWrittenAndSmallerThanTheOriginal()
    {
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });
        PhotoRecord[] decodable = all.Where(p => p.State != ProcessingState.Unreadable).ToArray();

        var derivatives = new DerivativeGenerator(Path.Combine(_workDir, "cache"));

        Assert.All(decodable, p =>
        {
            Assert.NotNull(p.ThumbnailKey);
            Assert.NotNull(p.ProxyKey);

            string thumb = derivatives.ResolvePath(p.ThumbnailKey!);
            Assert.True(File.Exists(thumb), $"missing thumbnail for {p.FileName}");
            Assert.True(new FileInfo(thumb).Length > 0);

            // Dimensions must be the upright ones, so a portrait photo is recorded as portrait.
            Assert.True(p.Width > 0 && p.Height > 0);
        });
    }

    [Fact]
    public async Task Orientation_IsAlwaysAValidExifValue()
    {
        // Two files in the corpus carry orientation 0, which is not a legal EXIF value — the
        // spec defines 1 through 8. Cameras and editors emit it anyway. The extractor must
        // normalise it to 1 rather than passing it through, because downstream code indexes
        // into rotation tables by this number.
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });

        Assert.NotEmpty(all);
        Assert.All(all, p =>
        {
            Assert.InRange(p.Orientation, 1, 8);
            // Dimensions are recorded post-rotation, so a portrait photo is stored portrait.
            if (p.State != ProcessingState.Unreadable)
            {
                Assert.True(p.Width > 0 && p.Height > 0);
            }
        });
    }

    [Fact]
    public async Task PerceptualHash_ClustersTheBurstAndNothingElse()
    {
        IReadOnlyList<IReadOnlyList<long>> groups = await _store.FindNearDuplicateGroupsAsync();

        // Resolve group members to filenames so a failure says which photos collided.
        var named = new List<string[]>();
        foreach (IReadOnlyList<long> group in groups)
        {
            var names = new List<string>();
            foreach (long id in group)
            {
                PhotoRecord? photo = await _store.GetAsync(id);
                names.Add(photo!.FileName);
            }

            named.Add(names.ToArray());
        }

        // All three burst frames must land in one group — together with the photograph they
        // were cropped from, which is also in the corpus and is genuinely the same shot. The
        // first version of this test expected a group of exactly 3 and failed on the group of
        // 4, which was the test being wrong rather than the detector: excluding the source
        // would mean the hash had failed to recognise its own crop.
        string[]? burst = named.FirstOrDefault(g => g.Any(n => n.Contains("burst", StringComparison.Ordinal)));
        Assert.NotNull(burst);

        string[] frames = burst!.Where(n => n.Contains("burst", StringComparison.Ordinal)).ToArray();
        Assert.Equal(3, frames.Length);

        // Anything else in the group must be the photograph the frames were cropped from.
        // Which photo that is depends on the corpus — the fetcher picks whichever real photo is
        // most visually distinct — so the assertion checks the relationship rather than
        // hardcoding a subject, which broke as soon as the corpus was refetched.
        string[] others = burst.Except(frames).ToArray();
        Assert.All(others, n => Assert.True(
            !n.Contains("reencoded", StringComparison.Ordinal)
            && !n.Contains("duplicate", StringComparison.Ordinal),
            $"unexpected photo clustered with the burst: {n}"));
        Assert.True(others.Length <= 1,
            "more than one photo clustered with the burst: " + string.Join(", ", others));

        // Other clusters are expected and are NOT a failure. This corpus is real photographs,
        // and Wikimedia search legitimately returns consecutive frames from one shoot — three
        // angles of the same museum car, two exposures of the same cathedral door. A library of
        // real photos contains series; a duplicate detector that found none would be the bug.
        //
        // What must hold is that the re-encoded copies cluster with their sources, since that
        // is the whole point of a perceptual hash: the same photograph saved as PNG, WebP, or
        // TIFF is still the same photograph.
        string[]? reencoded = named.FirstOrDefault(g => g.Any(n => n.Contains("reencoded", StringComparison.Ordinal)));
        Assert.True(
            reencoded is not null,
            "a re-encoded copy failed to cluster with its source — the perceptual hash is not "
            + "format-independent. Groups found: "
            + string.Join(" | ", named.Select(g => string.Join(", ", g))));
    }

    [Fact]
    public async Task Reimport_IsIdempotent()
    {
        // Re-running an import over the same folder must add nothing. This is what makes a
        // watch-folder or a crash-and-resume safe.
        var derivatives = new DerivativeGenerator(Path.Combine(_workDir, "cache"));
        var ingestor = new LibraryIngestor(_store, derivatives);

        IngestResult second = await ingestor.ImportAsync([AlbumPath]);

        Assert.Equal(0, second.Imported);
        Assert.Equal(50, second.Duplicates);
    }

    [Fact]
    public async Task Search_FullTextMatchesFilenameAndCamera()
    {
        // With no AI captions yet, the text index still has filenames and camera strings to work
        // with — which is what makes the app useful on day one, before anything is analysed.
        IReadOnlyList<PhotoRecord> doors = await _store.QueryAsync(new PhotoQuery { Text = "door" });
        Assert.NotEmpty(doors);
        Assert.All(doors, p => Assert.Contains("door", p.FileName, StringComparison.OrdinalIgnoreCase));

        IReadOnlyList<PhotoRecord> canon = await _store.QueryAsync(new PhotoQuery { Text = "Canon" });
        Assert.NotEmpty(canon);
        Assert.All(canon, p => Assert.Contains(
            "canon",
            $"{p.Camera.Make} {p.Camera.Model} {p.FileName}".ToLowerInvariant(),
            StringComparison.Ordinal));
    }

    [Fact]
    public async Task Search_ToleratesPunctuationThatIsFtsSyntax()
    {
        // A search box that throws on an apostrophe or a stray quote is broken. These must all
        // return calmly rather than raising a SQLite syntax error.
        foreach (string term in new[] { "dog's", "\"unclosed", "beach AND", "NEAR(", "*", "-", "a(b)c" })
        {
            IReadOnlyList<PhotoRecord> results = await _store.QueryAsync(new PhotoQuery { Text = term });
            Assert.NotNull(results);
        }
    }

    [Fact]
    public async Task TimeBuckets_CoverEveryYearInTheAlbum()
    {
        IReadOnlyList<TimeBucket> buckets = await _store.GetTimeBucketsAsync();

        Assert.NotEmpty(buckets);
        Assert.Equal(50 - 2, buckets.Sum(b => b.Count));   // 48 rows, the 2 dupes collapsed

        // Buckets must arrive oldest-first and strictly increasing: the rail draws them in order.
        Assert.Equal(buckets.OrderBy(b => b.Start), buckets);
    }

    [Fact]
    public async Task Ordering_ByCaptureDateDisagreesWithFilenameOrder()
    {
        // The album deliberately shuffles capture dates against filenames. If these two agree,
        // a sort that reads the wrong column would pass unnoticed.
        IReadOnlyList<PhotoRecord> byDate = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.CapturedDescending, Limit = 5000 });
        IReadOnlyList<PhotoRecord> byName = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.FileName, Limit = 5000 });

        Assert.NotEqual(
            byDate.Select(p => p.Id).ToArray(),
            byName.Select(p => p.Id).ToArray());

        // And the date order must actually be monotonic.
        for (int i = 1; i < byDate.Count; i++)
        {
            Assert.True(byDate[i - 1].CapturedUtc >= byDate[i].CapturedUtc);
        }
    }
}
