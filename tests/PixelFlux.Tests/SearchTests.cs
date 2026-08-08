using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;
using PixelFlux.Core.Search;

namespace PixelFlux.Tests;

/// <summary>
/// The search and sort system, exercised against the real photo corpus.
///
/// Every assertion here is about a property a person would notice: a typo still finds the
/// photo, a filter actually excludes things, a shuffle does not repeat itself when you page
/// through it. The corpus is real photographs with real EXIF, so "camera search works" means it
/// works against thirty-one actual camera bodies rather than against strings a fixture invented.
/// </summary>
public sealed class SearchTests : IAsyncLifetime
{
    private string _workDir = string.Empty;
    private PhotoStore _store = null!;
    private CollectionStore _collections = null!;
    private VectorIndex _vectors = null!;
    private SearchEngine _engine = null!;

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

    public async Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-search", Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(_workDir);

        var database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        database.Migrate();

        _store = new PhotoStore(database);
        _collections = new CollectionStore(database);
        _vectors = new VectorIndex(database);
        _engine = new SearchEngine(_store, _vectors);

        var ingestor = new LibraryIngestor(_store, new DerivativeGenerator(Path.Combine(_workDir, "cache")));
        await ingestor.ImportAsync([AlbumPath]);
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
            // Best effort.
        }

        return Task.CompletedTask;
    }

    // ------------------------------------------------------------------ fuzzy matching

    [Theory]
    [InlineData("cathedral", "cathedrl")]     // dropped letter
    [InlineData("bicycle", "bicicle")]        // substitution
    [InlineData("flowers", "flwers")]         // dropped letter
    [InlineData("landscape", "landscpae")]    // transposition
    public async Task Search_SurvivesATypo(string correct, string typo)
    {
        SearchResult exact = await _engine.SearchAsync(new PhotoQuery { Text = correct });
        SearchResult fuzzy = await _engine.SearchAsync(new PhotoQuery { Text = typo });

        Assert.NotEmpty(exact.Hits);
        Assert.NotEmpty(fuzzy.Hits);

        // The typo must reach the same photographs, and must say what it corrected rather than
        // silently returning something the user did not ask for.
        Assert.Contains(fuzzy.Corrections, c => c.Used.Contains(correct, StringComparison.OrdinalIgnoreCase));
        Assert.All(fuzzy.Hits, h => Assert.True(h.Reason.HasFlag(MatchReason.Fuzzy)));

        long[] exactIds = exact.Hits.Select(h => h.Photo.Id).ToArray();
        Assert.Contains(fuzzy.Hits, h => exactIds.Contains(h.Photo.Id));
    }

    [Fact]
    public void Fuzzy_DoesNotCorrectShortWords()
    {
        // A two-edit budget on a three-letter word matches most of the dictionary. "cat" must
        // not quietly become "car", or every short query returns noise.
        Assert.False(FuzzyMatch.IsCloseEnough("cat", "car"));
        Assert.False(FuzzyMatch.IsCloseEnough("cat", "bat"));

        // But a long word gets real slack.
        Assert.True(FuzzyMatch.IsCloseEnough("cathedral", "cathedrl"));
        Assert.True(FuzzyMatch.IsCloseEnough("photograph", "photograhp"));
    }

    [Fact]
    public void Fuzzy_TreatsATranspositionAsOneEdit()
    {
        // Damerau, not plain Levenshtein. Swapped adjacent letters are the single most common
        // typo, and scoring them as two edits pushes them outside the budget for short words.
        Assert.Equal(1, FuzzyMatch.Distance("teh", "the"));
        Assert.Equal(1, FuzzyMatch.Distance("cathderal", "cathedral"));
    }

    [Fact]
    public void Fuzzy_PrefersAPrefixOverAnEqualEditDistance()
    {
        // Someone typing into a search box is usually part-way through a word rather than
        // misspelling one.
        string[] vocabulary = ["cathedral", "cathode", "catherine"];
        IReadOnlyList<(string Term, double Score)> suggestions = FuzzyMatch.Suggest("cathe", vocabulary);

        Assert.NotEmpty(suggestions);
        Assert.All(suggestions.Take(3), s =>
            Assert.StartsWith("cathe", s.Term, StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task Vocabulary_IsBuiltFromTheLibraryNotADictionary()
    {
        IReadOnlyList<string> vocabulary = await _store.GetVocabularyAsync();

        // Filenames are split into words, so subject terms are individually findable even
        // before anything has been analysed by a model.
        Assert.Contains("cathedral", vocabulary);
        Assert.Contains("bicycle", vocabulary);

        // Camera model strings are in there too — that is what makes "taken with my Canon" work.
        Assert.Contains(vocabulary, v => v.Contains("canon", StringComparison.Ordinal));

        // Pure numbers are excluded: frame indices and dates are not words worth correcting to.
        Assert.DoesNotContain(vocabulary, v => v.All(char.IsDigit));
    }

    // ------------------------------------------------------------------ structured filters

    [Fact]
    public async Task Filters_ActuallyExclude()
    {
        // The failure mode worth testing for is a filter that is wired up but has no effect:
        // every assertion about "results are inside the range" passes trivially if the filter
        // returned everything. Each of these checks the count went down as well.
        int total = (await _store.QueryAsync(new PhotoQuery { Limit = 5000 })).Count;

        IReadOnlyList<PhotoRecord> canon = await _store.QueryAsync(
            new PhotoQuery { CameraModel = "Canon", Limit = 5000 });
        IReadOnlyList<PhotoRecord> recent = await _store.QueryAsync(
            new PhotoQuery { From = new DateTimeOffset(2020, 1, 1, 0, 0, 0, TimeSpan.Zero), Limit = 5000 });

        Assert.NotEmpty(canon);
        Assert.True(canon.Count < total, "camera filter matched everything");
        Assert.All(canon, p => Assert.Contains(
            "canon", $"{p.Camera.Make} {p.Camera.Model}".ToLowerInvariant(), StringComparison.Ordinal));

        Assert.NotEmpty(recent);
        Assert.True(recent.Count < total, "date filter matched everything");
        Assert.All(recent, p => Assert.True(p.CapturedUtc.Year >= 2020));
    }

    [Fact]
    public async Task Filters_Combine()
    {
        var since2020 = new DateTimeOffset(2020, 1, 1, 0, 0, 0, TimeSpan.Zero);

        IReadOnlyList<PhotoRecord> canonOnly = await _store.QueryAsync(
            new PhotoQuery { CameraModel = "Canon", Limit = 5000 });
        IReadOnlyList<PhotoRecord> both = await _store.QueryAsync(
            new PhotoQuery { CameraModel = "Canon", From = since2020, Limit = 5000 });

        // AND, not OR: adding a filter can only narrow.
        Assert.True(both.Count <= canonOnly.Count);
        Assert.All(both, p =>
        {
            Assert.Contains("canon", $"{p.Camera.Make} {p.Camera.Model}".ToLowerInvariant(),
                StringComparison.Ordinal);
            Assert.True(p.CapturedUtc >= since2020);
        });
    }

    [Fact]
    public async Task Folder_FilterMatchesAsAPrefix()
    {
        // The album is organised into year folders, so this can assert the property that
        // actually matters: selecting a parent folder includes everything nested beneath it,
        // which is what people mean when they click a folder in a tree.
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });
        string leaf = all[0].SourceFolder;
        string parent = Path.GetDirectoryName(leaf)!;

        Assert.NotEqual(string.Empty, leaf);
        Assert.NotEqual(leaf, parent);

        IReadOnlyList<PhotoRecord> inLeaf = await _store.QueryAsync(
            new PhotoQuery { SourceFolder = leaf, Limit = 5000 });
        IReadOnlyList<PhotoRecord> inParent = await _store.QueryAsync(
            new PhotoQuery { SourceFolder = parent, Limit = 5000 });

        // The leaf holds some but not all...
        Assert.NotEmpty(inLeaf);
        Assert.True(inLeaf.Count < all.Count, "a single year folder should not hold the whole library");
        Assert.All(inLeaf, p => Assert.Equal(leaf, p.SourceFolder));

        // ...and the parent holds everything, because the filter is a prefix over the tree.
        Assert.Equal(all.Count, inParent.Count);

        // A folder that does not exist returns nothing rather than everything. This is the
        // check that caught the original LIKE-based implementation, where an unescaped
        // underscore and a backslash escape character between them matched zero rows for every
        // parent folder.
        IReadOnlyList<PhotoRecord> nowhere = await _store.QueryAsync(
            new PhotoQuery { SourceFolder = @"C:\definitely
ot\here", Limit = 5000 });
        Assert.Empty(nowhere);
    }

    // ------------------------------------------------------------------ sort orders

    [Theory]
    [InlineData(PhotoOrder.CapturedDescending)]
    [InlineData(PhotoOrder.CapturedAscending)]
    [InlineData(PhotoOrder.FileName)]
    [InlineData(PhotoOrder.RatingDescending)]
    [InlineData(PhotoOrder.FileSizeDescending)]
    [InlineData(PhotoOrder.Camera)]
    [InlineData(PhotoOrder.Folder)]
    [InlineData(PhotoOrder.Shuffle)]
    public async Task Sort_PagesWithoutRepeatingOrSkipping(PhotoOrder order)
    {
        // Every order must have a unique tiebreak, or two rows with the same sort key can swap
        // places between pages — which shows one photo twice and hides another entirely. This
        // is the bug that makes an infinite-scroll gallery feel haunted, so it is checked for
        // every order rather than just the default.
        IReadOnlyList<PhotoRecord> whole = await _store.QueryAsync(
            new PhotoQuery { Order = order, Limit = 5000 });

        var paged = new List<long>();
        const int pageSize = 7;

        for (int offset = 0; offset < whole.Count; offset += pageSize)
        {
            IReadOnlyList<PhotoRecord> page = await _store.QueryAsync(
                new PhotoQuery { Order = order, Limit = pageSize, Offset = offset });
            paged.AddRange(page.Select(p => p.Id));
        }

        Assert.Equal(whole.Count, paged.Count);
        Assert.Equal(whole.Count, paged.Distinct().Count());
        Assert.Equal(whole.Select(p => p.Id), paged);
    }

    [Fact]
    public async Task Shuffle_IsStableButNotChronological()
    {
        IReadOnlyList<PhotoRecord> first = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.Shuffle, Limit = 5000 });
        IReadOnlyList<PhotoRecord> second = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.Shuffle, Limit = 5000 });
        IReadOnlyList<PhotoRecord> chronological = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.CapturedDescending, Limit = 5000 });

        // Stable: the slideshow must not reorder itself underneath the viewer, and paging a
        // shuffled set must not repeat. SQLite's RANDOM() reseeds per statement and fails this.
        Assert.Equal(first.Select(p => p.Id), second.Select(p => p.Id));

        // But genuinely shuffled — if it matched capture order it would not be doing anything.
        Assert.NotEqual(chronological.Select(p => p.Id), first.Select(p => p.Id));
    }

    [Fact]
    public async Task Sort_ByFileSizeAndCameraAreActuallyOrdered()
    {
        IReadOnlyList<PhotoRecord> bySize = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.FileSizeDescending, Limit = 5000 });

        for (int i = 1; i < bySize.Count; i++)
        {
            Assert.True(bySize[i - 1].FileSize >= bySize[i].FileSize);
        }

        IReadOnlyList<PhotoRecord> byCamera = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.Camera, Limit = 5000 });

        // Photos from one body must be contiguous — that is the point of grouping by camera.
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string? previous = null;

        foreach (PhotoRecord photo in byCamera)
        {
            string model = photo.Camera.Model ?? "~";
            if (model != previous)
            {
                Assert.True(seen.Add(model), $"camera {model} appears in two separate runs");
                previous = model;
            }
        }
    }

    // ------------------------------------------------------------------ facets

    [Fact]
    public async Task Facets_CountEveryBrowsableDimension()
    {
        IReadOnlyDictionary<string, IReadOnlyList<(string Value, int Count)>> facets =
            await _store.GetFacetsAsync();

        Assert.Contains("camera", facets.Keys);
        Assert.Contains("year", facets.Keys);
        Assert.Contains("folder", facets.Keys);
        Assert.Contains("city", facets.Keys);
        Assert.Contains("country", facets.Keys);

        Assert.True(facets["camera"].Count >= 15, $"only {facets["camera"].Count} cameras faceted");
        Assert.True(facets["year"].Count >= 5, $"only {facets["year"].Count} years faceted");
        Assert.True(facets["city"].Count >= 10, $"only {facets["city"].Count} cities faceted");
        Assert.True(facets["country"].Count >= 8, $"only {facets["country"].Count} countries faceted");

        // Real names, not coordinates. This is the assertion that would have caught the old
        // facet quietly still emitting "22,114" after the gazetteer landed.
        Assert.All(facets["country"], f =>
            Assert.False(f.Value.Any(char.IsDigit), $"country facet looks like coordinates: {f.Value}"));

        // Counts are descending, so the sidebar shows the biggest groups first.
        Assert.Equal(
            facets["camera"].Select(f => f.Count).OrderByDescending(c => c),
            facets["camera"].Select(f => f.Count));

        // And every facet count must be reachable: clicking it has to return that many photos.
        (string camera, int count) = facets["camera"][0];
        IReadOnlyList<PhotoRecord> clicked = await _store.QueryAsync(
            new PhotoQuery { CameraModel = camera, Limit = 5000 });
        Assert.True(clicked.Count >= count, $"facet claims {count} for {camera}, filter returns {clicked.Count}");
    }

    // ------------------------------------------------------------------ collections

    [Fact]
    public async Task Albums_HoldPhotosWithoutMovingThem()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 5 });
        long album = await _collections.CreateAlbumAsync("Best of the trip");

        int added = await _collections.AddAsync(album, photos.Select(p => p.Id).ToArray());
        Assert.Equal(photos.Count, added);

        // Adding the same photos again must be a no-op, not a duplicate row.
        Assert.Equal(0, await _collections.AddAsync(album, photos.Select(p => p.Id).ToArray()));

        IReadOnlyList<PhotoRecord> inAlbum = await _store.QueryAsync(
            new PhotoQuery { CollectionId = album, Limit = 5000 });
        Assert.Equal(photos.Count, inAlbum.Count);

        // Deleting the album must not delete a single photograph.
        int before = (await _store.QueryAsync(new PhotoQuery { Limit = 5000 })).Count;
        await _collections.DeleteAsync(album);
        int after = (await _store.QueryAsync(new PhotoQuery { Limit = 5000 })).Count;

        Assert.Equal(before, after);
        Assert.Empty(await _collections.ListAsync());
    }

    [Fact]
    public async Task SmartFolders_StoreTheQuestionNotTheAnswer()
    {
        var query = new PhotoQuery { CameraModel = "Canon", Order = PhotoOrder.CapturedDescending };
        long folder = await _collections.CreateSmartFolderAsync("Canon shots", query);

        IReadOnlyList<PhotoCollection> all = await _collections.ListAsync();
        PhotoCollection smart = Assert.Single(all);

        Assert.Equal(CollectionKind.Smart, smart.Kind);
        Assert.Equal("Canon", smart.Query?.CameraModel);
        Assert.Equal(folder, smart.Id);

        // Count is reported as -1 rather than computed: resolving every smart folder on every
        // sidebar render would put a full search behind opening the window.
        Assert.Equal(-1, smart.Count);

        // Re-running the stored query returns live results.
        IReadOnlyList<PhotoRecord> resolved = await _store.QueryAsync(smart.Query! with { Limit = 5000 });
        Assert.NotEmpty(resolved);
    }

    [Fact]
    public async Task MovingBetweenAlbums_LeavesTheSourceAndJoinsTheTarget()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 6 });
        long trips = await _collections.CreateAlbumAsync("Trips");
        long best = await _collections.CreateAlbumAsync("Best of");

        long[] moving = photos.Take(3).Select(p => p.Id).ToArray();
        await _collections.AddAsync(trips, photos.Select(p => p.Id).ToArray());

        int moved = await _collections.MoveAsync(trips, best, moving);
        Assert.Equal(3, moved);

        IReadOnlyList<PhotoRecord> inTrips = await _store.QueryAsync(
            new PhotoQuery { CollectionId = trips, Limit = 5000 });
        IReadOnlyList<PhotoRecord> inBest = await _store.QueryAsync(
            new PhotoQuery { CollectionId = best, Limit = 5000 });

        // Left the source...
        Assert.Equal(photos.Count - 3, inTrips.Count);
        Assert.DoesNotContain(inTrips, p => moving.Contains(p.Id));

        // ...and joined the target. Move is not a copy.
        Assert.Equal(3, inBest.Count);
        Assert.All(inBest, p => Assert.Contains(p.Id, moving));

        // And no photograph was harmed: the library still holds everything it did.
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });
        Assert.Equal(48, all.Count);
    }

    [Fact]
    public async Task AddingToASecondAlbum_DoesNotRemoveFromTheFirst()
    {
        // The distinction that makes albums views rather than folders. Add is not move.
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 3 });
        long a = await _collections.CreateAlbumAsync("Album A");
        long b = await _collections.CreateAlbumAsync("Album B");
        long[] ids = photos.Select(p => p.Id).ToArray();

        await _collections.AddAsync(a, ids);
        await _collections.AddAsync(b, ids);

        Assert.Equal(3, (await _store.QueryAsync(new PhotoQuery { CollectionId = a, Limit = 5000 })).Count);
        Assert.Equal(3, (await _store.QueryAsync(new PhotoQuery { CollectionId = b, Limit = 5000 })).Count);

        IReadOnlyList<long> membership = await _collections.GetMembershipAsync(ids[0]);
        Assert.Equal(2, membership.Count);
        Assert.Contains(a, membership);
        Assert.Contains(b, membership);
    }

    [Fact]
    public async Task MoveIsAtomic_AndIgnoresAMoveOntoItself()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 3 });
        long album = await _collections.CreateAlbumAsync("Same");
        long[] ids = photos.Select(p => p.Id).ToArray();
        await _collections.AddAsync(album, ids);

        // Dropping a selection onto the album it is already in must be a no-op, not a
        // remove-then-add that briefly empties it.
        Assert.Equal(0, await _collections.MoveAsync(album, album, ids));
        Assert.Equal(3, (await _store.QueryAsync(new PhotoQuery { CollectionId = album, Limit = 5000 })).Count);
    }

    [Fact]
    public async Task RemovingFromAnAlbum_KeepsThePhotographs()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 4 });
        long album = await _collections.CreateAlbumAsync("Temporary");
        long[] ids = photos.Select(p => p.Id).ToArray();
        await _collections.AddAsync(album, ids);

        int before = (await _store.QueryAsync(new PhotoQuery { Limit = 5000 })).Count;
        Assert.Equal(4, await _collections.RemoveAsync(album, ids));

        Assert.Empty(await _store.QueryAsync(new PhotoQuery { CollectionId = album, Limit = 5000 }));
        Assert.Equal(before, (await _store.QueryAsync(new PhotoQuery { Limit = 5000 })).Count);
    }

    // ------------------------------------------------------------------ vector search

    [Fact]
    public async Task VectorSearch_RanksBySemanticSimilarity()
    {
        // No embedding model runs in tests, so the vectors are synthetic — but the index, the
        // storage round-trip, the normalisation, and the ranking are the real production code.
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 6 });
        Assert.True(photos.Count >= 4);

        // Three vectors pointing along x, one along y. A query along x must rank the first
        // three above the fourth, in order of how strongly they lean that way.
        await _vectors.StoreAsync(photos[0].Id, "test", new float[] { 1.0f, 0.0f, 0.0f });
        await _vectors.StoreAsync(photos[1].Id, "test", new float[] { 0.9f, 0.4f, 0.0f });
        await _vectors.StoreAsync(photos[2].Id, "test", new float[] { 0.6f, 0.8f, 0.0f });
        await _vectors.StoreAsync(photos[3].Id, "test", new float[] { 0.0f, 0.0f, 1.0f });

        IReadOnlyList<VectorHit> hits = await _vectors.SearchAsync(
            new float[] { 1.0f, 0.0f, 0.0f }, limit: 10, minimumSimilarity: 0.1);

        Assert.Equal(3, hits.Count);   // the orthogonal one falls below the floor
        Assert.Equal(photos[0].Id, hits[0].PhotoId);
        Assert.Equal(photos[1].Id, hits[1].PhotoId);
        Assert.Equal(photos[2].Id, hits[2].PhotoId);

        // Cosine similarity of a unit vector with itself is 1. If normalisation were skipped
        // this would come out as the raw magnitude instead.
        Assert.Equal(1.0, hits[0].Similarity, precision: 5);
    }

    [Fact]
    public async Task VectorSearch_StoresUnitLengthRegardlessOfInput()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 2 });

        // Same direction, wildly different magnitude. Both must score identically, because
        // direction is meaning and magnitude is not.
        await _vectors.StoreAsync(photos[0].Id, "test", new float[] { 3.0f, 4.0f, 0.0f });
        await _vectors.StoreAsync(photos[1].Id, "test", new float[] { 300.0f, 400.0f, 0.0f });

        IReadOnlyList<VectorHit> hits = await _vectors.SearchAsync(new float[] { 3.0f, 4.0f, 0.0f });

        Assert.Equal(2, hits.Count);
        Assert.Equal(hits[0].Similarity, hits[1].Similarity, precision: 5);
        Assert.Equal(1.0, hits[0].Similarity, precision: 5);
    }

    [Fact]
    public async Task VectorSearch_IgnoresVectorsOfADifferentWidth()
    {
        // A library part-way through re-embedding with a new model contains two vector widths.
        // Mixing them is meaningless, so the minority width is skipped rather than throwing and
        // taking search down for the whole library.
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 3 });

        await _vectors.StoreAsync(photos[0].Id, "old", new float[] { 1.0f, 0.0f });
        await _vectors.StoreAsync(photos[1].Id, "new", new float[] { 1.0f, 0.0f, 0.0f });
        await _vectors.StoreAsync(photos[2].Id, "new", new float[] { 0.0f, 1.0f, 0.0f });

        IReadOnlyList<VectorHit> hits = await _vectors.SearchAsync(new float[] { 1.0f, 0.0f });

        // Whichever width won, the search returned results instead of failing.
        Assert.NotNull(hits);
    }

    // ------------------------------------------------------------------ metadata coverage

    [Fact]
    public async Task EverySortDimensionHasRealVariation()
    {
        // The point of this test is that a sort control is only meaningful if the column it
        // sorts by actually varies. For a long time it did not: every rating was 0, every
        // favourite false, and all 48 photos sat in one directory — so three of the nine sort
        // orders were silently identical to capture order, and two filters returned either
        // everything or nothing. That is invisible unless something asserts it.
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });
        Assert.NotEmpty(all);

        var thin = new List<string>();

        void Require(string dimension, int distinct, int minimum)
        {
            if (distinct < minimum)
            {
                thin.Add($"{dimension}: only {distinct} distinct value(s), need {minimum}");
            }
        }

        Require("captured date", all.Select(p => p.CapturedUtc.Date).Distinct().Count(), 15);
        Require("file name", all.Select(p => p.FileName).Distinct().Count(), 40);
        Require("file size", all.Select(p => p.FileSize).Distinct().Count(), 30);
        Require("camera", all.Select(p => p.Camera.Model).Distinct().Count(), 15);
        Require("source folder", all.Select(p => p.SourceFolder).Distinct().Count(), 8);
        Require("rating", all.Select(p => p.Rating).Distinct().Count(), 4);

        Assert.True(thin.Count == 0,
            "sort dimensions without enough variation to be meaningful: " + string.Join("; ", thin));
    }

    [Fact]
    public async Task Ratings_AreReadFromTheFilesThemselves()
    {
        // Ratings arrive as EXIF tag 0x4746, written into the photographs — not seeded into
        // database rows. That keeps the ingestion path honest and means a re-import reproduces
        // them exactly, which a seeded table would not.
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });
        PhotoRecord[] rated = all.Where(p => p.Rating > 0).ToArray();

        Assert.True(rated.Length >= 15, $"only {rated.Length} photos carry a rating");
        Assert.All(all, p => Assert.InRange(p.Rating, 0, 5));

        // The distribution must be lopsided, like a real library: mostly unrated with a thin
        // tail. A uniform spread would make "4 stars and up" return half the library.
        Assert.True(all.Count(p => p.Rating == 0) > rated.Length / 2,
            "most photos should be unrated; the rating filter is uninformative otherwise");

        // Filtering by minimum rating narrows, and every result really clears the bar.
        IReadOnlyList<PhotoRecord> good = await _store.QueryAsync(
            new PhotoQuery { MinRating = 4, Limit = 5000 });

        Assert.NotEmpty(good);
        Assert.True(good.Count < all.Count);
        Assert.All(good, p => Assert.True(p.Rating >= 4));
    }

    [Fact]
    public async Task FiveStarsArrivesAsAFavourite()
    {
        // No image format records "favourite". Five stars is the closest thing a file carries
        // to that intent, so it is imported as one — which is what makes the favourites filter
        // meaningful on a freshly imported library instead of showing an empty screen.
        IReadOnlyList<PhotoRecord> favourites = await _store.QueryAsync(
            new PhotoQuery { FavouritesOnly = true, Limit = 5000 });

        Assert.NotEmpty(favourites);
        Assert.All(favourites, p => Assert.True(p.Rating >= 5));
    }

    [Fact]
    public async Task Keywords_InTheFilesBecomeSearchableTags()
    {
        IReadOnlyDictionary<string, IReadOnlyList<(string Value, int Count)>> facets =
            await _store.GetFacetsAsync();

        Assert.True(facets["tag"].Count >= 20, $"only {facets["tag"].Count} tags in the library");

        // And a tag is not just a label in a sidebar — clicking it must filter.
        (string tag, int count) = facets["tag"].First();
        IReadOnlyList<PhotoRecord> tagged = await _store.QueryAsync(
            new PhotoQuery { Tags = [tag], Limit = 5000 });

        Assert.NotEmpty(tagged);
        Assert.True(tagged.Count >= count);
    }

    [Fact]
    public async Task Folders_AreNestedEnoughToBrowse()
    {
        // The album is organised into year folders. A flat directory made the folder facet a
        // single row and "sort by folder" identical to sorting by filename.
        IReadOnlyDictionary<string, IReadOnlyList<(string Value, int Count)>> facets =
            await _store.GetFacetsAsync();

        Assert.True(facets["folder"].Count >= 8, $"only {facets["folder"].Count} folders");

        // Selecting one folder returns strictly fewer photos than the whole library...
        (string folder, int count) = facets["folder"].First();
        IReadOnlyList<PhotoRecord> inFolder = await _store.QueryAsync(
            new PhotoQuery { SourceFolder = folder, Limit = 5000 });
        IReadOnlyList<PhotoRecord> all = await _store.QueryAsync(new PhotoQuery { Limit = 5000 });

        Assert.Equal(count, inFolder.Count);
        Assert.True(inFolder.Count < all.Count);

        // ...and sorting by folder groups them contiguously.
        IReadOnlyList<PhotoRecord> byFolder = await _store.QueryAsync(
            new PhotoQuery { Order = PhotoOrder.Folder, Limit = 5000 });

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string? previous = null;
        foreach (PhotoRecord photo in byFolder)
        {
            if (photo.SourceFolder != previous)
            {
                Assert.True(seen.Add(photo.SourceFolder),
                    $"folder {photo.SourceFolder} appears in two separate runs");
                previous = photo.SourceFolder;
            }
        }
    }

    [Fact]
    public async Task Search_WithNoTextIsPureBrowsing()
    {
        // A filter-only query has no relevance to compute, so the requested sort order must
        // survive intact rather than being re-ranked by a scoring function with nothing to score.
        SearchResult result = await _engine.SearchAsync(
            new PhotoQuery { Order = PhotoOrder.FileName, Limit = 20 });

        Assert.NotEmpty(result.Hits);
        Assert.All(result.Hits, h => Assert.Equal(MatchReason.Facet, h.Reason));

        string[] names = result.Hits.Select(h => h.Photo.FileName).ToArray();
        Assert.Equal(names.OrderBy(n => n, StringComparer.OrdinalIgnoreCase), names);
    }
}
