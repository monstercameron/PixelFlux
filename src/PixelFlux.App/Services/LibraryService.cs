using Microsoft.Extensions.Logging;
using PixelFlux.Core.Imaging;
using PixelFlux.Ai.Faces;
using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;
using PixelFlux.Core.Search;
using PixelFlux.Ai.Compute;

namespace PixelFlux.App.Services;

/// <summary>What a row in the time rail represents.</summary>
public enum RailRowKind
{
    /// <summary>A year heading. Clicking it selects the whole year.</summary>
    Year = 0,

    /// <summary>A single month, with a bar showing how many photos are in it.</summary>
    Month = 1,

    /// <summary>
    /// A run of consecutive months containing no photographs, drawn as one compressed marker.
    /// </summary>
    /// <remarks>
    /// These exist because the rail is a <em>timeline</em>, and a timeline that silently omits
    /// its empty stretches is lying about position. The first version only drew months that
    /// contained photos, so June 2007 sat immediately above July 2011 and the vertical axis
    /// meant nothing. Drawing every empty month instead would be honest and unusable — this
    /// corpus spans 230 months, most of them empty, which is three thousand pixels of nothing.
    /// Collapsing a run into one marked gap keeps the axis readable and still says out loud
    /// that time passed and no photographs were taken.
    /// </remarks>
    Gap = 2,
}

/// <summary>One row of the time rail.</summary>
/// <param name="Kind">Whether this is a year heading, a month, or a compressed empty run.</param>
/// <param name="Start">First instant of the period, UTC.</param>
/// <param name="End">Last instant of the period, UTC.</param>
/// <param name="Count">Photos captured in it.</param>
/// <param name="Share">
/// Count as a fraction of the busiest month, 0-1, square-root scaled. Precomputed here so the
/// rail draws bar widths without needing to know the library's maximum.
/// </param>
/// <param name="HasPendingWork">Whether anything in this period is still waiting to be analysed.</param>
/// <param name="GapMonths">For <see cref="RailRowKind.Gap"/>, how many empty months it covers.</param>
public readonly record struct RailRow(
    RailRowKind Kind,
    DateTimeOffset Start,
    DateTimeOffset End,
    int Count,
    double Share,
    bool HasPendingWork,
    int GapMonths);

/// <summary>What the status bar shows about the library and the machines working on it.</summary>
/// <param name="Total">Photos in the library.</param>
/// <param name="Pending">Photos waiting to be analysed.</param>
/// <param name="Processing">Photos currently claimed by a worker.</param>
/// <param name="Unreadable">Photos whose pixels could not be decoded.</param>
/// <param name="Working">Whether anything is in flight — drives the safelight.</param>
public readonly record struct LibraryStatus(
    int Total,
    int Pending,
    int Processing,
    int Unreadable,
    bool Working);

/// <summary>
/// The single seam between the UI and <c>PixelFlux.Core</c>.
/// </summary>
/// <remarks>
/// Razor components talk to this and nothing else — no component opens a database connection or
/// constructs a query object. That keeps the promised two-layer shape honest (component →
/// service → store) and means the view-shaped concerns that do not belong in Core, such as
/// turning a cache key into a URL the WebView can fetch, live in exactly one place.
/// </remarks>
public sealed class LibraryService
{
    /// <summary>
    /// Hostname the WebView maps to the derivative cache directory.
    /// </summary>
    /// <remarks>
    /// The WebView cannot load <c>file://</c> URLs from a page served over its own scheme, so
    /// thumbnails need a route. The options were base64 data URIs — which would put the whole
    /// visible grid through the JS bridge as text, several megabytes per screen — or a virtual
    /// host mapping, which hands the WebView a directory and lets it stream files natively with
    /// its own caching. The mapping is set up in <c>MainPage</c>; this constant is the contract
    /// between that call and the URLs built here.
    /// </remarks>
    public const string CacheHost = "pixelflux.cache";

    private readonly PhotoStore _store;
    private readonly PixelFlux.Core.Pipeline.SettingsStore _settings;
    private readonly CollectionStore _collections;
    private readonly SegmentStore _segments;
    private readonly FaceStore _faces;
    private readonly VectorIndex _vectors;
    private readonly SearchEngine _search;
    private readonly ComputeBackend _compute;
    private readonly ClipEmbedder _clip;
    private readonly QwenVisionDescriber _describer;
    private readonly SemaphoreSlim _calibrating = new(1, 1);
    private bool _calibrated;
    private readonly DerivativeGenerator _derivatives;
    private readonly ILogger<LibraryService> _log;

    /// <summary>Creates the service.</summary>
    /// <param name="store">The photo index.</param>
    /// <param name="collections">The album index.</param>
    /// <param name="segments">The segmentation index.</param>
    /// <param name="faces">The face index.</param>
    /// <param name="vectors">The embedding index.</param>
    /// <param name="derivatives">Derivative generator, used for its key resolution.</param>
    /// <param name="paths">Local storage locations.</param>
    /// <param name="compute">What hardware the models run on.</param>
    /// <param name="settings">Library settings, for preferences that travel with the photographs.</param>
    /// <param name="logger">Logger.</param>
    public LibraryService(
        PhotoStore store,
        CollectionStore collections,
        SegmentStore segments,
        FaceStore faces,
        VectorIndex vectors,
        DerivativeGenerator derivatives,
        LibraryPaths paths,
        ComputeBackend compute,
        PixelFlux.Core.Pipeline.SettingsStore settings,
        ILogger<LibraryService> logger)
    {
        ArgumentNullException.ThrowIfNull(paths);

        _store = store;
        _collections = collections;
        _segments = segments;
        _faces = faces;
        _vectors = vectors;
        _search = new SearchEngine(store, vectors);
        _derivatives = derivatives;

        // Constructed eagerly but opened lazily: this only stats four files, and the encoders
        // are loaded the first time somebody searches or runs a sweep.
        _compute = compute;
        _settings = settings;
        _clip = new ClipEmbedder(
            paths.ClipVisionModelPath,
            paths.ClipTextModelPath,
            paths.ClipVocabularyPath,
            paths.ClipMergesPath,
            compute: compute);

        // Also lazy: this only checks that a configuration file exists. The model behind it is
        // 1.4 GB and opens on the first photograph it is asked about.
        _describer = new QwenVisionDescriber(paths.VisionModelDirectory);
        _log = logger;
        Paths = paths;
    }

    /// <summary>Local storage locations.</summary>
    public LibraryPaths Paths { get; }

    /// <summary>Reads a library setting.</summary>
    /// <param name="key">The setting.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Its value, or null if never set.</returns>
    public Task<string?> GetSettingAsync(string key, CancellationToken cancellationToken = default) =>
        _settings.GetAsync(key, cancellationToken);

    /// <summary>Writes a library setting.</summary>
    /// <param name="key">The setting.</param>
    /// <param name="value">Its value.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public Task SetSettingAsync(
        string key,
        string value,
        CancellationToken cancellationToken = default) =>
        _settings.SetAsync(key, value, cancellationToken);

    /// <summary>Raised after an import finishes, so open views can refresh.</summary>
    public event Action? LibraryChanged;

    /// <summary>Builds the URL the WebView uses to fetch a cached derivative.</summary>
    /// <param name="key">A thumbnail or proxy key, or null.</param>
    /// <returns>An <c>https://</c> URL into the mapped cache host, or null when there is no derivative.</returns>
    public static string? DerivativeUrl(string? key)
        => string.IsNullOrEmpty(key) ? null : $"https://{CacheHost}/{key}";

    /// <summary>Runs a query against the library.</summary>
    /// <param name="query">The filter.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Matching photos.</returns>
    public Task<IReadOnlyList<PhotoRecord>> QueryAsync(
        PhotoQuery query,
        CancellationToken cancellationToken = default)
        => _store.QueryAsync(query, cancellationToken);

    /// <summary>Fetches one photo by id.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The row, or null if it has been removed.</returns>
    public Task<PhotoRecord?> GetPhotoAsync(long photoId, CancellationToken cancellationToken = default)
        => _store.GetAsync(photoId, cancellationToken);

    /// <summary>Fetches the tags on a photo.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Its tags.</returns>
    public Task<IReadOnlyList<PhotoTag>> GetTagsAsync(
        long photoId,
        CancellationToken cancellationToken = default)
        => _store.GetTagsAsync(photoId, cancellationToken);

    /// <summary>Reads the counters behind the status bar.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Current library status.</returns>
    public async Task<LibraryStatus> GetStatusAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyDictionary<ProcessingState, int> counts =
            await _store.GetStateCountsAsync(cancellationToken).ConfigureAwait(false);

        int pending = counts.GetValueOrDefault(ProcessingState.Pending);
        int processing = counts.GetValueOrDefault(ProcessingState.Processing);

        return new LibraryStatus(
            Total: counts.Values.Sum(),
            Pending: pending,
            Processing: processing,
            Unreadable: counts.GetValueOrDefault(ProcessingState.Unreadable),
            Working: processing > 0);
    }

    /// <summary>
    /// Builds the time rail: a continuous newest-first timeline of year headings, months, and
    /// compressed runs of empty months.
    /// </summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Rows in display order, newest first.</returns>
    /// <remarks>
    /// <para>
    /// The rail walks every month between the oldest and newest photograph rather than only the
    /// months that contain one, so vertical position actually corresponds to time. Runs of four
    /// or more empty months collapse into a single <see cref="RailRowKind.Gap"/> row — enough
    /// compression to keep a twenty-year library scrollable, while still showing that the
    /// silence happened.
    /// </para>
    /// <para>
    /// Bar length is square-root scaled against the busiest month, not linear. Real libraries
    /// are extremely spiky — one wedding outproduces the surrounding two years — and on a linear
    /// scale every ordinary month next to it renders as an invisible sliver.
    /// </para>
    /// </remarks>
    public async Task<IReadOnlyList<RailRow>> GetRailAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<TimeBucket> buckets =
            await _store.GetTimeBucketsAsync(cancellationToken).ConfigureAwait(false);

        if (buckets.Count == 0)
        {
            return [];
        }

        IReadOnlyList<PhotoRecord> pending = await _store
            .QueryAsync(new PhotoQuery { State = ProcessingState.Pending, Limit = 5000 }, cancellationToken)
            .ConfigureAwait(false);

        var pendingMonths = pending
            .Select(p => (p.CapturedUtc.Year, p.CapturedUtc.Month))
            .ToHashSet();

        var counts = buckets.ToDictionary(b => (b.Start.Year, b.Start.Month), b => b.Count);
        double busiest = buckets.Max(b => b.Count);

        var newest = new DateTime(buckets[^1].Start.Year, buckets[^1].Start.Month, 1);
        var oldest = new DateTime(buckets[0].Start.Year, buckets[0].Start.Month, 1);

        var rows = new List<RailRow>();
        int currentYear = 0;
        int gapRun = 0;
        DateTime gapEnd = default;

        void FlushGap()
        {
            if (gapRun == 0)
            {
                return;
            }

            // A gap of one to three months is not worth a row of its own; it reads as noise and
            // costs more vertical space than the months it replaces. Those are simply omitted,
            // and the year heading still anchors the position.
            if (gapRun >= 4)
            {
                DateTime gapStart = gapEnd.AddMonths(-(gapRun - 1));
                rows.Add(new RailRow(
                    RailRowKind.Gap,
                    new DateTimeOffset(gapStart, TimeSpan.Zero),
                    new DateTimeOffset(gapEnd.AddMonths(1).AddTicks(-1), TimeSpan.Zero),
                    0, 0, false, gapRun));
            }

            gapRun = 0;
        }

        for (DateTime month = newest; month >= oldest; month = month.AddMonths(-1))
        {
            if (month.Year != currentYear)
            {
                FlushGap();
                currentYear = month.Year;

                int yearTotal = counts
                    .Where(kv => kv.Key.Year == currentYear)
                    .Sum(kv => kv.Value);

                rows.Add(new RailRow(
                    RailRowKind.Year,
                    new DateTimeOffset(new DateTime(currentYear, 1, 1), TimeSpan.Zero),
                    new DateTimeOffset(new DateTime(currentYear, 12, 31, 23, 59, 59), TimeSpan.Zero),
                    yearTotal, 0,
                    pendingMonths.Any(m => m.Year == currentYear),
                    0));
            }

            if (!counts.TryGetValue((month.Year, month.Month), out int count) || count == 0)
            {
                if (gapRun == 0)
                {
                    gapEnd = month;
                }

                gapRun++;
                continue;
            }

            FlushGap();

            rows.Add(new RailRow(
                RailRowKind.Month,
                new DateTimeOffset(month, TimeSpan.Zero),
                new DateTimeOffset(month.AddMonths(1).AddTicks(-1), TimeSpan.Zero),
                count,
                Math.Sqrt(count / busiest),
                pendingMonths.Contains((month.Year, month.Month)),
                0));
        }

        FlushGap();
        return rows;
    }

    /// <summary>Counts photos per value of each browsable dimension, for the filter panel.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Facet name to its values and counts.</returns>
    /// <remarks>
    /// The detected-object facet lives in a different table, so it is merged in here rather than
    /// in SQL. Keeping the panel's contract as one dictionary means adding a browsing dimension
    /// never changes the component that renders them.
    /// </remarks>
    public async Task<IReadOnlyDictionary<string, IReadOnlyList<(string Value, int Count)>>> GetFacetsAsync(
        CancellationToken cancellationToken = default)
    {
        IReadOnlyDictionary<string, IReadOnlyList<(string Value, int Count)>> facets =
            await _store.GetFacetsAsync(cancellationToken).ConfigureAwait(false);

        var merged = new Dictionary<string, IReadOnlyList<(string Value, int Count)>>(facets, StringComparer.Ordinal)
        {
            ["object"] = await _segments.GetObjectFacetAsync(cancellationToken).ConfigureAwait(false),
        };

        return merged;
    }

    /// <summary>Fetches the regions a model located in a photograph, most prominent first.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Its segments.</returns>
    public Task<IReadOnlyList<PhotoSegmentRecord>> GetSegmentsAsync(
        long photoId,
        CancellationToken cancellationToken = default)
        => _segments.GetAsync(photoId, cancellationToken);

    // -------------------------------------------------------------------- search by meaning

    /// <summary>Whether CLIP is installed, so photographs can be searched by meaning.</summary>
    public bool MeaningSearchAvailable => _clip.IsAvailable;

    /// <summary>How many photographs have been described, and how many there are.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Described photographs, and the library total.</returns>
    public Task<(int Described, int Total)> MeaningCoverageAsync(CancellationToken cancellationToken = default)
        => _vectors.CoverageAsync(cancellationToken);

    /// <summary>
    /// Runs a search, blending words and meaning.
    /// </summary>
    /// <param name="query">Structured filters plus the typed text.</param>
    /// <param name="cancellationToken">Cancels the search.</param>
    /// <returns>Ranked results.</returns>
    /// <remarks>
    /// <para>
    /// The query text is turned into a vector here and handed to the engine, which keeps
    /// <c>PixelFlux.Core</c> free of any dependency on the model stack — the search layer stays
    /// testable with no ONNX runtime present, which is why the engine takes a vector rather than
    /// a phrase.
    /// </para>
    /// <para>
    /// Meaning never replaces words, it is added to them. An exact filename match must still win:
    /// somebody typing "IMG_4021" wants that file, not the photograph CLIP thinks most resembles
    /// the idea of a filename.
    /// </para>
    /// </remarks>
    public async Task<SearchResult> SearchAsync(PhotoQuery query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);

        float[]? vector = null;

        if (!string.IsNullOrWhiteSpace(query.Text) && _clip.IsAvailable)
        {
            await CalibrateOnceAsync(cancellationToken).ConfigureAwait(false);
            vector = await _clip.EmbedQueryAsync(query.Text, cancellationToken).ConfigureAwait(false);
        }

        return await _search.SearchAsync(query, vector, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Works out which photographs are agreeable to everything, once per session.
    /// </summary>
    /// <remarks>
    /// About thirty model runs, so roughly a second, and only on the first search of a session
    /// — not at startup, where it would delay the window appearing for a feature the user may
    /// never touch. A failure here is not fatal: search carries on uncorrected, which is how it
    /// behaved before the correction existed.
    /// </remarks>
    private async Task CalibrateOnceAsync(CancellationToken cancellationToken)
    {
        if (_calibrated)
        {
            return;
        }

        await _calibrating.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            if (_calibrated)
            {
                return;
            }

            var bank = new List<ReadOnlyMemory<float>>(ReferencePhrases.All.Count);

            foreach (string reference in ReferencePhrases.All)
            {
                if (await _clip.EmbedTextAsync(reference, cancellationToken).ConfigureAwait(false) is { } v)
                {
                    bank.Add(v);
                }
            }

            await _vectors.CalibrateAsync(bank, cancellationToken).ConfigureAwait(false);
            _calibrated = true;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _log.LogWarning(ex, "Could not calibrate the embedding index; searching uncorrected");
            _calibrated = true;
        }
        finally
        {
            _calibrating.Release();
        }
    }

    /// <summary>
    /// Gives every photograph that has none a vector, so it can be searched by meaning.
    /// </summary>
    /// <remarks>
    /// Distinct from <see cref="DescribeLibraryAsync" />, which writes English. This produces
    /// numbers: 512 of them per photograph, comparable against a phrase. The two are separate
    /// passes over the library because they cost wildly different amounts — 66 milliseconds a
    /// photograph against thirteen seconds — and pairing them would make the cheap one as slow
    /// as the expensive one.
    /// </remarks>
    /// <param name="progress">Progress sink.</param>
    /// <param name="cancellationToken">Stops the run; finished photographs are kept.</param>
    /// <returns>How many photographs were described.</returns>
    public async Task<int> IndexLibraryForMeaningAsync(
        IProgress<EmbeddingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!_clip.IsAvailable)
        {
            _log.LogInformation("CLIP is not installed; nothing to describe");
            return 0;
        }

        var worker = new EmbeddingWorker(_store, _vectors, _clip, Paths.CacheRoot);
        int done = await worker.RunAsync(int.MaxValue, progress, cancellationToken).ConfigureAwait(false);

        // The hub figures were computed for the old set of rows and mean nothing for the new one.
        _calibrated = false;

        LibraryChanged?.Invoke();
        return done;
    }

    // ------------------------------------------------------------------------- describing

    /// <summary>Whether the vision-language model is installed.</summary>
    public bool DescriptionsAvailable => _describer.IsAvailable;

    /// <summary>How many photographs have a written description, and how many there are.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Described photographs, and the library total.</returns>
    public Task<(int Described, int Total)> DescriptionCoverageAsync(
        CancellationToken cancellationToken = default)
        => _store.DescriptionCoverageAsync(cancellationToken);

    /// <summary>
    /// Writes a description of every photograph that has none.
    /// </summary>
    /// <param name="progress">Progress sink.</param>
    /// <param name="cancellationToken">Stops the run; finished photographs are kept.</param>
    /// <returns>How many photographs were described.</returns>
    /// <remarks>
    /// Minutes rather than seconds — around thirteen seconds a photograph. The caller is
    /// expected to show progress and let the user carry on; the describer deliberately runs on a
    /// third of the machine's cores so that is possible.
    /// </remarks>
    public async Task<int> DescribeLibraryAsync(
        IProgress<DescriptionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!_describer.IsAvailable)
        {
            _log.LogInformation("No vision-language model installed; nothing to describe");
            return 0;
        }

        var worker = new DescriptionWorker(_store, _describer, Paths.CacheRoot);
        int done = await worker.RunAsync(int.MaxValue, progress, cancellationToken).ConfigureAwait(false);

        LibraryChanged?.Invoke();
        return done;
    }

    // --------------------------------------------------------------------------- labelling

    /// <summary>Attaches a keyword the user typed to a photo.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="tag">The keyword.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>Whether anything was added.</returns>
    public Task<bool> AddUserTagAsync(long photoId, string tag, CancellationToken cancellationToken = default)
        => _store.AddUserTagAsync(photoId, tag, cancellationToken);

    /// <summary>Removes a keyword the user typed.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="tag">The keyword.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>Whether anything was removed.</returns>
    public Task<bool> RemoveUserTagAsync(long photoId, string tag, CancellationToken cancellationToken = default)
        => _store.RemoveUserTagAsync(photoId, tag, cancellationToken);

    /// <summary>Sets, or clears, what the user calls one detected region.</summary>
    /// <param name="segmentId">The region.</param>
    /// <param name="label">The user's word, or null to fall back to the model's.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>Whether the region was changed.</returns>
    public Task<bool> SetSegmentLabelAsync(
        long segmentId,
        string? label,
        CancellationToken cancellationToken = default)
        => _segments.SetUserLabelAsync(segmentId, label, cancellationToken);

    /// <summary>Words already in use in this library, for suggesting as the user types.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Tags and detected object names, alphabetically.</returns>
    /// <remarks>
    /// Suggestions come from the library, never from a fixed dictionary. Somebody labelling
    /// their own photographs is naming their own things, and the useful prompt is "you have
    /// called something this before", not a list of the eighty classes a model happens to know.
    /// </remarks>
    public async Task<IReadOnlyList<string>> LabelVocabularyAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<(string Label, int Count)> objects =
            await _segments.GetObjectFacetAsync(cancellationToken).ConfigureAwait(false);

        IReadOnlyList<string> tags = await _store.GetVocabularyAsync(cancellationToken).ConfigureAwait(false);

        return objects.Select(o => o.Label)
            .Concat(tags)
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .Take(400)
            .ToList();
    }

    // ------------------------------------------------------------------------------ faces

    /// <summary>Fetches the faces found in a photograph, most prominent first.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Its faces.</returns>
    public Task<IReadOnlyList<PhotoFaceRecord>> GetFacesAsync(
        long photoId,
        CancellationToken cancellationToken = default)
        => _faces.GetAsync(photoId, cancellationToken);

    /// <summary>Lists faces across the whole library, for the faces page.</summary>
    /// <param name="order">How to order them.</param>
    /// <param name="minimumConfidence">Drops faces the detector was less sure of than this.</param>
    /// <param name="limit">Most faces to return.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Faces with enough of their photograph to render a card.</returns>
    public Task<IReadOnlyList<FaceListing>> ListFacesAsync(
        FaceOrder order = FaceOrder.Prominence,
        double minimumConfidence = 0,
        int limit = 500,
        CancellationToken cancellationToken = default)
        => _faces.ListAsync(order, minimumConfidence, null, limit, 0, cancellationToken);

    /// <summary>Lists the library's faces collapsed to one entry per person.</summary>
    /// <param name="threshold">How alike two faces must be to count as one person.</param>
    /// <param name="order">How to order the people.</param>
    /// <param name="minimumConfidence">Drops faces the detector was less sure of than this.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Groups, the people who appear most often first.</returns>
    public Task<IReadOnlyList<FaceGroup>> ListPeopleAsync(
        double threshold,
        FaceOrder order = FaceOrder.Prominence,
        double minimumConfidence = 0,
        CancellationToken cancellationToken = default)
        => _faces.ListPeopleAsync(threshold, order, minimumConfidence, 500, cancellationToken);

    /// <summary>Counts the faces in the library, and the photographs holding them.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Total faces and how many photographs contain at least one.</returns>
    public Task<(int Faces, int Photographs)> CountFacesAsync(CancellationToken cancellationToken = default)
        => _faces.CountAsync(cancellationToken);

    /// <summary>
    /// Runs a face sweep over the photographs no detector has looked at yet.
    /// </summary>
    /// <param name="progress">Progress sink.</param>
    /// <param name="cancellationToken">Stops the sweep; finished photographs are kept.</param>
    /// <returns>How many photographs were examined and how many faces were found.</returns>
    /// <remarks>
    /// <para>
    /// The detector is constructed per sweep rather than held as a singleton. It owns an ONNX
    /// session and about 40 MB of arena, and a photo library is swept once and then left alone
    /// for weeks — paying that for the life of the process to save a hundred milliseconds on a
    /// button nobody presses twice would be the wrong way round.
    /// </para>
    /// <para>
    /// Nothing here touches the network. The model is a file on disk and the faces it finds are
    /// written to this machine only, which is the point: face data is the most sensitive thing
    /// in a photo library and the design gives it nowhere to go.
    /// </para>
    /// </remarks>
    public async Task<(int Examined, int Faces)> SweepFacesAsync(
        IProgress<FaceSweepProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        using var detector = new YuNetFaceDetector(Paths.FaceModelPath, compute: _compute);

        if (!detector.IsAvailable)
        {
            _log.LogInformation("No face model at {Path}; sweep skipped", Paths.FaceModelPath);
            return (0, 0);
        }

        // The recognizer is optional and much larger than the detector. Without it the sweep
        // still finds and crops every face; only grouping is missing.
        using var recognizer = new SFaceRecognizer(Paths.RecognitionModelPath, compute: _compute);
        var worker = new FaceWorker(_store, _faces, detector, Paths.CacheRoot, recognizer);
        (int examined, int found) = await worker
            .RunAsync(int.MaxValue, progress, cancellationToken).ConfigureAwait(false);

        LibraryChanged?.Invoke();
        return (examined, found);
    }

    /// <summary>Finds the faces that look like a given one.</summary>
    /// <param name="faceId">The face to search for.</param>
    /// <param name="threshold">Minimum similarity to count as the same person.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Matches, most alike first, including the face searched for.</returns>
    public Task<IReadOnlyList<FaceMatch>> FindSimilarFacesAsync(
        long faceId,
        double threshold,
        CancellationToken cancellationToken = default)
        => _faces.FindSimilarAsync(faceId, threshold, 500, cancellationToken);

    /// <summary>How many detected faces carry a comparable vector.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Faces with an embedding, and faces in total.</returns>
    public Task<(int Embedded, int Total)> FaceComparabilityAsync(CancellationToken cancellationToken = default)
        => _faces.EmbeddingCoverageAsync(cancellationToken);

    /// <summary>Whether the recognition model is installed, so faces can be compared.</summary>
    public bool FaceMatchingAvailable => File.Exists(Paths.RecognitionModelPath);

    /// <summary>Whether a face model is installed and the sweep can run.</summary>
    public bool FaceDetectionAvailable => File.Exists(Paths.FaceModelPath);

    /// <summary>Imports folders into the library.</summary>
    /// <param name="folders">Folders to walk.</param>
    /// <param name="progress">Progress sink.</param>
    /// <param name="cancellationToken">Cancels the import.</param>
    /// <returns>What was imported.</returns>
    public async Task<IngestResult> ImportAsync(
        IReadOnlyList<string> folders,
        IProgress<IngestProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var ingestor = new LibraryIngestor(_store, _derivatives, null);
        IngestResult result = await ingestor.ImportAsync(folders, progress, cancellationToken).ConfigureAwait(false);

        _log.LogInformation(
            "Imported {Imported} new photos ({Duplicates} already known, {Failed} failed) in {Elapsed}",
            result.Imported, result.Duplicates, result.Failed, result.Elapsed);

        LibraryChanged?.Invoke();
        return result;
    }

    // ---------------------------------------------------------------------------- albums

    /// <summary>Lists every album, with live counts.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Albums in display order.</returns>
    public Task<IReadOnlyList<PhotoCollection>> GetAlbumsAsync(CancellationToken cancellationToken = default)
        => _collections.ListAsync(cancellationToken);

    /// <summary>Creates an album.</summary>
    /// <param name="name">Display name.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>The new album's id, or null if the name is already taken.</returns>
    /// <remarks>
    /// A duplicate name returns null rather than throwing. The name column is uniquely indexed,
    /// so this is an ordinary thing for a user to do by accident, and it deserves a message in
    /// the dialog rather than the crash boundary.
    /// </remarks>
    public async Task<long?> CreateAlbumAsync(string name, CancellationToken cancellationToken = default)
    {
        try
        {
            long id = await _collections.CreateAlbumAsync(name, cancellationToken).ConfigureAwait(false);
            LibraryChanged?.Invoke();
            return id;
        }
        catch (Microsoft.Data.Sqlite.SqliteException ex) when (ex.SqliteErrorCode == 19)
        {
            return null;   // SQLITE_CONSTRAINT — the unique index on name
        }
    }

    /// <summary>Adds photos to an album, leaving any other membership intact.</summary>
    /// <param name="albumId">Target album.</param>
    /// <param name="photoIds">Photos to add.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many were newly added.</returns>
    public async Task<int> AddToAlbumAsync(
        long albumId,
        IReadOnlyList<long> photoIds,
        CancellationToken cancellationToken = default)
    {
        int added = await _collections.AddAsync(albumId, photoIds, cancellationToken).ConfigureAwait(false);
        LibraryChanged?.Invoke();
        return added;
    }

    /// <summary>Moves photos out of one album and into another.</summary>
    /// <param name="fromAlbumId">Album to take them out of.</param>
    /// <param name="toAlbumId">Album to put them into.</param>
    /// <param name="photoIds">Photos to move.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many moved.</returns>
    public async Task<int> MoveBetweenAlbumsAsync(
        long fromAlbumId,
        long toAlbumId,
        IReadOnlyList<long> photoIds,
        CancellationToken cancellationToken = default)
    {
        int moved = await _collections
            .MoveAsync(fromAlbumId, toAlbumId, photoIds, cancellationToken).ConfigureAwait(false);
        LibraryChanged?.Invoke();
        return moved;
    }

    /// <summary>
    /// Removes photos from an album. The photographs themselves are untouched.
    /// </summary>
    /// <param name="albumId">The album.</param>
    /// <param name="photoIds">Photos to remove.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many were removed.</returns>
    public async Task<int> RemoveFromAlbumAsync(
        long albumId,
        IReadOnlyList<long> photoIds,
        CancellationToken cancellationToken = default)
    {
        int removed = await _collections.RemoveAsync(albumId, photoIds, cancellationToken).ConfigureAwait(false);
        LibraryChanged?.Invoke();
        return removed;
    }

    /// <summary>Renames an album.</summary>
    /// <param name="albumId">The album.</param>
    /// <param name="name">New name.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task RenameAlbumAsync(long albumId, string name, CancellationToken cancellationToken = default)
    {
        await _collections.RenameAsync(albumId, name, cancellationToken).ConfigureAwait(false);
        LibraryChanged?.Invoke();
    }

    /// <summary>Deletes an album. No photographs are deleted.</summary>
    /// <param name="albumId">The album.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task DeleteAlbumAsync(long albumId, CancellationToken cancellationToken = default)
    {
        await _collections.DeleteAsync(albumId, cancellationToken).ConfigureAwait(false);
        LibraryChanged?.Invoke();
    }

    /// <summary>Absolute path to a cached derivative, for export and for diagnostics.</summary>
    /// <param name="key">A derivative key.</param>
    /// <returns>The absolute path.</returns>
    public string ResolveDerivativePath(string key) => _derivatives.ResolvePath(key);
}
