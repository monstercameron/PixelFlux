using PixelFlux.Core.Index;
using PixelFlux.Core.Model;

namespace PixelFlux.Core.Search;

/// <summary>How a photo came to be in a result set. Drives ranking and the "why" chip in the UI.</summary>
[Flags]
public enum MatchReason
{
    /// <summary>No text matching was involved — a pure structured filter.</summary>
    None = 0,

    /// <summary>Matched the full-text index exactly.</summary>
    Exact = 1,

    /// <summary>Matched only after a query word was corrected to a known term.</summary>
    Fuzzy = 2,

    /// <summary>Matched by embedding similarity rather than by any shared word.</summary>
    Semantic = 4,

    /// <summary>Matched a structured facet: camera, place, date, folder, tag.</summary>
    Facet = 8,
}

/// <summary>One photo in a ranked result set.</summary>
/// <param name="Photo">The photo.</param>
/// <param name="Score">Blended relevance, higher is better.</param>
/// <param name="Reason">Which strategies matched it.</param>
/// <param name="Explanation">Short human-readable note, for example <c>"cathederal" → cathedral</c>.</param>
public sealed record SearchHit(
    PhotoRecord Photo,
    double Score,
    MatchReason Reason,
    string? Explanation);

/// <summary>The complete outcome of a search, including what the engine did to the query.</summary>
/// <param name="Hits">Ranked results.</param>
/// <param name="TotalMatched">How many photos matched before the limit was applied.</param>
/// <param name="Corrections">Query words that were corrected, as (typed, used) pairs.</param>
/// <param name="UsedSemantic">Whether embedding search contributed.</param>
public sealed record SearchResult(
    IReadOnlyList<SearchHit> Hits,
    int TotalMatched,
    IReadOnlyList<(string Typed, string Used)> Corrections,
    bool UsedSemantic);

/// <summary>
/// Runs a search across every strategy the library supports and merges the results.
/// </summary>
/// <remarks>
/// <para>
/// Four strategies, applied in order of confidence:
/// </para>
/// <list type="number">
/// <item><description>
/// <b>Structured filters</b> — date range, camera, GPS box, rating, folder, tags. Exact,
/// cheap, and always applied. These narrow the candidate set before anything expensive runs.
/// </description></item>
/// <item><description>
/// <b>Exact full text</b> — FTS5 over titles, captions, tags, filenames, camera strings.
/// </description></item>
/// <item><description>
/// <b>Fuzzy widening</b> — only if exact returned little. Each unmatched query word is compared
/// against the library's own vocabulary and replaced with the nearest real term. Correcting
/// against terms that actually exist, rather than against a dictionary, is what stops
/// <c>"a7iv"</c> from being helpfully corrected into a word nobody's library contains.
/// </description></item>
/// <item><description>
/// <b>Semantic</b> — embedding similarity, when a model has produced vectors. This is the only
/// strategy that can match meaning rather than wording, and the only one that needs the AI
/// pipeline to have run.
/// </description></item>
/// </list>
/// <para>
/// Results are merged rather than concatenated: a photo found by two strategies scores higher
/// than one found by either alone, which is what puts the obvious answers on the first row.
/// </para>
/// <para>
/// <b>What this class deliberately does not do</b> is let anything generate SQL. Language in,
/// <see cref="PhotoQuery"/> out, fixed predicates from there — so no phrasing, and no model
/// output, can reach the database as executable text.
/// </para>
/// </remarks>
public sealed class SearchEngine
{
    // Weights. Tuned so a photo matching two strategies always outranks one matching a single
    // stronger strategy — the agreement between independent signals is worth more than either.
    private const double ExactWeight = 1.00;
    private const double FuzzyWeight = 0.55;
    private const double SemanticWeight = 0.80;

    /// <summary>
    /// How far above the library's average a photo must score to count as a semantic match.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without a floor, embedding search returns the whole library ranked — on a 132-photo
    /// library, a search for "wo" came back with 120 frames. Every photograph has some
    /// similarity to every phrase, so "top matches" with no cut-off means "everything, in an
    /// order". The floor is what turns a ranking into a result.
    /// </para>
    /// <para>
    /// Measured in standard deviations rather than raw similarity, because raw similarity is not
    /// comparable between queries — see <see cref="VectorHit.Standout" />. At 2.0, on the test
    /// library: "red car" returns five cars, "animal" five animals, "cat" three cats, "blonde
    /// hair" five portraits of the one blonde person, and "a submarine underwater" — of which
    /// there is none — returns nothing at all.
    /// </para>
    /// <para>
    /// Two is a deliberate compromise rather than the sharpest value available. A higher floor
    /// separates nonsense more cleanly, but it also drops real matches when a library holds many
    /// photographs of the same thing: those raise the average they are being measured against,
    /// so "blonde hair" scores 2.3 while "cat" scores 4.7 with both perfectly correct. Real
    /// queries with many right answers matter more than made-up ones.
    /// </para>
    /// </remarks>
    private const double SemanticStandoutFloor = 2.0;

    /// <summary>Standout at which a semantic match counts as good as an exact one.</summary>
    /// <remarks>
    /// The weights only mean anything if every strategy contributes on the same scale. The word
    /// strategies produce a rank between 0 and 1; raw cosine similarity does not — after the
    /// hubness correction a strong match scores about 0.09 and a weak one about 0.02, so a
    /// semantic contribution of 0.8 x 0.09 was invisible beside an exact hit's 1.0 x 1.0.
    /// Meaning was in the blend and could not affect it.
    ///
    /// Standout is already query-independent, so it maps onto 0..1 directly: at the floor a
    /// match contributes nothing, at five standard deviations it contributes its full weight.
    /// </remarks>
    private const double SemanticFullMark = 5.0;

    /// <summary>Puts a standout on the same 0-to-1 scale the word strategies use.</summary>
    private static double SemanticRank(double standout) => Math.Clamp(
        (standout - SemanticStandoutFloor) / (SemanticFullMark - SemanticStandoutFloor), 0, 1);

    // Below this many exact hits, widen with fuzzy matching. Not zero: someone who types a word
    // that matches two photos usually still wants the near-misses offered, and someone whose
    // query matched forty does not.
    private const int WidenBelow = 5;

    private readonly PhotoStore _store;
    private readonly VectorIndex _vectors;

    /// <summary>Creates the engine.</summary>
    /// <param name="store">The photo index.</param>
    /// <param name="vectors">The embedding index. Searches still work when it is empty.</param>
    public SearchEngine(PhotoStore store, VectorIndex vectors)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(vectors);

        _store = store;
        _vectors = vectors;
    }

    /// <summary>
    /// Runs a search.
    /// </summary>
    /// <param name="query">Structured filters plus optional free text.</param>
    /// <param name="queryVector">
    /// An embedding of the query text, when one is available. Supplied by the caller rather than
    /// computed here so that <c>PixelFlux.Core</c> keeps no dependency on the model stack — the
    /// search layer must remain usable, and testable, with no ONNX runtime present.
    /// </param>
    /// <param name="cancellationToken">Cancels the search.</param>
    /// <returns>Ranked results and a note of what the engine did to the query.</returns>
    public async Task<SearchResult> SearchAsync(
        PhotoQuery query,
        ReadOnlyMemory<float>? queryVector = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);

        // No text at all: this is pure browsing with filters, so relevance is meaningless and
        // the requested sort order is the whole answer.
        if (string.IsNullOrWhiteSpace(query.Text))
        {
            IReadOnlyList<PhotoRecord> browsed = await _store.QueryAsync(query, cancellationToken)
                .ConfigureAwait(false);

            return new SearchResult(
                browsed.Select(p => new SearchHit(p, 0, MatchReason.Facet, null)).ToArray(),
                browsed.Count,
                [],
                UsedSemantic: false);
        }

        var scores = new Dictionary<long, double>();
        var reasons = new Dictionary<long, MatchReason>();
        var found = new Dictionary<long, PhotoRecord>();
        var corrections = new List<(string Typed, string Used)>();

        void Accumulate(PhotoRecord photo, double weight, double rank, MatchReason reason)
        {
            found[photo.Id] = photo;
            scores[photo.Id] = scores.GetValueOrDefault(photo.Id) + (weight * rank);
            reasons[photo.Id] = reasons.GetValueOrDefault(photo.Id) | reason;
        }

        // ---- 1 + 2: structured filters and exact full text ---------------------------------
        IReadOnlyList<PhotoRecord> exact = await _store
            .QueryAsync(query with { Limit = Math.Max(query.Limit, 200) }, cancellationToken)
            .ConfigureAwait(false);

        for (int i = 0; i < exact.Count; i++)
        {
            // Rank decays with position so that, all else equal, the database's own ordering
            // survives into the blended result rather than being flattened.
            Accumulate(exact[i], ExactWeight, RankOf(i, exact.Count), MatchReason.Exact);
        }

        // ---- 3: meaning ----------------------------------------------------------------------
        bool usedSemantic = false;
        if (queryVector is { } vector && vector.Length > 0)
        {
            IReadOnlyList<VectorHit> hits = await _vectors
                .SearchAsync(vector, limit: 120, cancellationToken: cancellationToken)
                .ConfigureAwait(false);

            foreach (VectorHit hit in hits)
            {
                // Only photographs that stand clear of the library's average for this query.
                // Everything else is the long tail of "somewhat related to everything", which is
                // not a search result — without this, a 132-photo library returned 120 frames.
                if (hit.Standout < SemanticStandoutFloor)
                {
                    continue;
                }

                PhotoRecord? photo = found.GetValueOrDefault(hit.PhotoId)
                                     ?? await _store.GetAsync(hit.PhotoId, cancellationToken)
                                         .ConfigureAwait(false);
                if (photo is null)
                {
                    continue;
                }

                // Semantic hits must still respect the structured filters. A date range or a
                // camera filter is an instruction, not a hint, and a vector match that ignores it
                // would look like the filter is broken.
                if (!PassesFilters(photo, query))
                {
                    continue;
                }

                usedSemantic = true;
                Accumulate(photo, SemanticWeight, SemanticRank(hit.Standout), MatchReason.Semantic);
            }
        }

        // ---- 4: fuzzy widening -----------------------------------------------------------
        //
        // Last, and only when nothing else answered. Guessing at somebody's spelling is a
        // rescue, not a contribution — if either the word index or the meaning index found
        // something, a guess can only add wrong things.
        //
        // The condition is deliberately "no semantic hits at all", not "few results". "blonde
        // hair" matches three photographs by meaning and nothing by word, which is few enough to
        // look like a failed query; widening it turned "hair" into "chair" and buried the three
        // correct answers under furniture. Three right answers is a good result, and the engine
        // must not treat a short correct answer as a reason to start guessing.
        if (exact.Count < WidenBelow && !usedSemantic)
        {
            IReadOnlyList<string> vocabulary = await _store
                .GetVocabularyAsync(cancellationToken).ConfigureAwait(false);

            foreach (string word in Tokenise(query.Text!))
            {
                IReadOnlyList<(string Term, double Score)> suggestions =
                    FuzzyMatch.Correct(word, vocabulary, limit: 3);

                foreach ((string term, double similarity) in suggestions)
                {
                    if (term.Equals(word, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    corrections.Add((word, term));

                    IReadOnlyList<PhotoRecord> widened = await _store
                        .QueryAsync(query with { Text = term, Limit = 100 }, cancellationToken)
                        .ConfigureAwait(false);

                    for (int i = 0; i < widened.Count; i++)
                    {
                        Accumulate(
                            widened[i],
                            FuzzyWeight * similarity,
                            RankOf(i, widened.Count),
                            MatchReason.Fuzzy);
                    }
                }
            }
        }

        SearchHit[] ranked = scores
            .OrderByDescending(kv => kv.Value)
            .ThenByDescending(kv => found[kv.Key].CapturedUtc)
            .Select(kv => new SearchHit(
                found[kv.Key],
                kv.Value,
                reasons[kv.Key],
                Explain(reasons[kv.Key], corrections)))
            .ToArray();

        return new SearchResult(
            ranked.Take(query.Limit).ToArray(),
            ranked.Length,
            corrections.Distinct().ToArray(),
            usedSemantic);
    }

    /// <summary>Rank contribution of position <paramref name="index"/> in a list of <paramref name="count"/>.</summary>
    /// <remarks>
    /// A gentle reciprocal rather than a linear fall-off: the difference between first and third
    /// place should matter, the difference between fortieth and forty-second should not.
    /// </remarks>
    private static double RankOf(int index, int count)
        => count == 0 ? 0 : 1.0 / (1.0 + (index * 0.06));

    /// <summary>
    /// Re-checks the structured filters in memory, for hits that arrived from the vector index
    /// rather than from SQL.
    /// </summary>
    private static bool PassesFilters(PhotoRecord photo, PhotoQuery query)
    {
        if (query.From is { } from && photo.CapturedUtc < from)
        {
            return false;
        }

        if (query.To is { } to && photo.CapturedUtc > to)
        {
            return false;
        }

        if (query.State is { } state && photo.State != state)
        {
            return false;
        }

        if (query.FavouritesOnly == true && !photo.IsFavourite)
        {
            return false;
        }

        if (query.MinRating is { } rating && photo.Rating < rating)
        {
            return false;
        }

        if (!string.IsNullOrWhiteSpace(query.CameraModel))
        {
            string haystack = $"{photo.Camera.Make} {photo.Camera.Model}";
            if (!haystack.Contains(query.CameraModel, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        if (query.Bounds is { } bounds)
        {
            if (photo.Location is not { } location)
            {
                return false;
            }

            if (location.Latitude < bounds.South || location.Latitude > bounds.North ||
                location.Longitude < bounds.West || location.Longitude > bounds.East)
            {
                return false;
            }
        }

        return true;
    }

    private static IEnumerable<string> Tokenise(string text)
        => text.Split([' ', '\t', ',', ';', '.', '"', '\''], StringSplitOptions.RemoveEmptyEntries
                                                            | StringSplitOptions.TrimEntries)
               .Where(w => w.Length >= 2);

    private static string? Explain(MatchReason reason, List<(string Typed, string Used)> corrections)
    {
        if (reason.HasFlag(MatchReason.Exact))
        {
            return null;   // the obvious case needs no explanation
        }

        if (reason.HasFlag(MatchReason.Fuzzy) && corrections.Count > 0)
        {
            (string typed, string used) = corrections[0];
            return $"“{typed}” → {used}";
        }

        return reason.HasFlag(MatchReason.Semantic) ? "similar meaning" : null;
    }
}
