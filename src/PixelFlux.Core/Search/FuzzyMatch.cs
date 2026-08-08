namespace PixelFlux.Core.Search;

/// <summary>
/// Approximate string matching, for search that survives a typo.
/// </summary>
/// <remarks>
/// <para>
/// Full-text search is exact: type <c>lightouse</c> and SQLite's FTS5 returns nothing, because
/// no document contains that token. That is the correct behaviour for a database and the wrong
/// behaviour for a search box. This class is the layer that turns a near-miss into a hit.
/// </para>
/// <para>
/// Two measures, used for different jobs:
/// </para>
/// <list type="bullet">
/// <item><description>
/// <b>Damerau-Levenshtein</b> for short strings — a tag, a camera model, one query word. It
/// counts insertions, deletions, substitutions, <em>and transpositions</em>, the last of which
/// matters more than it sounds: <c>teh</c> for <c>the</c> and <c>Cnaon</c> for <c>Canon</c> are
/// the single most common human typo, and plain Levenshtein scores them as two edits, the same
/// as two unrelated errors.
/// </description></item>
/// <item><description>
/// <b>Trigram similarity</b> for longer text — a caption, a filename. It compares sets of
/// three-character slices, so it degrades gracefully with length and does not care about word
/// order, where edit distance on a forty-character caption is both slow and meaningless.
/// </description></item>
/// </list>
/// <para>
/// Neither is a substitute for the FTS index. The intended flow is: run the exact query first,
/// and only if it returns too little, widen with these against the vocabulary the library
/// actually contains. Fuzzy matching a query against every row would be both slow and worse —
/// it is a fallback, not a search engine.
/// </para>
/// </remarks>
public static class FuzzyMatch
{
    /// <summary>
    /// Computes Damerau-Levenshtein edit distance between two strings, case-insensitively.
    /// </summary>
    /// <param name="a">First string.</param>
    /// <param name="b">Second string.</param>
    /// <param name="maxDistance">
    /// Give up and return <paramref name="maxDistance"/> + 1 once the best possible result
    /// exceeds this. Callers only ever care about small distances, and the early exit turns the
    /// common "these two words are nothing alike" case from a full matrix fill into a few rows.
    /// </param>
    /// <returns>The number of edits, or <paramref name="maxDistance"/> + 1 if it exceeds the cap.</returns>
    public static int Distance(string a, string b, int maxDistance = 3)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        a = a.ToLowerInvariant();
        b = b.ToLowerInvariant();

        if (a == b)
        {
            return 0;
        }

        // A length difference alone is a lower bound on the distance, so this rejects the
        // hopeless cases before allocating anything.
        if (Math.Abs(a.Length - b.Length) > maxDistance)
        {
            return maxDistance + 1;
        }

        if (a.Length == 0)
        {
            return b.Length;
        }

        if (b.Length == 0)
        {
            return a.Length;
        }

        // Three rows rather than a full matrix: the recurrence only ever reads the two previous
        // rows, and the row before that is what makes transposition detectable.
        int width = b.Length + 1;
        Span<int> twoBack = width <= 128 ? stackalloc int[width] : new int[width];
        Span<int> oneBack = width <= 128 ? stackalloc int[width] : new int[width];
        Span<int> current = width <= 128 ? stackalloc int[width] : new int[width];

        for (int j = 0; j <= b.Length; j++)
        {
            oneBack[j] = j;
        }

        for (int i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            int rowBest = current[0];

            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;

                int value = Math.Min(
                    Math.Min(current[j - 1] + 1,   // insertion
                             oneBack[j] + 1),      // deletion
                    oneBack[j - 1] + cost);        // substitution

                // Transposition: the two characters are swapped relative to each other.
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1])
                {
                    value = Math.Min(value, twoBack[j - 2] + 1);
                }

                current[j] = value;
                rowBest = Math.Min(rowBest, value);
            }

            // Every subsequent row is >= the best of this one, so once the whole row is past
            // the cap the answer can only get worse.
            if (rowBest > maxDistance)
            {
                return maxDistance + 1;
            }

            Span<int> recycled = twoBack;
            twoBack = oneBack;
            oneBack = current;
            current = recycled;
        }

        return oneBack[b.Length];
    }

    /// <summary>
    /// Whether two words are close enough to be treated as the same search term.
    /// </summary>
    /// <param name="query">The word the user typed.</param>
    /// <param name="candidate">A word from the library's vocabulary.</param>
    /// <returns><see langword="true"/> when the two are within the length-scaled edit budget.</returns>
    /// <remarks>
    /// The budget scales with length, because a fixed edit distance is far too permissive on
    /// short words and far too strict on long ones. At distance 2, <c>cat</c> matches
    /// <c>car</c>, <c>can</c>, <c>bat</c>, <c>hat</c>, and <c>mat</c> — useless. The same budget
    /// on <c>cathedral</c> catches only genuine misspellings. So: no slack under 4 characters,
    /// one edit up to 7, two beyond that.
    /// </remarks>
    public static bool IsCloseEnough(string query, string candidate)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(candidate);

        int budget = query.Length switch
        {
            <= 3 => 0,
            <= 7 => 1,
            _ => 2,
        };

        return budget > 0 && Distance(query, candidate, budget) <= budget;
    }

    /// <summary>
    /// Computes trigram similarity between two strings: shared three-character slices over
    /// total distinct slices (a Jaccard index).
    /// </summary>
    /// <param name="a">First string.</param>
    /// <param name="b">Second string.</param>
    /// <returns>0 (nothing in common) to 1 (identical).</returns>
    /// <remarks>
    /// Both strings are padded so that word beginnings and endings produce trigrams too;
    /// without that, <c>cat</c> and <c>concatenate</c> would share every trigram <c>cat</c> has
    /// and score a perfect match on the shorter side.
    /// </remarks>
    public static double TrigramSimilarity(string a, string b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        HashSet<string> setA = Trigrams(a);
        HashSet<string> setB = Trigrams(b);

        if (setA.Count == 0 || setB.Count == 0)
        {
            return string.Equals(a, b, StringComparison.OrdinalIgnoreCase) ? 1 : 0;
        }

        int shared = setA.Count(t => setB.Contains(t));
        return (double)shared / (setA.Count + setB.Count - shared);
    }

    private static HashSet<string> Trigrams(string value)
    {
        string padded = "  " + value.ToLowerInvariant().Trim() + " ";
        var set = new HashSet<string>(StringComparer.Ordinal);

        for (int i = 0; i + 3 <= padded.Length; i++)
        {
            set.Add(padded.Substring(i, 3));
        }

        return set;
    }

    /// <summary>
    /// Finds words in the vocabulary that the query was probably a misspelling of.
    /// </summary>
    /// <param name="query">The word as typed.</param>
    /// <param name="vocabulary">Words the library actually contains.</param>
    /// <param name="limit">Most corrections to return.</param>
    /// <returns>Candidate corrections, best first.</returns>
    /// <remarks>
    /// <para>
    /// Deliberately not <see cref="Suggest" />. That method is autocomplete: it treats a prefix
    /// as an almost-certain match, which is right when somebody is halfway through typing and
    /// catastrophic when they have finished. Used as a corrector it turned "cat" into
    /// "cathedral" and "red" into "redmi" — both prefixes of words that happened to be in the
    /// library — so a search for a red car returned a plate of food and a living room.
    /// </para>
    /// <para>
    /// A correction is an edit, not an extension. Only genuine typos qualify: the same
    /// length-scaled budget as <see cref="IsCloseEnough" />, which gives a three-letter word no
    /// budget at all and therefore leaves short words alone. Short words are exactly where
    /// prefix expansion does the most damage, and exactly where a typo is least likely to be
    /// recoverable anyway.
    /// </para>
    /// </remarks>
    public static IReadOnlyList<(string Term, double Score)> Correct(
        string query,
        IEnumerable<string> vocabulary,
        int limit = 3)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(vocabulary);

        if (query.Length < 4)
        {
            return [];
        }

        var scored = new List<(string Term, double Score)>();

        foreach (string term in vocabulary.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (string.Equals(term, query, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (IsCloseEnough(query, term))
            {
                scored.Add((term, TrigramSimilarity(query, term)));
            }
        }

        return scored
            .OrderByDescending(s => s.Score)
            .ThenBy(s => s.Term, StringComparer.OrdinalIgnoreCase)
            .Take(limit)
            .ToArray();
    }

    /// <summary>
    /// Picks the entries of a vocabulary closest to a query word, for autocomplete.
    /// </summary>
    /// <param name="query">The word the user typed.</param>
    /// <param name="vocabulary">Known terms — the library's tags, camera models, filenames.</param>
    /// <param name="limit">Maximum suggestions to return.</param>
    /// <returns>Matches ordered best-first, each with a 0-1 score.</returns>
    /// <remarks>
    /// A prefix match outranks everything, because somebody typing into a box is usually part way
    /// through a word — someone who has typed <c>cathe</c> wants <c>cathedral</c>, not the
    /// similarly-distant <c>cache</c>.
    ///
    /// That makes this the wrong function for correcting a finished query, where a prefix is
    /// almost always a different word: see <see cref="Correct" />.
    /// </remarks>
    public static IReadOnlyList<(string Term, double Score)> Suggest(
        string query,
        IEnumerable<string> vocabulary,
        int limit = 8)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(vocabulary);

        if (query.Length < 2)
        {
            return [];
        }

        var scored = new List<(string Term, double Score)>();

        foreach (string term in vocabulary.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (term.StartsWith(query, StringComparison.OrdinalIgnoreCase))
            {
                // Longer completions of the same prefix are slightly less likely to be meant.
                scored.Add((term, 1.0 - (0.001 * Math.Min(term.Length, 100))));
                continue;
            }

            if (term.Contains(query, StringComparison.OrdinalIgnoreCase))
            {
                scored.Add((term, 0.80));
                continue;
            }

            if (IsCloseEnough(query, term))
            {
                scored.Add((term, 0.70));
                continue;
            }

            double similarity = TrigramSimilarity(query, term);
            if (similarity >= 0.42)
            {
                scored.Add((term, similarity * 0.65));
            }
        }

        return scored
            .OrderByDescending(s => s.Score)
            .ThenBy(s => s.Term, StringComparer.OrdinalIgnoreCase)
            .Take(limit)
            .ToArray();
    }
}
