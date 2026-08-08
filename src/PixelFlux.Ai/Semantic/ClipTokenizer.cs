using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace PixelFlux.Ai.Semantic;

/// <summary>
/// CLIP's byte-pair tokenizer, reimplemented.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this is hand-written.</b> Text and image only land in the same vector space if the text
/// is split into exactly the tokens the model was trained on. Get the tokenizer subtly wrong —
/// one missing <c>&lt;/w&gt;</c>, the wrong merge order, a byte mapped differently — and nothing
/// throws: the model returns a confident 512-dimensional vector describing a sentence nobody
/// wrote. There is no .NET package for CLIP's variant, so it is written out here and tested
/// against the token ids the reference implementation produces.
/// </para>
/// <para>
/// <b>The algorithm.</b> Lower-case and collapse whitespace, split on a fixed pattern that keeps
/// contractions and punctuation runs together, re-encode each piece byte-by-byte into a private
/// alphabet, mark the end of the word, then repeatedly merge the highest-ranked adjacent pair
/// until none of the remaining pairs appear in the merge table.
/// </para>
/// <para>
/// <b>The private alphabet.</b> Every byte becomes a printable character, so a token table of
/// text strings can address arbitrary bytes and any input — emoji, accents, a filename in
/// Japanese — tokenizes without a special case. It is the same mapping GPT-2 uses.
/// </para>
/// </remarks>
public sealed class ClipTokenizer
{
    /// <summary>Marks the start of the sequence.</summary>
    private const string StartOfText = "<|startoftext|>";

    /// <summary>Marks the end of the sequence, and is where the model reads its answer from.</summary>
    private const string EndOfText = "<|endoftext|>";

    /// <summary>
    /// Longest sequence the model has positions for.
    /// </summary>
    /// <remarks>
    /// CLIP learned 77 position embeddings and has no way to represent a 78th, so anything longer
    /// is truncated. A search query never comes close; this is a guard against a pathological
    /// input rather than a real limit.
    /// </remarks>
    public const int MaximumTokens = 77;

    /// <summary>
    /// How CLIP splits text before any merging happens.
    /// </summary>
    /// <remarks>
    /// The contractions are listed individually and first, so "don't" becomes "don" + "'t" rather
    /// than being cut at the apostrophe by the punctuation rule. Letters group into words, digits
    /// are taken <em>one at a time</em> — CLIP's own quirk, and the reason "2019" is four tokens —
    /// and everything else that is not whitespace runs together.
    /// </remarks>
    private static readonly Regex Pattern = new(
        @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex Whitespace = new(@"\s+", RegexOptions.Compiled);

    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<(string A, string B), int> _merges;
    private readonly Dictionary<byte, char> _byteToChar;
    private readonly Dictionary<string, int[]> _cache = [];

    /// <summary>Loads the tokenizer from CLIP's vocabulary and merge table.</summary>
    /// <param name="vocabularyPath">Path to <c>vocab.json</c>.</param>
    /// <param name="mergesPath">Path to <c>merges.txt</c>.</param>
    /// <exception cref="FileNotFoundException">Either file is missing.</exception>
    public ClipTokenizer(string vocabularyPath, string mergesPath)
    {
        _vocabulary = JsonSerializer.Deserialize<Dictionary<string, int>>(
            File.ReadAllText(vocabularyPath))
            ?? throw new InvalidDataException($"{vocabularyPath} is not a token table.");

        _merges = [];

        // Rank is position in the file: earlier merges bind tighter. The first line is a version
        // header, not a merge.
        int rank = 0;
        foreach (string line in File.ReadLines(mergesPath))
        {
            if (rank == 0 && line.StartsWith("#version", StringComparison.Ordinal))
            {
                continue;
            }

            string[] parts = line.Split(' ');
            if (parts.Length == 2)
            {
                _merges.TryAdd((parts[0], parts[1]), rank++);
            }
        }

        _byteToChar = BuildByteAlphabet();

        StartId = _vocabulary.GetValueOrDefault(StartOfText, 49406);
        EndId = _vocabulary.GetValueOrDefault(EndOfText, 49407);
    }

    /// <summary>Token id that opens every sequence.</summary>
    public int StartId { get; }

    /// <summary>Token id that closes every sequence.</summary>
    public int EndId { get; }

    /// <summary>How many distinct tokens the model knows.</summary>
    public int VocabularySize => _vocabulary.Count;

    /// <summary>
    /// Turns a phrase into the token ids the text encoder expects.
    /// </summary>
    /// <param name="text">What the user typed.</param>
    /// <returns>Start marker, the phrase's tokens, end marker.</returns>
    /// <remarks>
    /// Deliberately not padded to <see cref="MaximumTokens"/>. The exported graph pools its
    /// answer at the end marker, and the padding CLIP conventionally uses is that same token —
    /// so a padded sequence relies on the graph picking the first of several identical ids. The
    /// input length is dynamic, so feeding only the real tokens sidesteps the question entirely
    /// and gives the encoder nothing after the phrase to attend to.
    /// </remarks>
    public int[] Encode(string text)
    {
        var ids = new List<int>(16) { StartId };

        foreach (Match match in Pattern.Matches(Clean(text)))
        {
            foreach (int id in EncodePiece(match.Value))
            {
                if (ids.Count >= MaximumTokens - 1)
                {
                    break;
                }

                ids.Add(id);
            }
        }

        ids.Add(EndId);
        return [.. ids];
    }

    /// <summary>Lower-cases and collapses runs of whitespace.</summary>
    /// <remarks>
    /// Invariant lower-casing, explicitly. Under a Turkish locale the default would map "I" to a
    /// dotless "ı", which is not in CLIP's vocabulary — so searching for "IMAX" would tokenize
    /// differently depending on the machine's regional settings.
    /// </remarks>
    private static string Clean(string text)
        => Whitespace.Replace((text ?? string.Empty).Trim(), " ").ToLowerInvariant();

    private int[] EncodePiece(string piece)
    {
        if (_cache.TryGetValue(piece, out int[]? cached))
        {
            return cached;
        }

        // Bytes, not characters: this is what lets one token table cover every script.
        var encoded = new StringBuilder();
        foreach (byte b in Encoding.UTF8.GetBytes(piece))
        {
            encoded.Append(_byteToChar[b]);
        }

        if (encoded.Length == 0)
        {
            return _cache[piece] = [];
        }

        // Word-final marker on the last symbol. Without it "dog" inside "dogma" and "dog" as a
        // whole word would be the same token, and CLIP distinguishes them.
        var symbols = new List<string>(encoded.Length);
        for (int i = 0; i < encoded.Length - 1; i++)
        {
            symbols.Add(encoded[i].ToString());
        }

        symbols.Add(encoded[^1] + "</w>");

        Merge(symbols);

        var ids = new int[symbols.Count];
        for (int i = 0; i < symbols.Count; i++)
        {
            // Every byte of the alphabet is in the table, so a miss can only be a corrupt vocab
            // file. Falling back to the end marker keeps a bad file from crashing a search.
            ids[i] = _vocabulary.GetValueOrDefault(symbols[i], EndId);
        }

        return _cache[piece] = ids;
    }

    /// <summary>
    /// Repeatedly joins the highest-ranked adjacent pair until no pair is mergeable.
    /// </summary>
    /// <remarks>
    /// Rank order, not left-to-right. Applying merges in reading order produces a different and
    /// wrong tokenization for most words — the merge table is a priority list, and the whole
    /// scheme depends on always taking the tightest-binding pair currently present.
    /// </remarks>
    private void Merge(List<string> symbols)
    {
        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_merges.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                return;
            }

            symbols[bestIndex] += symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }
    }

    /// <summary>
    /// Maps all 256 byte values onto distinct printable characters.
    /// </summary>
    /// <remarks>
    /// Bytes that are already printable ASCII, Latin-1 letters, or Latin-1 symbols keep their own
    /// character; the remaining 68 — control codes, space, and the soft hyphen — are moved up
    /// into an unused block starting at U+0100. The point is that no byte maps to whitespace or
    /// to a character the splitter would treat specially, so the byte string round-trips through
    /// a text-keyed table without any escaping.
    /// </remarks>
    private static Dictionary<byte, char> BuildByteAlphabet()
    {
        var map = new Dictionary<byte, char>(256);
        var taken = new List<int>();

        for (int b = '!'; b <= '~'; b++) { taken.Add(b); }
        for (int b = 0xA1; b <= 0xAC; b++) { taken.Add(b); }
        for (int b = 0xAE; b <= 0xFF; b++) { taken.Add(b); }

        foreach (int b in taken)
        {
            map[(byte)b] = (char)b;
        }

        int next = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!map.ContainsKey((byte)b))
            {
                map[(byte)b] = (char)(256 + next);
                next++;
            }
        }

        return map;
    }
}
