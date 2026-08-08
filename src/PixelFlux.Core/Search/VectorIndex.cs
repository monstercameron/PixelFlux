using System.Buffers.Binary;
using System.Numerics;
using Microsoft.Data.Sqlite;
using PixelFlux.Core.Index;

namespace PixelFlux.Core.Search;

/// <summary>One photo matched by embedding similarity.</summary>
/// <param name="PhotoId">The photo.</param>
/// <param name="Similarity">Cosine similarity to the query, -1 to 1.</param>
/// <param name="Standout">
/// How far above the library's average this photo scored, in standard deviations.
/// </param>
/// <remarks>
/// <para>
/// <b>Why the raw similarity is not enough.</b> CLIP's text-image scores are squeezed into a
/// narrow band and the band moves with the query. Measured on this library: "red car" tops out
/// at 0.287 and "a submarine underwater" — of which there is none — still tops out at 0.200,
/// while the nonsense phrase "xyzzy quantum banana" reaches 0.237, higher than several genuine
/// matches for other queries. So no fixed cut-off can separate a hit from a miss: 0.237 is
/// excellent for one query and meaningless for another.
/// </para>
/// <para>
/// What does separate them is the shape of the distribution. When the library really contains
/// the thing, a few photos stand well clear of the rest; when it does not, every photo scores
/// about the same and the top one is merely the least arbitrary. Expressing each score in
/// standard deviations above the mean makes that comparable across queries, which is what a
/// relevance cut-off needs.
/// </para>
/// </remarks>
public readonly record struct VectorHit(long PhotoId, double Similarity, double Standout = 0);

/// <summary>
/// Semantic search over image embeddings.
/// </summary>
/// <remarks>
/// <para>
/// This is what makes <c>"dogs running on a beach at sunset"</c> find the right photograph even
/// though none of those words appear anywhere in its metadata. An embedding model places images
/// and text in the same vector space; the photo whose vector points most nearly in the same
/// direction as the query's is the closest match in meaning rather than in wording.
/// </para>
/// <para>
/// <b>Brute force, deliberately.</b> Every vector is loaded and compared on each search. The
/// obvious objection is that this does not scale, and the arithmetic answers it: a 768-dimension
/// float32 vector is 3 KB, so a 50,000-photo library is 150 MB resident and one search is 38
/// million multiply-adds — a few milliseconds with SIMD. An approximate index (HNSW, IVF) would
/// add a dependency, a build step, a staleness problem, and a recall trade-off in exchange for
/// speed this application does not need. Revisit past a few hundred thousand photos; not before.
/// </para>
/// <para>
/// <b>Vectors are stored normalised.</b> Cosine similarity between unit vectors is just their
/// dot product, so normalising once at write time removes two square roots from every
/// comparison at read time. <see cref="StoreAsync"/> enforces it rather than trusting callers.
/// </para>
/// </remarks>
public sealed class VectorIndex
{
    private readonly PhotoDatabase _database;

    // The loaded vectors, held as one flat array rather than an array of arrays.
    //
    // Contiguity is the whole trick: scoring walks memory in a straight line, which keeps the
    // prefetcher useful and lets each row be sliced as a Span with no allocation. A
    // float[][] of 50,000 rows would be 50,000 separate heap objects scattered across the
    // generation, and the scan would be dominated by cache misses rather than arithmetic.
    private float[] _vectors = [];
    private long[] _ids = [];
    private int _dimensions;
    private bool _loaded;

    /// <summary>Per-photo general agreeableness, or null when never calibrated.</summary>
    private double[]? _hub;

    /// <summary>Row count the hub figures were computed for, so a reload invalidates them.</summary>
    private int _hubRows;

    private readonly Lock _gate = new();

    /// <summary>Creates an index over a database.</summary>
    /// <param name="database">The database holding the embeddings table.</param>
    public VectorIndex(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    /// <summary>Number of photos currently loaded into memory.</summary>
    public int Count => _ids.Length;

    /// <summary>Dimensionality of the loaded vectors, or 0 when none are loaded.</summary>
    public int Dimensions => _dimensions;

    /// <summary>
    /// Stores a photo's embedding, normalising it to unit length first.
    /// </summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="model">Identifier of the model that produced the vector, for example <c>siglip2-base</c>.</param>
    /// <param name="vector">The embedding. Copied; the caller may reuse the array.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <exception cref="ArgumentException">The vector is empty or has zero magnitude.</exception>
    /// <remarks>
    /// The model identifier is stored per row rather than globally on purpose. Two models
    /// produce vectors that are not comparable even at the same dimensionality, and a library
    /// part-way through a re-embedding run contains both. Recording which is which is what lets
    /// that migration happen in the background instead of requiring a stop-the-world rebuild.
    /// </remarks>
    public async Task StoreAsync(
        long photoId,
        string model,
        ReadOnlyMemory<float> vector,
        CancellationToken cancellationToken = default)
    {
        if (vector.Length == 0)
        {
            throw new ArgumentException("An embedding cannot be empty.", nameof(vector));
        }

        float[] unit = Normalise(vector.Span);

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO photo_embeddings (photo_id, model, dims, vector)
            VALUES ($id, $model, $dims, $vec)
            ON CONFLICT(photo_id) DO UPDATE SET
                model = excluded.model, dims = excluded.dims, vector = excluded.vector;
            """;
        command.Parameters.AddWithValue("$id", photoId);
        command.Parameters.AddWithValue("$model", model);
        command.Parameters.AddWithValue("$dims", unit.Length);
        command.Parameters.AddWithValue("$vec", ToBlob(unit));

        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

        // The in-memory copy is now stale. Dropping it is cheaper and far less error-prone than
        // splicing one row into the flat array, and ingestion writes in bulk anyway.
        Invalidate();
    }

    /// <summary>Photographs with no vector from a given model.</summary>
    /// <param name="model">The model about to run.</param>
    /// <param name="limit">Most ids to return.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Ids still to describe, oldest-indexed first.</returns>
    /// <remarks>
    /// Keyed on the model, not on mere presence of a row. Vectors from two models are not
    /// comparable, so installing a better one has to re-describe the library — and asking only
    /// "does this photograph have a vector" would report the job already done and leave the
    /// library permanently split between two spaces.
    /// </remarks>
    public async Task<IReadOnlyList<long>> PendingAsync(
        string model,
        int limit = 100000,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT p.id FROM photos p
            WHERE NOT EXISTS (
                SELECT 1 FROM photo_embeddings e WHERE e.photo_id = p.id AND e.model = $model)
            ORDER BY p.id
            LIMIT $limit;
            """;
        command.Parameters.AddWithValue("$model", model);
        command.Parameters.AddWithValue("$limit", limit);

        var ids = new List<long>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            ids.Add(reader.GetInt64(0));
        }

        return ids;
    }

    /// <summary>How many photographs carry a vector, and how many there are.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Described photographs, and the library total.</returns>
    public async Task<(int Described, int Total)> CoverageAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            "SELECT (SELECT COUNT(*) FROM photo_embeddings), (SELECT COUNT(*) FROM photos);";

        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        return await reader.ReadAsync(cancellationToken).ConfigureAwait(false)
            ? (reader.GetInt32(0), reader.GetInt32(1))
            : (0, 0);
    }

    /// <summary>
    /// Teaches the index which photographs are indiscriminately popular, so it can discount them.
    /// </summary>
    /// <param name="bank">
    /// Vectors for a spread of unrelated reference phrases. Supplied by the caller because
    /// producing them needs the model stack, which this assembly deliberately does not depend on.
    /// </param>
    /// <param name="cancellationToken">Cancels the calibration.</param>
    /// <remarks>
    /// <para>
    /// <b>The problem this solves is called hubness.</b> In a high-dimensional space a few points
    /// end up unusually close to everything, and in CLIP's image space those points are
    /// photographs with a lot going on — a crowded restaurant table, a busy street. Measured on
    /// this library, the same three photographs of a meal came top for "xyzzy quantum banana"
    /// and for "a submarine underwater", scoring higher than the genuine best match for
    /// "animal". They are not similar to those queries; they are similar to everything.
    /// </para>
    /// <para>
    /// <b>The correction.</b> Score each photograph against a bank of unrelated reference
    /// phrases and keep its average. That average is how agreeable the photograph is in general,
    /// with nothing to do with any particular search, so subtracting it from a query's score
    /// leaves only what is specific to that query. A hub loses most of its score; a photograph
    /// that is unremarkable in general but a strong match here keeps almost all of it.
    /// </para>
    /// <para>
    /// Calibration is optional. Without it the index behaves exactly as it did before, which
    /// matters because the bank costs a model run per phrase and a caller may not have one.
    /// </para>
    /// </remarks>
    public async Task CalibrateAsync(
        IReadOnlyList<ReadOnlyMemory<float>> bank,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(bank);
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);

        if (_ids.Length == 0 || bank.Count == 0)
        {
            return;
        }

        var hub = new double[_ids.Length];
        int used = 0;

        foreach (ReadOnlyMemory<float> phrase in bank)
        {
            if (phrase.Length != _dimensions)
            {
                continue;
            }

            float[] unit = Normalise(phrase.Span);
            used++;

            for (int row = 0; row < _ids.Length; row++)
            {
                hub[row] += Dot(unit, new ReadOnlySpan<float>(_vectors, row * _dimensions, _dimensions));
            }
        }

        if (used == 0)
        {
            return;
        }

        for (int row = 0; row < hub.Length; row++)
        {
            hub[row] /= used;
        }

        lock (_gate)
        {
            _hub = hub;
            _hubRows = _ids.Length;
        }
    }

    /// <summary>Discards the in-memory copy so the next search reloads from the database.</summary>
    public void Invalidate()
    {
        lock (_gate)
        {
            _loaded = false;
            _vectors = [];
            _ids = [];
            _dimensions = 0;
        }
    }

    /// <summary>
    /// Finds the photos whose embeddings point most nearly in the same direction as a query.
    /// </summary>
    /// <param name="query">The query vector. Normalised internally; need not be unit length.</param>
    /// <param name="limit">Maximum hits to return.</param>
    /// <param name="minimumSimilarity">
    /// Discard anything below this. Cosine similarity has no natural zero point for "unrelated"
    /// — in a typical image/text space, genuinely unrelated pairs sit around 0.1-0.2 rather than
    /// 0 — so a floor is what stops a nonsense query from confidently returning the whole
    /// library in arbitrary order.
    /// </param>
    /// <param name="cancellationToken">Cancels the search.</param>
    /// <returns>Hits ordered by similarity, highest first.</returns>
    public async Task<IReadOnlyList<VectorHit>> SearchAsync(
        ReadOnlyMemory<float> query,
        int limit = 60,
        double minimumSimilarity = 0.18,
        CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);

        if (_ids.Length == 0 || query.Length != _dimensions)
        {
            return [];
        }

        float[] unit = Normalise(query.Span);

        // Every photo is scored before anything is filtered, because the filter itself depends
        // on the whole distribution. This is one pass over a contiguous array — the same work
        // the search was already doing.
        // The hub figures are per-row, so they are only meaningful for the set of rows they were
        // computed against. A library that has grown since falls back to uncorrected scores
        // rather than to misaligned ones.
        double[]? hub = _hub is { } h && _hubRows == _ids.Length ? h : null;

        var all = new double[_ids.Length];
        double total = 0;

        for (int row = 0; row < _ids.Length; row++)
        {
            var candidate = new ReadOnlySpan<float>(_vectors, row * _dimensions, _dimensions);
            all[row] = Dot(unit, candidate) - (hub?[row] ?? 0);
            total += all[row];
        }

        double mean = total / all.Length;
        double variance = 0;

        foreach (double value in all)
        {
            double delta = value - mean;
            variance += delta * delta;
        }

        // A library where every photo scores identically has no standout to speak of; guarding
        // the divisor keeps that case at zero rather than at infinity.
        double deviation = Math.Sqrt(variance / all.Length);
        bool spread = deviation > 1e-6;

        var hits = new List<VectorHit>(Math.Min(_ids.Length, limit * 4));

        for (int row = 0; row < _ids.Length; row++)
        {
            // The corrected score is a difference, not a cosine, so the caller's absolute floor
            // no longer applies to it. Once calibrated, relevance is judged by standout instead —
            // which is the comparable measure, and the reason the floor defaults to being loose.
            if (hub is null && all[row] < minimumSimilarity)
            {
                continue;
            }

            hits.Add(new VectorHit(
                _ids[row],
                all[row],
                spread ? (all[row] - mean) / deviation : 0));
        }

        hits.Sort((x, y) => y.Similarity.CompareTo(x.Similarity));
        return hits.Count > limit ? hits.GetRange(0, limit) : hits;
    }

    /// <summary>
    /// Finds the photos most similar to a given photo — the "more like this" operation.
    /// </summary>
    /// <param name="photoId">The photo to match against.</param>
    /// <param name="limit">Maximum hits.</param>
    /// <param name="cancellationToken">Cancels the search.</param>
    /// <returns>Similar photos, excluding the source, ordered by similarity.</returns>
    public async Task<IReadOnlyList<VectorHit>> SimilarToAsync(
        long photoId,
        int limit = 24,
        CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);

        int row = Array.IndexOf(_ids, photoId);
        if (row < 0)
        {
            return [];
        }

        float[] source = new ReadOnlySpan<float>(_vectors, row * _dimensions, _dimensions).ToArray();

        IReadOnlyList<VectorHit> hits = await SearchAsync(
            source, limit + 1, minimumSimilarity: 0.0, cancellationToken).ConfigureAwait(false);

        return hits.Where(h => h.PhotoId != photoId).Take(limit).ToArray();
    }

    private async Task EnsureLoadedAsync(CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            if (_loaded)
            {
                return;
            }
        }

        var ids = new List<long>();
        var blobs = new List<byte[]>();
        int dimensions = 0;

        await using (SqliteConnection connection = _database.Open())
        {
            await using SqliteCommand command = connection.CreateCommand();

            // Ordered by dims so that a library mid-migration between two embedding models
            // groups cleanly; only the majority dimensionality is loaded (see below).
            command.CommandText = "SELECT photo_id, dims, vector FROM photo_embeddings ORDER BY dims, photo_id;";

            await using SqliteDataReader reader =
                await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                int rowDimensions = reader.GetInt32(1);

                // Vectors of different widths cannot be compared at all. Rather than throwing —
                // which would break search for the whole library because one photo was embedded
                // by a different model — the first width seen wins and the rest are skipped
                // until they are re-embedded.
                if (dimensions == 0)
                {
                    dimensions = rowDimensions;
                }
                else if (rowDimensions != dimensions)
                {
                    continue;
                }

                ids.Add(reader.GetInt64(0));
                blobs.Add((byte[])reader[2]);
            }
        }

        float[] flat = new float[ids.Count * dimensions];
        for (int row = 0; row < blobs.Count; row++)
        {
            FromBlob(blobs[row], flat.AsSpan(row * dimensions, dimensions));
        }

        lock (_gate)
        {
            _ids = ids.ToArray();
            _vectors = flat;
            _dimensions = dimensions;
            _loaded = true;
        }
    }

    /// <summary>Dot product of two equal-length spans, vectorised where the hardware allows.</summary>
    /// <remarks>
    /// Both inputs are unit vectors, so this <em>is</em> the cosine similarity — no division by
    /// magnitudes is needed, which is the entire reason vectors are normalised at write time.
    /// </remarks>
    private static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int width = Vector<float>.Count;
        var accumulator = Vector<float>.Zero;
        int i = 0;

        for (; i + width <= a.Length; i += width)
        {
            accumulator += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
        }

        float total = Vector.Dot(accumulator, Vector<float>.One);

        for (; i < a.Length; i++)
        {
            total += a[i] * b[i];
        }

        return total;
    }

    private static float[] Normalise(ReadOnlySpan<float> vector)
    {
        double sumOfSquares = 0;
        foreach (float value in vector)
        {
            sumOfSquares += value * (double)value;
        }

        double magnitude = Math.Sqrt(sumOfSquares);
        if (magnitude < 1e-9)
        {
            throw new ArgumentException("An embedding with zero magnitude has no direction.", nameof(vector));
        }

        float[] unit = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            unit[i] = (float)(vector[i] / magnitude);
        }

        return unit;
    }

    // Little-endian float32, explicitly. The default BitConverter layout is host-dependent, and
    // this database travels: a library on a NAS is read by whatever machine mounts it.
    private static byte[] ToBlob(ReadOnlySpan<float> vector)
    {
        byte[] blob = new byte[vector.Length * sizeof(float)];
        for (int i = 0; i < vector.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(blob.AsSpan(i * sizeof(float)), vector[i]);
        }

        return blob;
    }

    private static void FromBlob(ReadOnlySpan<byte> blob, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = BinaryPrimitives.ReadSingleLittleEndian(blob[(i * sizeof(float))..]);
        }
    }
}
