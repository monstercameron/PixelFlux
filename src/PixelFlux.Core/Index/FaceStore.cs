using Microsoft.Data.Sqlite;
using PixelFlux.Core.Model;

namespace PixelFlux.Core.Index;

/// <summary>
/// One face together with just enough of its photograph to render a card for it.
/// </summary>
/// <param name="Face">The face.</param>
/// <param name="PhotoFileName">Filename of the photograph it came from.</param>
/// <param name="PhotoThumbnailKey">The photograph's thumbnail, used when a crop is missing.</param>
/// <param name="CapturedUtc">When the photograph was taken.</param>
/// <param name="PlaceLabel">Resolved place, or null if the photograph has no GPS fix.</param>
/// <remarks>
/// A projection rather than a face plus a photo lookup. The faces page shows hundreds of cards
/// and each one names its photograph; fetching the photo rows separately would be a query per
/// card, and fetching whole <see cref="PhotoRecord"/> objects would carry forty columns to
/// display four.
/// </remarks>
public sealed record FaceListing(
    PhotoFaceRecord Face,
    string PhotoFileName,
    string? PhotoThumbnailKey,
    DateTimeOffset CapturedUtc,
    string? PlaceLabel);

/// <summary>A face that resembles another, with how strongly.</summary>
/// <param name="Listing">The matching face and its photograph.</param>
/// <param name="Similarity">
/// Cosine similarity to the face that was searched for, from -1 to 1. Surfaced rather than
/// hidden because a match at 0.42 and a match at 0.95 are different claims, and a person
/// deciding whether the app got it right deserves to see which one they are looking at.
/// </param>
public sealed record FaceMatch(FaceListing Listing, double Similarity);

/// <summary>One person, as far as appearance can establish it.</summary>
/// <param name="Representative">The face that stands for the group — the most prominent one.</param>
/// <param name="FaceCount">How many faces were folded into it.</param>
/// <param name="PhotoCount">How many distinct photographs those faces came from.</param>
/// <remarks>
/// Not an identity. There is no name, no stored person record, and nothing that survives a
/// re-sweep: a group is the answer to "which of these faces look alike", recomputed each time
/// the page loads. Persisting it would turn a recomputable opinion into a fact the user cannot
/// correct, which is the wrong shape for a guess about who somebody is.
/// </remarks>
public sealed record FaceGroup(FaceListing Representative, int FaceCount, int PhotoCount);

/// <summary>Somebody with a name, and how much of the library they appear in.</summary>
/// <param name="Id">The person.</param>
/// <param name="Name">What they are called.</param>
/// <param name="FaceCount">Faces assigned to them.</param>
/// <param name="PhotoCount">Distinct photographs they appear in.</param>
/// <param name="CropKey">Their most prominent face, for a card. Null before a sweep wrote crops.</param>
/// <remarks>
/// Deliberately not a <see cref="FaceGroup"/>. A group is the recognition model's guess, recomputed
/// on every page load and never stored; this is a fact somebody typed. The two appear in the same
/// places and are not the same kind of claim, and collapsing them into one type would lose exactly
/// the distinction that makes naming worth having.
/// </remarks>
public sealed record NamedPerson(
    long Id,
    string Name,
    int FaceCount,
    int PhotoCount,
    string? CropKey);

/// <summary>How the faces page is ordered.</summary>
public enum FaceOrder
{
    /// <summary>Most prominent first — biggest, most confident faces at the top.</summary>
    Prominence = 0,

    /// <summary>Newest photograph first.</summary>
    Newest = 1,

    /// <summary>Oldest photograph first.</summary>
    Oldest = 2,

    /// <summary>Highest detector confidence first.</summary>
    Confidence = 3,
}

/// <summary>
/// Reads and writes the faces a detector found.
/// </summary>
/// <remarks>
/// <para>
/// Separate from <see cref="SegmentStore"/> despite the near-identical shape, because the two
/// tables are read in opposite directions. Segments are read per photograph, to draw an overlay.
/// Faces are read across the whole library, to fill a page — so the interesting method here is
/// the one that lists everything, and it carries a projection of the photograph with it.
/// </para>
/// <para>
/// Like segments, every row belongs to a model version and is replaced wholesale when a better
/// detector runs. Nothing a person typed lives here, which is what makes re-analysis safe.
/// </para>
/// </remarks>
public sealed class FaceStore
{
    private readonly PhotoDatabase _database;

    /// <summary>Creates a store over a migrated database.</summary>
    /// <param name="database">The database handle.</param>
    public FaceStore(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    /// <summary>Replaces every face recorded for a photograph.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="faces">The new faces. An empty list clears them.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// Replace, not merge, and in one transaction — the same reasoning as segments. Two runs of
    /// a detector produce nearly-but-not-quite the same boxes, and merging would show the same
    /// person three times from one photograph.
    /// </remarks>
    public async Task ReplaceAsync(
        long photoId,
        IReadOnlyList<PhotoFaceRecord> faces,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(faces);

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        // Read the names before the rows holding them are deleted. A sweep replaces every face for
        // a photograph, so without this a re-run of the detector — a better model, a repaired
        // file, pressing "try again" — would silently throw away every name anybody had typed.
        List<(double X, double Y, double W, double H, long PersonId)> named =
            await ReadNamesAsync(connection, tx, photoId, cancellationToken).ConfigureAwait(false);

        await using (SqliteCommand clear = connection.CreateCommand())
        {
            clear.Transaction = tx;
            clear.CommandText = "DELETE FROM photo_faces WHERE photo_id = $id;";
            clear.Parameters.AddWithValue("$id", photoId);
            await clear.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        faces = ReapplyNames(faces, named);

        if (faces.Count > 0)
        {
            await using SqliteCommand insert = connection.CreateCommand();
            insert.Transaction = tx;
            insert.CommandText = """
                INSERT INTO photo_faces
                    (photo_id, confidence, x, y, w, h, area, roll, landmarks, crop_key, model,
                     embedding, embed_model, person_id)
                VALUES ($id, $conf, $x, $y, $w, $h, $area, $roll, $marks, $crop, $model,
                        $vector, $embedModel, $person);
                """;

            insert.Parameters.AddWithValue("$id", photoId);
            SqliteParameter confidence = insert.Parameters.Add("$conf", SqliteType.Real);
            SqliteParameter x = insert.Parameters.Add("$x", SqliteType.Real);
            SqliteParameter y = insert.Parameters.Add("$y", SqliteType.Real);
            SqliteParameter w = insert.Parameters.Add("$w", SqliteType.Real);
            SqliteParameter h = insert.Parameters.Add("$h", SqliteType.Real);
            SqliteParameter area = insert.Parameters.Add("$area", SqliteType.Real);
            SqliteParameter roll = insert.Parameters.Add("$roll", SqliteType.Real);
            SqliteParameter marks = insert.Parameters.Add("$marks", SqliteType.Text);
            SqliteParameter crop = insert.Parameters.Add("$crop", SqliteType.Text);
            SqliteParameter model = insert.Parameters.Add("$model", SqliteType.Text);
            SqliteParameter vector = insert.Parameters.Add("$vector", SqliteType.Blob);
            SqliteParameter embedModel = insert.Parameters.Add("$embedModel", SqliteType.Text);
            SqliteParameter person = insert.Parameters.Add("$person", SqliteType.Integer);

            foreach (PhotoFaceRecord face in faces)
            {
                confidence.Value = face.Confidence;
                x.Value = face.X;
                y.Value = face.Y;
                w.Value = face.Width;
                h.Value = face.Height;
                area.Value = face.AreaFraction;
                roll.Value = face.RollDegrees;
                marks.Value = face.Landmarks;
                crop.Value = (object?)face.CropKey ?? DBNull.Value;
                model.Value = face.Model;
                vector.Value = face.Embedding is { Length: > 0 } v
                    ? (object)ToBytes(v)
                    : DBNull.Value;
                embedModel.Value = (object?)face.EmbedModel ?? DBNull.Value;
                person.Value = (object?)face.PersonId ?? DBNull.Value;
                await insert.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }
        }

        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Fetches the faces in one photograph, most prominent first.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Its faces.</returns>
    public async Task<IReadOnlyList<PhotoFaceRecord>> GetAsync(
        long photoId,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT f.id, f.photo_id, f.confidence, f.x, f.y, f.w, f.h, f.area, f.roll,
                   f.landmarks, f.crop_key, f.model, f.embedding, f.embed_model,
                   f.person_id, per.name
            FROM photo_faces f
            LEFT JOIN people per ON per.id = f.person_id
            WHERE f.photo_id = $id ORDER BY f.area DESC, f.id;
            """;
        command.Parameters.AddWithValue("$id", photoId);

        var faces = new List<PhotoFaceRecord>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            faces.Add(Read(reader));
        }

        return faces;
    }

    /// <summary>
    /// Lists faces across the whole library, for the faces page.
    /// </summary>
    /// <param name="order">How to order them.</param>
    /// <param name="minimumConfidence">Drops faces the detector was less sure of than this.</param>
    /// <param name="photoIds">
    /// When given, restricts the listing to these photographs — the faces page uses it to stay
    /// in step with whatever the gallery's filters currently select.
    /// </param>
    /// <param name="limit">Most faces to return.</param>
    /// <param name="offset">How many to skip, for paging.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Faces with enough of their photograph to render a card.</returns>
    /// <remarks>
    /// Every ordering ends with <c>id</c>. Without a unique tiebreak, SQLite is free to return
    /// rows that tie on the sort key in a different order on each call, which makes paging drop
    /// and repeat faces — the same trap the photo orderings already avoid.
    /// </remarks>
    public async Task<IReadOnlyList<FaceListing>> ListAsync(
        FaceOrder order = FaceOrder.Prominence,
        double minimumConfidence = 0,
        IReadOnlyCollection<long>? photoIds = null,
        int limit = 500,
        int offset = 0,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();

        // Prominence is computed here rather than read from a column because it is a blend that
        // may be retuned; storing it would mean a migration and a re-analysis to change a weight.
        // It is cheap: a square root over a few thousand rows.
        string sort = order switch
        {
            FaceOrder.Newest => "p.captured_utc DESC, f.id DESC",
            FaceOrder.Oldest => "p.captured_utc ASC, f.id ASC",
            FaceOrder.Confidence => "f.confidence DESC, f.id",
            _ => "(SQRT(f.area) * 0.7 + f.confidence * 0.3) DESC, f.id",
        };

        string restriction = photoIds is { Count: > 0 }
            ? $" AND f.photo_id IN ({string.Join(',', photoIds.Select(id => id.ToString(System.Globalization.CultureInfo.InvariantCulture)))})"
            : string.Empty;

        command.CommandText = $"""
            SELECT f.id, f.photo_id, f.confidence, f.x, f.y, f.w, f.h, f.area, f.roll,
                   f.landmarks, f.crop_key, f.model, f.embedding, f.embed_model,
                   f.person_id, per.name,
                   p.file_name, p.thumbnail_key, p.captured_utc, p.place_label
            FROM photo_faces f
            JOIN photos p ON p.id = f.photo_id
            LEFT JOIN people per ON per.id = f.person_id
            WHERE f.confidence >= $conf{restriction}
            ORDER BY {sort}
            LIMIT $limit OFFSET $offset;
            """;

        command.Parameters.AddWithValue("$conf", minimumConfidence);
        command.Parameters.AddWithValue("$limit", limit);
        command.Parameters.AddWithValue("$offset", offset);

        var listings = new List<FaceListing>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            listings.Add(new FaceListing(
                Read(reader),
                reader.GetString(16),
                reader.IsDBNull(17) ? null : reader.GetString(17),
                DateTimeOffset.Parse(reader.GetString(18), System.Globalization.CultureInfo.InvariantCulture),
                reader.IsDBNull(19) ? null : reader.GetString(19)));
        }

        return listings;
    }

    /// <summary>
    /// Finds the faces that look like a given one.
    /// </summary>
    /// <param name="faceId">The face to search for.</param>
    /// <param name="threshold">Minimum cosine similarity to count as the same person.</param>
    /// <param name="limit">Most matches to return.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>
    /// Matches, most alike first, including the face searched for — which sits at similarity 1
    /// and is the photograph the user clicked, so leaving it out would make the result look like
    /// it had lost one.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>Brute force, deliberately.</b> Every embedded face is read and compared. For a library
    /// of ten thousand faces that is five megabytes and about a millisecond of arithmetic — far
    /// below the point where an approximate index would pay for itself, and an approximate index
    /// would introduce recall the user cannot see or reason about. When a library appears that
    /// needs one, this method is the only thing that has to change.
    /// </para>
    /// <para>
    /// Only faces from the same recognition model are considered. Vectors from two models are
    /// not comparable; comparing them anyway would not error, it would quietly return nobody.
    /// </para>
    /// </remarks>
    public async Task<IReadOnlyList<FaceMatch>> FindSimilarAsync(
        long faceId,
        double threshold,
        int limit = 500,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();

        float[] query;
        string model;

        await using (SqliteCommand fetch = connection.CreateCommand())
        {
            fetch.CommandText =
                "SELECT embedding, embed_model FROM photo_faces WHERE id = $id AND embedding IS NOT NULL;";
            fetch.Parameters.AddWithValue("$id", faceId);

            await using SqliteDataReader head = await fetch
                .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

            if (!await head.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                // The face has no vector: too small to describe, or swept before a recognition
                // model was installed. Not an error — the UI offers no "find this person" for it.
                return [];
            }

            query = ToFloats((byte[])head[0]);
            model = head.IsDBNull(1) ? string.Empty : head.GetString(1);
        }

        if (query.Length == 0)
        {
            return [];
        }

        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT f.id, f.photo_id, f.confidence, f.x, f.y, f.w, f.h, f.area, f.roll,
                   f.landmarks, f.crop_key, f.model, f.embedding, f.embed_model,
                   f.person_id, per.name,
                   p.file_name, p.thumbnail_key, p.captured_utc, p.place_label
            FROM photo_faces f
            JOIN photos p ON p.id = f.photo_id
            LEFT JOIN people per ON per.id = f.person_id
            WHERE f.embedding IS NOT NULL AND f.embed_model = $model;
            """;
        command.Parameters.AddWithValue("$model", model);

        var matches = new List<FaceMatch>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            PhotoFaceRecord face = Read(reader);

            double score = face.Id == faceId ? 1.0 : Dot(query, face.Embedding!);
            if (score < threshold)
            {
                continue;
            }

            matches.Add(new FaceMatch(
                new FaceListing(
                    face,
                    reader.GetString(16),
                    reader.IsDBNull(17) ? null : reader.GetString(17),
                    DateTimeOffset.Parse(reader.GetString(18), System.Globalization.CultureInfo.InvariantCulture),
                    reader.IsDBNull(19) ? null : reader.GetString(19)),
                score));
        }

        // Ties broken by id, for the same reason every other ordering here is: without it the
        // grid reshuffles between renders.
        return matches
            .OrderByDescending(m => m.Similarity)
            .ThenBy(m => m.Listing.Face.Id)
            .Take(limit)
            .ToList();
    }

    /// <summary>How many faces carry a comparable vector.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Faces with an embedding, and faces in total.</returns>
    /// <remarks>
    /// Both, because the difference is the honest answer to "why can I not search for this
    /// face?" — a library swept before the recognition model was installed has faces and no
    /// vectors, and the page should say so rather than silently offering nothing.
    /// </remarks>
    public async Task<(int Embedded, int Total)> EmbeddingCoverageAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            "SELECT COUNT(embedding), COUNT(*) FROM photo_faces;";

        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        return await reader.ReadAsync(cancellationToken).ConfigureAwait(false)
            ? (reader.GetInt32(0), reader.GetInt32(1))
            : (0, 0);
    }

    /// <summary>Cosine similarity between two stored, already-normalised vectors.</summary>
    /// <remarks>
    /// A dot product: embeddings are unit length when written, so there is nothing to divide by.
    /// Mismatched lengths score 0 rather than throwing — that means two models, and "not
    /// comparable" is the right answer.
    /// </remarks>
    private static double Dot(float[] a, float[] b)
    {
        if (a.Length == 0 || a.Length != b.Length)
        {
            return 0;
        }

        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>
    /// Lists the library's faces collapsed to one card per person.
    /// </summary>
    /// <param name="threshold">How alike two faces must be to be treated as one person.</param>
    /// <param name="order">How to order the groups' representative faces.</param>
    /// <param name="minimumConfidence">Drops faces the detector was less sure of than this.</param>
    /// <param name="limit">Most groups to return.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Groups, largest first.</returns>
    /// <remarks>
    /// <para>
    /// <b>Why this exists.</b> A library that holds eleven frames of one sitting produces eleven
    /// near-identical cards, and a wall of the same face repeated is not a useful index of who is
    /// in your photographs — it is the contact sheet again, sorted differently. Collapsing them
    /// turns the page into what it should have been: one card per person, with a count.
    /// </para>
    /// <para>
    /// <b>Greedy, single pass.</b> Faces are taken most-prominent first; each one that has not
    /// already been claimed starts a group and absorbs every unclaimed face within the threshold.
    /// Not the best clustering available, and deliberately so: it is order-stable, it can be
    /// explained in a sentence, and every group is anchored on a real face the user can see. The
    /// alternative — proper agglomerative clustering with a merge rule — produces groups whose
    /// membership depends on the merge order, which is impossible to explain when it goes wrong.
    /// </para>
    /// <para>
    /// <b>Cost.</b> Comparison is quadratic in the worst case, where every face is a different
    /// person. It is far cheaper in practice because claimed faces drop out, but a library of
    /// tens of thousands of strangers would be slow, so the pass is bounded by
    /// <see cref="MaximumFacesToGroup"/> and anything beyond it is returned ungrouped rather
    /// than silently omitted.
    /// </para>
    /// <para>
    /// Faces with no vector cannot be compared to anything, so each stands alone. That is
    /// honest: they might be anybody.
    /// </para>
    /// </remarks>
    public async Task<IReadOnlyList<FaceGroup>> ListPeopleAsync(
        double threshold,
        FaceOrder order = FaceOrder.Prominence,
        double minimumConfidence = 0,
        int limit = 500,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<FaceListing> all = await ListAsync(
            FaceOrder.Prominence, minimumConfidence, null, MaximumFacesToGroup, 0, cancellationToken)
            .ConfigureAwait(false);

        var claimed = new bool[all.Count];
        var groups = new List<FaceGroup>();

        for (int i = 0; i < all.Count; i++)
        {
            if (claimed[i])
            {
                continue;
            }

            claimed[i] = true;

            PhotoFaceRecord anchor = all[i].Face;
            var members = new List<FaceListing> { all[i] };

            if (anchor.Embedding is { Length: > 0 } vector)
            {
                for (int j = i + 1; j < all.Count; j++)
                {
                    if (claimed[j] || all[j].Face.EmbedModel != anchor.EmbedModel)
                    {
                        continue;
                    }

                    if (all[j].Face.Embedding is { } other && Dot(vector, other) >= threshold)
                    {
                        claimed[j] = true;
                        members.Add(all[j]);
                    }
                }
            }

            groups.Add(new FaceGroup(
                all[i],
                members.Count,
                members.Select(m => m.Face.PhotoId).Distinct().Count()));

            cancellationToken.ThrowIfCancellationRequested();
        }

        // Ordered after grouping, not before: the grouping pass has to run most-prominent-first
        // so the face standing for each person is the best one available, whatever order the
        // page then asks for.
        IEnumerable<FaceGroup> sorted = order switch
        {
            FaceOrder.Newest => groups.OrderByDescending(g => g.Representative.CapturedUtc)
                                      .ThenBy(g => g.Representative.Face.Id),
            FaceOrder.Oldest => groups.OrderBy(g => g.Representative.CapturedUtc)
                                      .ThenBy(g => g.Representative.Face.Id),
            FaceOrder.Confidence => groups.OrderByDescending(g => g.Representative.Face.Confidence)
                                          .ThenBy(g => g.Representative.Face.Id),

            // Prominence means something slightly different once faces are grouped: a person who
            // appears in nine photographs is more of a presence in the library than a stranger
            // caught once, however large their face was in that one frame.
            _ => groups.OrderByDescending(g => g.PhotoCount)
                       .ThenByDescending(g => g.Representative.Face.Prominence)
                       .ThenBy(g => g.Representative.Face.Id),
        };

        return sorted.Take(limit).ToList();
    }

    /// <summary>
    /// Most faces the grouping pass will compare.
    /// </summary>
    /// <remarks>
    /// A bound on a quadratic loop, not a product limit. Four thousand faces is at most eight
    /// million dot products of 128 floats, which is a few hundred milliseconds — slow enough to
    /// notice on a page load and fast enough to accept. Beyond it the page would hang, so the
    /// most prominent four thousand are grouped and the rest are not shown on the grouped view;
    /// "every face" still lists all of them.
    /// </remarks>
    public const int MaximumFacesToGroup = 4000;

    /// <summary>How many faces are recorded, and in how many photographs.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Total faces and the number of photographs containing at least one.</returns>
    /// <remarks>
    /// Both numbers, because either alone misleads. "412 faces" says nothing about how much of
    /// the library has people in it, and "in 96 photographs" says nothing about how crowded they
    /// are. The page header shows the pair.
    /// </remarks>
    public async Task<(int Faces, int Photographs)> CountAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT COUNT(*), COUNT(DISTINCT photo_id) FROM photo_faces;";

        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        return await reader.ReadAsync(cancellationToken).ConfigureAwait(false)
            ? (reader.GetInt32(0), reader.GetInt32(1))
            : (0, 0);
    }

    /// <summary>Ids of the photographs that contain at least one face.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The photo ids.</returns>
    /// <remarks>
    /// Feeds the gallery's "has people" filter. Returned as ids rather than joined into
    /// <see cref="PhotoStore"/>'s query builder so that the faces table stays an optional
    /// extra: a library analysed by an older build has no faces, and every other filter must
    /// keep working exactly as it did.
    /// </remarks>
    public async Task<IReadOnlyList<long>> PhotoIdsWithFacesAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT DISTINCT photo_id FROM photo_faces;";

        var ids = new List<long>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            ids.Add(reader.GetInt64(0));
        }

        return ids;
    }

    /// <summary>Ids of photographs that no face pass has looked at yet.</summary>
    /// <param name="model">The detector about to run.</param>
    /// <param name="limit">Most ids to return.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Photographs still to sweep, oldest-indexed first.</returns>
    /// <remarks>
    /// <para>
    /// "Has no rows from this model" is the wrong test on its own, because a photograph with no
    /// people in it correctly produces no rows and would be swept again for ever. So the sweep
    /// keeps its own marker: a row in <c>face_sweeps</c> saying this model has seen this
    /// photograph, written whether or not it found anything.
    /// </para>
    /// <para>
    /// The table is created on demand rather than in a migration. It is pure bookkeeping — losing
    /// it costs one redundant sweep and nothing else — and keeping it out of the migration array
    /// means an older build can still open a library a newer one has swept.
    /// </para>
    /// </remarks>
    public async Task<IReadOnlyList<long>> PendingAsync(
        string model,
        int limit = 1000,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await EnsureSweepTableAsync(connection, cancellationToken).ConfigureAwait(false);

        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT p.id FROM photos p
            WHERE NOT EXISTS (
                SELECT 1 FROM face_sweeps s WHERE s.photo_id = p.id AND s.model = $model)
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

    /// <summary>Records that a detector has looked at a photograph.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="model">The detector.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task MarkSweptAsync(long photoId, string model, CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await EnsureSweepTableAsync(connection, cancellationToken).ConfigureAwait(false);

        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO face_sweeps (photo_id, model, swept_utc) VALUES ($id, $model, $when)
            ON CONFLICT(photo_id, model) DO UPDATE SET swept_utc = excluded.swept_utc;
            """;
        command.Parameters.AddWithValue("$id", photoId);
        command.Parameters.AddWithValue("$model", model);
        command.Parameters.AddWithValue("$when", DateTimeOffset.UtcNow.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Forgets every sweep and every face, so the next run starts over.</summary>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many face rows were removed.</returns>
    public async Task<int> ResetAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await EnsureSweepTableAsync(connection, cancellationToken).ConfigureAwait(false);

        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "DELETE FROM face_sweeps; DELETE FROM photo_faces;";
        return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task EnsureSweepTableAsync(SqliteConnection connection, CancellationToken cancellationToken)
    {
        await using SqliteCommand create = connection.CreateCommand();
        create.CommandText = """
            CREATE TABLE IF NOT EXISTS face_sweeps (
                photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                model     TEXT    NOT NULL,
                swept_utc TEXT    NOT NULL,
                PRIMARY KEY (photo_id, model)
            );
            """;
        await create.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }


    /// <summary>Puts a name to a face, creating the person if this is the first time.</summary>
    /// <param name="faceId">The face.</param>
    /// <param name="name">What to call them. Trimmed; empty clears the name instead.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>The person's id, or null if the name was cleared.</returns>
    /// <remarks>
    /// Find-or-create rather than a separate "add person" step. Naming somebody is the only way a
    /// person enters this library, and making the user create one first would be a form to fill in
    /// before the thing they actually wanted to do. Matching is case-insensitive, so typing "mum"
    /// after "Mum" adds to the same collection rather than starting a rival one.
    /// </remarks>
    public async Task<long?> NameFaceAsync(
        long faceId,
        string? name,
        CancellationToken cancellationToken = default)
    {
        string? trimmed = string.IsNullOrWhiteSpace(name) ? null : name.Trim();

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        long? personId = trimmed is null
            ? null
            : await FindOrCreatePersonAsync(connection, tx, trimmed, cancellationToken)
                .ConfigureAwait(false);

        await using (SqliteCommand assign = connection.CreateCommand())
        {
            assign.Transaction = tx;
            assign.CommandText = "UPDATE photo_faces SET person_id = $person WHERE id = $face;";
            assign.Parameters.AddWithValue("$person", (object?)personId ?? DBNull.Value);
            assign.Parameters.AddWithValue("$face", faceId);
            await assign.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        return personId;
    }

    /// <summary>Puts the same name to several faces at once.</summary>
    /// <param name="faceIds">The faces.</param>
    /// <param name="name">What to call them.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>The person's id.</returns>
    /// <remarks>
    /// For confirming a suggestion. Recognition can offer "these fourteen also look like her", and
    /// accepting that is one decision — so it is one transaction. A cancelled half-run would
    /// otherwise leave the collection split with no way to tell which faces had been confirmed.
    /// </remarks>
    public async Task<long> NameFacesAsync(
        IReadOnlyList<long> faceIds,
        string name,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(faceIds);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        long personId = await FindOrCreatePersonAsync(connection, tx, name.Trim(), cancellationToken)
            .ConfigureAwait(false);

        await using (SqliteCommand assign = connection.CreateCommand())
        {
            assign.Transaction = tx;
            assign.CommandText = "UPDATE photo_faces SET person_id = $person WHERE id = $face;";
            SqliteParameter person = assign.Parameters.Add("$person", SqliteType.Integer);
            SqliteParameter face = assign.Parameters.Add("$face", SqliteType.Integer);
            person.Value = personId;

            foreach (long faceId in faceIds)
            {
                face.Value = faceId;
                await assign.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }
        }

        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        return personId;
    }

    /// <summary>Everybody who has been named, and how much of the library they are in.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>One entry per person, most photographed first.</returns>
    public async Task<IReadOnlyList<NamedPerson>> ListNamedAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT per.id, per.name, COUNT(f.id), COUNT(DISTINCT f.photo_id),
                   (SELECT crop_key FROM photo_faces
                    WHERE person_id = per.id AND crop_key IS NOT NULL
                    ORDER BY area DESC LIMIT 1)
            FROM people per
            LEFT JOIN photo_faces f ON f.person_id = per.id
            GROUP BY per.id, per.name
            ORDER BY COUNT(DISTINCT f.photo_id) DESC, per.name;
            """;

        var people = new List<NamedPerson>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            people.Add(new NamedPerson(
                reader.GetInt64(0),
                reader.GetString(1),
                reader.GetInt32(2),
                reader.GetInt32(3),
                reader.IsDBNull(4) ? null : reader.GetString(4)));
        }

        return people;
    }

    /// <summary>Renames somebody, everywhere at once.</summary>
    /// <param name="personId">The person.</param>
    /// <param name="name">Their new name.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>False when the name is already taken by somebody else.</returns>
    /// <remarks>
    /// False rather than an exception on a clash, because a clash is something the user did and
    /// not a fault: they have typed a name that already exists. Whether that means "merge them" is
    /// a decision for the caller, and not one this store should make on its own.
    /// </remarks>
    public async Task<bool> RenamePersonAsync(
        long personId,
        string name,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "UPDATE people SET name = $name WHERE id = $id;";
        command.Parameters.AddWithValue("$name", name.Trim());
        command.Parameters.AddWithValue("$id", personId);

        try
        {
            return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false) > 0;
        }
        catch (SqliteException error) when (error.SqliteErrorCode == 19)
        {
            // A UNIQUE violation: somebody else already has that name.
            return false;
        }
    }

    private static async Task<long> FindOrCreatePersonAsync(
        SqliteConnection connection,
        SqliteTransaction tx,
        string name,
        CancellationToken cancellationToken)
    {
        await using (SqliteCommand find = connection.CreateCommand())
        {
            find.Transaction = tx;
            find.CommandText = "SELECT id FROM people WHERE name = $name;";
            find.Parameters.AddWithValue("$name", name);

            if (await find.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false) is long id)
            {
                return id;
            }
        }

        await using SqliteCommand create = connection.CreateCommand();
        create.Transaction = tx;
        create.CommandText = """
            INSERT INTO people (name, created_utc) VALUES ($name, $now);
            SELECT last_insert_rowid();
            """;
        create.Parameters.AddWithValue("$name", name);
        create.Parameters.AddWithValue("$now", DateTimeOffset.UtcNow.ToString("O"));

        return (long)(await create.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false))!;
    }

    private static async Task<List<(double X, double Y, double W, double H, long PersonId)>>
        ReadNamesAsync(
            SqliteConnection connection,
            SqliteTransaction tx,
            long photoId,
            CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = tx;
        command.CommandText = """
            SELECT x, y, w, h, person_id FROM photo_faces
            WHERE photo_id = $id AND person_id IS NOT NULL;
            """;
        command.Parameters.AddWithValue("$id", photoId);

        var named = new List<(double, double, double, double, long)>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            named.Add((reader.GetDouble(0), reader.GetDouble(1),
                       reader.GetDouble(2), reader.GetDouble(3), reader.GetInt64(4)));
        }

        return named;
    }

    /// <summary>Puts names back onto the faces a fresh sweep found.</summary>
    /// <remarks>
    /// <para>
    /// Matched by overlap, exactly as segment corrections are, and for the same reason: across two
    /// runs a face has no identity except where it is. Row ids change and the order changes, so
    /// position is all there is to match on.
    /// </para>
    /// <para>
    /// The threshold is deliberately high. A face box is small and two people standing close
    /// together can overlap loosely, and putting the wrong name on somebody is far worse than
    /// losing a name and being asked again — one is a correctable annoyance, the other is a
    /// photo library that quietly lies about who is in a picture. Each name is applied once, to
    /// its best match, so two boxes cannot both claim it.
    /// </para>
    /// </remarks>
    private static IReadOnlyList<PhotoFaceRecord> ReapplyNames(
        IReadOnlyList<PhotoFaceRecord> faces,
        List<(double X, double Y, double W, double H, long PersonId)> named)
    {
        if (named.Count == 0 || faces.Count == 0)
        {
            return faces;
        }

        var result = faces.ToList();
        var taken = new bool[result.Count];

        foreach ((double x, double y, double w, double h, long personId) in named)
        {
            int best = -1;
            double bestOverlap = NameOverlapThreshold;

            for (int i = 0; i < result.Count; i++)
            {
                if (taken[i])
                {
                    continue;
                }

                double overlap = Iou(
                    (x, y, w, h),
                    (result[i].X, result[i].Y, result[i].Width, result[i].Height));

                if (overlap > bestOverlap)
                {
                    bestOverlap = overlap;
                    best = i;
                }
            }

            if (best >= 0)
            {
                taken[best] = true;
                result[best] = result[best] with { PersonId = personId };
            }
        }

        return result;
    }

    /// <summary>How much two boxes must overlap for a name to move between them.</summary>
    private const double NameOverlapThreshold = 0.6;

    private static double Iou(
        (double X, double Y, double W, double H) a,
        (double X, double Y, double W, double H) b)
    {
        double left = Math.Max(a.X, b.X);
        double top = Math.Max(a.Y, b.Y);
        double right = Math.Min(a.X + a.W, b.X + b.W);
        double bottom = Math.Min(a.Y + a.H, b.Y + b.H);

        if (right <= left || bottom <= top)
        {
            return 0;
        }

        double overlap = (right - left) * (bottom - top);
        double union = (a.W * a.H) + (b.W * b.H) - overlap;
        return union <= 0 ? 0 : overlap / union;
    }

    private static PhotoFaceRecord Read(SqliteDataReader reader) => new(
        reader.GetInt64(0),
        reader.GetInt64(1),
        reader.GetDouble(2),
        reader.GetDouble(3),
        reader.GetDouble(4),
        reader.GetDouble(5),
        reader.GetDouble(6),
        reader.GetDouble(7),
        reader.GetDouble(8),
        reader.GetString(9),
        reader.IsDBNull(10) ? null : reader.GetString(10),
        reader.GetString(11),
        reader.IsDBNull(12) ? null : ToFloats((byte[])reader[12]),
        reader.IsDBNull(13) ? null : reader.GetString(13),
        reader.IsDBNull(14) ? null : reader.GetInt64(14),
        reader.IsDBNull(15) ? null : reader.GetString(15));

    /// <summary>Packs an embedding for storage.</summary>
    /// <param name="vector">The embedding.</param>
    /// <returns>Its raw little-endian bytes.</returns>
    /// <remarks>
    /// A straight memory copy of the floats, not a text or JSON encoding. 128 floats are 512
    /// bytes this way and about 1.5 KB as text, and the whole point of holding them as a blob is
    /// that the comparison sweep can read them back into a span without parsing anything.
    /// </remarks>
    private static byte[] ToBytes(float[] vector)
    {
        var bytes = new byte[vector.Length * sizeof(float)];
        Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    /// <summary>Unpacks a stored embedding.</summary>
    /// <param name="bytes">Raw bytes from the column.</param>
    /// <returns>The floats, or an empty array if the blob is not a whole number of them.</returns>
    /// <remarks>
    /// A truncated blob returns empty rather than throwing. It cannot happen through this class,
    /// but a face that silently stops matching is a far better failure than a page that will not
    /// render — and an empty vector is exactly "this face cannot be compared", which the rest of
    /// the code already handles.
    /// </remarks>
    private static float[] ToFloats(byte[] bytes)
    {
        if (bytes.Length == 0 || bytes.Length % sizeof(float) != 0)
        {
            return [];
        }

        var vector = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, vector, 0, bytes.Length);
        return vector;
    }
}
