using Microsoft.Data.Sqlite;
using PixelFlux.Core.Model;

namespace PixelFlux.Core.Index;

/// <summary>
/// Reads and writes the labelled regions a segmentation model found in each photograph.
/// </summary>
/// <remarks>
/// Separate from <see cref="PhotoStore"/> because the ownership rule is different and worth
/// keeping visible: every row here belongs to a model version and is deleted wholesale when a
/// better model re-runs. Nothing a person typed lives in this table, so replacing its contents
/// is always safe — which is exactly what makes it safe to re-analyse a library.
/// </remarks>
public sealed class SegmentStore
{
    private readonly PhotoDatabase _database;

    /// <summary>Creates a store over a migrated database.</summary>
    /// <param name="database">The database handle.</param>
    public SegmentStore(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    /// <summary>
    /// Replaces every segment for a photograph.
    /// </summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="segments">The new segments. An empty list clears them.</param>
    /// <param name="model">Identifier of the model that produced them.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// Replace, not merge. Two runs of a detector on the same photograph produce
    /// nearly-but-not-quite the same boxes, and merging would accumulate three slightly
    /// different dogs. The delete and the inserts share a transaction so a photo is never left
    /// with half of one run and half of another.
    /// </remarks>
    public async Task ReplaceAsync(
        long photoId,
        IReadOnlyList<PhotoSegmentRecord> segments,
        string model,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(segments);

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        // Corrections are read before the delete and re-applied to whatever the new run found
        // in the same place. A person who has said "that is a car, not a truck" should not have
        // to say it again because a better model was installed — and a re-analysis that silently
        // discarded their work would make them stop correcting anything.
        List<(double X, double Y, double W, double H, string Label)> corrections =
            await ReadCorrectionsAsync(connection, tx, photoId, cancellationToken).ConfigureAwait(false);

        await using (SqliteCommand clear = connection.CreateCommand())
        {
            clear.Transaction = tx;
            clear.CommandText = "DELETE FROM photo_segments WHERE photo_id = $id;";
            clear.Parameters.AddWithValue("$id", photoId);
            await clear.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        IReadOnlyList<PhotoSegmentRecord> incoming = corrections.Count == 0
            ? segments
            : ReapplyCorrections(segments, corrections);

        if (incoming.Count > 0)
        {
            await using SqliteCommand insert = connection.CreateCommand();
            insert.Transaction = tx;
            insert.CommandText = """
                INSERT INTO photo_segments
                    (photo_id, label, confidence, x, y, w, h, area, prominence, mask_key, model,
                     user_label)
                VALUES ($id, $label, $conf, $x, $y, $w, $h, $area, $prom, $mask, $model, $user);
                """;

            insert.Parameters.AddWithValue("$id", photoId);
            SqliteParameter label = insert.Parameters.Add("$label", SqliteType.Text);
            SqliteParameter confidence = insert.Parameters.Add("$conf", SqliteType.Real);
            SqliteParameter x = insert.Parameters.Add("$x", SqliteType.Real);
            SqliteParameter y = insert.Parameters.Add("$y", SqliteType.Real);
            SqliteParameter w = insert.Parameters.Add("$w", SqliteType.Real);
            SqliteParameter h = insert.Parameters.Add("$h", SqliteType.Real);
            SqliteParameter area = insert.Parameters.Add("$area", SqliteType.Real);
            SqliteParameter prominence = insert.Parameters.Add("$prom", SqliteType.Real);
            SqliteParameter mask = insert.Parameters.Add("$mask", SqliteType.Text);
            SqliteParameter user = insert.Parameters.Add("$user", SqliteType.Text);
            insert.Parameters.AddWithValue("$model", model);

            foreach (PhotoSegmentRecord segment in incoming)
            {
                label.Value = segment.Label;
                confidence.Value = segment.Confidence;
                x.Value = segment.X;
                y.Value = segment.Y;
                w.Value = segment.Width;
                h.Value = segment.Height;
                area.Value = segment.AreaFraction;
                prominence.Value = segment.Prominence;
                mask.Value = (object?)segment.MaskKey ?? DBNull.Value;
                user.Value = (object?)segment.UserLabel ?? DBNull.Value;
                await insert.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }
        }

        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Sets, or clears, what a person calls one region.</summary>
    /// <param name="segmentId">The region.</param>
    /// <param name="label">The person's word for it, or null/blank to go back to the model's.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>Whether a row was changed.</returns>
    /// <remarks>
    /// Trimmed and length-capped, but otherwise taken as typed. This is somebody naming a thing
    /// in their own photograph; it is not a controlled vocabulary and it is not the model's list
    /// of eighty classes. Lower-casing it or forcing it onto a known class would be the
    /// application telling the user they are wrong about their own picture.
    /// </remarks>
    public async Task<bool> SetUserLabelAsync(
        long segmentId,
        string? label,
        CancellationToken cancellationToken = default)
    {
        string? cleaned = string.IsNullOrWhiteSpace(label)
            ? null
            : label.Trim()[..Math.Min(label.Trim().Length, 60)];

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "UPDATE photo_segments SET user_label = $label WHERE id = $id;";
        command.Parameters.AddWithValue("$label", (object?)cleaned ?? DBNull.Value);
        command.Parameters.AddWithValue("$id", segmentId);

        return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false) > 0;
    }

    private static async Task<List<(double X, double Y, double W, double H, string Label)>> ReadCorrectionsAsync(
        SqliteConnection connection,
        SqliteTransaction tx,
        long photoId,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = tx;
        command.CommandText = """
            SELECT x, y, w, h, user_label FROM photo_segments
            WHERE photo_id = $id AND user_label IS NOT NULL;
            """;
        command.Parameters.AddWithValue("$id", photoId);

        var corrections = new List<(double, double, double, double, string)>();
        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            corrections.Add((reader.GetDouble(0), reader.GetDouble(1),
                             reader.GetDouble(2), reader.GetDouble(3), reader.GetString(4)));
        }

        return corrections;
    }

    /// <summary>
    /// Carries a person's corrections onto the regions a fresh run found.
    /// </summary>
    /// <remarks>
    /// Matched by overlap, because that is the only identity a region has across two runs of two
    /// different models: row ids change, the order changes, and even the class changes — that
    /// last being the whole reason the correction exists. Half-overlap is the threshold; below
    /// it the new run has found something else in roughly the same place, and inheriting a name
    /// meant for a different thing would be worse than losing it.
    ///
    /// Each correction is applied once, to its best match, so two boxes that both overlap a
    /// corrected region cannot both claim the name.
    /// </remarks>
    private static List<PhotoSegmentRecord> ReapplyCorrections(
        IReadOnlyList<PhotoSegmentRecord> segments,
        List<(double X, double Y, double W, double H, string Label)> corrections)
    {
        var result = segments.ToList();
        var taken = new bool[result.Count];

        foreach ((double x, double y, double w, double h, string label) in corrections)
        {
            int best = -1;
            double bestOverlap = 0.5;

            for (int i = 0; i < result.Count; i++)
            {
                if (taken[i])
                {
                    continue;
                }

                double overlap = Iou((x, y, w, h),
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
                result[best] = result[best] with { UserLabel = label };
            }
        }

        return result;
    }

    private static double Iou(
        (double X, double Y, double W, double H) a,
        (double X, double Y, double W, double H) b)
    {
        double left = Math.Max(a.X, b.X);
        double top = Math.Max(a.Y, b.Y);
        double right = Math.Min(a.X + a.W, b.X + b.W);
        double bottom = Math.Min(a.Y + a.H, b.Y + b.H);

        double overlap = Math.Max(0, right - left) * Math.Max(0, bottom - top);
        double union = (a.W * a.H) + (b.W * b.H) - overlap;

        return union <= 0 ? 0 : overlap / union;
    }

    /// <summary>Fetches a photograph's segments, most prominent first.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Its segments.</returns>
    public async Task<IReadOnlyList<PhotoSegmentRecord>> GetAsync(
        long photoId,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT id, photo_id, label, confidence, x, y, w, h, area, prominence, mask_key, model,
                   user_label
            FROM photo_segments WHERE photo_id = $id ORDER BY prominence DESC;
            """;
        command.Parameters.AddWithValue("$id", photoId);

        var segments = new List<PhotoSegmentRecord>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            segments.Add(new PhotoSegmentRecord(
                reader.GetInt64(0),
                reader.GetInt64(1),
                reader.GetString(2),
                reader.GetDouble(3),
                reader.GetDouble(4),
                reader.GetDouble(5),
                reader.GetDouble(6),
                reader.GetDouble(7),
                reader.GetDouble(8),
                reader.GetDouble(9),
                reader.IsDBNull(10) ? null : reader.GetString(10),
                reader.GetString(11),
                reader.IsDBNull(12) ? null : reader.GetString(12)));
        }

        return segments;
    }

    /// <summary>
    /// Counts photographs containing each detected object, for the facet panel.
    /// </summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Label and the number of photographs containing it, commonest first.</returns>
    /// <remarks>
    /// Counts <em>photographs</em>, not segments. A photo of a crowd holds thirty "person"
    /// segments and is still one photo, and a facet claiming thirty would be lying about how
    /// much clicking it narrows the view.
    /// </remarks>
    public async Task<IReadOnlyList<(string Label, int Count)>> GetObjectFacetAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT COALESCE(user_label, label) AS shown, COUNT(DISTINCT photo_id)
            FROM photo_segments
            GROUP BY shown ORDER BY COUNT(DISTINCT photo_id) DESC, shown LIMIT 40;
            """;

        var facet = new List<(string, int)>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            facet.Add((reader.GetString(0), reader.GetInt32(1)));
        }

        return facet;
    }

    /// <summary>Deletes every segment produced by models other than the current one.</summary>
    /// <param name="currentModel">The model whose results to keep.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many rows were removed.</returns>
    /// <remarks>
    /// Used when upgrading models. Stale segments are worse than none: their masks point at
    /// cache files a newer run has already overwritten, and their labels sit in the facet
    /// promising photographs that no longer match.
    /// </remarks>
    public async Task<int> PurgeOtherModelsAsync(
        string currentModel,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "DELETE FROM photo_segments WHERE model <> $model;";
        command.Parameters.AddWithValue("$model", currentModel);
        return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }
}
