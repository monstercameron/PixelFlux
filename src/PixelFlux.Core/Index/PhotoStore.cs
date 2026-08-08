using System.Globalization;
using Microsoft.Data.Sqlite;
using PixelFlux.Core.Model;

namespace PixelFlux.Core.Index;

/// <summary>How a gallery query should be ordered.</summary>
public enum PhotoOrder
{
    /// <summary>Newest capture first. The default the gallery opens on.</summary>
    CapturedDescending = 0,

    /// <summary>Oldest capture first.</summary>
    CapturedAscending = 1,

    /// <summary>Most recently added to the library first.</summary>
    IndexedDescending = 2,

    /// <summary>Filename, A-Z.</summary>
    FileName = 3,

    /// <summary>Highest rated first, then newest. Unrated photos come last.</summary>
    RatingDescending = 4,

    /// <summary>Largest file first — the practical way to find what is filling a disk.</summary>
    FileSizeDescending = 5,

    /// <summary>Grouped by camera body, then by date within each.</summary>
    Camera = 6,

    /// <summary>Grouped by source folder, then by filename — mirrors how the files sit on disk.</summary>
    Folder = 7,

    /// <summary>
    /// Stable shuffle. Feeds the slideshow, and rediscovers photographs that chronological
    /// browsing buries.
    /// </summary>
    /// <remarks>
    /// Ordered by a hash of the content hash rather than by <c>RANDOM()</c>. SQLite's RANDOM()
    /// reseeds per statement, so paging through a shuffled set would show duplicates on page two
    /// and skip others entirely. Hashing a stable column gives an arbitrary but repeatable
    /// order, which is what paging needs.
    /// </remarks>
    Shuffle = 8,
}

/// <summary>A filter over the library. Every property is optional and they combine with AND.</summary>
/// <remarks>
/// This is the "constrained query representation" the natural-language layer targets. Parsed
/// language produces one of these; it never produces SQL. That boundary is the security
/// property of the search feature — a model can choose <em>which</em> filters to set, but the
/// mapping from filter to predicate is fixed code, so no phrasing can reach the database as
/// executable text.
/// </remarks>
public sealed record PhotoQuery
{
    /// <summary>Full-text terms matched against title, caption, description, tags, filename, camera.</summary>
    public string? Text { get; init; }

    /// <summary>Only photos captured on or after this instant.</summary>
    public DateTimeOffset? From { get; init; }

    /// <summary>Only photos captured on or before this instant.</summary>
    public DateTimeOffset? To { get; init; }

    /// <summary>Substring match against the camera model.</summary>
    public string? CameraModel { get; init; }

    /// <summary>Only photos carrying all of these tags.</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>Only photos with a GPS fix inside this bounding box (south, west, north, east).</summary>
    /// <remarks>
    /// Retained for map-style queries and for tests. The interface filters by
    /// <see cref="City"/> and <see cref="Country"/> instead, because those are what a person
    /// means when they say where a photograph was taken.
    /// </remarks>
    public (double South, double West, double North, double East)? Bounds { get; init; }

    /// <summary>Only photos resolved to this city.</summary>
    public string? City { get; init; }

    /// <summary>Only photos resolved to this country.</summary>
    public string? Country { get; init; }

    /// <summary>Only photos in which the model found this object, for example <c>dog</c>.</summary>
    public string? Object { get; init; }

    /// <summary>Only favourites.</summary>
    public bool? FavouritesOnly { get; init; }

    /// <summary>Minimum star rating.</summary>
    public int? MinRating { get; init; }

    /// <summary>Only photos in this processing state.</summary>
    public ProcessingState? State { get; init; }

    /// <summary>Only photos imported from this folder, matched as a path prefix.</summary>
    /// <remarks>
    /// A prefix, so selecting <c>C:\Photos\2024</c> includes everything beneath it. People
    /// organise photographs in nested folders and expect a parent to mean "and its children".
    /// </remarks>
    public string? SourceFolder { get; init; }

    /// <summary>Only photos in this collection.</summary>
    public long? CollectionId { get; init; }

    /// <summary>Ordering.</summary>
    public PhotoOrder Order { get; init; } = PhotoOrder.CapturedDescending;

    /// <summary>Maximum rows to return.</summary>
    public int Limit { get; init; } = 500;

    /// <summary>Rows to skip, for paging.</summary>
    public int Offset { get; init; }
}

/// <summary>One bucket of the time rail: how many photos fall in a period.</summary>
/// <param name="Start">Start of the period, UTC.</param>
/// <param name="Count">Photos captured within it.</param>
public readonly record struct TimeBucket(DateTimeOffset Start, int Count);

/// <summary>
/// Reads and writes photo rows. The only type in PixelFlux that speaks SQL.
/// </summary>
/// <remarks>
/// Concentrating SQL here is the one-abstraction-layer rule applied honestly: there is no
/// repository interface over this, no unit-of-work, and no entity mapper. Callers get a class
/// with methods that return records. If a query needs to change, it changes in this file.
/// </remarks>
public sealed class PhotoStore
{
    private readonly PhotoDatabase _database;

    /// <summary>Creates a store over a migrated database.</summary>
    /// <param name="database">The database handle.</param>
    public PhotoStore(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    // SQLite has no date type. Every timestamp is stored as an ISO-8601 UTC string with fixed
    // width, which sorts lexicographically in exactly capture order — so ORDER BY on the text
    // column is correct and can use the index, with no conversion at query time.
    private const string TimeFormat = "yyyy-MM-ddTHH:mm:ss.fffZ";

    private static string Iso(DateTimeOffset value)
        => value.ToUniversalTime().ToString(TimeFormat, CultureInfo.InvariantCulture);

    private static DateTimeOffset ParseIso(string value)
        => DateTimeOffset.ParseExact(value, TimeFormat, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);

    /// <summary>
    /// Inserts a photo, or returns the existing row's id if the content hash is already known.
    /// </summary>
    /// <param name="photo">The record to insert. Its <see cref="PhotoRecord.Id"/> is ignored.</param>
    /// <param name="tags">Tags to attach.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>The row id, and whether this call created it.</returns>
    /// <remarks>
    /// The <c>ON CONFLICT DO NOTHING</c> on <c>content_hash</c> is what makes ingestion
    /// idempotent and therefore safe to interrupt: re-scanning a folder after a crash inserts
    /// nothing and costs one index probe per file.
    /// </remarks>
    public async Task<(long Id, bool Inserted)> UpsertAsync(
        PhotoRecord photo,
        IReadOnlyList<PhotoTag>? tags = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(photo);

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        long id;
        bool inserted;

        await using (SqliteCommand insert = connection.CreateCommand())
        {
            insert.Transaction = tx;
            insert.CommandText = """
                INSERT INTO photos (
                    content_hash, perceptual_hash, original_path, file_name, source_folder, mime_type,
                    width, height, file_size, captured_utc, capture_exact, file_modified_utc,
                    indexed_utc, camera_make, camera_model, lens_model, iso, f_number,
                    exposure_seconds, focal_length_mm, gps_lat, gps_lon, gps_alt,
                    place_city, place_country, place_code, place_label, orientation,
                    thumbnail_key, proxy_key, state, state_detail, model_version,
                    ai_caption, ai_description, user_title, user_notes, rating, is_favourite, revision
                ) VALUES (
                    $hash, $phash, $path, $name, $folder, $mime,
                    $w, $h, $size, $captured, $exact, $modified,
                    $indexed, $make, $model, $lens, $iso, $fnum,
                    $exposure, $focal, $lat, $lon, $alt,
                    $city, $country, $ccode, $plabel, $orient,
                    $thumb, $proxy, $state, $detail, $modelver,
                    $aicap, $aidesc, $title, $notes, $rating, $fav, 1
                )
                ON CONFLICT(content_hash) DO NOTHING
                RETURNING id;
                """;

            BindPhoto(insert, photo);

            object? scalar = await insert.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            if (scalar is not null and not DBNull)
            {
                id = Convert.ToInt64(scalar, CultureInfo.InvariantCulture);
                inserted = true;
            }
            else
            {
                // The conflict path: fetch the id of the row that was already there.
                await using SqliteCommand existing = connection.CreateCommand();
                existing.Transaction = tx;
                existing.CommandText = "SELECT id FROM photos WHERE content_hash = $hash;";
                existing.Parameters.AddWithValue("$hash", photo.ContentHash);
                id = Convert.ToInt64(
                    await existing.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false),
                    CultureInfo.InvariantCulture);
                inserted = false;
            }
        }

        if (inserted)
        {
            if (tags is { Count: > 0 })
            {
                await WriteTagsAsync(connection, tx, id, tags, cancellationToken).ConfigureAwait(false);
            }

            await ReindexTextAsync(connection, tx, id, cancellationToken).ConfigureAwait(false);
        }

        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        return (id, inserted);
    }

    private static void BindPhoto(SqliteCommand command, PhotoRecord photo)
    {
        command.Parameters.AddWithValue("$hash", photo.ContentHash);
        command.Parameters.AddWithValue("$phash", photo.PerceptualHash);
        command.Parameters.AddWithValue("$path", photo.OriginalPath);
        command.Parameters.AddWithValue("$name", photo.FileName);
        command.Parameters.AddWithValue("$folder", photo.SourceFolder);
        command.Parameters.AddWithValue("$mime", photo.MimeType);
        command.Parameters.AddWithValue("$w", photo.Width);
        command.Parameters.AddWithValue("$h", photo.Height);
        command.Parameters.AddWithValue("$size", photo.FileSize);
        command.Parameters.AddWithValue("$captured", Iso(photo.CapturedUtc));
        command.Parameters.AddWithValue("$exact", photo.CaptureTimeIsExact ? 1 : 0);
        command.Parameters.AddWithValue("$modified", Iso(photo.FileModifiedUtc));
        command.Parameters.AddWithValue("$indexed", Iso(photo.IndexedUtc));
        command.Parameters.AddWithValue("$make", (object?)photo.Camera.Make ?? DBNull.Value);
        command.Parameters.AddWithValue("$model", (object?)photo.Camera.Model ?? DBNull.Value);
        command.Parameters.AddWithValue("$lens", (object?)photo.Camera.Lens ?? DBNull.Value);
        command.Parameters.AddWithValue("$iso", (object?)photo.Camera.Iso ?? DBNull.Value);
        command.Parameters.AddWithValue("$fnum", (object?)photo.Camera.FNumber ?? DBNull.Value);
        command.Parameters.AddWithValue("$exposure", (object?)photo.Camera.ExposureSeconds ?? DBNull.Value);
        command.Parameters.AddWithValue("$focal", (object?)photo.Camera.FocalLengthMm ?? DBNull.Value);
        command.Parameters.AddWithValue("$lat", (object?)photo.Location?.Latitude ?? DBNull.Value);
        command.Parameters.AddWithValue("$lon", (object?)photo.Location?.Longitude ?? DBNull.Value);
        command.Parameters.AddWithValue("$alt", (object?)photo.Location?.Altitude ?? DBNull.Value);
        command.Parameters.AddWithValue("$city", (object?)photo.Place?.City ?? DBNull.Value);
        command.Parameters.AddWithValue("$country", (object?)photo.Place?.Country ?? DBNull.Value);
        command.Parameters.AddWithValue("$ccode", (object?)photo.Place?.CountryCode ?? DBNull.Value);
        command.Parameters.AddWithValue("$plabel", (object?)photo.Place?.Label ?? DBNull.Value);
        command.Parameters.AddWithValue("$orient", photo.Orientation);
        command.Parameters.AddWithValue("$thumb", (object?)photo.ThumbnailKey ?? DBNull.Value);
        command.Parameters.AddWithValue("$proxy", (object?)photo.ProxyKey ?? DBNull.Value);
        command.Parameters.AddWithValue("$state", (int)photo.State);
        command.Parameters.AddWithValue("$detail", (object?)photo.StateDetail ?? DBNull.Value);
        command.Parameters.AddWithValue("$modelver", (object?)photo.ModelVersion ?? DBNull.Value);
        command.Parameters.AddWithValue("$aicap", (object?)photo.AiCaption ?? DBNull.Value);
        command.Parameters.AddWithValue("$aidesc", (object?)photo.AiDescription ?? DBNull.Value);
        command.Parameters.AddWithValue("$title", (object?)photo.UserTitle ?? DBNull.Value);
        command.Parameters.AddWithValue("$notes", (object?)photo.UserNotes ?? DBNull.Value);
        command.Parameters.AddWithValue("$rating", photo.Rating);
        command.Parameters.AddWithValue("$fav", photo.IsFavourite ? 1 : 0);
    }

    private static async Task WriteTagsAsync(
        SqliteConnection connection,
        SqliteTransaction tx,
        long photoId,
        IReadOnlyList<PhotoTag> tags,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = tx;
        command.CommandText = """
            INSERT INTO photo_tags (photo_id, tag, confidence, source)
            VALUES ($id, $tag, $conf, $src)
            ON CONFLICT(photo_id, tag, source) DO UPDATE SET confidence = excluded.confidence;
            """;

        SqliteParameter id = command.Parameters.Add("$id", SqliteType.Integer);
        SqliteParameter tag = command.Parameters.Add("$tag", SqliteType.Text);
        SqliteParameter conf = command.Parameters.Add("$conf", SqliteType.Real);
        SqliteParameter src = command.Parameters.Add("$src", SqliteType.Integer);

        id.Value = photoId;
        foreach (PhotoTag t in tags)
        {
            tag.Value = t.Tag.Trim().ToLowerInvariant();
            conf.Value = t.Confidence;
            src.Value = (int)t.Source;
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Rebuilds the full-text row for one photo from its current columns and tags.
    /// </summary>
    /// <remarks>
    /// Called explicitly rather than driven by SQL triggers. Triggers would keep FTS in step
    /// automatically, but they would also fire once per tag insert during ingestion — thousands
    /// of redundant rebuilds. Doing it once, after a photo's tags are written, is both faster
    /// and easier to follow when the index looks wrong.
    /// </remarks>
    private static async Task ReindexTextAsync(
        SqliteConnection connection,
        SqliteTransaction? tx,
        long photoId,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = tx;
        command.CommandText = """
            DELETE FROM photo_fts WHERE rowid IN (SELECT rowid FROM photo_fts_map WHERE photo_id = $id);
            DELETE FROM photo_fts_map WHERE photo_id = $id;

            INSERT INTO photo_fts (title, caption, description, tags, filename, camera)
            SELECT
                COALESCE(p.user_title, ''),
                COALESCE(p.ai_caption, ''),
                COALESCE(p.ai_description, '') || ' ' || COALESCE(p.user_notes, ''),
                COALESCE((SELECT GROUP_CONCAT(t.tag, ' ') FROM photo_tags t WHERE t.photo_id = p.id), ''),
                p.file_name,
                TRIM(COALESCE(p.camera_make, '') || ' ' || COALESCE(p.camera_model, '') || ' '
                     || COALESCE(p.lens_model, '') || ' ' || COALESCE(p.place_city, '') || ' '
                     || COALESCE(p.place_country, ''))
            FROM photos p WHERE p.id = $id;

            INSERT INTO photo_fts_map (rowid, photo_id) VALUES (last_insert_rowid(), $id);
            """;
        command.Parameters.AddWithValue("$id", photoId);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Rebuilds the full-text row for a photo after its metadata changed.</summary>
    /// <param name="photoId">The photo to reindex.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task ReindexTextAsync(long photoId, CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await ReindexTextAsync(connection, null, photoId, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Runs a query and returns matching photos.</summary>
    /// <param name="query">The filter.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Matching photos, ordered as the query asked.</returns>
    public async Task<IReadOnlyList<PhotoRecord>> QueryAsync(
        PhotoQuery query,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();

        var where = new List<string>();

        if (!string.IsNullOrWhiteSpace(query.Text))
        {
            where.Add("""
                p.id IN (
                    SELECT m.photo_id FROM photo_fts f
                    JOIN photo_fts_map m ON m.rowid = f.rowid
                    WHERE photo_fts MATCH $text
                )
                """);
            command.Parameters.AddWithValue("$text", ToFtsQuery(query.Text));
        }

        if (query.From is { } from)
        {
            where.Add("p.captured_utc >= $from");
            command.Parameters.AddWithValue("$from", Iso(from));
        }

        if (query.To is { } to)
        {
            where.Add("p.captured_utc <= $to");
            command.Parameters.AddWithValue("$to", Iso(to));
        }

        if (!string.IsNullOrWhiteSpace(query.CameraModel))
        {
            where.Add("(p.camera_model LIKE $camera OR p.camera_make LIKE $camera)");
            command.Parameters.AddWithValue("$camera", $"%{query.CameraModel.Trim()}%");
        }

        if (query.Bounds is { } bounds)
        {
            where.Add("p.gps_lat BETWEEN $south AND $north AND p.gps_lon BETWEEN $west AND $east");
            command.Parameters.AddWithValue("$south", bounds.South);
            command.Parameters.AddWithValue("$north", bounds.North);
            command.Parameters.AddWithValue("$west", bounds.West);
            command.Parameters.AddWithValue("$east", bounds.East);
        }

        if (query.FavouritesOnly == true)
        {
            where.Add("p.is_favourite = 1");
        }

        if (query.MinRating is { } rating)
        {
            where.Add("p.rating >= $rating");
            command.Parameters.AddWithValue("$rating", rating);
        }

        if (query.State is { } state)
        {
            where.Add("p.state = $state");
            command.Parameters.AddWithValue("$state", (int)state);
        }

        if (!string.IsNullOrWhiteSpace(query.Object))
        {
            // COALESCE so a corrected region answers to the name a person gave it. Filtering on
            // the raw model label would mean the object facet offers "car", the user clicks it,
            // and the photograph they themselves relabelled as a car is missing from the result.
            where.Add("EXISTS (SELECT 1 FROM photo_segments g "
                    + "WHERE g.photo_id = p.id AND COALESCE(g.user_label, g.label) = $object)");
            command.Parameters.AddWithValue("$object", query.Object.Trim().ToLowerInvariant());
        }

        if (!string.IsNullOrWhiteSpace(query.City))
        {
            where.Add("p.place_city = $city");
            command.Parameters.AddWithValue("$city", query.City.Trim());
        }

        if (!string.IsNullOrWhiteSpace(query.Country))
        {
            where.Add("p.place_country = $country");
            command.Parameters.AddWithValue("$country", query.Country.Trim());
        }

        if (!string.IsNullOrWhiteSpace(query.SourceFolder))
        {
            // A range scan, not LIKE.
            //
            // LIKE looked obvious and was wrong twice over on Windows paths. Its wildcards are
            // '%' and '_', and '_' is extremely common in folder names, so an unescaped prefix
            // silently matches folders it should not. Escaping needs an ESCAPE clause, and the
            // only sane escape character — backslash — is the path separator, so every path
            // would need doubling and any mistake produces a filter that matches nothing. The
            // first version made exactly that mistake and returned zero rows for every parent
            // folder.
            //
            // Two comparisons have none of those problems: no metacharacters, no escaping, and
            // they use the index on source_folder as an ordinary range.
            where.Add("p.source_folder >= $folderLo AND p.source_folder < $folderHi");
            string prefix = query.SourceFolder.TrimEnd('\\', '/');
            command.Parameters.AddWithValue("$folderLo", prefix);
            // ￿ sorts above every character that can appear in a path, so the upper bound
            // catches the folder itself and everything nested beneath it.
            command.Parameters.AddWithValue("$folderHi", prefix + "￿");
        }

        if (query.CollectionId is { } collection)
        {
            where.Add("EXISTS (SELECT 1 FROM collection_photos cp "
                    + "WHERE cp.photo_id = p.id AND cp.collection_id = $collection)");
            command.Parameters.AddWithValue("$collection", collection);
        }

        for (int i = 0; i < query.Tags.Count; i++)
        {
            // One EXISTS per tag gives AND semantics ("dog AND beach"). A single IN clause would
            // give OR, which is almost never what someone typing two tags means.
            string parameter = $"$tag{i}";
            where.Add($"EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id = p.id AND t.tag = {parameter})");
            command.Parameters.AddWithValue(parameter, query.Tags[i].Trim().ToLowerInvariant());
        }

        string clause = where.Count > 0 ? "WHERE " + string.Join(" AND ", where) : string.Empty;

        // Order and limit are not parameterisable in SQLite, so they are mapped from a closed
        // enum and clamped integers — never interpolated from caller text.
        // Every branch is a literal from a closed enum — no caller text reaches the SQL text.
        // Each order ends with a unique tiebreak (id) so paging is stable: without one, two rows
        // with the same sort key can swap places between pages and a photo appears twice.
        string order = query.Order switch
        {
            PhotoOrder.CapturedAscending => "p.captured_utc ASC, p.id ASC",
            PhotoOrder.IndexedDescending => "p.indexed_utc DESC, p.id DESC",
            PhotoOrder.FileName => "p.file_name COLLATE NOCASE ASC, p.id ASC",
            PhotoOrder.RatingDescending => "p.rating DESC, p.captured_utc DESC, p.id DESC",
            PhotoOrder.FileSizeDescending => "p.file_size DESC, p.id DESC",
            PhotoOrder.Camera =>
                "COALESCE(p.camera_model, '~') COLLATE NOCASE ASC, p.captured_utc DESC, p.id DESC",
            PhotoOrder.Folder =>
                "p.source_folder COLLATE NOCASE ASC, p.file_name COLLATE NOCASE ASC, p.id ASC",
            // A stable pseudo-shuffle: order by a slice of the content hash. SQLite's RANDOM()
            // reseeds per statement, so paging a randomised set repeats and skips rows.
            PhotoOrder.Shuffle => "SUBSTR(p.content_hash, 7, 8) ASC, p.id ASC",
            _ => "p.captured_utc DESC, p.id DESC",
        };

        int limit = Math.Clamp(query.Limit, 1, 5000);
        int offset = Math.Max(query.Offset, 0);

        command.CommandText = $"SELECT p.* FROM photos p {clause} ORDER BY {order} LIMIT {limit} OFFSET {offset};";

        var results = new List<PhotoRecord>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            results.Add(Read(reader));
        }

        return results;
    }

    /// <summary>Counts rows matching a query, ignoring its limit and offset.</summary>
    /// <param name="query">The filter.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The total match count.</returns>
    public async Task<int> CountAsync(PhotoQuery query, CancellationToken cancellationToken = default)
    {
        IReadOnlyList<PhotoRecord> rows = await QueryAsync(
            query with { Limit = 5000, Offset = 0 }, cancellationToken).ConfigureAwait(false);
        return rows.Count;
    }

    /// <summary>Fetches one photo by id.</summary>
    /// <param name="id">The row id.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The photo, or null if no such row.</returns>
    public async Task<PhotoRecord?> GetAsync(long id, CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT * FROM photos WHERE id = $id;";
        command.Parameters.AddWithValue("$id", id);

        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        return await reader.ReadAsync(cancellationToken).ConfigureAwait(false) ? Read(reader) : null;
    }

    /// <summary>Fetches the tags attached to a photo, highest confidence first.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The tags.</returns>
    public async Task<IReadOnlyList<PhotoTag>> GetTagsAsync(
        long photoId,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT tag, confidence, source FROM photo_tags
            WHERE photo_id = $id ORDER BY source DESC, confidence DESC, tag;
            """;
        command.Parameters.AddWithValue("$id", photoId);

        var tags = new List<PhotoTag>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            tags.Add(new PhotoTag(reader.GetString(0), reader.GetDouble(1), (MetadataSource)reader.GetInt32(2)));
        }

        return tags;
    }

    /// <summary>
    /// Counts photos per calendar month, oldest first. Feeds the time rail.
    /// </summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>One bucket per month that contains at least one photo.</returns>
    /// <remarks>
    /// Months with no photos are omitted rather than returned as zero. The rail draws the gaps
    /// itself from the span between buckets, which keeps this query proportional to the data
    /// rather than to the length of the timeline — a library with one photo from 1994 and the
    /// rest from last year should not produce three hundred empty rows.
    /// </remarks>
    public async Task<IReadOnlyList<TimeBucket>> GetTimeBucketsAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT SUBSTR(captured_utc, 1, 7) AS ym, COUNT(*)
            FROM photos GROUP BY ym ORDER BY ym ASC;
            """;

        var buckets = new List<TimeBucket>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            string ym = reader.GetString(0);
            if (DateTimeOffset.TryParseExact(ym + "-01T00:00:00.000Z", TimeFormat,
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                    out DateTimeOffset start))
            {
                buckets.Add(new TimeBucket(start, reader.GetInt32(1)));
            }
        }

        return buckets;
    }

    /// <summary>
    /// Groups photos whose perceptual hashes are within
    /// <see cref="Imaging.ImageHashing.NearDuplicateThreshold"/> of each other.
    /// </summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Groups of two or more photo ids that look like the same shot.</returns>
    /// <remarks>
    /// Deliberately an in-memory O(n²) sweep rather than SQL. Hamming distance is not something
    /// a B-tree can index, and at realistic library sizes the scan is milliseconds — 50,000
    /// photos is 1.25 billion 64-bit XOR-popcounts, which is a few seconds once, in the
    /// background. If that ever stops being true the answer is a BK-tree, not a cleverer query.
    /// </remarks>
    public async Task<IReadOnlyList<IReadOnlyList<long>>> FindNearDuplicateGroupsAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT id, perceptual_hash FROM photos ORDER BY captured_utc;";

        var items = new List<(long Id, string Hash)>();
        await using (SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false))
        {
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                items.Add((reader.GetInt64(0), reader.GetString(1)));
            }
        }

        var groups = new List<IReadOnlyList<long>>();
        var claimed = new HashSet<long>();

        for (int i = 0; i < items.Count; i++)
        {
            if (!claimed.Add(items[i].Id))
            {
                continue;
            }

            var group = new List<long> { items[i].Id };
            for (int j = i + 1; j < items.Count; j++)
            {
                if (!claimed.Contains(items[j].Id) &&
                    Imaging.ImageHashing.AreNearDuplicates(items[i].Hash, items[j].Hash))
                {
                    group.Add(items[j].Id);
                    claimed.Add(items[j].Id);
                }
            }

            if (group.Count > 1)
            {
                groups.Add(group);
            }
        }

        return groups;
    }

    /// <summary>
    /// Every distinct term the library actually contains: tags, camera makes and models, lens
    /// names, folder names, and words from filenames and captions.
    /// </summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Distinct lowercase terms, longest first.</returns>
    /// <remarks>
    /// <para>
    /// This is what fuzzy correction matches against, and matching against the library's own
    /// vocabulary rather than a dictionary is the point. A dictionary would "correct" a query
    /// for <c>a7iv</c> or <c>ILCE-7M4</c> into some real English word the user does not own a
    /// single photo of. Correcting only towards terms that exist here means every suggestion is
    /// guaranteed to return something.
    /// </para>
    /// <para>
    /// Cached by the caller, not here — it changes only on import, and the search path should
    /// not pay for a full scan on every keystroke.
    /// </para>
    /// </remarks>
    public async Task<IReadOnlyList<string>> GetVocabularyAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT DISTINCT tag FROM photo_tags
            UNION SELECT DISTINCT LOWER(camera_make)  FROM photos WHERE camera_make  IS NOT NULL
            UNION SELECT DISTINCT LOWER(camera_model) FROM photos WHERE camera_model IS NOT NULL
            UNION SELECT DISTINCT LOWER(lens_model)   FROM photos WHERE lens_model   IS NOT NULL
            UNION SELECT DISTINCT LOWER(place_city)   FROM photos WHERE place_city   IS NOT NULL
            UNION SELECT DISTINCT LOWER(place_country) FROM photos WHERE place_country IS NOT NULL
            UNION SELECT DISTINCT LOWER(file_name)    FROM photos
            UNION SELECT DISTINCT LOWER(ai_caption)   FROM photos WHERE ai_caption   IS NOT NULL;
            """;

        var terms = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            if (reader.IsDBNull(0))
            {
                continue;
            }

            // Filenames and captions are split into words. A filename is often the only
            // description an unanalysed photo has, and "020_car_sunbeam-talbot.jpg" is only
            // useful to a searcher once "sunbeam" and "talbot" are separately findable.
            foreach (string word in reader.GetString(0)
                         .Split(['-', '_', ' ', '.', ',', '/', '\\', '(', ')'],
                                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                // Two characters is below the useful floor, and pure digits are frame numbers
                // and dates rather than words worth correcting towards.
                if (word.Length >= 3 && !word.All(char.IsDigit))
                {
                    terms.Add(word.ToLowerInvariant());
                }
            }
        }

        return terms.OrderByDescending(t => t.Length).ToArray();
    }

    /// <summary>
    /// Counts photos per value of a browsable dimension, for the sidebar and filter chips.
    /// </summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Facet name to its values and counts, each ordered by count descending.</returns>
    /// <remarks>
    /// Facets are how someone narrows a library without knowing what to type. Every dimension
    /// here is one a person actually thinks in — where the files live, what took them, when,
    /// and what is in them — and each maps to a field on <see cref="PhotoQuery"/>, so clicking
    /// a facet is exactly equivalent to typing the filter.
    /// </remarks>
    public async Task<IReadOnlyDictionary<string, IReadOnlyList<(string Value, int Count)>>> GetFacetsAsync(
        CancellationToken cancellationToken = default)
    {
        var facets = new Dictionary<string, IReadOnlyList<(string, int)>>(StringComparer.Ordinal);

        await using SqliteConnection connection = _database.Open();

        async Task<IReadOnlyList<(string, int)>> CollectAsync(string sql)
        {
            await using SqliteCommand command = connection.CreateCommand();
            command.CommandText = sql;

            var rows = new List<(string, int)>();
            await using SqliteDataReader reader =
                await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                if (!reader.IsDBNull(0))
                {
                    rows.Add((reader.GetString(0), reader.GetInt32(1)));
                }
            }

            return rows;
        }

        facets["folder"] = await CollectAsync("""
            SELECT source_folder, COUNT(*) FROM photos
            WHERE source_folder <> '' GROUP BY source_folder ORDER BY COUNT(*) DESC, source_folder;
            """).ConfigureAwait(false);

        facets["camera"] = await CollectAsync("""
            SELECT camera_model, COUNT(*) FROM photos
            WHERE camera_model IS NOT NULL GROUP BY camera_model ORDER BY COUNT(*) DESC, camera_model;
            """).ConfigureAwait(false);

        facets["year"] = await CollectAsync("""
            SELECT SUBSTR(captured_utc, 1, 4), COUNT(*) FROM photos
            GROUP BY 1 ORDER BY 1 DESC;
            """).ConfigureAwait(false);

        facets["tag"] = await CollectAsync("""
            SELECT tag, COUNT(*) FROM photo_tags GROUP BY tag ORDER BY COUNT(*) DESC, tag LIMIT 60;
            """).ConfigureAwait(false);

        // Real place names, resolved at ingest from the embedded gazetteer. This used to be
        // one-degree coordinate cells labelled "22,114", which is not a place anybody has been
        // to. Cities and countries are separate facets because they are different questions:
        // "photos from Japan" and "photos in Kyoto" narrow by very different amounts.
        facets["city"] = await CollectAsync("""
            SELECT place_city, COUNT(*) FROM photos
            WHERE place_city IS NOT NULL AND place_city <> ''
            GROUP BY place_city ORDER BY COUNT(*) DESC, place_city LIMIT 40;
            """).ConfigureAwait(false);

        facets["country"] = await CollectAsync("""
            SELECT place_country, COUNT(*) FROM photos
            WHERE place_country IS NOT NULL
            GROUP BY place_country ORDER BY COUNT(*) DESC, place_country LIMIT 40;
            """).ConfigureAwait(false);

        return facets;
    }

    /// <summary>
    /// Moves a photograph to a new processing state.
    /// </summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="state">The new state.</param>
    /// <param name="detail">Failure reason or claiming device, where relevant.</param>
    /// <param name="model">Model version responsible, recorded so a re-run can find stale rows.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// Bumps the revision as well. Processing state is something other devices need to see —
    /// it is how they avoid analysing a photograph another machine has already done — so it has
    /// to be part of what the sync layer publishes.
    /// </remarks>
    public async Task SetStateAsync(
        long photoId,
        ProcessingState state,
        string? detail = null,
        string? model = null,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            UPDATE photos
            SET state = $state,
                state_detail = $detail,
                model_version = COALESCE($model, model_version),
                revision = revision + 1
            WHERE id = $id;
            """;
        command.Parameters.AddWithValue("$state", (int)state);
        command.Parameters.AddWithValue("$detail", (object?)detail ?? DBNull.Value);
        command.Parameters.AddWithValue("$model", (object?)model ?? DBNull.Value);
        command.Parameters.AddWithValue("$id", photoId);

        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Attaches tags to a photograph and refreshes its full-text row.
    /// </summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="tags">Tags to attach.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// AI-sourced tags replace previous AI-sourced tags for the same photograph, because a new
    /// model run supersedes the old one and merging would accumulate every label any model has
    /// ever guessed. File- and user-sourced tags are left alone — that is what the provenance
    /// column is for.
    /// </remarks>
    public async Task AddTagsAsync(
        long photoId,
        IReadOnlyList<PhotoTag> tags,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(tags);

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        if (tags.Any(t => t.Source == MetadataSource.Ai))
        {
            await using SqliteCommand clear = connection.CreateCommand();
            clear.Transaction = tx;
            clear.CommandText =
                "DELETE FROM photo_tags WHERE photo_id = $id AND source = $ai;";
            clear.Parameters.AddWithValue("$id", photoId);
            clear.Parameters.AddWithValue("$ai", (int)MetadataSource.Ai);
            await clear.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        if (tags.Count > 0)
        {
            await WriteTagsAsync(connection, tx, photoId, tags, cancellationToken).ConfigureAwait(false);
        }

        await ReindexTextAsync(connection, tx, photoId, cancellationToken).ConfigureAwait(false);
        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Attaches one keyword a person typed.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="tag">The keyword.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>Whether anything was added.</returns>
    /// <remarks>
    /// <para>
    /// Separate from <see cref="AddTagsAsync"/>, which replaces the whole AI-sourced set on every
    /// call. A person adding one word must not clear the model's, and a model re-run must not
    /// clear theirs — the provenance column is what keeps the two apart, and this method is the
    /// only way a <c>User</c> tag ever gets written singly.
    /// </para>
    /// <para>
    /// Lower-cased and trimmed. Tags are for finding things, and a library where "Beach",
    /// "beach" and "beach " are three different tags is a library where the facet list is
    /// nonsense and half the searches miss.
    /// </para>
    /// </remarks>
    public async Task<bool> AddUserTagAsync(
        long photoId,
        string tag,
        CancellationToken cancellationToken = default)
    {
        string cleaned = (tag ?? string.Empty).Trim().ToLowerInvariant();
        if (cleaned.Length == 0)
        {
            return false;
        }

        cleaned = cleaned[..Math.Min(cleaned.Length, 60)];

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        await WriteTagsAsync(connection, tx, photoId,
            [new PhotoTag(cleaned, 1.0, MetadataSource.User)], cancellationToken).ConfigureAwait(false);

        await ReindexTextAsync(connection, tx, photoId, cancellationToken).ConfigureAwait(false);
        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        return true;
    }

    /// <summary>Removes a keyword a person typed.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="tag">The keyword.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>Whether anything was removed.</returns>
    /// <remarks>
    /// Only <c>User</c> tags. A person can take back their own word; they cannot delete what the
    /// model found or what the file already carried, because a re-run would put it straight back
    /// and the deletion would look like a bug. Suppressing a model's tag is a different feature
    /// and needs somewhere to record the suppression.
    /// </remarks>
    public async Task<bool> RemoveUserTagAsync(
        long photoId,
        string tag,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        int removed;

        await using (SqliteCommand command = connection.CreateCommand())
        {
            command.Transaction = tx;
            command.CommandText =
                "DELETE FROM photo_tags WHERE photo_id = $id AND tag = $tag AND source = $user;";
            command.Parameters.AddWithValue("$id", photoId);
            command.Parameters.AddWithValue("$tag", (tag ?? string.Empty).Trim().ToLowerInvariant());
            command.Parameters.AddWithValue("$user", (int)MetadataSource.User);
            removed = await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        await ReindexTextAsync(connection, tx, photoId, cancellationToken).ConfigureAwait(false);
        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        return removed > 0;
    }

    /// <summary>Stores what a vision model wrote about a photograph.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="description">The description, or null to clear it.</param>
    /// <param name="model">Identifier of the model that wrote it.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// Reindexes the full-text row, which is the entire point: a description nobody can search
    /// is a paragraph of text taking up disk. It lands in the <c>description</c> FTS column
    /// alongside the user's own notes, so a query for "deckchair" reaches a photograph where
    /// nothing but the model ever mentioned one.
    /// </remarks>
    public async Task SetDescriptionAsync(
        long photoId,
        string? description,
        string model,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        await using (SqliteCommand command = connection.CreateCommand())
        {
            command.Transaction = tx;
            command.CommandText = """
                UPDATE photos SET ai_description = $text, model_version = $model
                WHERE id = $id;
                """;
            command.Parameters.AddWithValue("$text", (object?)description ?? DBNull.Value);
            command.Parameters.AddWithValue("$model", model);
            command.Parameters.AddWithValue("$id", photoId);
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        await ReindexTextAsync(connection, tx, photoId, cancellationToken).ConfigureAwait(false);
        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Photographs nobody has described yet.</summary>
    /// <param name="limit">Most ids to return.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Ids still to describe, oldest-indexed first.</returns>
    /// <remarks>
    /// Presence of text is the marker, not a model version. Describing is slow — seconds per
    /// photograph — so a better model arriving is not a reason to silently redo a library
    /// overnight; that is a deliberate act, and clearing the column is how it is asked for.
    /// </remarks>
    public async Task<IReadOnlyList<long>> UndescribedAsync(
        int limit = 100000,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT id FROM photos
            WHERE ai_description IS NULL OR TRIM(ai_description) = ''
            ORDER BY id
            LIMIT $limit;
            """;
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

    /// <summary>How many photographs have been described, and how many there are.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Described photographs, and the library total.</returns>
    public async Task<(int Described, int Total)> DescriptionCoverageAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT SUM(CASE WHEN ai_description IS NOT NULL AND TRIM(ai_description) <> ''
                            THEN 1 ELSE 0 END),
                   COUNT(*)
            FROM photos;
            """;

        await using SqliteDataReader reader = await command
            .ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        return await reader.ReadAsync(cancellationToken).ConfigureAwait(false)
            ? (reader.IsDBNull(0) ? 0 : reader.GetInt32(0), reader.GetInt32(1))
            : (0, 0);
    }

    /// <summary>Counts photos in each processing state.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>A map of state to count; states with no photos are absent.</returns>
    public async Task<IReadOnlyDictionary<ProcessingState, int>> GetStateCountsAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT state, COUNT(*) FROM photos GROUP BY state;";

        var counts = new Dictionary<ProcessingState, int>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            counts[(ProcessingState)reader.GetInt32(0)] = reader.GetInt32(1);
        }

        return counts;
    }

    /// <summary>
    /// Turns user-typed text into an FTS5 query, neutralising the operator syntax.
    /// </summary>
    /// <remarks>
    /// FTS5 has its own expression language — <c>NEAR</c>, <c>*</c>, <c>-</c>, <c>"</c>, and
    /// parentheses — and a search box that exposes it produces syntax errors the moment someone
    /// types an apostrophe. Each term is quoted as a literal and joined with AND, so the box
    /// behaves the way people expect a search box to behave. A trailing <c>*</c> is added to the
    /// final term so results narrow as you type rather than vanishing mid-word.
    /// </remarks>
    private static string ToFtsQuery(string text)
    {
        string[] terms = text
            .Split([' ', '\t', '\n', ',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.Trim('"', '\'', '(', ')', '*', '-', '^'))
            .Where(t => t.Length > 0)
            .ToArray();

        if (terms.Length == 0)
        {
            return "\"\"";
        }

        var quoted = terms.Select(t => "\"" + t.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"").ToArray();
        quoted[^1] += "*";
        return string.Join(" AND ", quoted);
    }

    private static PhotoRecord Read(SqliteDataReader reader)
    {
        string? Text(string column)
        {
            int index = reader.GetOrdinal(column);
            return reader.IsDBNull(index) ? null : reader.GetString(index);
        }

        int? Int(string column)
        {
            int index = reader.GetOrdinal(column);
            return reader.IsDBNull(index) ? null : reader.GetInt32(index);
        }

        double? Real(string column)
        {
            int index = reader.GetOrdinal(column);
            return reader.IsDBNull(index) ? null : reader.GetDouble(index);
        }

        double? latitude = Real("gps_lat");
        double? longitude = Real("gps_lon");

        return new PhotoRecord
        {
            Id = reader.GetInt64(reader.GetOrdinal("id")),
            ContentHash = reader.GetString(reader.GetOrdinal("content_hash")),
            PerceptualHash = reader.GetString(reader.GetOrdinal("perceptual_hash")),
            OriginalPath = reader.GetString(reader.GetOrdinal("original_path")),
            FileName = reader.GetString(reader.GetOrdinal("file_name")),
            SourceFolder = Text("source_folder") ?? string.Empty,
            MimeType = reader.GetString(reader.GetOrdinal("mime_type")),
            Width = reader.GetInt32(reader.GetOrdinal("width")),
            Height = reader.GetInt32(reader.GetOrdinal("height")),
            FileSize = reader.GetInt64(reader.GetOrdinal("file_size")),
            CapturedUtc = ParseIso(reader.GetString(reader.GetOrdinal("captured_utc"))),
            CaptureTimeIsExact = reader.GetInt32(reader.GetOrdinal("capture_exact")) != 0,
            FileModifiedUtc = ParseIso(reader.GetString(reader.GetOrdinal("file_modified_utc"))),
            IndexedUtc = ParseIso(reader.GetString(reader.GetOrdinal("indexed_utc"))),
            Camera = new CameraInfo
            {
                Make = Text("camera_make"),
                Model = Text("camera_model"),
                Lens = Text("lens_model"),
                Iso = Int("iso"),
                FNumber = Real("f_number"),
                ExposureSeconds = Real("exposure_seconds"),
                FocalLengthMm = Real("focal_length_mm"),
            },
            Location = latitude is not null && longitude is not null
                ? new GeoPoint(latitude.Value, longitude.Value, Real("gps_alt"))
                : null,
            Place = Text("place_country") is { } resolvedCountry
                ? new PlaceName(
                    Text("place_city") ?? string.Empty,
                    resolvedCountry,
                    Text("place_code") ?? string.Empty,
                    Text("place_label") ?? resolvedCountry)
                : null,
            Orientation = reader.GetInt32(reader.GetOrdinal("orientation")),
            ThumbnailKey = Text("thumbnail_key"),
            ProxyKey = Text("proxy_key"),
            State = (ProcessingState)reader.GetInt32(reader.GetOrdinal("state")),
            StateDetail = Text("state_detail"),
            ModelVersion = Text("model_version"),
            AiCaption = Text("ai_caption"),
            AiDescription = Text("ai_description"),
            UserTitle = Text("user_title"),
            UserNotes = Text("user_notes"),
            Rating = reader.GetInt32(reader.GetOrdinal("rating")),
            IsFavourite = reader.GetInt32(reader.GetOrdinal("is_favourite")) != 0,
            Revision = reader.GetInt64(reader.GetOrdinal("revision")),
        };
    }
}
