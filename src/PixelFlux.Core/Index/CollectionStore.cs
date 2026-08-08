using System.Globalization;
using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace PixelFlux.Core.Index;

/// <summary>What kind of collection this is.</summary>
public enum CollectionKind
{
    /// <summary>A hand-picked list of photos, in an order the user chose.</summary>
    Manual = 0,

    /// <summary>A stored query, re-evaluated every time it is opened.</summary>
    Smart = 1,
}

/// <summary>A named group of photos — an album, or a saved search.</summary>
/// <param name="Id">Row id.</param>
/// <param name="Name">Display name. Unique, case-insensitively.</param>
/// <param name="Kind">Whether membership is hand-picked or computed.</param>
/// <param name="Query">The stored query for a smart collection; null for a manual one.</param>
/// <param name="CoverPhotoId">Photo to show as the cover, if one was chosen.</param>
/// <param name="Count">Number of photos in it.</param>
/// <param name="UpdatedUtc">When it last changed.</param>
public sealed record PhotoCollection(
    long Id,
    string Name,
    CollectionKind Kind,
    PhotoQuery? Query,
    long? CoverPhotoId,
    int Count,
    DateTimeOffset UpdatedUtc);

/// <summary>
/// Reads and writes collections: manual albums and smart folders.
/// </summary>
/// <remarks>
/// <para>
/// The two kinds share a table because they are the same thing from the system's point of view:
/// a name, and a way of producing a list of photos. A manual album stores its list; a smart
/// folder stores the question and asks it again on each open. Everything above — the sidebar,
/// rename, delete, reorder, slideshow, export — is written once and works for both.
/// </para>
/// <para>
/// A photo can belong to any number of collections, and belonging to one never moves or copies
/// the file. Collections are views over the library, not folders on disk; the originals stay
/// exactly where the user put them.
/// </para>
/// </remarks>
public sealed class CollectionStore
{
    private const string TimeFormat = "yyyy-MM-ddTHH:mm:ss.fffZ";

    private readonly PhotoDatabase _database;

    /// <summary>Creates a store over a migrated database.</summary>
    /// <param name="database">The database handle.</param>
    public CollectionStore(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    private static string Now() => DateTimeOffset.UtcNow.ToString(TimeFormat, CultureInfo.InvariantCulture);

    /// <summary>Creates a manual album.</summary>
    /// <param name="name">Display name.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>The new collection's id.</returns>
    public Task<long> CreateAlbumAsync(string name, CancellationToken cancellationToken = default)
        => CreateAsync(name, CollectionKind.Manual, null, cancellationToken);

    /// <summary>Creates a smart folder from a query.</summary>
    /// <param name="name">Display name.</param>
    /// <param name="query">The query to store and re-run on each open.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>The new collection's id.</returns>
    public Task<long> CreateSmartFolderAsync(
        string name,
        PhotoQuery query,
        CancellationToken cancellationToken = default)
        => CreateAsync(name, CollectionKind.Smart, query, cancellationToken);

    private async Task<long> CreateAsync(
        string name,
        CollectionKind kind,
        PhotoQuery? query,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A collection needs a name.", nameof(name));
        }

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO collections (name, kind, query_json, position, created_utc, updated_utc)
            VALUES ($name, $kind, $query, COALESCE((SELECT MAX(position) + 1 FROM collections), 0), $now, $now)
            RETURNING id;
            """;
        command.Parameters.AddWithValue("$name", name.Trim());
        command.Parameters.AddWithValue("$kind", (int)kind);
        command.Parameters.AddWithValue("$query",
            query is null ? DBNull.Value : JsonSerializer.Serialize(query));
        command.Parameters.AddWithValue("$now", Now());

        object? id = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
        return Convert.ToInt64(id, CultureInfo.InvariantCulture);
    }

    /// <summary>Lists every collection with its current photo count.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Collections in display order.</returns>
    /// <remarks>
    /// The count for a smart folder is reported as -1 rather than computed here. Working it out
    /// means running that folder's query, and doing so for every smart folder on every sidebar
    /// render would put a full search behind an operation the user experiences as "the window
    /// opened". The caller fills these in lazily, for the ones actually on screen.
    /// </remarks>
    public async Task<IReadOnlyList<PhotoCollection>> ListAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT c.id, c.name, c.kind, c.query_json, c.cover_photo, c.updated_utc,
                   (SELECT COUNT(*) FROM collection_photos cp WHERE cp.collection_id = c.id)
            FROM collections c
            ORDER BY c.position, c.name COLLATE NOCASE;
            """;

        var results = new List<PhotoCollection>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            var kind = (CollectionKind)reader.GetInt32(2);

            PhotoQuery? query = null;
            if (!reader.IsDBNull(3))
            {
                try
                {
                    query = JsonSerializer.Deserialize<PhotoQuery>(reader.GetString(3));
                }
                catch (JsonException)
                {
                    // A smart folder saved by a newer version with a query shape this build
                    // cannot parse. Showing it as an empty folder is better than refusing to
                    // list any collections at all.
                }
            }

            results.Add(new PhotoCollection(
                reader.GetInt64(0),
                reader.GetString(1),
                kind,
                query,
                reader.IsDBNull(4) ? null : reader.GetInt64(4),
                kind == CollectionKind.Smart ? -1 : reader.GetInt32(6),
                DateTimeOffset.ParseExact(reader.GetString(5), TimeFormat, CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal)));
        }

        return results;
    }

    /// <summary>Adds photos to a manual album, appending them in the order given.</summary>
    /// <param name="collectionId">The album.</param>
    /// <param name="photoIds">Photos to add. Ones already present are left where they are.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many were newly added.</returns>
    public async Task<int> AddAsync(
        long collectionId,
        IReadOnlyList<long> photoIds,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(photoIds);

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        int nextPosition;
        await using (SqliteCommand max = connection.CreateCommand())
        {
            max.Transaction = tx;
            max.CommandText =
                "SELECT COALESCE(MAX(position) + 1, 0) FROM collection_photos WHERE collection_id = $c;";
            max.Parameters.AddWithValue("$c", collectionId);
            nextPosition = Convert.ToInt32(
                await max.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false),
                CultureInfo.InvariantCulture);
        }

        int added = 0;
        await using (SqliteCommand insert = connection.CreateCommand())
        {
            insert.Transaction = tx;
            insert.CommandText = """
                INSERT INTO collection_photos (collection_id, photo_id, position, added_utc)
                VALUES ($c, $p, $pos, $now)
                ON CONFLICT(collection_id, photo_id) DO NOTHING;
                """;

            SqliteParameter c = insert.Parameters.Add("$c", SqliteType.Integer);
            SqliteParameter p = insert.Parameters.Add("$p", SqliteType.Integer);
            SqliteParameter pos = insert.Parameters.Add("$pos", SqliteType.Integer);
            insert.Parameters.AddWithValue("$now", Now());

            c.Value = collectionId;
            foreach (long photoId in photoIds)
            {
                p.Value = photoId;
                pos.Value = nextPosition++;
                added += await insert.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }
        }

        await TouchAsync(connection, tx, collectionId, cancellationToken).ConfigureAwait(false);
        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        return added;
    }

    /// <summary>Removes photos from an album. The photos themselves are untouched.</summary>
    /// <param name="collectionId">The album.</param>
    /// <param name="photoIds">Photos to remove.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many were removed.</returns>
    public async Task<int> RemoveAsync(
        long collectionId,
        IReadOnlyList<long> photoIds,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(photoIds);

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            "DELETE FROM collection_photos WHERE collection_id = $c AND photo_id = $p;";

        SqliteParameter c = command.Parameters.Add("$c", SqliteType.Integer);
        SqliteParameter p = command.Parameters.Add("$p", SqliteType.Integer);
        c.Value = collectionId;

        int removed = 0;
        foreach (long photoId in photoIds)
        {
            p.Value = photoId;
            removed += await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        return removed;
    }

    /// <summary>
    /// Moves photos from one album to another: removes them from the source and adds them to
    /// the target, as a single transaction.
    /// </summary>
    /// <param name="fromCollectionId">Album to take them out of.</param>
    /// <param name="toCollectionId">Album to put them into.</param>
    /// <param name="photoIds">Photos to move.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many photos ended up in the target album as a result.</returns>
    /// <remarks>
    /// <para>
    /// Move is a distinct operation from add-then-remove, and the difference is the transaction.
    /// Doing it in two calls means a failure between them leaves photos in neither album — the
    /// user's curation silently deleted. Here either both sides happen or neither does.
    /// </para>
    /// <para>
    /// Note that moving is only meaningful <em>between</em> albums. A photo does not live in an
    /// album the way a file lives in a folder — membership is a view, and a photo can be in any
    /// number of them — so there is no "move out of the library". Removing from the last album
    /// a photo belongs to leaves the photograph exactly where it was on disk.
    /// </para>
    /// </remarks>
    public async Task<int> MoveAsync(
        long fromCollectionId,
        long toCollectionId,
        IReadOnlyList<long> photoIds,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(photoIds);

        if (fromCollectionId == toCollectionId || photoIds.Count == 0)
        {
            return 0;
        }

        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        int nextPosition;
        await using (SqliteCommand max = connection.CreateCommand())
        {
            max.Transaction = tx;
            max.CommandText =
                "SELECT COALESCE(MAX(position) + 1, 0) FROM collection_photos WHERE collection_id = $c;";
            max.Parameters.AddWithValue("$c", toCollectionId);
            nextPosition = Convert.ToInt32(
                await max.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false),
                CultureInfo.InvariantCulture);
        }

        int moved = 0;

        await using (SqliteCommand insert = connection.CreateCommand())
        await using (SqliteCommand delete = connection.CreateCommand())
        {
            insert.Transaction = tx;
            insert.CommandText = """
                INSERT INTO collection_photos (collection_id, photo_id, position, added_utc)
                VALUES ($to, $p, $pos, $now)
                ON CONFLICT(collection_id, photo_id) DO NOTHING;
                """;
            SqliteParameter to = insert.Parameters.Add("$to", SqliteType.Integer);
            SqliteParameter insertPhoto = insert.Parameters.Add("$p", SqliteType.Integer);
            SqliteParameter position = insert.Parameters.Add("$pos", SqliteType.Integer);
            insert.Parameters.AddWithValue("$now", Now());
            to.Value = toCollectionId;

            delete.Transaction = tx;
            delete.CommandText =
                "DELETE FROM collection_photos WHERE collection_id = $from AND photo_id = $p;";
            SqliteParameter from = delete.Parameters.Add("$from", SqliteType.Integer);
            SqliteParameter deletePhoto = delete.Parameters.Add("$p", SqliteType.Integer);
            from.Value = fromCollectionId;

            foreach (long photoId in photoIds)
            {
                insertPhoto.Value = photoId;
                position.Value = nextPosition++;
                await insert.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

                deletePhoto.Value = photoId;
                await delete.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
                moved++;
            }
        }

        await TouchAsync(connection, tx, fromCollectionId, cancellationToken).ConfigureAwait(false);
        await TouchAsync(connection, tx, toCollectionId, cancellationToken).ConfigureAwait(false);
        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);

        return moved;
    }

    /// <summary>Returns which albums a photo currently belongs to.</summary>
    /// <param name="photoId">The photo.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>Collection ids.</returns>
    /// <remarks>
    /// Drives the tick marks in the "add to album" menu, so the menu shows where a photo already
    /// is rather than offering to add it somewhere it already lives.
    /// </remarks>
    public async Task<IReadOnlyList<long>> GetMembershipAsync(
        long photoId,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT collection_id FROM collection_photos WHERE photo_id = $p;";
        command.Parameters.AddWithValue("$p", photoId);

        var ids = new List<long>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            ids.Add(reader.GetInt64(0));
        }

        return ids;
    }

    /// <summary>Renames a collection.</summary>
    /// <param name="collectionId">The collection.</param>
    /// <param name="name">The new name.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task RenameAsync(long collectionId, string name, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A collection needs a name.", nameof(name));
        }

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "UPDATE collections SET name = $name, updated_utc = $now WHERE id = $id;";
        command.Parameters.AddWithValue("$name", name.Trim());
        command.Parameters.AddWithValue("$now", Now());
        command.Parameters.AddWithValue("$id", collectionId);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Deletes a collection. The photos in it are not deleted.</summary>
    /// <param name="collectionId">The collection.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task DeleteAsync(long collectionId, CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        // Membership rows go with it via ON DELETE CASCADE; the photos table is untouched.
        command.CommandText = "DELETE FROM collections WHERE id = $id;";
        command.Parameters.AddWithValue("$id", collectionId);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task TouchAsync(
        SqliteConnection connection,
        SqliteTransaction tx,
        long collectionId,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = tx;
        command.CommandText = "UPDATE collections SET updated_utc = $now WHERE id = $id;";
        command.Parameters.AddWithValue("$now", Now());
        command.Parameters.AddWithValue("$id", collectionId);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }
}
