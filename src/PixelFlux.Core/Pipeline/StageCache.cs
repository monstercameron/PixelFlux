using Microsoft.Data.Sqlite;
using PixelFlux.Core.Index;

namespace PixelFlux.Core.Pipeline;

/// <summary>
/// Remembers what a stage produced, keyed by the bytes it looked at.
/// </summary>
/// <remarks>
/// <para>
/// The expensive stages are pure functions: the same image through the same model gives the same
/// answer. Describing a photograph costs about sixteen seconds of processor, and until now every
/// one of those seconds was spent again whenever the row holding the answer went away.
/// </para>
/// <para>
/// The key is (content hash, stage, model), and the interesting choice is that the photograph's id
/// is not in it. An id is a fact about this database; a hash is a fact about the picture. That
/// makes an analysis outlive the row it was computed for — remove a photograph and put it back,
/// reorganise the folders it lives in, rebuild the index from scratch, and the work comes back
/// with it rather than being done again.
/// </para>
/// <para>
/// What it is <i>not</i> for, since the obvious guess is wrong: importing the same file twice.
/// <c>photos.content_hash</c> is unique, so two copies of one picture are one row, and there is no
/// second analysis to save. That claim was written here first and a test removed it.
/// </para>
/// <para>
/// Payloads are the stage's own JSON, opaque here. This class knows how to store bytes against a
/// key and nothing about descriptions, segments or vectors — which is why adding a stage needs no
/// change to it.
/// </para>
/// </remarks>
public sealed class StageCache
{
    private readonly PhotoDatabase _database;

    /// <summary>Creates a cache over a migrated database.</summary>
    /// <param name="database">The database handle.</param>
    public StageCache(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    /// <summary>Looks for a stored result.</summary>
    /// <param name="contentHash">Hash of the image the stage would look at.</param>
    /// <param name="stage">The stage.</param>
    /// <param name="model">The model version asking. An older entry is not a hit.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The stored payload, or null if this exact combination has never been computed.</returns>
    public async Task<string?> GetAsync(
        string contentHash,
        PipelineStage stage,
        string model,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT payload FROM stage_cache
            WHERE content_hash = $hash AND stage = $stage AND model = $model;
            """;
        command.Parameters.AddWithValue("$hash", contentHash);
        command.Parameters.AddWithValue("$stage", stage.Slug());
        command.Parameters.AddWithValue("$model", model);

        object? found = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
        return found as string;
    }

    /// <summary>Stores a result.</summary>
    /// <param name="contentHash">Hash of the image the stage looked at.</param>
    /// <param name="stage">The stage.</param>
    /// <param name="model">The model version that produced it.</param>
    /// <param name="payload">The stage's own serialised output.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task PutAsync(
        string contentHash,
        PipelineStage stage,
        string model,
        string payload,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO stage_cache (content_hash, stage, model, payload, created_at)
            VALUES ($hash, $stage, $model, $payload, $now)
            ON CONFLICT(content_hash, stage, model) DO UPDATE SET
                payload = excluded.payload, created_at = excluded.created_at;
            """;
        command.Parameters.AddWithValue("$hash", contentHash);
        command.Parameters.AddWithValue("$stage", stage.Slug());
        command.Parameters.AddWithValue("$model", model);
        command.Parameters.AddWithValue("$payload", payload);
        command.Parameters.AddWithValue("$now", DateTimeOffset.UtcNow.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>How many results are stored, and how much room they take.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The entry count and the total payload size in bytes.</returns>
    public async Task<(int Entries, long Bytes)> SizeAsync(
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM stage_cache;";

        await using SqliteDataReader reader =
            await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        return await reader.ReadAsync(cancellationToken).ConfigureAwait(false)
            ? (reader.GetInt32(0), reader.GetInt64(1))
            : (0, 0);
    }

    /// <summary>Drops entries no photograph in the library could use.</summary>
    /// <param name="keepOtherModels">
    /// When true, entries from older model versions are kept as long as their image is still in the
    /// library. They cost little and make "re-run with the previous model" possible.
    /// </param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many entries were removed.</returns>
    public async Task<int> PruneAsync(
        bool keepOtherModels = true,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        // Orphans only: an entry whose image is no longer anywhere in the library. Removing a
        // photograph and putting it back is common enough — reorganising folders does it — that
        // pruning on deletion would throw away the answer just before it is asked for again.
        command.CommandText = keepOtherModels
            ? """
              DELETE FROM stage_cache
              WHERE content_hash NOT IN (SELECT content_hash FROM photos);
              """
            : """
              DELETE FROM stage_cache
              WHERE content_hash NOT IN (SELECT content_hash FROM photos)
                 OR model NOT IN (SELECT DISTINCT model FROM photo_jobs WHERE model IS NOT NULL);
              """;
        return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }
}
