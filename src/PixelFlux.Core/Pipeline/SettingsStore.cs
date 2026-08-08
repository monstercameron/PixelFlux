using Microsoft.Data.Sqlite;
using PixelFlux.Core.Index;

namespace PixelFlux.Core.Pipeline;

/// <summary>
/// Key-and-value settings that belong to the photo library rather than to the machine.
/// </summary>
/// <remarks>
/// Deliberately thin. Anything with structure — a schedule, a model choice — serialises itself to a
/// string and parses itself back, so this class never grows a method per preference and adding one
/// never touches the schema. The parsing lives with the thing being parsed, where the rules are.
/// </remarks>
public sealed class SettingsStore
{
    private readonly PhotoDatabase _database;

    /// <summary>Creates a settings store over a migrated database.</summary>
    /// <param name="database">The database handle.</param>
    public SettingsStore(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    /// <summary>Reads a setting.</summary>
    /// <param name="key">The setting's name.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The stored value, or null if it has never been set.</returns>
    public async Task<string?> GetAsync(string key, CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT value FROM app_settings WHERE key = $key;";
        command.Parameters.AddWithValue("$key", key);
        return await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false) as string;
    }

    /// <summary>Writes a setting.</summary>
    /// <param name="key">The setting's name.</param>
    /// <param name="value">The value to store.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task SetAsync(
        string key,
        string value,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO app_settings (key, value) VALUES ($key, $value)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value;
            """;
        command.Parameters.AddWithValue("$key", key);
        command.Parameters.AddWithValue("$value", value);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }
}
