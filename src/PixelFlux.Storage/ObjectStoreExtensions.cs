using System.Text.Json;

namespace PixelFlux.Storage;

/// <summary>
/// Convenience wrappers over <see cref="IObjectStore"/> for the shapes PixelFlux actually
/// stores: small byte arrays and JSON documents.
/// </summary>
/// <remarks>
/// These live as extension methods rather than as interface members on purpose. Every one of
/// them is expressible in terms of the six core operations, so keeping them out of the
/// interface means a new store implementation has six methods to write instead of a dozen, and
/// cannot accidentally give <c>ExistsAsync</c> different semantics from <c>StatAsync</c>.
/// </remarks>
public static class ObjectStoreExtensions
{
    /// <summary>
    /// JSON settings used for every record PixelFlux writes to shared storage.
    /// </summary>
    /// <remarks>
    /// Two choices matter here and should not be changed casually, because objects written by
    /// one version of the app are read by another:
    /// <list type="bullet">
    /// <item><description>
    /// Property names are written verbatim (no camel-case policy), so the C# record definition
    /// is the wire format and there is no second naming convention to keep in your head.
    /// </description></item>
    /// <item><description>
    /// Unknown members are ignored on read, which is what lets a newer device add a field to a
    /// revision record without breaking an older device that is still syncing the same folder.
    /// </description></item>
    /// </list>
    /// </remarks>
    public static readonly JsonSerializerOptions RecordJson = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = null,
        PropertyNameCaseInsensitive = true,
    };

    /// <summary>Returns whether an object exists at <paramref name="key"/>.</summary>
    /// <param name="store">The store to query.</param>
    /// <param name="key">The object key.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns><see langword="true"/> if the object exists.</returns>
    public static async Task<bool> ExistsAsync(
        this IObjectStore store,
        string key,
        CancellationToken cancellationToken = default)
        => await store.StatAsync(key, cancellationToken).ConfigureAwait(false) is not null;

    /// <summary>Reads an entire object into memory.</summary>
    /// <param name="store">The store to read from.</param>
    /// <param name="key">The object key.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The object bytes, or <see langword="null"/> if it does not exist.</returns>
    public static async Task<byte[]?> ReadAllBytesAsync(
        this IObjectStore store,
        string key,
        CancellationToken cancellationToken = default)
    {
        await using Stream? source = await store.OpenReadAsync(key, cancellationToken).ConfigureAwait(false);
        if (source is null)
        {
            return null;
        }

        using var buffer = new MemoryStream();
        await source.CopyToAsync(buffer, cancellationToken).ConfigureAwait(false);
        return buffer.ToArray();
    }

    /// <summary>Writes a byte array as an object, replacing any existing one.</summary>
    /// <param name="store">The store to write to.</param>
    /// <param name="key">The object key.</param>
    /// <param name="content">The bytes to write.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public static async Task WriteAllBytesAsync(
        this IObjectStore store,
        string key,
        ReadOnlyMemory<byte> content,
        CancellationToken cancellationToken = default)
    {
        using var buffer = new MemoryStream(content.ToArray(), writable: false);
        await store.WriteAsync(key, buffer, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Reads and deserialises a JSON object.</summary>
    /// <typeparam name="T">The record type to deserialise into.</typeparam>
    /// <param name="store">The store to read from.</param>
    /// <param name="key">The object key.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>
    /// The deserialised value, or <see langword="null"/> if the object does not exist
    /// <em>or</em> its contents could not be parsed.
    /// </returns>
    /// <remarks>
    /// Malformed JSON is reported as <see langword="null"/> rather than as an exception. Shared
    /// storage is a place where partially-synced and conflict-renamed files genuinely appear,
    /// and a single corrupt record must not be able to stop a device from syncing the rest.
    /// Callers that need to distinguish "absent" from "corrupt" should
    /// <see cref="ExistsAsync">stat the key</see> as well.
    /// </remarks>
    public static async Task<T?> ReadJsonAsync<T>(
        this IObjectStore store,
        string key,
        CancellationToken cancellationToken = default)
        where T : class
    {
        byte[]? bytes = await store.ReadAllBytesAsync(key, cancellationToken).ConfigureAwait(false);
        if (bytes is null || bytes.Length == 0)
        {
            return null;
        }

        try
        {
            return JsonSerializer.Deserialize<T>(bytes, RecordJson);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>Serialises a value to JSON and writes it, replacing any existing object.</summary>
    /// <typeparam name="T">The record type to serialise.</typeparam>
    /// <param name="store">The store to write to.</param>
    /// <param name="key">The object key.</param>
    /// <param name="value">The value to serialise.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public static Task WriteJsonAsync<T>(
        this IObjectStore store,
        string key,
        T value,
        CancellationToken cancellationToken = default)
        => store.WriteAllBytesAsync(key, JsonSerializer.SerializeToUtf8Bytes(value, RecordJson), cancellationToken);

    /// <summary>
    /// Atomically creates a JSON object only if the key is currently free.
    /// </summary>
    /// <typeparam name="T">The record type to serialise.</typeparam>
    /// <param name="store">The store to write to.</param>
    /// <param name="key">The object key to claim.</param>
    /// <param name="value">The claim record to serialise.</param>
    /// <param name="cancellationToken">Cancels the attempt.</param>
    /// <returns><see langword="true"/> if this caller created the object.</returns>
    public static Task<bool> TryCreateJsonAsync<T>(
        this IObjectStore store,
        string key,
        T value,
        CancellationToken cancellationToken = default)
        => store.TryCreateAsync(key, JsonSerializer.SerializeToUtf8Bytes(value, RecordJson), cancellationToken);

    /// <summary>
    /// Materialises a prefix listing into a list.
    /// </summary>
    /// <param name="store">The store to enumerate.</param>
    /// <param name="prefix">Key prefix to match.</param>
    /// <param name="cancellationToken">Cancels the enumeration.</param>
    /// <returns>Every matching entry, in unspecified order.</returns>
    /// <remarks>
    /// Only appropriate for prefixes known to be small — job queues and revision folders, which
    /// hold hundreds of entries at most. Never call this on an empty prefix against a real
    /// library.
    /// </remarks>
    public static async Task<List<ObjectEntry>> ListAllAsync(
        this IObjectStore store,
        string prefix,
        CancellationToken cancellationToken = default)
    {
        var results = new List<ObjectEntry>();
        await foreach (ObjectEntry entry in store.ListAsync(prefix, cancellationToken).ConfigureAwait(false))
        {
            results.Add(entry);
        }

        return results;
    }
}
