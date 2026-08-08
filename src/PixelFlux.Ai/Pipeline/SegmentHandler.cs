using System.Text.Json;
using PixelFlux.Ai.Segmentation;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using PixelFlux.Core.Pipeline;

namespace PixelFlux.Ai.Pipeline;

/// <summary>
/// The stage that finds objects in a photograph and outlines them.
/// </summary>
/// <remarks>
/// Runs after the description because that is the sequence, not because it needs one. Its masks
/// live in the derivative cache under the photograph's content hash, which is the same key the
/// result cache uses — so a cached set of segments always has its overlay images sitting on disk
/// already, and applying one is a database write and nothing more.
/// </remarks>
public sealed class SegmentHandler : IStageHandler
{
    private readonly SegmentationWorker _worker;
    private readonly PhotoStore _photos;

    /// <summary>Creates the handler.</summary>
    /// <param name="worker">The segmentation worker, used one photograph at a time.</param>
    /// <param name="photos">The photo index.</param>
    public SegmentHandler(SegmentationWorker worker, PhotoStore photos)
    {
        ArgumentNullException.ThrowIfNull(worker);
        ArgumentNullException.ThrowIfNull(photos);

        _worker = worker;
        _photos = photos;
    }

    /// <inheritdoc/>
    public PipelineStage Stage => PipelineStage.Segment;

    /// <inheritdoc/>
    public string? ModelVersion => _worker.IsAvailable ? _worker.ModelVersion : null;

    /// <inheritdoc/>
    public async Task ApplyAsync(long photoId, string payload, CancellationToken cancellationToken)
    {
        List<PhotoSegmentRecord> records = StagePayload.Read<PhotoSegmentRecord>(payload);

        // The stored records carry whichever photograph they were computed for, which on a cache
        // hit is a row that no longer exists: the entry outlived it. Rebinding to the row asking
        // now is what makes reuse safe.
        await _worker.RecordAsync(
            photoId,
            [.. records.Select(record => record with { PhotoId = photoId })],
            _worker.ModelVersion,
            cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task<string?> ExecuteAsync(long photoId, CancellationToken cancellationToken)
    {
        PhotoRecord? photo = await _photos.GetAsync(photoId, cancellationToken)
            .ConfigureAwait(false);
        if (photo is null)
        {
            return null;
        }

        IReadOnlyList<PhotoSegmentRecord> records =
            await _worker.ExamineAsync(photo, cancellationToken).ConfigureAwait(false);

        await _worker.RecordAsync(photoId, records, _worker.ModelVersion, cancellationToken)
            .ConfigureAwait(false);

        // An empty result is a real result — this photograph contains none of the eighty classes
        // the model knows — and caching it is what stops a landscape being re-segmented forever.
        return StagePayload.Write(records);
    }
}

/// <summary>Turns a stage's records into cache payloads and back.</summary>
/// <remarks>
/// JSON rather than anything denser. These payloads are read once each and are small — a few
/// hundred bytes for a segmented photograph — so the thing worth optimising is not size but the
/// ability to look at the cache and see what is in it when something goes wrong.
/// </remarks>
public static class StagePayload
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = false };

    /// <summary>Serialises records for the cache.</summary>
    /// <typeparam name="T">The record type.</typeparam>
    /// <param name="records">What the stage produced.</param>
    /// <returns>A JSON array.</returns>
    public static string Write<T>(IReadOnlyList<T> records) =>
        JsonSerializer.Serialize(records, Options);

    /// <summary>Reads records back out of a cache payload.</summary>
    /// <typeparam name="T">The record type.</typeparam>
    /// <param name="payload">A value from <see cref="Write"/>.</param>
    /// <returns>The records, or an empty list when the payload cannot be read.</returns>
    /// <remarks>
    /// An unreadable payload returns empty rather than throwing. The cache is an optimisation; a
    /// corrupt entry should cost a re-analysis, not the photograph.
    /// </remarks>
    public static List<T> Read<T>(string payload)
    {
        try
        {
            return JsonSerializer.Deserialize<List<T>>(payload, Options) ?? [];
        }
        catch (JsonException)
        {
            return [];
        }
    }
}
