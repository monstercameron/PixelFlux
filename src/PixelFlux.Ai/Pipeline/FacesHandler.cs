using PixelFlux.Ai.Faces;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using PixelFlux.Core.Pipeline;

namespace PixelFlux.Ai.Pipeline;

/// <summary>
/// The stage that finds faces and measures each one.
/// </summary>
/// <remarks>
/// <para>
/// The model version is both models — detector and recogniser — because a photograph swept by the
/// detector alone is not finished once a recognition model appears: its faces carry no vectors and
/// never will unless the sweep runs again. Folding both into one string means installing the
/// recogniser later makes every photograph outstanding again automatically, which is exactly what
/// should happen and needs no code to make it happen.
/// </para>
/// <para>
/// Nothing here leaves the machine. Faces are the most sensitive thing in a photo library, and the
/// detector, the recogniser, the crops and the vectors are all local — as is the cache, which is a
/// table in the library's own database.
/// </para>
/// </remarks>
public sealed class FacesHandler : IStageHandler
{
    private readonly FaceWorker _worker;
    private readonly PhotoStore _photos;

    /// <summary>Creates the handler.</summary>
    /// <param name="worker">The face worker, used one photograph at a time.</param>
    /// <param name="photos">The photo index.</param>
    public FacesHandler(FaceWorker worker, PhotoStore photos)
    {
        ArgumentNullException.ThrowIfNull(worker);
        ArgumentNullException.ThrowIfNull(photos);

        _worker = worker;
        _photos = photos;
    }

    /// <inheritdoc/>
    public PipelineStage Stage => PipelineStage.Faces;

    /// <inheritdoc/>
    public string? ModelVersion => _worker.IsAvailable ? _worker.SweepVersion : null;

    /// <inheritdoc/>
    public async Task ApplyAsync(long photoId, string payload, CancellationToken cancellationToken)
    {
        List<PhotoFaceRecord> records = StagePayload.Read<PhotoFaceRecord>(payload);

        await _worker.RecordAsync(
            photoId,
            [.. records.Select(record => record with { PhotoId = photoId })],
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

        IReadOnlyList<PhotoFaceRecord> records =
            await _worker.ExamineAsync(photo, cancellationToken).ConfigureAwait(false);

        await _worker.RecordAsync(photoId, records, cancellationToken).ConfigureAwait(false);
        return StagePayload.Write(records);
    }
}
