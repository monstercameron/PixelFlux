using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Ai.Faces;

/// <summary>Progress from a face sweep.</summary>
/// <param name="Total">Photographs queued when the sweep started.</param>
/// <param name="Done">Photographs finished.</param>
/// <param name="Found">Faces written so far.</param>
/// <param name="Current">Name of the photograph being examined.</param>
public readonly record struct FaceSweepProgress(int Total, int Done, int Found, string? Current);

/// <summary>
/// Sweeps a library for faces and writes a crop of each one.
/// </summary>
/// <remarks>
/// <para>
/// Sequential and resumable, for the same reasons as the segmentation worker: ONNX Runtime is
/// already using several cores, and this runs on a machine somebody is working on. Each
/// photograph is committed as it finishes, so stopping the sweep keeps what it has done.
/// </para>
/// <para>
/// A photograph is marked swept whether or not it contained anyone. Without that marker, every
/// photograph of a door would be re-examined on every run for ever — "no rows" and "not looked
/// at yet" are the same thing in the faces table and have to be told apart somewhere else.
/// </para>
/// </remarks>
public sealed class FaceWorker
{
    /// <summary>
    /// Edge length of the square crop written for each face.
    /// </summary>
    /// <remarks>
    /// The faces page shows these at around 96 CSS pixels, so 256 covers a 2x display with room
    /// to spare and still costs about 8 KB each. Going larger would make the cache bigger than
    /// the thumbnails it sits beside, for pixels nothing displays.
    /// </remarks>
    private const int CropSize = 256;

    /// <summary>
    /// How much context to include around the detected box, as a multiple of its size.
    /// </summary>
    /// <remarks>
    /// The detector's box is tight on the features — it clips the forehead, the chin and both
    /// ears. That is right for recognition and wrong for looking at: a wall of cropped foreheads
    /// is genuinely hard to read. 1.6 restores the head and a little shoulder, which is roughly
    /// what a passport photograph frames.
    /// </remarks>
    private const double CropPadding = 1.6;

    private readonly PhotoStore _photos;
    private readonly FaceStore _faces;
    private readonly IFaceDetector _detector;
    private readonly IFaceRecognizer? _recognizer;
    private readonly string _cacheRoot;
    private readonly ILogger<FaceWorker> _log;

    /// <summary>Creates a worker.</summary>
    /// <param name="photos">The photo index.</param>
    /// <param name="faces">Where faces are written.</param>
    /// <param name="detector">The model that finds faces.</param>
    /// <param name="cacheRoot">Derivative cache directory; crops are written beneath it.</param>
    /// <param name="recognizer">
    /// Optional. When present, every face also gets a vector, which is what makes "find this
    /// person" possible. Absent, the sweep still finds and crops faces and the faces page works
    /// exactly as before — grouping is the part that goes missing, and it goes missing quietly.
    /// </param>
    /// <param name="logger">Optional logger.</param>
    public FaceWorker(
        PhotoStore photos,
        FaceStore faces,
        IFaceDetector detector,
        string cacheRoot,
        IFaceRecognizer? recognizer = null,
        ILogger<FaceWorker>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(photos);
        ArgumentNullException.ThrowIfNull(faces);
        ArgumentNullException.ThrowIfNull(detector);

        _photos = photos;
        _faces = faces;
        _detector = detector;
        _recognizer = recognizer is { IsAvailable: true } ? recognizer : null;
        _cacheRoot = cacheRoot;
        _log = logger ?? NullLogger<FaceWorker>.Instance;
    }

    /// <summary>Whether a detector is installed and this stage can run at all.</summary>
    public bool IsAvailable => _detector.IsAvailable;

    /// <summary>
    /// Identifies what a sweep of this library was capable of, for the resume marker.
    /// </summary>
    /// <remarks>
    /// Both models, because a photograph swept by the detector alone is not finished once a
    /// recognition model appears — its faces have no vectors and never will unless the sweep
    /// runs again.
    /// </remarks>
    public string SweepVersion =>
        _recognizer is null ? _detector.ModelVersion : $"{_detector.ModelVersion}+{_recognizer.ModelVersion}";

    /// <summary>Cache key for one face crop.</summary>
    /// <param name="contentHash">The photograph's content hash.</param>
    /// <param name="index">Face index within that photograph.</param>
    /// <returns>A store-relative key such as <c>face/3f/3f9c...-01.jpg</c>.</returns>
    /// <remarks>
    /// Keyed by content hash, like thumbnails and masks, so crops survive the index being deleted
    /// and rebuilt. JPEG rather than PNG: these are photographic content, and a PNG of a face is
    /// roughly six times the size for no visible gain.
    /// </remarks>
    public static string CropKey(string contentHash, int index)
        => $"face/{contentHash[..2]}/{contentHash}-{index:00}.jpg";

    /// <summary>Sweeps every photograph no face pass has looked at yet.</summary>
    /// <param name="limit">Most photographs to examine in this run.</param>
    /// <param name="progress">Optional progress sink.</param>
    /// <param name="cancellationToken">Stops the sweep. Finished photographs are kept.</param>
    /// <returns>How many photographs were examined and how many faces were found.</returns>
    public async Task<(int Examined, int Faces)> RunAsync(
        int limit = 1000,
        IProgress<FaceSweepProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!_detector.IsAvailable)
        {
            _log.LogInformation("No face model installed; nothing to do");
            return (0, 0);
        }

        // The marker names both models, so installing a recognition model later invalidates
        // every previous sweep and the library gets embedded on the next run. Keying on the
        // detector alone would leave a fully-swept library permanently without vectors, and the
        // only symptom would be that "find this person" silently found nobody.
        IReadOnlyList<long> pending = await _faces
            .PendingAsync(SweepVersion, limit, cancellationToken).ConfigureAwait(false);

        int examined = 0;
        int found = 0;

        foreach (long photoId in pending)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            PhotoRecord? photo = await _photos.GetAsync(photoId, cancellationToken).ConfigureAwait(false);
            if (photo is null)
            {
                continue;
            }

            progress?.Report(new FaceSweepProgress(pending.Count, examined, found, photo.FileName));

            try
            {
                found += await SweepOneAsync(photo, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                // One unreadable photograph must not stop the sweep. It is still marked swept,
                // so the next run moves past it instead of failing on it again — a corrupt file
                // does not become readable by being retried.
                _log.LogWarning(ex, "Could not examine {File} for faces", photo.FileName);
                await _faces.MarkSweptAsync(photoId, SweepVersion, cancellationToken)
                    .ConfigureAwait(false);
            }

            examined++;
        }

        progress?.Report(new FaceSweepProgress(pending.Count, examined, found, null));
        return (examined, found);
    }

    /// <summary>Finds and describes every face in one photograph, without recording anything.</summary>
    /// <param name="photo">The photograph.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The faces found, ready to be stored.</returns>
    /// <remarks>
    /// Split from the writing half so the analysis queue can cache it. Detection and recognition
    /// depend only on the image bytes and the models, which means the same photograph imported
    /// twice can reuse this result — and the crops it writes are keyed by content hash, so they
    /// are already on disk when it does.
    /// </remarks>
    public async Task<IReadOnlyList<PhotoFaceRecord>> ExamineAsync(
        PhotoRecord photo,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(photo);
        // Detect on the proxy. The graph resizes to 640x640 regardless, so a 45-megapixel decode
        // would buy nothing — but the crop is cut from whatever was detected on, so the proxy's
        // 2560px long edge is also what limits crop quality. At 256px square that is ample.
        string source = ResolveSource(photo);

        FaceResult result = await _detector.DetectAsync(source, cancellationToken).ConfigureAwait(false);

        var records = new List<PhotoFaceRecord>(result.Faces.Count);

        if (result.Faces.Count > 0)
        {
            // Decoded once, at half the proxy's size, and used for both the crops and the
            // recognition vectors.
            //
            // This file was being decoded twice: once inside the detector and again here at full
            // resolution. The second decode was the single most expensive thing in the face
            // stage — 429 ms a photograph against about 80 ms for detection itself.
            //
            // 1280 rather than a rounder number because the proxy's long edge is 2560, and a JPEG
            // decoder can only rescale by halves — asking for 1600 would silently get the full
            // 2560 and save nothing. A crop is 256 square with 1.6x padding, so this still has
            // pixels to spare for any face large enough to be worth a card.
            using Image<Rgb24> image = Compute.AnalysisImage.Load(source, 1280);

            for (int i = 0; i < result.Faces.Count; i++)
            {
                DetectedFace face = result.Faces[i];
                string key = CropKey(photo.ContentHash, i);
                string? written = null;

                try
                {
                    await WriteCropAsync(image, face, key, cancellationToken).ConfigureAwait(false);
                    written = key;
                }
                catch (Exception ex) when (ex is IOException or ImageProcessingException)
                {
                    // A missing crop is a cosmetic problem — the page falls back to the
                    // photograph's thumbnail — so it must not cost us the detection itself.
                    _log.LogWarning(ex, "Could not write face crop {Key}", key);
                }

                // Embedded from the source photograph and its landmarks, not from the crop
                // written above. The crop is padded, straightened, and resampled for a person to
                // look at; the model wants the face warped onto its own canonical landmark
                // positions. Feeding it the pretty version would align a second time on top of
                // the first and describe the resampling as much as the face.
                float[]? vector = null;

                try
                {
                    vector = _recognizer?.Embed(image, face);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    // A face that cannot be described is still a face worth showing. It simply
                    // does not take part in grouping, which the record already expresses by
                    // carrying a null vector.
                    _log.LogWarning(ex, "Could not describe a face in {File}", photo.FileName);
                }

                records.Add(new PhotoFaceRecord(
                    0,
                    photo.Id,
                    face.Confidence,
                    face.X,
                    face.Y,
                    face.Width,
                    face.Height,
                    face.AreaFraction,
                    face.RollDegrees,
                    PhotoFaceRecord.FormatLandmarks(face.Landmarks),
                    written,
                    result.ModelVersion,
                    vector,
                    vector is null ? null : _recognizer!.ModelVersion));
            }
        }

        return records;
    }

    /// <summary>Records faces against a photograph.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="records">Faces from <see cref="ExamineAsync"/>, or from the queue's cache.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many faces were stored.</returns>
    public async Task<int> RecordAsync(
        long photoId,
        IReadOnlyList<PhotoFaceRecord> records,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);

        await _faces.ReplaceAsync(photoId, records, cancellationToken).ConfigureAwait(false);
        await _faces.MarkSweptAsync(photoId, SweepVersion, cancellationToken).ConfigureAwait(false);

        // A photograph with people in it becomes findable by typing "person", without anyone
        // having to know there is a separate faces page. Ai-sourced, so a later run replaces it
        // and nothing a person typed is touched.
        if (records.Count > 0)
        {
            await _photos.AddTagsAsync(
                photoId,
                [new PhotoTag("person", 1.0, MetadataSource.Ai)],
                cancellationToken).ConfigureAwait(false);
        }

        return records.Count;
    }

    private async Task<int> SweepOneAsync(PhotoRecord photo, CancellationToken cancellationToken)
    {
        IReadOnlyList<PhotoFaceRecord> records =
            await ExamineAsync(photo, cancellationToken).ConfigureAwait(false);
        return await RecordAsync(photo.Id, records, cancellationToken).ConfigureAwait(false);
    }

    private string ResolveSource(PhotoRecord photo)
    {
        if (photo.ProxyKey is { } proxy)
        {
            string path = Path.Combine(_cacheRoot, proxy.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(path))
            {
                return path;
            }
        }

        return photo.OriginalPath;
    }

    /// <summary>
    /// Cuts one square, straightened crop out of a photograph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Rotate first, then crop.</b> The head is straightened by rotating the whole image about
    /// the face's centre before the square is taken, so the crop's edges stay parallel to the
    /// screen. Cropping first and rotating the little square instead would swing the corners in
    /// and leave four triangles of blank canvas.
    /// </para>
    /// <para>
    /// <b>Square, always.</b> A grid of faces only reads as a grid if every cell is the same
    /// shape. The square is taken from the longer side of the padded box and clamped to the
    /// image, so a face at the very edge of a frame yields a smaller crop rather than one padded
    /// with black.
    /// </para>
    /// <para>
    /// Small tilts are left alone. Rotation resamples the whole image, which is the most
    /// expensive thing in this method, and below a few degrees nobody can see the difference.
    /// </para>
    /// </remarks>
    private async Task WriteCropAsync(
        Image<Rgb24> image,
        DetectedFace face,
        string key,
        CancellationToken cancellationToken)
    {
        string path = Path.Combine(_cacheRoot, key.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);

        double centreX = (face.X + (face.Width / 2)) * image.Width;
        double centreY = (face.Y + (face.Height / 2)) * image.Height;
        double side = Math.Max(face.Width * image.Width, face.Height * image.Height) * CropPadding;

        using Image<Rgb24> work = image.Clone();

        if (Math.Abs(face.RollDegrees) > 3)
        {
            // Rotating about the image centre moves the face; the offset below puts it back.
            work.Mutate(ctx => ctx.Rotate((float)-face.RollDegrees));

            double radians = -face.RollDegrees * Math.PI / 180;
            double cos = Math.Cos(radians);
            double sin = Math.Sin(radians);

            double dx = centreX - (image.Width / 2.0);
            double dy = centreY - (image.Height / 2.0);

            centreX = (work.Width / 2.0) + ((dx * cos) - (dy * sin));
            centreY = (work.Height / 2.0) + ((dx * sin) + (dy * cos));
        }

        int half = (int)Math.Round(side / 2);
        int left = (int)Math.Round(centreX) - half;
        int top = (int)Math.Round(centreY) - half;
        int size = half * 2;

        // Clamp into the image. Shrinking the square is better than padding it: a smaller face
        // still looks like a face, and a black band down one side looks like a bug.
        size = Math.Min(size, Math.Min(work.Width, work.Height));
        left = Math.Clamp(left, 0, Math.Max(0, work.Width - size));
        top = Math.Clamp(top, 0, Math.Max(0, work.Height - size));

        if (size < 8)
        {
            return;
        }

        work.Mutate(ctx => ctx
            .Crop(new Rectangle(left, top, size, size))
            .Resize(CropSize, CropSize));

        string temp = $"{path}.{Guid.NewGuid():n}.tmp";

        try
        {
            await work.SaveAsync(temp, new JpegEncoder { Quality = 82 }, cancellationToken)
                .ConfigureAwait(false);
            File.Move(temp, path, overwrite: true);
        }
        catch
        {
            try
            {
                File.Delete(temp);
            }
            catch (IOException)
            {
                // A stray .tmp is never read.
            }

            throw;
        }
    }
}
