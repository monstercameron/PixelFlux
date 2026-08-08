using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace PixelFlux.Ai.Segmentation;

/// <summary>Progress from an analysis run.</summary>
/// <param name="Total">Photographs queued when the run started.</param>
/// <param name="Done">Photographs finished.</param>
/// <param name="Found">Segments written so far.</param>
/// <param name="Current">Name of the photograph being analysed.</param>
public readonly record struct AnalysisProgress(int Total, int Done, int Found, string? Current);

/// <summary>
/// Works through the photographs waiting to be analysed, one at a time.
/// </summary>
/// <remarks>
/// <para>
/// Deliberately sequential. Segmentation is already multi-threaded inside ONNX Runtime, so
/// running several photographs at once contends for the same cores and finishes no sooner — and
/// this is designed to run in the background on a machine the user is also using, where being
/// slower and less intrusive is the better trade.
/// </para>
/// <para>
/// Analysis is resumable and idempotent. Each photograph is committed as it completes, so
/// interrupting the run keeps the work already done, and re-running only picks up what is still
/// pending. On a library of any size that matters more than raw throughput: nobody gets to leave
/// a laptop untouched for eight hours.
/// </para>
/// </remarks>
public sealed class SegmentationWorker
{
    private readonly PhotoStore _photos;
    private readonly SegmentStore _segments;
    private readonly ISegmenter _segmenter;
    private readonly string _cacheRoot;
    private readonly ILogger<SegmentationWorker> _log;

    /// <summary>Creates a worker.</summary>
    /// <param name="photos">The photo index.</param>
    /// <param name="segments">Where results are written.</param>
    /// <param name="segmenter">The model.</param>
    /// <param name="cacheRoot">Derivative cache directory; masks are written beneath it.</param>
    /// <param name="logger">Optional logger.</param>
    public SegmentationWorker(
        PhotoStore photos,
        SegmentStore segments,
        ISegmenter segmenter,
        string cacheRoot,
        ILogger<SegmentationWorker>? logger = null)
    {
        _photos = photos;
        _segments = segments;
        _segmenter = segmenter;
        _cacheRoot = cacheRoot;
        _log = logger ?? NullLogger<SegmentationWorker>.Instance;
    }

    /// <summary>Whether a segmenter is installed and this stage can run at all.</summary>
    public bool IsAvailable => _segmenter.IsAvailable;

    /// <summary>The segmenter version results are stamped with.</summary>
    public string ModelVersion => _segmenter.ModelVersion;

    /// <summary>Converts an HSL hue to RGB at the overlay's fixed saturation and lightness.</summary>
    /// <param name="hue">Hue in degrees, 0-359.</param>
    /// <returns>The colour to bake into the mask.</returns>
    /// <remarks>
    /// Saturation and lightness match what the label chips use, so a chip and its mask are
    /// visibly the same colour — which is the whole mechanism tying one to the other.
    /// </remarks>
    private static (byte R, byte G, byte B) HueToRgb(int hue)
    {
        const double saturation = 0.85;
        const double lightness = 0.58;

        double c = (1 - Math.Abs((2 * lightness) - 1)) * saturation;
        double h = hue / 60.0;
        double x = c * (1 - Math.Abs((h % 2) - 1));
        double m = lightness - (c / 2);

        (double r, double g, double b) = (int)h switch
        {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };

        return ((byte)Math.Round((r + m) * 255),
                (byte)Math.Round((g + m) * 255),
                (byte)Math.Round((b + m) * 255));
    }

    /// <summary>Cache key for one segment's mask.</summary>
    /// <param name="contentHash">The photograph's content hash.</param>
    /// <param name="index">Segment index within that photograph.</param>
    /// <returns>A store-relative key such as <c>mask/3f/3f9c...-02.png</c>.</returns>
    /// <remarks>
    /// Keyed by content hash rather than row id so masks survive the index being deleted and
    /// rebuilt — the same reason thumbnails are.
    /// </remarks>
    public static string MaskKey(string contentHash, int index)
        => $"mask/{contentHash[..2]}/{contentHash}-{index:00}.png";

    /// <summary>
    /// Analyses every photograph still waiting.
    /// </summary>
    /// <param name="limit">Most photographs to process in this run.</param>
    /// <param name="progress">Optional progress sink.</param>
    /// <param name="cancellationToken">Stops the run. Completed photographs are kept.</param>
    /// <returns>How many photographs were analysed and how many segments were found.</returns>
    public async Task<(int Analysed, int Segments)> RunAsync(
        int limit = 1000,
        IProgress<AnalysisProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!_segmenter.IsAvailable)
        {
            _log.LogInformation("No segmentation model installed; nothing to do");
            return (0, 0);
        }

        IReadOnlyList<PhotoRecord> pending = await _photos.QueryAsync(
            new PhotoQuery { State = ProcessingState.Pending, Limit = limit },
            cancellationToken).ConfigureAwait(false);

        int analysed = 0;
        int found = 0;

        foreach (PhotoRecord photo in pending)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            progress?.Report(new AnalysisProgress(pending.Count, analysed, found, photo.FileName));

            try
            {
                found += await AnalyseOneAsync(photo, cancellationToken).ConfigureAwait(false);
                analysed++;
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                // One unanalysable photograph must not stop the run. Marked Failed so the next
                // pass skips it rather than retrying the same failure for ever.
                _log.LogWarning(ex, "Could not analyse {File}", photo.FileName);
                await _photos.SetStateAsync(
                    photo.Id, ProcessingState.Failed, ex.Message, _segmenter.ModelVersion, cancellationToken)
                    .ConfigureAwait(false);
                analysed++;
            }
        }

        progress?.Report(new AnalysisProgress(pending.Count, analysed, found, null));
        return (analysed, found);
    }

    /// <summary>Finds the objects in one photograph and writes their masks, recording nothing.</summary>
    /// <param name="photo">The photograph.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The segments found, ready to be stored.</returns>
    /// <remarks>
    /// Split from the writing half so the analysis queue can cache it. The masks this writes are
    /// keyed by content hash, so a cached result always has its overlay images already on disk —
    /// which is what makes reusing the result safe rather than merely fast.
    /// </remarks>
    public async Task<IReadOnlyList<PhotoSegmentRecord>> ExamineAsync(
        PhotoRecord photo,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(photo);
        // Segment the proxy, not the original.
        //
        // The network resizes its input to 640x640 regardless, so feeding it a 45-megapixel
        // original buys nothing and costs a very expensive decode. The proxy is already on disk,
        // already upright, and already the right order of magnitude.
        string source = photo.ProxyKey is { } proxy
            ? Path.Combine(_cacheRoot, proxy.Replace('/', Path.DirectorySeparatorChar))
            : photo.OriginalPath;

        if (!File.Exists(source))
        {
            source = photo.OriginalPath;
        }

        SegmentationResult result = await _segmenter.SegmentAsync(source, cancellationToken)
            .ConfigureAwait(false);

        var records = new List<PhotoSegmentRecord>(result.Segments.Count);

        for (int i = 0; i < result.Segments.Count; i++)
        {
            PhotoSegment segment = result.Segments[i];
            string key = MaskKey(photo.ContentHash, i);

            await WriteMaskAsync(key, segment, cancellationToken).ConfigureAwait(false);

            records.Add(new PhotoSegmentRecord(
                0, photo.Id, segment.Label, segment.Confidence,
                segment.X, segment.Y, segment.Width, segment.Height,
                segment.AreaFraction, segment.Prominence, key, result.ModelVersion));
        }

        return records;
    }

    /// <summary>Records segments against a photograph.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="records">Segments from <see cref="ExamineAsync"/>, or from the queue's cache.</param>
    /// <param name="modelVersion">The segmenter version the records came from.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many segments were stored.</returns>
    public async Task<int> RecordAsync(
        long photoId,
        IReadOnlyList<PhotoSegmentRecord> records,
        string modelVersion,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);

        await _segments.ReplaceAsync(photoId, records, modelVersion, cancellationToken)
            .ConfigureAwait(false);

        // Detected labels also become searchable tags, so that typing "dog" finds the photo
        // without the user having to know there is a separate object filter. Marked Ai-sourced,
        // so a later model run replaces them and nothing a person typed is disturbed.
        PhotoTag[] tags = records
            .Select(r => r.Label)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(label => new PhotoTag(label.ToLowerInvariant(), 1.0, MetadataSource.Ai))
            .ToArray();

        await _photos.AddTagsAsync(photoId, tags, cancellationToken).ConfigureAwait(false);
        await _photos.SetStateAsync(
            photoId, ProcessingState.Complete, null, modelVersion, cancellationToken)
            .ConfigureAwait(false);

        return records.Count;
    }

    private async Task<int> AnalyseOneAsync(PhotoRecord photo, CancellationToken cancellationToken)
    {
        IReadOnlyList<PhotoSegmentRecord> records =
            await ExamineAsync(photo, cancellationToken).ConfigureAwait(false);
        return await RecordAsync(photo.Id, records, _segmenter.ModelVersion, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Writes one mask as a finished, already-coloured RGBA overlay in the derivative cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>A picture, not a stencil.</b> This began as a greyscale stencil applied with CSS
    /// <c>mask-image</c>, and it went wrong twice. First the PNG had no alpha channel, so the
    /// mask was fully opaque everywhere and tinted whole bounding boxes. Then, with alpha fixed
    /// and rendering correctly in a test harness, it still showed nothing in the application —
    /// because a CSS mask is a cross-origin subresource, and it has to satisfy both the page's
    /// content policy and the WebView's host-resource rules, neither of which an
    /// <c>&lt;img&gt;</c> from the same mapped host has to.
    /// </para>
    /// <para>
    /// Baking the label's colour in makes each mask an ordinary image that the overlay drops in
    /// with an <c>&lt;img&gt;</c> tag — exactly what the thumbnails already do, and therefore
    /// exactly as reliable. The cost is that changing the palette means re-analysing, which is
    /// a fair trade for a feature that otherwise silently renders nothing.
    /// </para>
    /// </remarks>
    private async Task WriteMaskAsync(string key, PhotoSegment segment, CancellationToken cancellationToken)
    {
        string path = Path.Combine(_cacheRoot, key.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);

        // The label's own colour is baked in, so the file is a finished overlay rather than a
        // stencil. See the remarks above for why this is not left to CSS.
        (byte r, byte g, byte b) = HueToRgb(PhotoSegmentRecord.HueFor(segment.Label));

        using var image = new Image<Rgba32>(segment.MaskWidth, segment.MaskHeight);

        image.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < segment.MaskHeight; y++)
            {
                Span<Rgba32> row = rows.GetRowSpan(y);
                int offset = y * segment.MaskWidth;
                for (int x = 0; x < segment.MaskWidth; x++)
                {
                    row[x] = new Rgba32(r, g, b, segment.Mask[offset + x]);
                }
            }
        });

        string temp = $"{path}.{Guid.NewGuid():n}.tmp";

        try
        {
            await image.SaveAsync(temp, new PngEncoder
            {
                ColorType = PngColorType.RgbWithAlpha,
                BitDepth = PngBitDepth.Bit8,
                CompressionLevel = PngCompressionLevel.BestCompression,
            }, cancellationToken).ConfigureAwait(false);

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
