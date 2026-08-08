using PixelFlux.Ai.Compute;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Ai.Segmentation;

/// <summary>
/// Instance segmentation with a YOLO-family ONNX model.
/// </summary>
/// <remarks>
/// <para>
/// <b>Licensing.</b> The model this is designed against — <c>yolo11n-seg.onnx</c> from
/// Ultralytics — is AGPL-3.0. That is workable for local and personal use and is a genuine
/// constraint on redistributing PixelFlux. The model is therefore a loose file the user
/// supplies, not an embedded resource, and everything here is written against
/// <see cref="ISegmenter"/> so a differently-licensed model can replace it without touching
/// storage or the interface.
/// </para>
/// <para>
/// <b>Output shape.</b> A YOLO segmentation graph emits two tensors:
/// </para>
/// <list type="bullet">
/// <item><description>
/// <c>output0</c>, <c>[1, 4 + classes + 32, anchors]</c> — for every candidate box: centre x,
/// centre y, width, height, one score per class, and 32 mask coefficients.
/// </description></item>
/// <item><description>
/// <c>output1</c>, <c>[1, 32, 160, 160]</c> — 32 mask prototypes shared by the whole image.
/// </description></item>
/// </list>
/// <para>
/// A candidate's mask is the weighted sum of those prototypes using its own 32 coefficients,
/// squashed through a sigmoid. That is the whole trick, and it is why the network can emit a
/// per-instance mask without predicting a full-resolution one per object.
/// </para>
/// </remarks>
public sealed class YoloSegmenter : ISegmenter, IDisposable
{
    /// <summary>Network input edge in pixels. YOLO segmentation models are trained at 640.</summary>
    private const int InputSize = 640;

    /// <summary>Prototype mask edge, always a quarter of the input.</summary>
    private const int ProtoSize = 160;

    /// <summary>Number of mask prototypes. Fixed by the architecture.</summary>
    private const int ProtoCount = 32;

    /// <summary>
    /// Minimum score for a detection to be kept.
    /// </summary>
    /// <remarks>
    /// 0.35 rather than the more common 0.25. This drives a user-visible overlay and a search
    /// facet, and a wrong label on someone's photograph is far more annoying than a missed one —
    /// people notice "toaster" on a picture of their dog and do not notice the absence of a
    /// third chair. Recall is cheap to recover by re-running with a lower threshold; a lost
    /// trust in the labels is not.
    /// </remarks>
    private const float ConfidenceThreshold = 0.35f;

    /// <summary>Boxes overlapping more than this are treated as the same object.</summary>
    private const float NmsIouThreshold = 0.45f;

    /// <summary>Mask probability above which a pixel belongs to the segment.</summary>
    private const float MaskThreshold = 0.5f;

    /// <summary>Most segments to keep per photograph.</summary>
    /// <remarks>
    /// A crowd scene can produce sixty people. Storing and overlaying all of them helps nobody:
    /// the overlay becomes unreadable and the tag list becomes "person" twenty times. The list
    /// is sorted by prominence before truncation, so what survives is what fills the frame.
    /// </remarks>
    private const int MaxSegments = 20;

    /// <summary>The 80 COCO classes these models are trained on, in index order.</summary>
    private static readonly string[] CocoLabels =
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ];

    // Chosen once for the whole application rather than per model, so "run on the neural
    // processor" is one setting and not four places to keep in step. Null means nobody supplied
    // one, which is the processor with the same options this always used.
    private readonly ComputeBackend? _compute;

    private readonly InferenceSession? _session;
    private readonly ILogger<YoloSegmenter> _log;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Creates a segmenter over a model file.</summary>
    /// <param name="modelPath">Path to the ONNX model, or null to run unavailable.</param>
    /// <param name="logger">Optional logger.</param>
    /// <param name="compute">Where models run. Null means the processor, exactly as before.</param>
    public YoloSegmenter(
        string? modelPath,
        ILogger<YoloSegmenter>? logger = null,
        ComputeBackend? compute = null)
    {
        _compute = compute;
        _log = logger ?? NullLogger<YoloSegmenter>.Instance;
        ModelVersion = modelPath is null ? "none" : Path.GetFileNameWithoutExtension(modelPath);

        if (modelPath is null || !File.Exists(modelPath))
        {
            _log.LogInformation("No segmentation model at {Path}; segmentation is off", modelPath);
            return;
        }

        try
        {
            // One thread per pair of cores. Segmentation runs in the background while the
            // machine is otherwise idle, but "idle" is a scheduler decision — if the user comes
            // back mid-batch the app must not be the reason their laptop is unusable. The backend
            // decides whether any of it lands on an accelerator instead.
            using SessionOptions options =
                (_compute ?? new ComputeBackend()).CreateSessionOptions(
                    Environment.ProcessorCount / 2,
                    Path.GetFileNameWithoutExtension(modelPath));

            _session = new InferenceSession(modelPath, options);
            _log.LogInformation("Segmentation model loaded: {Model}", ModelVersion);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or FileNotFoundException or DllNotFoundException)
        {
            // A corrupt download or a missing native runtime must not stop the app from
            // starting; the library is still fully browsable without segmentation.
            _log.LogWarning(ex, "Could not load segmentation model {Path}", modelPath);
            _session = null;
        }
    }

    /// <inheritdoc />
    public bool IsAvailable => _session is not null;

    /// <inheritdoc />
    public string ModelVersion { get; }

    /// <inheritdoc />
    public async Task<SegmentationResult> SegmentAsync(
        string imagePath,
        CancellationToken cancellationToken = default)
    {
        if (_session is null)
        {
            return SegmentationResult.Empty();
        }

        // One inference at a time. ONNX Runtime sessions are thread-safe, but running several
        // 640x640 passes concurrently on a fanless laptop competes with the ingestion decode
        // pool for the same cores and finishes no sooner.
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await Task.Run(() => Segment(imagePath, cancellationToken), cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    private SegmentationResult Segment(string imagePath, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();

        using Image<Rgb24> source = AnalysisImage.Load(imagePath, InputSize);
        int originalWidth = source.Width;
        int originalHeight = source.Height;

        // Letterbox: scale to fit 640x640 preserving aspect, pad the remainder grey. Stretching
        // instead would distort every object and cost real accuracy on portrait photographs.
        double scale = Math.Min((double)InputSize / originalWidth, (double)InputSize / originalHeight);
        int scaledWidth = (int)Math.Round(originalWidth * scale);
        int scaledHeight = (int)Math.Round(originalHeight * scale);
        int padX = (InputSize - scaledWidth) / 2;
        int padY = (InputSize - scaledHeight) / 2;

        using Image<Rgb24> letterboxed = source.Clone(ctx => ctx
            .Resize(scaledWidth, scaledHeight)
            .Pad(InputSize, InputSize, Color.FromRgb(114, 114, 114)));

        var input = new DenseTensor<float>([1, 3, InputSize, InputSize]);
        letterboxed.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < InputSize; y++)
            {
                Span<Rgb24> row = rows.GetRowSpan(y);
                for (int x = 0; x < InputSize; x++)
                {
                    // CHW layout, RGB order, scaled to 0-1 — what the graph expects.
                    input[0, 0, y, x] = row[x].R / 255f;
                    input[0, 1, y, x] = row[x].G / 255f;
                    input[0, 2, y, x] = row[x].B / 255f;
                }
            }
        });

        cancellationToken.ThrowIfCancellationRequested();

        string inputName = _session!.InputMetadata.Keys.First();
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs =
            _session.Run([NamedOnnxValue.CreateFromTensor(inputName, input)]);

        DisposableNamedOnnxValue[] ordered = outputs.ToArray();

        // output0 is the wide one (predictions), output1 the 4-D one (prototypes). Matching by
        // rank rather than by name because exporters disagree about the names.
        Tensor<float> predictions = ordered.First(o => o.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>();
        Tensor<float> prototypes = ordered.First(o => o.AsTensor<float>().Dimensions.Length == 4).AsTensor<float>();

        int channels = predictions.Dimensions[1];
        int anchors = predictions.Dimensions[2];
        int classCount = channels - 4 - ProtoCount;

        var candidates = new List<Candidate>();

        for (int a = 0; a < anchors; a++)
        {
            int bestClass = -1;
            float bestScore = ConfidenceThreshold;

            for (int c = 0; c < classCount; c++)
            {
                float score = predictions[0, 4 + c, a];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            if (bestClass < 0)
            {
                continue;
            }

            float cx = predictions[0, 0, a];
            float cy = predictions[0, 1, a];
            float w = predictions[0, 2, a];
            float h = predictions[0, 3, a];

            var coefficients = new float[ProtoCount];
            for (int k = 0; k < ProtoCount; k++)
            {
                coefficients[k] = predictions[0, 4 + classCount + k, a];
            }

            candidates.Add(new Candidate(
                bestClass, bestScore,
                cx - (w / 2), cy - (h / 2), w, h,
                coefficients));
        }

        cancellationToken.ThrowIfCancellationRequested();

        List<Candidate> kept = NonMaximumSuppression(candidates);
        var segments = new List<PhotoSegment>(kept.Count);

        foreach (Candidate candidate in kept)
        {
            PhotoSegment? segment = BuildSegment(
                candidate, prototypes, classCount,
                scale, padX, padY, originalWidth, originalHeight);

            if (segment is not null)
            {
                segments.Add(segment);
            }
        }

        // Most prominent first: that ordering is what the overlay draws, what the tag list
        // shows, and what truncation keeps.
        segments.Sort((x, y) => y.Prominence.CompareTo(x.Prominence));
        if (segments.Count > MaxSegments)
        {
            segments.RemoveRange(MaxSegments, segments.Count - MaxSegments);
        }

        watch.Stop();
        return new SegmentationResult(segments, ModelVersion, watch.ElapsedMilliseconds);
    }

    /// <summary>
    /// Turns one surviving candidate into a segment with a real mask.
    /// </summary>
    /// <remarks>
    /// The mask is the sigmoid of the dot product between the candidate's 32 coefficients and
    /// the 32 shared prototypes, evaluated only inside the candidate's own box. Evaluating the
    /// whole 160x160 grid and cropping afterwards would be simpler and roughly ten times the
    /// arithmetic for a small object.
    /// </remarks>
    private static PhotoSegment? BuildSegment(
        Candidate candidate,
        Tensor<float> prototypes,
        int classCount,
        double scale,
        int padX,
        int padY,
        int originalWidth,
        int originalHeight)
    {
        // Letterboxed pixel space -> original image pixel space.
        double left = (candidate.X - padX) / scale;
        double top = (candidate.Y - padY) / scale;
        double width = candidate.Width / scale;
        double height = candidate.Height / scale;

        // Clip to the frame. Boxes routinely overhang the edge, and a negative origin would
        // produce a mask rectangle that cannot be indexed.
        double clippedLeft = Math.Clamp(left, 0, originalWidth);
        double clippedTop = Math.Clamp(top, 0, originalHeight);
        double clippedRight = Math.Clamp(left + width, 0, originalWidth);
        double clippedBottom = Math.Clamp(top + height, 0, originalHeight);

        if (clippedRight - clippedLeft < 2 || clippedBottom - clippedTop < 2)
        {
            return null;
        }

        // Mask raster size. Capped so a full-frame segment on a 45-megapixel photograph does not
        // become a 45-megabyte byte array; 256 on the long edge is ample for an overlay that is
        // drawn semi-transparent over the picture.
        double boxWidth = clippedRight - clippedLeft;
        double boxHeight = clippedBottom - clippedTop;
        double maskScale = Math.Min(1.0, 256.0 / Math.Max(boxWidth, boxHeight));
        int maskWidth = Math.Max(1, (int)Math.Round(boxWidth * maskScale));
        int maskHeight = Math.Max(1, (int)Math.Round(boxHeight * maskScale));

        // Resolve the prototype combination ONCE onto the 160x160 grid, then interpolate that
        // small float grid per output pixel.
        //
        // The obvious implementation — dot the 32 coefficients with the 32 prototypes at every
        // output pixel — costs maskWidth * maskHeight * 32 multiply-adds, and bilinear
        // interpolation quadruples that. Measured, it took inference from 493 ms to 2300 ms per
        // photograph: the mask, not the network, became the bottleneck.
        //
        // Collapsing first costs 160*160*32 regardless of mask size, and every output pixel is
        // then four array reads. For a 256-pixel mask that is roughly a tenth of the work, and it
        // gets relatively cheaper the larger the segment.
        float[] grid = CollapsePrototypes(prototypes, candidate.Coefficients);

        var mask = new byte[maskWidth * maskHeight];
        int covered = 0;

        for (int my = 0; my < maskHeight; my++)
        {
            // Mask pixel -> original image -> letterboxed -> prototype grid.
            double imageY = clippedTop + ((my + 0.5) / maskHeight * boxHeight);
            double protoY = ((((imageY * scale) + padY) / InputSize) * ProtoSize) - 0.5;

            for (int mx = 0; mx < maskWidth; mx++)
            {
                double imageX = clippedLeft + ((mx + 0.5) / maskWidth * boxWidth);
                double protoX = ((((imageX * scale) + padX) / InputSize) * ProtoSize) - 0.5;

                // Bilinear, not nearest. The prototype grid is only 160x160 for a 640-pixel
                // input, so nearest sampling quantises every outline to that grid and produces
                // visibly stair-stepped edges — which matters more here than it usually would,
                // because the mask is drawn over the photograph at full size and the jaggedness
                // is the first thing anyone notices.
                if (Sigmoid(SampleGrid(grid, protoX, protoY)) > MaskThreshold)
                {
                    mask[(my * maskWidth) + mx] = 255;
                    covered++;
                }
            }
        }

        if (covered == 0)
        {
            // A box whose mask is empty is a detection the mask head disagreed with. Dropping it
            // is better than drawing an outline around nothing.
            return null;
        }

        double frameArea = (double)originalWidth * originalHeight;
        double maskArea = covered / (double)(maskWidth * maskHeight) * boxWidth * boxHeight;

        return new PhotoSegment(
            candidate.ClassId < CocoLabels.Length ? CocoLabels[candidate.ClassId] : $"class {candidate.ClassId}",
            candidate.Score,
            clippedLeft / originalWidth,
            clippedTop / originalHeight,
            boxWidth / originalWidth,
            boxHeight / originalHeight,
            maskArea / frameArea,
            mask,
            maskWidth,
            maskHeight);
    }

    /// <summary>
    /// Combines the 32 shared prototypes into one 160x160 grid using a candidate's coefficients.
    /// </summary>
    /// <remarks>
    /// Reads the tensor through its backing buffer rather than the four-index accessor.
    /// <c>prototypes[0, k, y, x]</c> recomputes a strided offset and bounds-checks on every one
    /// of the 819,200 reads here, and that indexer was the single largest cost in mask building
    /// before this was flattened.
    /// </remarks>
    private static float[] CollapsePrototypes(Tensor<float> prototypes, float[] coefficients)
    {
        const int plane = ProtoSize * ProtoSize;
        var grid = new float[plane];

        ReadOnlySpan<float> source = prototypes is DenseTensor<float> dense
            ? dense.Buffer.Span
            : prototypes.ToArray();

        for (int k = 0; k < ProtoCount; k++)
        {
            float coefficient = coefficients[k];
            if (coefficient == 0)
            {
                continue;
            }

            ReadOnlySpan<float> prototype = source.Slice(k * plane, plane);
            for (int i = 0; i < plane; i++)
            {
                grid[i] += coefficient * prototype[i];
            }
        }

        return grid;
    }

    /// <summary>Bilinearly samples the collapsed grid at a fractional position.</summary>
    /// <remarks>Four array reads. The expensive part already happened in <see cref="CollapsePrototypes"/>.</remarks>
    private static float SampleGrid(float[] grid, double x, double y)
    {
        int x0 = (int)Math.Floor(x);
        int y0 = (int)Math.Floor(y);
        float fx = (float)(x - x0);
        float fy = (float)(y - y0);

        int cx0 = Math.Clamp(x0, 0, ProtoSize - 1);
        int cy0 = Math.Clamp(y0, 0, ProtoSize - 1);
        int cx1 = Math.Clamp(x0 + 1, 0, ProtoSize - 1);
        int cy1 = Math.Clamp(y0 + 1, 0, ProtoSize - 1);

        float topLeft = grid[(cy0 * ProtoSize) + cx0];
        float topRight = grid[(cy0 * ProtoSize) + cx1];
        float bottomLeft = grid[(cy1 * ProtoSize) + cx0];
        float bottomRight = grid[(cy1 * ProtoSize) + cx1];

        float top = topLeft + ((topRight - topLeft) * fx);
        float bottom = bottomLeft + ((bottomRight - bottomLeft) * fx);
        return top + ((bottom - top) * fy);
    }

    private static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));

    /// <summary>
    /// Greedy non-maximum suppression, per class.
    /// </summary>
    /// <remarks>
    /// Per class rather than globally: a person holding a dog produces two heavily overlapping
    /// boxes, and class-agnostic suppression would throw one of them away. Suppression is only
    /// ever meant to collapse duplicate detections of the <em>same</em> thing.
    /// </remarks>
    private static List<Candidate> NonMaximumSuppression(List<Candidate> candidates)
    {
        var kept = new List<Candidate>();

        foreach (IGrouping<int, Candidate> byClass in candidates.GroupBy(c => c.ClassId))
        {
            Candidate[] ordered = byClass.OrderByDescending(c => c.Score).ToArray();
            var suppressed = new bool[ordered.Length];

            for (int i = 0; i < ordered.Length; i++)
            {
                if (suppressed[i])
                {
                    continue;
                }

                kept.Add(ordered[i]);

                for (int j = i + 1; j < ordered.Length; j++)
                {
                    if (!suppressed[j] && IntersectionOverUnion(ordered[i], ordered[j]) > NmsIouThreshold)
                    {
                        suppressed[j] = true;
                    }
                }
            }
        }

        return kept;
    }

    private static float IntersectionOverUnion(Candidate a, Candidate b)
    {
        float left = Math.Max(a.X, b.X);
        float top = Math.Max(a.Y, b.Y);
        float right = Math.Min(a.X + a.Width, b.X + b.Width);
        float bottom = Math.Min(a.Y + a.Height, b.Y + b.Height);

        float overlap = Math.Max(0, right - left) * Math.Max(0, bottom - top);
        float union = (a.Width * a.Height) + (b.Width * b.Height) - overlap;

        return union <= 0 ? 0 : overlap / union;
    }

    private readonly record struct Candidate(
        int ClassId,
        float Score,
        float X,
        float Y,
        float Width,
        float Height,
        float[] Coefficients);

    /// <summary>Releases the inference session.</summary>
    public void Dispose()
    {
        _session?.Dispose();
        _gate.Dispose();
    }
}
