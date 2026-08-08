using PixelFlux.Ai.Compute;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Ai.Faces;

/// <summary>One face found in a photograph.</summary>
/// <param name="Confidence">Detector confidence, 0-1.</param>
/// <param name="X">Left edge of the face box as a fraction of image width.</param>
/// <param name="Y">Top edge as a fraction of image height.</param>
/// <param name="Width">Box width as a fraction of image width.</param>
/// <param name="Height">Box height as a fraction of image height.</param>
/// <param name="Landmarks">
/// Five points as fractions of the image: right eye, left eye, nose, right mouth corner, left
/// mouth corner, in that order.
/// </param>
/// <remarks>
/// The landmarks are kept even though nothing draws them yet. They are what a recognition model
/// needs in order to align a face before embedding it, and re-running detection over a whole
/// library later just to recover five numbers per face would be a waste.
/// </remarks>
public sealed record DetectedFace(
    double Confidence,
    double X,
    double Y,
    double Width,
    double Height,
    IReadOnlyList<(double X, double Y)> Landmarks)
{
    /// <summary>Share of the frame this face occupies.</summary>
    public double AreaFraction => Width * Height;

    /// <summary>
    /// Angle of the line between the eyes, in degrees.
    /// </summary>
    /// <remarks>
    /// Used to straighten a face when cropping it for the faces page. A gallery of heads at
    /// arbitrary angles is noticeably harder to scan than one where they all sit upright.
    /// </remarks>
    public double RollDegrees
    {
        get
        {
            if (Landmarks.Count < 2)
            {
                return 0;
            }

            (double rx, double ry) = Landmarks[0];
            (double lx, double ly) = Landmarks[1];
            return Math.Atan2(ly - ry, lx - rx) * 180 / Math.PI;
        }
    }
}

/// <summary>Everything a face pass found in one photograph.</summary>
/// <param name="Faces">Faces, largest first.</param>
/// <param name="ModelVersion">Identifier of the model that produced them.</param>
/// <param name="ElapsedMs">Wall-clock inference time.</param>
public sealed record FaceResult(IReadOnlyList<DetectedFace> Faces, string ModelVersion, long ElapsedMs)
{
    /// <summary>An empty result, for when no model is installed or nothing was found.</summary>
    /// <param name="modelVersion">Model identifier, or a marker such as <c>none</c>.</param>
    /// <returns>A result with no faces.</returns>
    public static FaceResult Empty(string modelVersion = "none") => new([], modelVersion, 0);
}

/// <summary>Finds faces in a photograph.</summary>
/// <remarks>
/// An interface so the model can be replaced without touching storage or the UI. Implementations
/// must be safe to call concurrently.
/// </remarks>
public interface IFaceDetector
{
    /// <summary>Whether a usable model is installed.</summary>
    bool IsAvailable { get; }

    /// <summary>Identifier of the model in use.</summary>
    string ModelVersion { get; }

    /// <summary>Finds the faces in one image file.</summary>
    /// <param name="imagePath">Absolute path to the image.</param>
    /// <param name="cancellationToken">Cancels inference.</param>
    /// <returns>What was found, or an empty result when no model is installed.</returns>
    Task<FaceResult> DetectAsync(string imagePath, CancellationToken cancellationToken = default);
}

/// <summary>
/// Face detection with YuNet.
/// </summary>
/// <remarks>
/// <para>
/// <b>Licensing.</b> YuNet comes from the OpenCV Model Zoo under Apache 2.0 — a deliberate
/// contrast with the AGPL segmentation model. It is also 232 KB, which is small enough to ship
/// without asking anyone to download anything, and it runs in a few milliseconds. For a feature
/// that has to sweep an entire library, "small, fast, permissive" beat "slightly more accurate".
/// </para>
/// <para>
/// <b>Nothing leaves the machine.</b> Faces are the most sensitive thing in a photo library, and
/// this runs entirely locally like every other part of PixelFlux. No image, crop, or embedding
/// is ever sent anywhere.
/// </para>
/// <para>
/// <b>Output decoding.</b> YuNet emits twelve tensors — <c>cls</c>, <c>obj</c>, <c>bbox</c> and
/// <c>kps</c> at strides 8, 16 and 32. Each stride is a grid over the input, and a detection's
/// position is its grid cell plus the predicted offset, scaled by the stride. The confidence is
/// the geometric mean of the class and objectness scores, which is what the reference
/// implementation uses and what the thresholds below are calibrated against.
/// </para>
/// </remarks>
public sealed class YuNetFaceDetector : IFaceDetector, IDisposable
{
    /// <summary>
    /// Square input the image is letterboxed into.
    /// </summary>
    /// <remarks>
    /// <para>
    /// YuNet as an architecture accepts any input size, and larger would find smaller faces. This
    /// exported graph does not: its input dimensions are baked in at 640x640, and feeding it
    /// anything else is rejected outright by the runtime rather than handled. So the image is
    /// letterboxed — scaled to fit and padded — instead of resized freely.
    /// </para>
    /// <para>
    /// Padding goes bottom and right, never centred. With the picture pinned at the origin, the
    /// mapping back to image coordinates is a single division by the scaled size, and there is no
    /// offset to forget. Centred padding is prettier and has caused more off-by-half bugs than it
    /// has ever been worth.
    /// </para>
    /// </remarks>
    private const int InputSize = 640;

    /// <summary>Default minimum confidence for a detection to be kept.</summary>
    /// <remarks>
    /// <para>
    /// Measured, not inherited: the reference implementation defaults to 0.6, and a sweep over the
    /// test album (see the calibration test) says that is too loose here. Recall over the
    /// photographs that contain people is flat at eight faces from 0.80 all the way down to 0.40,
    /// so nothing below 0.80 buys a single extra face — it only buys false positives, which climb
    /// 3, 4, 7, 8, 11 as the threshold falls. Above 0.80 recall collapses: 0.90 finds three faces
    /// of eight.
    /// </para>
    /// <para>
    /// 0.80 is therefore the knee, and it is where this sits. The remaining detections on
    /// people-free photographs are a painted portrait hanging in a country-house living room and a
    /// figure in the background of a street corner — arguably faces, and both are things a person
    /// looking at the faces page would understand rather than be baffled by.
    /// </para>
    /// </remarks>
    public const float DefaultConfidenceThreshold = 0.8f;

    /// <summary>Boxes overlapping more than this are treated as the same face.</summary>
    private const float NmsIouThreshold = 0.3f;

    /// <summary>
    /// Default smallest face worth keeping, as a fraction of the image's shorter edge.
    /// </summary>
    /// <remarks>
    /// A face twelve pixels across in a 4000-pixel photograph is a smudge in a crowd. It cannot
    /// be recognised, it cannot be cropped into anything a person would identify, and it would
    /// bury the real subjects on the faces page under hundreds of unusable thumbnails.
    /// </remarks>
    public const double DefaultMinimumFaceFraction = 0.018;

    private static readonly int[] Strides = [8, 16, 32];

    // Chosen once for the whole application rather than per model, so "run on the neural
    // processor" is one setting and not four places to keep in step. Null means nobody supplied
    // one, which is the processor with the same options this always used.
    private readonly ComputeBackend? _compute;

    private readonly InferenceSession? _session;
    private readonly ILogger<YuNetFaceDetector> _log;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly float _confidenceThreshold;
    private readonly double _minimumFaceFraction;

    /// <summary>Creates a detector over a model file.</summary>
    /// <param name="modelPath">Path to the ONNX model, or null to run unavailable.</param>
    /// <param name="logger">Optional logger.</param>
    /// <param name="confidenceThreshold">
    /// Minimum confidence, or null for <see cref="DefaultConfidenceThreshold" />. Exposed so the
    /// calibration test can sweep it; the defaults were chosen by running that sweep.
    /// </param>
    /// <param name="minimumFaceFraction">
    /// <param name="compute">Where models run. Null means the processor, exactly as before.</param>
    /// Smallest face to keep, or null for <see cref="DefaultMinimumFaceFraction" />.
    /// </param>
    public YuNetFaceDetector(
        string? modelPath,
        ILogger<YuNetFaceDetector>? logger = null,
        float? confidenceThreshold = null,
        double? minimumFaceFraction = null,
        ComputeBackend? compute = null)
    {
        _compute = compute;
        _log = logger ?? NullLogger<YuNetFaceDetector>.Instance;
        _confidenceThreshold = confidenceThreshold ?? DefaultConfidenceThreshold;
        _minimumFaceFraction = minimumFaceFraction ?? DefaultMinimumFaceFraction;
        ModelVersion = modelPath is null ? "none" : Path.GetFileNameWithoutExtension(modelPath);

        if (modelPath is null || !File.Exists(modelPath))
        {
            _log.LogInformation("No face model at {Path}; face detection is off", modelPath);
            return;
        }

        try
        {
            _session = new InferenceSession(
                modelPath,
                (_compute ?? new ComputeBackend()).CreateSessionOptions(
                    Environment.ProcessorCount / 2,
                    Path.GetFileNameWithoutExtension(modelPath)));

            _log.LogInformation("Face model loaded: {Model}", ModelVersion);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or FileNotFoundException or DllNotFoundException)
        {
            _log.LogWarning(ex, "Could not load face model {Path}", modelPath);
            _session = null;
        }
    }

    /// <inheritdoc />
    public bool IsAvailable => _session is not null;

    /// <inheritdoc />
    public string ModelVersion { get; }

    /// <inheritdoc />
    public async Task<FaceResult> DetectAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        if (_session is null)
        {
            return FaceResult.Empty();
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await Task.Run(() => Detect(imagePath, cancellationToken), cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    private FaceResult Detect(string imagePath, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();

        using Image<Rgb24> source = AnalysisImage.Load(imagePath, InputSize);

        // Letterbox: scale the long edge to the fixed input and pad the rest with black. The
        // scaled size is kept because it, not the padded square, is what box coordinates are a
        // fraction of — dividing by 640 instead would squash every box towards the origin by the
        // aspect ratio.
        double scale = Math.Min((double)InputSize / source.Width, (double)InputSize / source.Height);
        int fitWidth = Math.Max(1, Math.Min(InputSize, (int)Math.Round(source.Width * scale)));
        int fitHeight = Math.Max(1, Math.Min(InputSize, (int)Math.Round(source.Height * scale)));

        using Image<Rgb24> resized = source.Clone(ctx => ctx.Resize(fitWidth, fitHeight));

        var input = new DenseTensor<float>([1, 3, InputSize, InputSize]);
        resized.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < fitHeight; y++)
            {
                Span<Rgb24> row = rows.GetRowSpan(y);
                for (int x = 0; x < fitWidth; x++)
                {
                    // BGR, 0-255, unnormalised. YuNet was trained this way; feeding it RGB or
                    // 0-1 values produces detections that look almost plausible and are wrong.
                    input[0, 0, y, x] = row[x].B;
                    input[0, 1, y, x] = row[x].G;
                    input[0, 2, y, x] = row[x].R;
                }
            }
        });

        cancellationToken.ThrowIfCancellationRequested();

        string inputName = _session!.InputMetadata.Keys.First();
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs =
            _session.Run([NamedOnnxValue.CreateFromTensor(inputName, input)]);

        var byName = outputs.ToDictionary(o => o.Name, o => o.AsTensor<float>(), StringComparer.Ordinal);
        var candidates = new List<DetectedFace>();

        foreach (int stride in Strides)
        {
            Tensor<float> cls = byName[$"cls_{stride}"];
            Tensor<float> obj = byName[$"obj_{stride}"];
            Tensor<float> box = byName[$"bbox_{stride}"];
            Tensor<float> kps = byName[$"kps_{stride}"];

            int cols = InputSize / stride;
            int rows = InputSize / stride;

            for (int i = 0; i < rows * cols; i++)
            {
                // Geometric mean of class and objectness, as the reference decoder uses.
                float score = MathF.Sqrt(Math.Clamp(cls[0, i, 0], 0, 1) * Math.Clamp(obj[0, i, 0], 0, 1));
                if (score < _confidenceThreshold)
                {
                    continue;
                }

                int cx = i % cols;
                int cy = i / cols;

                // Position is the grid cell plus the predicted offset, times the stride. Size is
                // exponential, which is how the network encodes scale.
                //
                // The predicted point is the CENTRE of the face, not its top-left corner. Reading
                // it as a corner is a mistake that hides well: every box is then shifted down and
                // right by half its own size, which still overlaps the face enough to pass a
                // plausibility check, still suppresses correctly, and still scores the same. It
                // only becomes obvious when you cut a crop out and see the head in the top-left
                // quarter of it.
                float centreX = (cx + box[0, i, 0]) * stride;
                float centreY = (cy + box[0, i, 1]) * stride;
                float w = MathF.Exp(box[0, i, 2]) * stride;
                float h = MathF.Exp(box[0, i, 3]) * stride;
                float left = centreX - (w / 2);
                float top = centreY - (h / 2);

                var landmarks = new List<(double, double)>(5);
                for (int k = 0; k < 5; k++)
                {
                    landmarks.Add((
                        (cx + kps[0, i, k * 2]) * stride / fitWidth,
                        (cy + kps[0, i, (k * 2) + 1]) * stride / fitHeight));
                }

                candidates.Add(new DetectedFace(
                    score,
                    left / fitWidth, top / fitHeight, w / fitWidth, h / fitHeight,
                    landmarks));
            }
        }

        List<DetectedFace> faces = Suppress(candidates)
            .Where(f => Math.Min(f.Width, f.Height) >= _minimumFaceFraction)
            .Where(f => f is { X: > -0.05, Y: > -0.05 } && f.X + f.Width < 1.05 && f.Y + f.Height < 1.05)
            .OrderByDescending(f => f.AreaFraction)
            .ToList();

        watch.Stop();
        return new FaceResult(faces, ModelVersion, watch.ElapsedMilliseconds);
    }

    /// <summary>Greedy non-maximum suppression over face boxes.</summary>
    /// <remarks>
    /// Class-agnostic here, unlike the segmentation pass — every detection is the same class, and
    /// two boxes that overlap heavily are the same face seen at two strides.
    /// </remarks>
    private static List<DetectedFace> Suppress(List<DetectedFace> candidates)
    {
        DetectedFace[] ordered = candidates.OrderByDescending(c => c.Confidence).ToArray();
        var suppressed = new bool[ordered.Length];
        var kept = new List<DetectedFace>();

        for (int i = 0; i < ordered.Length; i++)
        {
            if (suppressed[i])
            {
                continue;
            }

            kept.Add(ordered[i]);

            for (int j = i + 1; j < ordered.Length; j++)
            {
                if (!suppressed[j] && Iou(ordered[i], ordered[j]) > NmsIouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    private static double Iou(DetectedFace a, DetectedFace b)
    {
        double left = Math.Max(a.X, b.X);
        double top = Math.Max(a.Y, b.Y);
        double right = Math.Min(a.X + a.Width, b.X + b.Width);
        double bottom = Math.Min(a.Y + a.Height, b.Y + b.Height);

        double overlap = Math.Max(0, right - left) * Math.Max(0, bottom - top);
        double union = a.AreaFraction + b.AreaFraction - overlap;

        return union <= 0 ? 0 : overlap / union;
    }

    /// <summary>Releases the inference session.</summary>
    public void Dispose()
    {
        _session?.Dispose();
        _gate.Dispose();
    }
}
