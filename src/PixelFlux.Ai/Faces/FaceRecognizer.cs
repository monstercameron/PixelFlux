using PixelFlux.Ai.Compute;
using System.Numerics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Ai.Faces;

/// <summary>Turns a detected face into a vector that can be compared with other faces.</summary>
/// <remarks>
/// An interface so the model can be swapped without touching storage or the UI, and so tests can
/// stand in a deterministic fake for a 38 MB download.
/// </remarks>
public interface IFaceRecognizer
{
    /// <summary>Whether a usable model is installed.</summary>
    bool IsAvailable { get; }

    /// <summary>Identifier of the model in use.</summary>
    string ModelVersion { get; }

    /// <summary>Length of the vectors this model produces.</summary>
    int Dimensions { get; }

    /// <summary>
    /// Describes one face as a unit-length vector.
    /// </summary>
    /// <param name="image">The photograph the face was found in.</param>
    /// <param name="face">The face, with the landmarks used to align it.</param>
    /// <returns>A normalised embedding, or null when no model is installed or the face is unusable.</returns>
    float[]? Embed(Image<Rgb24> image, DetectedFace face);
}

/// <summary>
/// Face recognition with SFace.
/// </summary>
/// <remarks>
/// <para>
/// <b>What this does and does not claim.</b> It produces a vector per face such that two
/// photographs of one person land close together and two people land far apart. It does not
/// identify anybody: there are no names here, and no model of who exists. "Same person" is a
/// distance falling under a threshold, and that threshold was measured rather than assumed —
/// see the calibration test.
/// </para>
/// <para>
/// <b>Licensing.</b> SFace comes from the OpenCV Model Zoo under Apache 2.0, matching the
/// detector. It is 38 MB, which is too large to ship inside the application, so it is a file the
/// user supplies and everything degrades quietly without it.
/// </para>
/// <para>
/// <b>Nothing leaves the machine.</b> Face vectors are the most identifying data a photo library
/// can hold. They are computed here, stored here, compared here, and sent nowhere.
/// </para>
/// </remarks>
public sealed class SFaceRecognizer : IFaceRecognizer, IDisposable
{
    /// <summary>Edge of the aligned crop the model expects.</summary>
    private const int AlignedSize = 112;

    /// <summary>
    /// Where the five landmarks must land in the aligned crop.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are not arbitrary — they are the canonical positions SFace was trained against, and
    /// the whole method depends on hitting them. Aligning to this template is what makes two
    /// photographs of the same person comparable: it removes head tilt, in-plane rotation, and
    /// the difference between a close portrait and a distant one, so the vector describes the
    /// face rather than the framing.
    /// </para>
    /// <para>
    /// Order matches the detector's landmark order: right eye, left eye, nose, right mouth
    /// corner, left mouth corner. "Right" means the subject's right, which is the left of the
    /// image — which is why the first x is the smaller one.
    /// </para>
    /// </remarks>
    private static readonly (float X, float Y)[] Template =
    [
        (38.2946f, 51.6963f),
        (73.5318f, 51.5014f),
        (56.0252f, 71.7366f),
        (41.5493f, 92.3655f),
        (70.7299f, 92.2041f),
    ];

    /// <summary>
    /// Smallest face worth embedding, as a fraction of the shorter edge.
    /// </summary>
    /// <remarks>
    /// Below this the aligned crop is upsampled from a few dozen pixels and the vector describes
    /// interpolation artefacts more than a person. Such a face is still shown on the faces page
    /// — it is a real face — it simply cannot take part in "find this person", and saying so is
    /// better than returning confident nonsense.
    /// </remarks>
    private const double MinimumFaceFraction = 0.035;

    // Chosen once for the whole application rather than per model, so "run on the neural
    // processor" is one setting and not four places to keep in step. Null means nobody supplied
    // one, which is the processor with the same options this always used.
    private readonly ComputeBackend? _compute;

    private readonly InferenceSession? _session;
    private readonly ILogger<SFaceRecognizer> _log;
    private readonly object _gate = new();
    private readonly string _inputName;

    /// <summary>Creates a recognizer over a model file.</summary>
    /// <param name="modelPath">Path to the ONNX model, or null to run unavailable.</param>
    /// <param name="logger">Optional logger.</param>
    /// <param name="compute">Where models run. Null means the processor, exactly as before.</param>
    public SFaceRecognizer(
        string? modelPath,
        ILogger<SFaceRecognizer>? logger = null,
        ComputeBackend? compute = null)
    {
        _compute = compute;
        _log = logger ?? NullLogger<SFaceRecognizer>.Instance;
        ModelVersion = modelPath is null ? "none" : Path.GetFileNameWithoutExtension(modelPath);
        _inputName = "data";

        if (modelPath is null || !File.Exists(modelPath))
        {
            _log.LogInformation("No recognition model at {Path}; grouping faces is off", modelPath);
            return;
        }

        try
        {
            _session = new InferenceSession(
                modelPath,
                (_compute ?? new ComputeBackend()).CreateSessionOptions(
                    Environment.ProcessorCount / 2,
                    Path.GetFileNameWithoutExtension(modelPath)));

            _inputName = _session.InputMetadata.Keys.First();
            Dimensions = _session.OutputMetadata.Values.First().Dimensions is [_, int d] && d > 0 ? d : 128;

            _log.LogInformation("Recognition model loaded: {Model}, {Dims} dimensions",
                ModelVersion, Dimensions);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or FileNotFoundException or DllNotFoundException)
        {
            _log.LogWarning(ex, "Could not load recognition model {Path}", modelPath);
            _session = null;
        }
    }

    /// <inheritdoc />
    public bool IsAvailable => _session is not null;

    /// <inheritdoc />
    public string ModelVersion { get; }

    /// <inheritdoc />
    public int Dimensions { get; } = 128;

    /// <summary>
    /// How alike two face vectors are, from -1 to 1.
    /// </summary>
    /// <param name="a">One embedding.</param>
    /// <param name="b">Another.</param>
    /// <returns>Cosine similarity, or 0 if the two cannot be compared.</returns>
    /// <remarks>
    /// A plain dot product, because embeddings are stored normalised. Vectors of different
    /// lengths score 0 rather than throwing: that happens when a library holds results from two
    /// models, and the honest answer is "these are not comparable", not a crash.
    /// </remarks>
    public static double Similarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length == 0 || a.Length != b.Length)
        {
            return 0;
        }

        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <inheritdoc />
    public float[]? Embed(Image<Rgb24> image, DetectedFace face)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(face);

        if (_session is null || face.Landmarks.Count < 5)
        {
            return null;
        }

        if (Math.Min(face.Width, face.Height) < MinimumFaceFraction)
        {
            return null;
        }

        using Image<Rgb24> aligned = Align(image, face);

        var input = new DenseTensor<float>([1, 3, AlignedSize, AlignedSize]);

        aligned.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < AlignedSize; y++)
            {
                Span<Rgb24> row = rows.GetRowSpan(y);
                for (int x = 0; x < AlignedSize; x++)
                {
                    // BGR, 0-255, unnormalised — the same convention as the detector, and the
                    // one SFace was trained with. Feeding RGB produces vectors that look
                    // perfectly reasonable and cluster the wrong people together.
                    input[0, 0, y, x] = row[x].B;
                    input[0, 1, y, x] = row[x].G;
                    input[0, 2, y, x] = row[x].R;
                }
            }
        });

        float[] vector;

        // One session, serialised. ONNX Runtime sessions are thread-safe for Run, but the
        // worker is sequential anyway and a lock costs nothing here while removing a whole
        // category of question.
        lock (_gate)
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs =
                _session.Run([NamedOnnxValue.CreateFromTensor(_inputName, input)]);

            vector = outputs.First().AsTensor<float>().ToArray();
        }

        Normalise(vector);
        return vector;
    }

    /// <summary>
    /// Warps the face onto the model's canonical landmark positions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A similarity transform — rotation, uniform scale, translation — fitted by least squares
    /// to all five landmark pairs. Not an affine fit: allowing shear and independent axis scaling
    /// would let the transform squash a face towards the template and erase the very differences
    /// that distinguish one person from another.
    /// </para>
    /// <para>
    /// Five points over-determine a four-parameter transform, which is the point. Landmarks are
    /// individually noisy, and a least-squares fit over all five is far steadier than the exact
    /// solution from any two of them.
    /// </para>
    /// </remarks>
    private static Image<Rgb24> Align(Image<Rgb24> image, DetectedFace face)
    {
        // Landmarks are stored as fractions so they survive the image being resized; the fit
        // needs pixels.
        var source = new (double X, double Y)[5];
        for (int i = 0; i < 5; i++)
        {
            source[i] = (face.Landmarks[i].X * image.Width, face.Landmarks[i].Y * image.Height);
        }

        Matrix3x2 matrix = FitSimilarity(source, Template);

        return image.Clone(ctx => ctx.Transform(
            new Rectangle(0, 0, image.Width, image.Height),
            matrix,
            new Size(AlignedSize, AlignedSize),
            KnownResamplers.Bicubic));
    }

    /// <summary>
    /// Least-squares similarity transform taking <paramref name="source"/> onto <paramref name="target"/>.
    /// </summary>
    /// <param name="source">Points in the source image, in pixels.</param>
    /// <param name="target">Where they should land.</param>
    /// <returns>The transform, as a matrix the imaging library can apply.</returns>
    /// <remarks>
    /// The model is <c>u = ax - by + tx</c>, <c>v = bx + ay + ty</c> — four unknowns, which is
    /// rotation and uniform scale folded into <c>(a, b)</c> plus a translation. Because it is
    /// linear in all four, the least-squares solution is closed form and needs no iteration and
    /// no matrix decomposition; the normal equations reduce to the two expressions below.
    /// </remarks>
    private static Matrix3x2 FitSimilarity(
        IReadOnlyList<(double X, double Y)> source,
        IReadOnlyList<(float X, float Y)> target)
    {
        int n = source.Count;
        double sx = 0, sy = 0, su = 0, sv = 0, z = 0, p = 0, q = 0;

        for (int i = 0; i < n; i++)
        {
            (double x, double y) = source[i];
            double u = target[i].X;
            double v = target[i].Y;

            sx += x;
            sy += y;
            su += u;
            sv += v;
            z += (x * x) + (y * y);
            p += (u * x) + (v * y);
            q += (u * y) - (v * x);
        }

        double den = (n * z) - (sx * sx) - (sy * sy);

        if (Math.Abs(den) < 1e-9)
        {
            // Every landmark at one point — a degenerate detection. Identity leaves the caller
            // with a meaningless crop rather than a divide by zero, and the similarity it
            // produces will simply not match anything.
            return Matrix3x2.Identity;
        }

        double a = ((n * p) - (sx * su) - (sy * sv)) / den;
        double b = ((sy * su) - (sx * sv) - (n * q)) / den;
        double tx = (su - (a * sx) + (b * sy)) / n;
        double ty = (sv - (b * sx) - (a * sy)) / n;

        // System.Numerics lays the matrix out as x' = x*M11 + y*M21 + M31, so the rotation
        // block is transposed relative to how the equations above are written.
        return new Matrix3x2((float)a, (float)b, (float)-b, (float)a, (float)tx, (float)ty);
    }

    /// <summary>Scales a vector to unit length, in place.</summary>
    /// <remarks>
    /// Done once at write time so every later comparison is a dot product. A library with tens
    /// of thousands of faces compares one vector against all of them on every click; paying for
    /// two square roots per comparison would be paying them a hundred thousand times.
    /// </remarks>
    private static void Normalise(float[] vector)
    {
        double sum = 0;
        foreach (float value in vector)
        {
            sum += value * value;
        }

        if (sum <= 0)
        {
            return;
        }

        float scale = (float)(1.0 / Math.Sqrt(sum));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= scale;
        }
    }

    /// <summary>Releases the inference session.</summary>
    public void Dispose() => _session?.Dispose();
}
