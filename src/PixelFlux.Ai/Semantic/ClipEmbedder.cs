using PixelFlux.Ai.Compute;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Ai.Semantic;

/// <summary>Turns photographs and phrases into vectors that can be compared with each other.</summary>
/// <remarks>
/// An interface so the model can be swapped, and so tests can stand in a deterministic fake
/// rather than a 600 MB download.
/// </remarks>
public interface IImageTextEmbedder
{
    /// <summary>Whether a usable model is installed.</summary>
    bool IsAvailable { get; }

    /// <summary>Identifier of the model, recorded beside every vector it produces.</summary>
    string ModelVersion { get; }

    /// <summary>Length of the vectors this model produces.</summary>
    int Dimensions { get; }

    /// <summary>Describes a photograph.</summary>
    /// <param name="imagePath">Absolute path to the image.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>A unit-length vector, or null when no model is installed.</returns>
    Task<float[]?> EmbedImageAsync(string imagePath, CancellationToken cancellationToken = default);

    /// <summary>Describes a phrase, in the same space as <see cref="EmbedImageAsync" />.</summary>
    /// <param name="text">The exact text to embed.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>A unit-length vector, or null when no model is installed or the text is empty.</returns>
    /// <remarks>
    /// Embeds precisely what it is given. For a search box use <see cref="EmbedQueryAsync" />,
    /// which is the same thing with the caption phrasing CLIP expects.
    /// </remarks>
    Task<float[]?> EmbedTextAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>Describes what somebody typed into a search box.</summary>
    /// <param name="query">The user's words.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>A unit-length vector, or null when no model is installed or the query is empty.</returns>
    Task<float[]?> EmbedQueryAsync(string query, CancellationToken cancellationToken = default);
}

/// <summary>
/// Search by meaning, using CLIP.
/// </summary>
/// <remarks>
/// <para>
/// <b>What this buys.</b> "red car" finds a red car in a photograph nobody tagged, that no
/// detector labelled red, and whose filename says nothing. The existing search matches words
/// against words — filenames, EXIF, tags, the detector's eighty classes. This matches a
/// description against the picture itself, which is the difference between finding photographs
/// you have already described and finding photographs you have not.
/// </para>
/// <para>
/// <b>How it can work at all.</b> CLIP is trained so that a photograph and a sentence describing
/// it land near each other in one shared 512-dimensional space. Two separate networks — one for
/// pixels, one for words — are trained together until their outputs agree. So a text query and
/// an image can be compared with a dot product, which is what makes search over a whole library
/// a single sweep rather than a model run per photograph.
/// </para>
/// <para>
/// <b>Licensing and size.</b> CLIP ViT-B/32 is MIT, exported to ONNX by the transformers.js
/// project. The two encoders are about 580 MB together, far too large to ship, so they are files
/// the user supplies — and every part of the application degrades quietly without them.
/// </para>
/// <para>
/// <b>Nothing leaves the machine.</b> Both encoders run locally. A search phrase is turned into
/// a vector here and compared here.
/// </para>
/// </remarks>
public sealed class ClipEmbedder : IImageTextEmbedder, IDisposable
{
    /// <summary>Square input the vision encoder expects.</summary>
    private const int ImageSize = 224;

    /// <summary>
    /// Per-channel mean subtracted from each pixel, in RGB order.
    /// </summary>
    /// <remarks>
    /// CLIP's own normalisation constants, not ImageNet's. They are close enough that using the
    /// wrong pair produces vectors which still look reasonable and rank slightly wrongly — the
    /// worst kind of mistake, because nothing fails.
    /// </remarks>
    private static readonly float[] Mean = [0.48145466f, 0.4578275f, 0.40821073f];

    private static readonly float[] StandardDeviation = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>
    /// The phrasings a search query is rewritten into before it is embedded.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why a bare word is not enough.</b> CLIP learned from captions — "a photo of a golden
    /// retriever in the snow" — so a caption-shaped phrase lands where photographs live and a
    /// bare noun lands somewhere else entirely. Measured on this library, "animal" put two
    /// portraits of a man in a face mask above the cat; "a photo of an animal" returned a bird,
    /// a cat, and a man holding a koala. "blonde hair" led with a bald man in sunglasses;
    /// "a photo of a person with blonde hair" returned six portraits of the same blonde woman.
    /// Same model, same vectors, same library — only the phrasing changed.
    /// </para>
    /// <para>
    /// <b>Why several, averaged.</b> Any single template is a guess about grammar the user did
    /// not supply: "a photo of a {}" reads correctly for "red car" and badly for "blonde hair".
    /// Averaging across templates cancels most of that — the article is right in some and wrong
    /// in others, and what survives the average is the subject. This is the prompt-ensembling
    /// technique from the CLIP paper, cut down from eighty templates to five that suit a photo
    /// library rather than an object-recognition benchmark.
    /// </para>
    /// <para>
    /// The bare query is one of them, so a user who types a full sentence of their own is not
    /// overruled by the scaffolding.
    /// </para>
    /// <para>
    /// English only, and knowingly. CLIP's training data is overwhelmingly English, so a query
    /// in another language is weak before any template is applied; wrapping it in English words
    /// neither helps nor meaningfully hurts.
    /// </para>
    /// </remarks>
    private static readonly string[] QueryTemplates =
    [
        "{0}",
        "a photo of {0}",
        "a photo of a {0}",
        "a photograph of {0}",
        "a close-up photo of {0}",
    ];

    // Chosen once for the whole application rather than per model, so "run on the neural
    // processor" is one setting and not four places to keep in step. Null means nobody supplied
    // one, which is the processor with the same options this always used.
    private readonly ComputeBackend? _compute;

    private readonly string? _visionPath;
    private readonly string? _textPath;
    private readonly string? _vocabularyPath;
    private readonly string? _mergesPath;

    private InferenceSession? _vision;
    private InferenceSession? _text;
    private ClipTokenizer? _tokenizer;

    private readonly ILogger<ClipEmbedder> _log;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Creates an embedder over the CLIP exports.</summary>
    /// <param name="visionModelPath">Path to the vision encoder.</param>
    /// <param name="textModelPath">Path to the text encoder.</param>
    /// <param name="vocabularyPath">Path to <c>vocab.json</c>.</param>
    /// <param name="mergesPath">Path to <c>merges.txt</c>.</param>
    /// <param name="logger">Optional logger.</param>
    /// <param name="compute">Where models run. Null means the processor, exactly as before.</param>
    /// <remarks>
    /// All four or nothing. A vision encoder without a tokenizer could still describe
    /// photographs, but nothing could be searched for, so the half-installed state is treated as
    /// not installed rather than as a feature that silently returns no results.
    /// </remarks>
    public ClipEmbedder(
        string? visionModelPath,
        string? textModelPath,
        string? vocabularyPath,
        string? mergesPath,
        ILogger<ClipEmbedder>? logger = null,
        ComputeBackend? compute = null)
    {
        _compute = compute;
        _log = logger ?? NullLogger<ClipEmbedder>.Instance;
        ModelVersion = "clip-vit-base-patch32";

        bool present = visionModelPath is not null && File.Exists(visionModelPath)
                    && textModelPath is not null && File.Exists(textModelPath)
                    && vocabularyPath is not null && File.Exists(vocabularyPath)
                    && mergesPath is not null && File.Exists(mergesPath);

        IsAvailable = present;

        if (!present)
        {
            _log.LogInformation("CLIP is not installed; search by meaning is off");
            return;
        }

        _visionPath = visionModelPath;
        _textPath = textModelPath;
        _vocabularyPath = vocabularyPath;
        _mergesPath = mergesPath;
    }

    /// <inheritdoc />
    /// <remarks>
    /// True as soon as the four files are on disk — the models themselves are opened on first
    /// use. Availability is a question the interface asks on every render, and it must not be
    /// the thing that loads 580 MB.
    /// </remarks>
    public bool IsAvailable { get; }

    /// <inheritdoc />
    public string ModelVersion { get; }

    /// <inheritdoc />
    /// <remarks>
    /// A constant for this model rather than something read off the graph, so that asking for it
    /// does not force a 350 MB load. CLIP ViT-B/32 projects to 512 in both directions; a
    /// different export would be a different <see cref="ModelVersion" /> anyway, and vectors
    /// from two models are never mixed.
    /// </remarks>
    public int Dimensions => 512;

    /// <inheritdoc />
    public async Task<float[]?> EmbedImageAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        if (!IsAvailable)
        {
            return null;
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await Task.Run(() => EmbedImage(imagePath), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public async Task<float[]?> EmbedQueryAsync(string query, CancellationToken cancellationToken = default)
    {
        if (!IsAvailable || string.IsNullOrWhiteSpace(query))
        {
            return null;
        }

        string trimmed = query.Trim();

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await Task.Run(
                () =>
                {
                    var sum = new float[Dimensions];

                    foreach (string template in QueryTemplates)
                    {
                        float[] one = EmbedText(string.Format(
                            System.Globalization.CultureInfo.InvariantCulture, template, trimmed));

                        for (int i = 0; i < sum.Length; i++)
                        {
                            sum[i] += one[i];
                        }
                    }

                    // Averaging unit vectors gives something shorter than unit length; the
                    // direction is the answer, so it is renormalised rather than divided by the
                    // count. Everything downstream assumes unit length and compares by dot
                    // product.
                    Normalise(sum);
                    return sum;
                },
                cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public async Task<float[]?> EmbedTextAsync(string text, CancellationToken cancellationToken = default)
    {
        if (!IsAvailable || string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await Task.Run(() => EmbedText(text), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Opens an encoder the first time it is needed.
    /// </summary>
    /// <remarks>
    /// The two encoders are opened separately and only on demand, because the two things people
    /// do with them are lopsided. Searching needs the text encoder — 250 MB, opened once, then
    /// a few milliseconds per query. Describing the library needs the vision encoder — 350 MB,
    /// used heavily for one sweep and then never again until new photographs arrive. Loading
    /// both to answer "is this feature available" would cost 580 MB of resident memory in an
    /// application whose main job is displaying photographs.
    ///
    /// Callers hold the gate, so no double-open is possible.
    /// </remarks>
    private InferenceSession Open(string path)
    {
        using SessionOptions options =
            (_compute ?? new ComputeBackend()).CreateSessionOptions(
                Environment.ProcessorCount / 2,
                Path.GetFileNameWithoutExtension(path));

        return new InferenceSession(path, options);
    }

    private float[] EmbedImage(string imagePath)
    {
        _vision ??= Open(_visionPath!);

        using Image<Rgb24> source = AnalysisImage.Load(imagePath, ImageSize);

        // Shortest edge to 224, then centre-crop — CLIP's own preprocessing. Squashing the whole
        // frame into a square instead would distort every subject, and the model has never seen
        // a stretched photograph.
        using Image<Rgb24> square = source.Clone(ctx => ctx
            .Resize(new ResizeOptions
            {
                Size = new Size(ImageSize, ImageSize),
                Mode = ResizeMode.Crop,
                Position = AnchorPositionMode.Center,
                Sampler = KnownResamplers.Bicubic,
            }));

        var input = new DenseTensor<float>([1, 3, ImageSize, ImageSize]);

        square.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < ImageSize; y++)
            {
                Span<Rgb24> row = rows.GetRowSpan(y);
                for (int x = 0; x < ImageSize; x++)
                {
                    Rgb24 p = row[x];
                    input[0, 0, y, x] = ((p.R / 255f) - Mean[0]) / StandardDeviation[0];
                    input[0, 1, y, x] = ((p.G / 255f) - Mean[1]) / StandardDeviation[1];
                    input[0, 2, y, x] = ((p.B / 255f) - Mean[2]) / StandardDeviation[2];
                }
            }
        });

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = _vision.Run(
            [NamedOnnxValue.CreateFromTensor(_vision.InputMetadata.Keys.First(), input)]);

        float[] vector = outputs.First().AsTensor<float>().ToArray();
        Normalise(vector);
        return vector;
    }

    private float[] EmbedText(string text)
    {
        _text ??= Open(_textPath!);
        _tokenizer ??= new ClipTokenizer(_vocabularyPath!, _mergesPath!);

        int[] ids = _tokenizer.Encode(text);

        var input = new DenseTensor<long>([1, ids.Length]);
        for (int i = 0; i < ids.Length; i++)
        {
            input[0, i] = ids[i];
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = _text.Run(
            [NamedOnnxValue.CreateFromTensor(_text.InputMetadata.Keys.First(), input)]);

        float[] vector = outputs.First().AsTensor<float>().ToArray();
        Normalise(vector);
        return vector;
    }

    /// <summary>Scales a vector to unit length, in place.</summary>
    /// <remarks>
    /// Done here so every comparison downstream is a plain dot product. CLIP's raw outputs are
    /// not unit length, and comparing them unnormalised would rank by a mixture of similarity
    /// and magnitude — which mostly measures how much is going on in the photograph.
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

    /// <summary>Releases whichever inference sessions were opened.</summary>
    public void Dispose()
    {
        _vision?.Dispose();
        _text?.Dispose();
        _gate.Dispose();
    }
}
