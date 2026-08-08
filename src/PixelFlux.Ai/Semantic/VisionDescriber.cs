using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntimeGenAI;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Ai.Semantic;

/// <summary>Writes a description of what is in a photograph.</summary>
/// <remarks>
/// An interface so the model can be replaced, and so tests can stand in a fake rather than a
/// 1.4 GB download and fifteen seconds per photograph.
/// </remarks>
public interface IPhotoDescriber
{
    /// <summary>Whether a usable model is installed.</summary>
    bool IsAvailable { get; }

    /// <summary>Identifier of the model, recorded beside every description it writes.</summary>
    string ModelVersion { get; }

    /// <summary>Describes one photograph.</summary>
    /// <param name="imagePath">Absolute path to the image.</param>
    /// <param name="cancellationToken">Stops generation.</param>
    /// <returns>A paragraph of prose, or null when no model is installed.</returns>
    Task<string?> DescribeAsync(string imagePath, CancellationToken cancellationToken = default);
}

/// <summary>
/// Describes photographs with Qwen3-VL, a vision-language model running locally.
/// </summary>
/// <remarks>
/// <para>
/// <b>What this is for.</b> Everything else in PixelFlux recognises a fixed list: eighty object
/// classes, faces, places. This reads the picture and writes about it — "a red and white striped
/// deckchair", "a white marquee", the registration on a number plate. That text goes into the
/// word index, so the library becomes searchable by things nobody thought to make a category
/// for. It is the difference between finding photographs you have already described and finding
/// photographs nobody has.
/// </para>
/// <para>
/// <b>Why the runtime and not raw ONNX.</b> A vision-language model is not one graph you call.
/// It is a vision encoder, an embedding table and a decoder, plus an autoregressive loop, a
/// key-value cache, a chat template, and image tokens spliced into the text stream at positions
/// the processor decides. ONNX Runtime GenAI owns all of that. Hand-rolling it is days of work
/// and its failures are fluent nonsense rather than exceptions.
/// </para>
/// <para>
/// <b>Nothing leaves the machine.</b> The model is a directory on disk and every photograph is
/// read, described and stored here.
/// </para>
/// </remarks>
public sealed class QwenVisionDescriber : IPhotoDescriber, IDisposable
{
    /// <summary>
    /// Longest edge the photograph is scaled to before the model sees it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Qwen3-VL uses dynamic resolution — every 32x32 block of the input becomes one token — so
    /// this is really a token budget in disguise. A 2560-pixel proxy produced 5,664 image tokens
    /// before a single word was generated.
    /// </para>
    /// <para>
    /// Bigger is not better, which was a surprise worth recording. Measured on one photograph
    /// against a known number plate: 448px read "UVT 224" in 12.8 seconds, 672px read it in 16.5
    /// seconds, and 1024px took 30.8 seconds and read it as "UVT 2224". More pixels cost time
    /// and bought a hallucination. 672 sits in the middle with headroom for finer text than a
    /// number plate.
    /// </para>
    /// </remarks>
    /// <summary>Where the time went on the last photograph described.</summary>
    /// <param name="Prefill">
    /// Everything up to the first generated token: encoding the image and attending over its
    /// tokens. Falls when the input resolution falls.
    /// </param>
    /// <param name="Decode">The rest, one token at a time. Falls when the description is shorter.</param>
    /// <param name="Tokens">How many tokens were generated.</param>
    public readonly record struct DescribeTiming(TimeSpan Prefill, TimeSpan Decode, int Tokens)
    {
        /// <summary>Milliseconds spent per generated token.</summary>
        public double MillisecondsPerToken => Tokens == 0 ? 0 : Decode.TotalMilliseconds / Tokens;
    }

    private const int DefaultLongEdge = 672;

    /// <summary>
    /// How many of the machine's cores to give the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Six, on an eighteen-core machine, and this is not a typo. Decoding a 4-bit model is bound
    /// by memory bandwidth rather than arithmetic, so beyond a handful of threads they contend
    /// for the same bytes and make each other slower; the X2's cores are also not uniform, so
    /// every synchronisation point runs at the speed of the slowest one.
    /// </para>
    /// <para>
    /// Measured, generating eighty tokens: 4 threads 2.36s, <b>6 threads 2.19s</b>, 8 threads
    /// 2.89s, 12 threads 5.47s, 18 threads 13.62s, and the runtime's own default 4.77s. Six is
    /// six times faster than letting it have the whole machine, and twice as fast as the
    /// default. It also leaves twelve cores free, which matters because this runs while somebody
    /// is using the computer.
    /// </para>
    /// </remarks>
    private const int Threads = 6;

    /// <summary>Most tokens of description to generate.</summary>
    /// <remarks>
    /// Enough for a full paragraph — around 150 words — which is what the word index wants.
    /// Generation stops early at the end-of-turn token; this is the ceiling for a model that
    /// decides to keep going.
    /// </remarks>
    private const int DefaultMaximumTokens = 220;

    /// <summary>
    /// How much a token already used is discouraged from being used again.
    /// </summary>
    /// <remarks>
    /// Greedy decoding with no penalty loops. Measured on the test album: descriptions reliably
    /// reach the token cap by repeating a sentence verbatim three to five times — "There is a
    /// wooden box on the floor" five times over — so the last third of every description is
    /// wasted, and it is wasted at about 27 ms a token.
    /// </remarks>
    private const double DefaultRepetitionPenalty = 1.0;

    /// <summary>
    /// What the model is asked for.
    /// </summary>
    /// <remarks>
    /// Written for an index, not for a reader. Plain prose because headings and bullet points
    /// become noise in a full-text column; the explicit list of things to cover because a model
    /// left to itself writes three sentences about the mood; "read out any visible text" because
    /// signs, number plates and shopfronts are exactly the details a person searches for and
    /// nothing else in the application can extract them.
    /// </remarks>
    private const string Instruction =
        "Describe this photograph for someone searching a photo library. Write one paragraph of "
        + "plain prose. Name the subject, the setting, the colours, materials, actions and mood, "
        + "and read out any visible text such as signs or number plates. Be specific and "
        + "concrete. Do not use headings or bullet points.";

    // Both are adjustable so the two halves of the cost can be swept independently. Prefill
    // — about 5 s a photograph, measured — falls with the long edge; decode, about 6 s, falls with
    // the token count. A single "make it faster" knob would have hidden that they are separate.
    private readonly int _longEdge;
    private readonly int _maximumTokens;
    private readonly double _repetitionPenalty;

    private readonly string _modelDirectory;
    private readonly ILogger<QwenVisionDescriber> _log;
    private readonly SemaphoreSlim _gate = new(1, 1);

    private Model? _model;
    private MultiModalProcessor? _processor;

    /// <summary>Creates a describer over a model directory.</summary>
    /// <param name="modelDirectory">Directory holding <c>genai_config.json</c> and its graphs.</param>
    /// <param name="logger">Optional logger.</param>
    /// <param name="longEdge">Longest edge the photograph is scaled to. Drives prefill cost.</param>
    /// <param name="maximumTokens">Most tokens to generate. Drives decode cost.</param>
    /// <param name="repetitionPenalty">How much to discourage repeating tokens. 1.0 is off.</param>
    public QwenVisionDescriber(
        string? modelDirectory,
        ILogger<QwenVisionDescriber>? logger = null,
        int longEdge = DefaultLongEdge,
        int maximumTokens = DefaultMaximumTokens,
        double repetitionPenalty = DefaultRepetitionPenalty)
    {
        _log = logger ?? NullLogger<QwenVisionDescriber>.Instance;
        _longEdge = longEdge;
        _maximumTokens = maximumTokens;
        _repetitionPenalty = repetitionPenalty;
        _modelDirectory = modelDirectory ?? string.Empty;
        ModelVersion = "qwen3-vl-2b-instruct";

        IsAvailable = modelDirectory is not null
                   && File.Exists(Path.Combine(modelDirectory, "genai_config.json"));

        if (!IsAvailable)
        {
            _log.LogInformation("No vision-language model at {Path}; descriptions are off", modelDirectory);
        }
    }


    /// <summary>Where the time went on the last photograph described.</summary>
    public DescribeTiming LastTiming { get; private set; }

    /// <summary>Longest edge the photograph is scaled to before the model sees it.</summary>
    public int LongEdge => _longEdge;

    /// <summary>Most tokens of description generated.</summary>
    public int MaximumTokens => _maximumTokens;

    /// <summary>How strongly repeated tokens are discouraged. 1.0 is off.</summary>
    public double RepetitionPenalty => _repetitionPenalty;

    /// <inheritdoc />
    /// <remarks>
    /// True as soon as the configuration file is on disk. The model itself weighs 1.4 GB and is
    /// loaded on first use — this is asked on every render and must not be what loads it.
    /// </remarks>
    public bool IsAvailable { get; }

    /// <inheritdoc />
    public string ModelVersion { get; }

    /// <inheritdoc />
    public async Task<string?> DescribeAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        if (!IsAvailable)
        {
            return null;
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await Task.Run(() => Describe(imagePath, cancellationToken), cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    private string? Describe(string imagePath, CancellationToken cancellationToken)
    {
        Load();

        // The processor reads from a path, so the scaled copy has to exist as a file. Named with
        // a GUID because two libraries could be described at once by two processes sharing a
        // temp directory.
        string scaled = Path.Combine(Path.GetTempPath(), $"pixelflux-vlm-{Guid.NewGuid():n}.jpg");

        try
        {
            using (Image source = Image.Load(imagePath))
            {
                source.Mutate(ctx => ctx.Resize(new ResizeOptions
                {
                    Size = new Size(_longEdge, _longEdge),
                    Mode = ResizeMode.Max,
                    Sampler = KnownResamplers.Bicubic,
                }));

                source.Save(scaled);
            }

            long started = Stopwatch.GetTimestamp();

            using Images images = Images.Load([scaled]);
            using TokenizerStream stream = _processor!.CreateStream();

            string prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                          + Instruction
                          + "<|im_end|>\n<|im_start|>assistant\n";

            using NamedTensors inputs = _processor.ProcessImages(prompt, images);

            using var parameters = new GeneratorParams(_model!);

            // max_length counts the prompt, and the prompt is mostly image — several hundred
            // tokens before a word is written. The real limit is the generated count below.
            parameters.SetSearchOption("max_length", 8192);

            // Greedy. A description feeding a search index should be the same every time the
            // same photograph is described; sampling would make re-running the sweep produce a
            // different library.
            parameters.SetSearchOption("do_sample", false);

            // Discourages the loop that greedy decoding falls into. Still deterministic — the same
            // photograph gives the same description, which the search index depends on.
            if (_repetitionPenalty > 1.0)
            {
                parameters.SetSearchOption("repetition_penalty", _repetitionPenalty);
            }

            using var generator = new Generator(_model!, parameters);
            generator.SetInputs(inputs);

            var text = new System.Text.StringBuilder(1024);
            int produced = 0;

            // Split the clock at the first token, because the two halves respond to completely
            // different knobs and the total tells you nothing about which to turn. Everything up
            // to the first token is prefill — encoding the image and attending over several
            // hundred image tokens — and shrinks by lowering the input resolution. Everything
            // after is decode, one token at a time, and shrinks by asking for a shorter
            // description. Optimising the wrong one is a day spent for no change.
            //
            // The clock starts above, at Images.Load, not here. Started at the loop it reported a
            // prefill of zero every time: the runtime does the prompt pass inside SetInputs, so by
            // the time the first GenerateNextToken is called the expensive part is already done.
            TimeSpan prefill = TimeSpan.Zero;

            // Sentences already written, so a loop can be cut off rather than run to the cap.
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            int sentenceStart = 0;

            while (!generator.IsDone() && produced < _maximumTokens)
            {
                cancellationToken.ThrowIfCancellationRequested();

                generator.GenerateNextToken();

                if (produced == 0)
                {
                    prefill = Stopwatch.GetElapsedTime(started);
                }

                text.Append(stream.Decode(generator.GetSequence(0)[^1]));
                produced++;

                if (text.Length == 0 || text[^1] != '.')
                {
                    continue;
                }

                int previousStart = sentenceStart;
                string sentence = text.ToString(sentenceStart, text.Length - sentenceStart).Trim();
                sentenceStart = text.Length;

                if (sentence.Length > 12 && !seen.Add(sentence))
                {
                    // Drop the duplicate itself, not just what would have followed it. The repeat
                    // is only detectable once it has been written, and leaving it in would put one
                    // redundant sentence into the search index for no reason.
                    text.Length = previousStart;
                    // Said this already. Greedy decoding falls into a loop near the end of most
                    // descriptions — "There is a wooden box on the floor" five times over — and
                    // the repeats cost about 27 ms a token for text that carries nothing.
                    //
                    // Stopping rather than applying a repetition penalty is deliberate, and it was
                    // measured. A penalty does end the loop, and it also starts inventing: at 1.05
                    // a badge reading CULTURE became "Kreuz", and at 1.15 the model produced a
                    // "SCHULTE" sign, a box labelled "102" and a name tag reading "M." — three
                    // specific strings with no more of the photograph behind them. For text that
                    // becomes a search index that is the worse failure by far. A repeated sentence
                    // is noise; an invented proper noun makes a photograph findable by a word that
                    // is not in it.
                    break;
                }
            }

            LastTiming = new DescribeTiming(
                prefill, Stopwatch.GetElapsedTime(started) - prefill, produced);

            return Tidy(text.ToString());
        }
        finally
        {
            try
            {
                File.Delete(scaled);
            }
            catch (IOException)
            {
                // A stray temp file is never read.
            }
        }
    }

    /// <summary>Opens the model, once.</summary>
    /// <remarks>
    /// The thread count is applied by overlaying the packaged configuration rather than editing
    /// it, so the downloaded model directory stays exactly as published and can be replaced
    /// without losing the setting.
    /// </remarks>
    private void Load()
    {
        if (_model is not null)
        {
            return;
        }

        using var config = new Config(_modelDirectory);

        config.Overlay($$"""
            { "model": { "decoder": { "session_options": {
                "intra_op_num_threads": {{Threads}} } } } }
            """);

        _model = new Model(config);
        _processor = new MultiModalProcessor(_model);

        _log.LogInformation("Vision-language model loaded on {Threads} threads", Threads);
    }

    /// <summary>
    /// Trims the model's output down to the description itself.
    /// </summary>
    /// <remarks>
    /// Two things to remove. Chat turn markers, which occasionally survive decoding. And a
    /// trailing half-sentence, which happens whenever the token ceiling is reached mid-flow —
    /// storing "the car's hood is slightly" helps nobody, and a search index is one place where
    /// a truncated clause is worse than none.
    /// </remarks>
    private static string? Tidy(string raw)
    {
        string text = raw.Replace("<|im_end|>", string.Empty, StringComparison.Ordinal)
                         .Replace("<|endoftext|>", string.Empty, StringComparison.Ordinal)
                         .Trim();

        if (text.Length == 0)
        {
            return null;
        }

        int lastStop = text.LastIndexOfAny(['.', '!', '?']);

        // Only trim back to the last full stop when there is a substantial description before
        // it. Cutting a two-sentence answer down to one to avoid a dangling clause would lose
        // more than it saves.
        if (lastStop > 0 && lastStop < text.Length - 1 && lastStop > text.Length / 2)
        {
            text = text[..(lastStop + 1)];
        }

        // The model sometimes ends a sentence and then emits another stop; keeping both leaves
        // "..automobiles.." in the index and looks like a bug in the application rather than a
        // quirk of the model.
        while (text.EndsWith("..", StringComparison.Ordinal))
        {
            text = text[..^1];
        }

        return text;
    }

    /// <summary>Releases the model.</summary>
    public void Dispose()
    {
        _processor?.Dispose();
        _model?.Dispose();
        _gate.Dispose();
    }
}
