using System.Globalization;
using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using PixelFlux.Core.Pipeline;
using PixelFlux.Core.Search;

namespace PixelFlux.Ai.Pipeline;

/// <summary>
/// The stage that turns a photograph — and what the vision model said about it — into one search
/// vector.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why the description is in here.</b> CLIP's image encoder is good at gist and weak at
/// specifics: it reliably separates a beach from a boardroom and unreliably separates blonde hair
/// from brown. Its text encoder, given the sentence "a woman with blonde hair standing at a
/// lectern", places that sentence exactly where the query "blonde hair" is looking. The vision
/// model has already written that sentence for every photograph, so the words are sitting there
/// unused — folding them into the vector costs one text encode and buys the precision the image
/// encoder does not have.
/// </para>
/// <para>
/// <b>Why blend rather than store two vectors.</b> Two vectors means two searches and a rule for
/// combining their scores, and that rule would have to be tuned per query — captions win on
/// "blonde hair", images win on "sunset". One blended vector puts the tuning in one number, applied
/// once, offline, where it can be measured. It also keeps the search path exactly as it was.
/// </para>
/// <para>
/// <b>Why sentences, not the whole paragraph.</b> CLIP's text encoder takes 77 tokens and the
/// descriptions run to about 200. Truncating throws away most of the detail — which is the entire
/// point of having a description. Embedding each sentence and averaging keeps all of it, at the
/// cost of a few more encoder passes that take milliseconds each.
/// </para>
/// </remarks>
public sealed class EmbedHandler : IStageHandler
{
    /// <summary>
    /// How much of the blended vector comes from the image rather than from the description.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measured, not chosen — <c>pixelflux pipeline sweep</c> encodes every photograph and every
    /// description once, blends them at each weight, and scores the same queries against each.
    /// Precision at five over the test album:
    /// </para>
    /// <code>
    /// weight    red car   animal
    /// 0.0          0.80     0.00     captions only
    /// 0.2          1.00     0.00
    /// 0.6          1.00     0.20
    /// 0.8          1.00     0.40
    /// 1.0          0.80     0.60     image only
    /// </code>
    /// <para>
    /// The two ends both lose, which is the finding: "red car" needs the description, because the
    /// image encoder knows a car and not a red one, and "animal" is hurt by it, because the
    /// descriptions of the wildlife photographs talk about lagoons and light rather than saying
    /// the word. 0.8 is where the query that needs captions is fully served and the query that
    /// dislikes them has given up least.
    /// </para>
    /// <para>
    /// Worth knowing how thin this is. Five queries went in and three could not tell the weights
    /// apart — one because the album has no photographs matching it at all. Two discriminating
    /// queries is enough to reject the extremes and not enough to distinguish 0.7 from 0.8; the
    /// tie was broken by argument rather than evidence, on the grounds that a description is a
    /// model's opinion and can be confidently wrong, whereas the image encoder is merely vague.
    /// </para>
    /// </remarks>
    public const double ImageWeight = 0.8;

    /// <summary>How many sentences of a description are folded in.</summary>
    /// <remarks>
    /// Descriptions front-load: the first sentences say what the photograph is of and the last
    /// ones speculate about mood and intent. Eight covers the substance of every description
    /// measured and cuts the tail where the model starts editorialising.
    /// </remarks>
    public const int MaximumSentences = 8;

    private readonly PhotoStore _photos;
    private readonly VectorIndex _vectors;
    private readonly IImageTextEmbedder _embedder;
    private readonly string _cacheRoot;

    /// <summary>Creates the handler.</summary>
    /// <param name="photos">The photo index.</param>
    /// <param name="vectors">Where vectors are stored.</param>
    /// <param name="embedder">The CLIP encoders.</param>
    /// <param name="cacheRoot">Derivative cache, where proxies live.</param>
    public EmbedHandler(
        PhotoStore photos,
        VectorIndex vectors,
        IImageTextEmbedder embedder,
        string cacheRoot)
    {
        ArgumentNullException.ThrowIfNull(photos);
        ArgumentNullException.ThrowIfNull(vectors);
        ArgumentNullException.ThrowIfNull(embedder);

        _photos = photos;
        _vectors = vectors;
        _embedder = embedder;
        _cacheRoot = cacheRoot;
    }

    /// <inheritdoc/>
    public PipelineStage Stage => PipelineStage.Embed;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Suffixed, because a vector blended with a caption is not the vector the plain encoder used
    /// to produce, and vectors from two recipes are not comparable. The suffix is what makes every
    /// existing vector outstanding the first time this runs, without a migration.
    /// </para>
    /// <para>
    /// <b>The weight is in the suffix, and it has to be.</b> Without it, changing
    /// <see cref="ImageWeight"/> and requeueing does nothing at all: the queue asks the cache for
    /// this photograph at this version, the cache still has the vector computed at the old weight,
    /// and hands it straight back. That happened — a whole library re-embedded in a second, every
    /// vector unchanged, no error anywhere. The rule the cache enforces is that the version names
    /// the recipe, and the weight is part of the recipe.
    /// </para>
    /// </remarks>
    public string? ModelVersion => _embedder.IsAvailable
        ? $"{_embedder.ModelVersion}+cap{ImageWeight:F2}"
        : null;

    /// <inheritdoc/>
    public async Task ApplyAsync(long photoId, string payload, CancellationToken cancellationToken)
    {
        float[] vector = ReadVector(payload);
        if (vector.Length == 0)
        {
            return;
        }

        await _vectors.StoreAsync(photoId, ModelVersion!, vector, cancellationToken)
            .ConfigureAwait(false);
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

        float[]? image = await _embedder
            .EmbedImageAsync(StageSource.For(photo, _cacheRoot), cancellationToken)
            .ConfigureAwait(false);

        if (image is null)
        {
            return null;
        }

        float[]? caption = await EmbedDescriptionAsync(photo, cancellationToken)
            .ConfigureAwait(false);

        // No description — the vision model is not installed, or it declined this photograph — so
        // the vector is the image alone. Still stored under the blended model version: a
        // photograph that has been through this stage has been through it, and re-running would
        // produce the same answer until a description appears. When one does, the description
        // stage completing does not on its own requeue this — see PipelineRunner's ordering, and
        // the `pipeline redo embed` command for the deliberate re-run.
        float[] vector = caption is null ? image : Blend(image, caption, ImageWeight);

        await _vectors.StoreAsync(photoId, ModelVersion!, vector, cancellationToken)
            .ConfigureAwait(false);

        return WriteVector(vector);
    }

    /// <summary>Averages the sentences of a photograph's description into one vector.</summary>
    /// <param name="photo">The photograph.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>A unit-length vector, or null when there is no usable description.</returns>
    private async Task<float[]?> EmbedDescriptionAsync(
        PhotoRecord photo,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(photo.AiDescription))
        {
            return null;
        }

        float[]? total = null;
        int counted = 0;

        foreach (string sentence in Sentences(photo.AiDescription).Take(MaximumSentences))
        {
            float[]? piece = await _embedder.EmbedTextAsync(sentence, cancellationToken)
                .ConfigureAwait(false);
            if (piece is null)
            {
                continue;
            }

            total ??= new float[piece.Length];
            for (int i = 0; i < piece.Length && i < total.Length; i++)
            {
                total[i] += piece[i];
            }

            counted++;
        }

        return counted == 0 ? null : Normalise(total!);
    }

    /// <summary>Splits a description into sentences worth embedding.</summary>
    /// <param name="text">The description.</param>
    /// <returns>Trimmed sentences, skipping fragments too short to carry meaning.</returns>
    /// <remarks>
    /// Deliberately crude. A real sentence splitter would handle "Dr." and "U.S." correctly, and
    /// getting those wrong here costs nothing — the pieces are averaged, so a sentence broken in
    /// the wrong place contributes very nearly the same vector as one broken in the right place.
    /// </remarks>
    public static IEnumerable<string> Sentences(string text) =>
        text.Split(['.', '!', '?', '\n'], StringSplitOptions.RemoveEmptyEntries)
            .Select(part => part.Trim())
            .Where(part => part.Length >= 12);

    /// <summary>Mixes two unit vectors and renormalises.</summary>
    /// <param name="image">The image vector.</param>
    /// <param name="caption">The description vector.</param>
    /// <param name="imageWeight">Share of the result taken from the image, 0 to 1.</param>
    /// <returns>A unit-length vector in the same space as both inputs.</returns>
    /// <remarks>
    /// Renormalising matters. Search ranks by cosine similarity, which is a dot product only when
    /// both sides are unit length; a blended vector is shorter than either input — two unit vectors
    /// thirty degrees apart average to about 0.97 — and skipping this step would quietly rank every
    /// photograph whose description agrees with its image below one whose description does not.
    /// </remarks>
    public static float[] Blend(float[] image, float[] caption, double imageWeight)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(caption);

        int length = Math.Min(image.Length, caption.Length);
        var mixed = new float[length];

        for (int i = 0; i < length; i++)
        {
            mixed[i] = (float)((image[i] * imageWeight) + (caption[i] * (1 - imageWeight)));
        }

        return Normalise(mixed);
    }

    private static float[] Normalise(float[] vector)
    {
        double sum = 0;
        foreach (float value in vector)
        {
            sum += value * value;
        }

        double length = Math.Sqrt(sum);
        if (length <= 1e-9)
        {
            return vector;
        }

        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = (float)(vector[i] / length);
        }

        return vector;
    }

    /// <summary>Renders a vector for the cache.</summary>
    /// <remarks>
    /// Plain text rather than JSON or base64. A 512-float vector is the largest payload any stage
    /// produces and this is the format that stays legible in a database viewer, which has been
    /// worth more than the bytes it costs every time something has gone wrong with vectors.
    /// </remarks>
    private static string WriteVector(float[] vector) =>
        string.Join(' ', vector.Select(v => v.ToString("R", CultureInfo.InvariantCulture)));

    private static float[] ReadVector(string payload)
    {
        string[] parts = payload.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var vector = new float[parts.Length];

        for (int i = 0; i < parts.Length; i++)
        {
            if (!float.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture,
                    out vector[i]))
            {
                return [];
            }
        }

        return vector;
    }
}
