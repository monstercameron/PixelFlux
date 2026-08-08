using PixelFlux.Ai.Pipeline;
using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;

namespace PixelFlux.Cli;

/// <summary>
/// Measures how much of a search vector should come from the photograph and how much from the
/// description a vision model wrote about it.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="EmbedHandler.ImageWeight"/> is one number that decides how the whole library is
/// searched, and picking it by reasoning about it produces a number that sounds right. This
/// encodes each photograph and each description exactly once, then blends them in memory at every
/// weight and scores the same queries against each — so the answer costs one pass, not one pass per
/// weight, and it is an answer rather than an opinion.
/// </para>
/// <para>
/// Scoring is precision at five against filename prefixes, because the test album names its
/// subjects. That is a crude ground truth and it is the honest one available: it cannot be tuned
/// after the fact, which is exactly the property a benchmark needs.
/// </para>
/// </remarks>
public static class BlendSweep
{
    /// <summary>A query and the filenames a correct answer would contain.</summary>
    /// <param name="Text">What somebody would type.</param>
    /// <param name="Expect">Filename fragments that mark a hit as right.</param>
    private sealed record Probe(string Text, string[] Expect);

    private static readonly Probe[] Probes =
    [
        // Cam's own examples, plus two the image encoder was already good at — a weight that
        // improves the hard queries by ruining the easy ones is not an improvement.
        new("red car", ["_car_", "car."]),
        new("blonde hair", ["vonderleyen"]),
        new("animal", ["_wildlife_", "_pet_", "_animal_"]),
        new("a city street at night", ["_street_", "_night_", "_city_"]),

        // "a beach" was here and has been removed: the album has no beach photographs, so it
        // scored zero at every weight and did nothing but drag the mean down uniformly. A probe
        // that cannot distinguish the thing being measured is not a hard test, it is noise.
    ];

    private static readonly double[] Weights = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];

    /// <summary>Runs the sweep and prints a table.</summary>
    /// <param name="photos">The photo index.</param>
    /// <param name="clip">The CLIP encoders.</param>
    /// <param name="cacheRoot">Derivative cache, where proxies live.</param>
    /// <returns>A process exit code.</returns>
    public static async Task<int> RunAsync(PhotoStore photos, ClipEmbedder clip, string cacheRoot)
    {
        ArgumentNullException.ThrowIfNull(photos);
        ArgumentNullException.ThrowIfNull(clip);

        if (!clip.IsAvailable)
        {
            Console.Error.WriteLine("CLIP is not installed.");
            return 1;
        }

        IReadOnlyList<PhotoRecord> all = await photos.QueryAsync(
            new PhotoQuery { Limit = 100000 }).ConfigureAwait(false);

        var rows = new List<(string Name, float[] Image, float[]? Caption)>();

        Console.Write($"encoding {all.Count} photos ");

        foreach (PhotoRecord photo in all)
        {
            float[]? image;

            try
            {
                image = await clip
                    .EmbedImageAsync(StageSource.For(photo, cacheRoot)).ConfigureAwait(false);
            }
            catch (Exception error) when (error is not OperationCanceledException)
            {
                // The album contains a deliberately truncated JPEG. A benchmark that dies on it
                // measures nothing; leaving it out measures the other 131.
                Console.Write('x');
                continue;
            }

            if (image is null)
            {
                continue;
            }

            float[]? caption = null;

            if (!string.IsNullOrWhiteSpace(photo.AiDescription))
            {
                float[]? total = null;
                int counted = 0;

                foreach (string sentence in EmbedHandler.Sentences(photo.AiDescription)
                             .Take(EmbedHandler.MaximumSentences))
                {
                    if (await clip.EmbedTextAsync(sentence).ConfigureAwait(false) is not { } piece)
                    {
                        continue;
                    }

                    total ??= new float[piece.Length];
                    for (int i = 0; i < piece.Length; i++)
                    {
                        total[i] += piece[i];
                    }

                    counted++;
                }

                caption = counted == 0 ? null : total;
            }

            rows.Add((photo.FileName, image, caption));

            if (rows.Count % 10 == 0)
            {
                Console.Write('.');
            }
        }

        Console.WriteLine();
        Console.WriteLine($"{rows.Count} encoded, {rows.Count(r => r.Caption is not null)} with descriptions");
        Console.WriteLine();

        var queries = new List<(Probe Probe, float[] Vector)>();
        foreach (Probe probe in Probes)
        {
            if (await clip.EmbedQueryAsync(probe.Text).ConfigureAwait(false) is { } vector)
            {
                queries.Add((probe, vector));
            }
        }

        Console.Write($"{"weight",-8}");
        foreach ((Probe probe, _) in queries)
        {
            Console.Write($"{Shorten(probe.Text),12}");
        }

        Console.WriteLine($"{"mean",8}");

        foreach (double weight in Weights)
        {
            Console.Write($"{weight,-8:F1}");
            double total = 0;

            foreach ((Probe probe, float[] query) in queries)
            {
                double score = PrecisionAtFive(rows, query, weight, probe.Expect);
                total += score;
                Console.Write($"{score,12:F2}");
            }

            Console.WriteLine($"{total / queries.Count,8:F3}");
        }

        Console.WriteLine();
        Console.WriteLine($"in use: {EmbedHandler.ImageWeight:F1}");
        return 0;
    }

    private static double PrecisionAtFive(
        List<(string Name, float[] Image, float[]? Caption)> rows,
        float[] query,
        double imageWeight,
        string[] expect)
    {
        List<string> top =
        [
            .. rows
                .Select(row => (row.Name, Score: Dot(query, Mix(row.Image, row.Caption, imageWeight))))
                .OrderByDescending(hit => hit.Score)
                .Take(5)
                .Select(hit => hit.Name),
        ];

        return top.Count(name =>
            expect.Any(fragment => name.Contains(fragment, StringComparison.OrdinalIgnoreCase)))
            / 5.0;
    }

    private static float[] Mix(float[] image, float[]? caption, double imageWeight) =>
        caption is null || imageWeight >= 1
            ? image
            : EmbedHandler.Blend(image, Unit(caption), imageWeight);

    private static float[] Unit(float[] vector)
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

        var unit = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            unit[i] = (float)(vector[i] / length);
        }

        return unit;
    }

    private static double Dot(float[] a, float[] b)
    {
        double sum = 0;
        int length = Math.Min(a.Length, b.Length);

        for (int i = 0; i < length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    private static string Shorten(string text) =>
        text.Length <= 11 ? text : text[..11];
}
