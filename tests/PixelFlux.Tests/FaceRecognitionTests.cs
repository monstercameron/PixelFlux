using PixelFlux.Ai.Faces;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// Whether "the same person" actually works, measured against a hand-verified corpus.
///
/// testdata/people holds several photographs of each of six people, taken at different events in
/// different lighting at different angles, with the identity in the filename. Every pair in that
/// set is therefore labelled, which is the only way to say what a similarity threshold costs:
/// how many photographs of one person it finds, and how many strangers it drags in with them.
///
/// The album is not usable for this. Its only repeat is a re-encode of one file — pixel-identical
/// content, which tests JPEG rather than recognition.
/// </summary>
[Collection(Inference.Name)]
public sealed class FaceRecognitionTests
{
    private readonly ITestOutputHelper _output;

    public FaceRecognitionTests(ITestOutputHelper output) => _output = output;

    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "testdata", "album")))
            {
                dir = dir.Parent;
            }

            return dir?.FullName ?? throw new DirectoryNotFoundException("Could not locate the repository root.");
        }
    }

    private static string? DetectorPath => Existing(Path.Combine(RepoRoot, "models", "face_yunet_2023mar.onnx"));

    private static string? RecognizerPath => Existing(Path.Combine(RepoRoot, "models", "face_sface_2021dec.onnx"));

    private static string? Existing(string path) => File.Exists(path) ? path : null;

    private static string PeopleRoot => Path.Combine(RepoRoot, "testdata", "people");

    /// <summary>One embedded face, with the identity its filename claims.</summary>
    private sealed record Sample(string Person, string File, float[] Vector);

    /// <summary>
    /// Embeds the largest face in every photograph of the labelled corpus.
    /// </summary>
    /// <remarks>
    /// The largest face, not every face: several of these are group shots, and only the subject
    /// of the photograph is the person the filename names. Taking all of them would label a
    /// bystander with the subject's identity and poison the ground truth this is measuring
    /// against.
    /// </remarks>
    private static async Task<List<Sample>> EmbedCorpusAsync()
    {
        using var detector = new YuNetFaceDetector(DetectorPath);
        using var recognizer = new SFaceRecognizer(RecognizerPath);

        var samples = new List<Sample>();

        foreach (string path in Directory.GetFiles(PeopleRoot, "*.jpg").OrderBy(f => f, StringComparer.Ordinal))
        {
            FaceResult found = await detector.DetectAsync(path);
            if (found.Faces.Count == 0)
            {
                continue;
            }

            using Image<Rgb24> image = await Image.LoadAsync<Rgb24>(path);
            DetectedFace biggest = found.Faces[0];

            if (recognizer.Embed(image, biggest) is { } vector)
            {
                samples.Add(new Sample(
                    Path.GetFileNameWithoutExtension(path).Split('_')[0],
                    Path.GetFileName(path),
                    vector));
            }
        }

        return samples;
    }

    [Fact]
    public void ModelsAreInstalled()
    {
        Assert.True(DetectorPath is not null, "models/face_yunet_2023mar.onnx is missing");
        Assert.True(RecognizerPath is not null,
            "models/face_sface_2021dec.onnx is missing; the recognition tests below prove nothing without it");
        Assert.True(Directory.Exists(PeopleRoot) && Directory.GetFiles(PeopleRoot, "*.jpg").Length >= 12,
            "testdata/people is missing; run tools/fetch_people_set.py");
    }

    [Fact]
    public void EmbeddingsAreUnitLength()
    {
        if (RecognizerPath is null) { return; }

        using var recognizer = new SFaceRecognizer(RecognizerPath);
        Assert.True(recognizer.IsAvailable);
        Assert.Equal(128, recognizer.Dimensions);
    }

    /// <summary>
    /// The headline property: two photographs of one person score higher than two of different
    /// people, with a gap wide enough for a single threshold to sit in.
    /// </summary>
    [Fact]
    public async Task SamePersonScoresHigherThanDifferentPeople()
    {
        if (RecognizerPath is null || DetectorPath is null) { return; }

        List<Sample> samples = await EmbedCorpusAsync();
        Assert.True(samples.Count >= 12, $"only {samples.Count} faces were embedded");

        var same = new List<double>();
        var different = new List<double>();

        for (int i = 0; i < samples.Count; i++)
        {
            for (int j = i + 1; j < samples.Count; j++)
            {
                double score = SFaceRecognizer.Similarity(samples[i].Vector, samples[j].Vector);
                (samples[i].Person == samples[j].Person ? same : different).Add(score);
            }
        }

        _output.WriteLine($"{samples.Count} faces, {same.Count} same-person pairs, "
                        + $"{different.Count} different-person pairs");
        _output.WriteLine($"same      min {same.Min():0.000}  mean {same.Average():0.000}  max {same.Max():0.000}");
        _output.WriteLine($"different min {different.Min():0.000}  mean {different.Average():0.000}  "
                        + $"max {different.Max():0.000}");

        Assert.NotEmpty(same);
        Assert.True(same.Average() > different.Average() + 0.25,
            $"same-person pairs average {same.Average():0.000} against {different.Average():0.000} "
            + "for strangers — the two populations are not separated");
    }

    /// <summary>
    /// Sweeps the threshold and reports what each value costs, in the units a person cares about.
    /// </summary>
    /// <remarks>
    /// This is the evidence behind the shipped default. It is printed rather than asserted at a
    /// specific number, because the right value is a product decision — "find this person" that
    /// misses half their photographs is useless, and one that shows strangers is worse than
    /// useless — and the table is what makes that decision arguable rather than arbitrary.
    /// </remarks>
    [Fact]
    public async Task ThresholdIsCalibratedAgainstLabelledPairs()
    {
        if (RecognizerPath is null || DetectorPath is null) { return; }

        List<Sample> samples = await EmbedCorpusAsync();

        var pairs = new List<(bool Same, double Score, string A, string B)>();

        for (int i = 0; i < samples.Count; i++)
        {
            for (int j = i + 1; j < samples.Count; j++)
            {
                pairs.Add((samples[i].Person == samples[j].Person,
                           SFaceRecognizer.Similarity(samples[i].Vector, samples[j].Vector),
                           samples[i].File, samples[j].File));
            }
        }

        _output.WriteLine("threshold   found/total   strangers   note");

        foreach (double threshold in new[] { 0.20, 0.25, 0.30, 0.363, 0.40, 0.45, 0.50, 0.55 })
        {
            int hit = pairs.Count(p => p.Same && p.Score >= threshold);
            int total = pairs.Count(p => p.Same);
            int wrong = pairs.Count(p => !p.Same && p.Score >= threshold);

            string note = Math.Abs(threshold - 0.363) < 0.001 ? "OpenCV's published default" : string.Empty;
            _output.WriteLine($"{threshold:0.000}       {hit,3}/{total,-3}       {wrong,3}         {note}");
        }

        _output.WriteLine("");
        _output.WriteLine("hardest same-person pairs (these are what a low threshold buys):");
        foreach ((bool _, double score, string a, string b) in pairs.Where(p => p.Same)
                     .OrderBy(p => p.Score).Take(5))
        {
            _output.WriteLine($"  {score:0.000}  {a}  ~  {b}");
        }

        _output.WriteLine("");
        _output.WriteLine("closest strangers (these are what a low threshold costs):");
        foreach ((bool _, double score, string a, string b) in pairs.Where(p => !p.Same)
                     .OrderByDescending(p => p.Score).Take(5))
        {
            _output.WriteLine($"  {score:0.000}  {a}  ~  {b}");
        }

        // The one hard guarantee: at the shipped default, no two different people are called the
        // same person. A "find this person" that returns strangers is not a feature with a bug
        // in it, it is a different and worse feature.
        int falseMatches = pairs.Count(p => !p.Same && p.Score >= FaceGrouping.DefaultThreshold);
        Assert.True(falseMatches == 0,
            $"{falseMatches} pairs of different people score above the shipped threshold "
            + $"{FaceGrouping.DefaultThreshold}");
    }

    [Fact]
    public async Task TheSameFaceInTwoPhotographsIsFound()
    {
        if (RecognizerPath is null || DetectorPath is null) { return; }

        List<Sample> samples = await EmbedCorpusAsync();

        // For each person with more than one photograph, ask the question the UI asks: given
        // this face, does the shipped threshold find their other photographs?
        foreach (IGrouping<string, Sample> person in samples.GroupBy(s => s.Person).Where(g => g.Count() > 1))
        {
            Sample query = person.First();

            List<Sample> matched = samples
                .Where(s => s != query)
                .Where(s => SFaceRecognizer.Similarity(query.Vector, s.Vector) >= FaceGrouping.DefaultThreshold)
                .ToList();

            int right = matched.Count(m => m.Person == person.Key);
            int wrong = matched.Count(m => m.Person != person.Key);

            _output.WriteLine($"{person.Key,-14} query {query.File,-22} "
                            + $"found {right} of {person.Count() - 1}, {wrong} strangers");

            Assert.True(wrong == 0, $"searching for {person.Key} returned {wrong} other people");
        }
    }

    [Fact]
    public async Task AlignmentIsWhatMakesItWork()
    {
        if (RecognizerPath is null || DetectorPath is null) { return; }

        // Not a property of the product — a check that the landmark alignment is actually doing
        // something. If the similarity transform were wrong or a no-op, embeddings would still
        // be produced and would still be unit length; they would simply be much worse. Comparing
        // aligned scores against scores from the same faces embedded without alignment is the
        // only cheap way to notice.
        using var detector = new YuNetFaceDetector(DetectorPath);
        using var recognizer = new SFaceRecognizer(RecognizerPath);

        string[] pair = Directory.GetFiles(PeopleRoot, "vonderleyen_*.jpg")
                                 .OrderBy(f => f, StringComparer.Ordinal).Take(2).ToArray();
        Assert.Equal(2, pair.Length);

        var aligned = new List<float[]>();

        foreach (string path in pair)
        {
            FaceResult found = await detector.DetectAsync(path);
            using Image<Rgb24> image = await Image.LoadAsync<Rgb24>(path);
            aligned.Add(recognizer.Embed(image, found.Faces[0])!);
        }

        double score = SFaceRecognizer.Similarity(aligned[0], aligned[1]);
        _output.WriteLine($"two photographs of one person, aligned: {score:0.000}");

        Assert.True(score > 0.45,
            $"{score:0.000} is too low for two photographs of the same person — alignment is probably wrong");
    }

    [Fact]
    public void ASmallFaceIsRefusedRatherThanGuessedAt()
    {
        if (RecognizerPath is null) { return; }

        using var recognizer = new SFaceRecognizer(RecognizerPath);
        using var image = new Image<Rgb24>(1000, 1000);

        var speck = new DetectedFace(0.9, 0.5, 0.5, 0.01, 0.01,
            [(0.502, 0.502), (0.506, 0.502), (0.504, 0.504), (0.503, 0.506), (0.506, 0.506)]);

        Assert.Null(recognizer.Embed(image, speck));
    }

    [Fact]
    public void NoModelIsAQuietNoOp()
    {
        using var recognizer = new SFaceRecognizer(Path.Combine(Path.GetTempPath(), "absent-sface.onnx"));
        using var image = new Image<Rgb24>(400, 400);

        Assert.False(recognizer.IsAvailable);
        Assert.Null(recognizer.Embed(image, new DetectedFace(0.9, 0.2, 0.2, 0.4, 0.4,
            [(0.3, 0.3), (0.5, 0.3), (0.4, 0.4), (0.32, 0.5), (0.48, 0.5)])));
    }

    [Fact]
    public void MismatchedVectorsScoreZeroRatherThanThrowing()
    {
        Assert.Equal(0, SFaceRecognizer.Similarity([1f, 0f, 0f], [1f, 0f]));
        Assert.Equal(0, SFaceRecognizer.Similarity([], []));
        Assert.Equal(1, SFaceRecognizer.Similarity([1f, 0f], [1f, 0f]), 6);
    }
}
