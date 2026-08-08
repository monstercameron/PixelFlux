using PixelFlux.Ai.Faces;
using Xunit.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// Face detection against real photographs of real people.
///
/// The corpus has portrait photographs, street scenes with crowds, and photographs with no
/// people in them at all — which makes both halves checkable: that faces are found where there
/// are faces, and that none are invented where there are none. The second half matters more for
/// this feature than the first, because a faces page full of doorknobs is worse than a short one.
/// </summary>
[Collection(Inference.Name)]
public sealed class FaceDetectionTests
{
    private readonly ITestOutputHelper _output;

    public FaceDetectionTests(ITestOutputHelper output) => _output = output;

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

    private static string? ModelPath
    {
        get
        {
            string path = Path.Combine(RepoRoot, "models", "face_yunet_2023mar.onnx");
            return File.Exists(path) ? path : null;
        }
    }

    private static string[] Album(string pattern) =>
        Directory.GetFiles(Path.Combine(RepoRoot, "testdata", "album"), pattern, SearchOption.AllDirectories)
                 .OrderBy(f => f, StringComparer.Ordinal)
                 .ToArray();

    [Fact]
    public void ModelIsInstalled()
    {
        Assert.True(ModelPath is not null,
            "models/face_yunet_2023mar.onnx is missing; the tests below would prove nothing without it");
    }

    [Fact]
    public async Task FindsFacesInPortraitPhotographs()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);
        string[] portraits = Album("*_face_*.jpg");
        Assert.NotEmpty(portraits);

        int withFaces = 0;

        foreach (string photo in portraits)
        {
            FaceResult result = await detector.DetectAsync(photo);
            _output.WriteLine(
                $"{Path.GetFileName(photo),-58} {result.Faces.Count} face(s)  {result.ElapsedMs} ms");

            foreach (DetectedFace face in result.Faces)
            {
                _output.WriteLine(
                    $"    conf {face.Confidence:0.00}  area {face.AreaFraction:0.0000}  "
                    + $"roll {face.RollDegrees:+0.0;-0.0}°");
            }

            if (result.Faces.Count > 0)
            {
                withFaces++;
            }
        }

        // Every one of these is a photograph whose subject is a person's face. Missing more than
        // one means the detector is not doing its job on the easy case.
        Assert.True(withFaces >= portraits.Length - 1,
            $"only {withFaces} of {portraits.Length} portrait photographs yielded a face");
    }

    [Fact]
    public async Task DoesNotInventFacesInPhotographsWithoutPeople()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);

        // Doors, flowers and close-up food. A false positive here is what would fill the faces
        // page with pictures of doorknobs.
        string[] noPeople = Album("*_door_*.jpg").Concat(Album("*_flowers_*.jpg")).ToArray();
        Assert.NotEmpty(noPeople);

        var spurious = new List<string>();

        foreach (string photo in noPeople)
        {
            FaceResult result = await detector.DetectAsync(photo);
            if (result.Faces.Count > 0)
            {
                spurious.Add($"{Path.GetFileName(photo)}: {result.Faces.Count} "
                           + $"(max conf {result.Faces.Max(f => f.Confidence):0.00})");
            }
        }

        _output.WriteLine(spurious.Count == 0
            ? "no false positives"
            : "false positives:\n  " + string.Join("\n  ", spurious));

        Assert.True(spurious.Count == 0, string.Join("; ", spurious));
    }

    [Fact]
    public async Task GeometryStaysOnTheImage()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);

        foreach (string photo in Album("*.jpg").Take(20))
        {
            FaceResult result = await detector.DetectAsync(photo);

            foreach (DetectedFace face in result.Faces)
            {
                Assert.InRange(face.X, -0.05, 1);
                Assert.InRange(face.Y, -0.05, 1);
                Assert.InRange(face.Width, 0, 1.05);
                Assert.InRange(face.Height, 0, 1.05);
                Assert.InRange(face.Confidence, 0, 1);

                // Five landmarks, all on or near the picture. These are what a recognition model
                // will use to align a face later; garbage here would be invisible until then.
                Assert.Equal(5, face.Landmarks.Count);
                Assert.All(face.Landmarks, p =>
                {
                    Assert.InRange(p.X, -0.1, 1.1);
                    Assert.InRange(p.Y, -0.1, 1.1);
                });

                // A face is roughly as tall as it is wide. Wildly non-square boxes mean the
                // exponential size decoding has gone wrong.
                double ratio = face.Width / face.Height;
                Assert.InRange(ratio, 0.4, 2.2);
            }
        }
    }

    /// <summary>
    /// The landmarks must land inside the box, in the arrangement of an actual face.
    /// </summary>
    /// <remarks>
    /// This is the test that catches a mis-decoded box, and it exists because the first version
    /// read YuNet's predicted point as a top-left corner when it is a centre. Every box came out
    /// shifted down and right by half its size — which passed a bounds check, passed suppression,
    /// and scored identically. Only the crops gave it away, and only by looking at them.
    ///
    /// Landmarks are decoded by a different path from boxes, so they are an independent witness:
    /// if the two disagree about where the face is, one of them is wrong.
    /// </remarks>
    [Fact]
    public async Task LandmarksAgreeWithTheBox()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);
        int checked_ = 0;

        foreach (string photo in Album("*face_*.jpg").Concat(Album("*_street_*.jpg")))
        {
            foreach (DetectedFace face in (await detector.DetectAsync(photo)).Faces)
            {
                (double eyeRightX, double eyeRightY) = face.Landmarks[0];
                (double eyeLeftX, double eyeLeftY) = face.Landmarks[1];
                (double noseX, double noseY) = face.Landmarks[2];
                (double mouthX, double mouthY) = face.Landmarks[3];

                // Both eyes inside the box.
                foreach ((double x, double y) in new[] { (eyeRightX, eyeRightY), (eyeLeftX, eyeLeftY) })
                {
                    Assert.InRange(x, face.X, face.X + face.Width);
                    Assert.InRange(y, face.Y, face.Y + face.Height);
                }

                // Eyes above the middle, mouth below it. A half-box shift puts the eyes at or
                // above the top edge instead.
                double middle = face.Y + (face.Height / 2);
                Assert.True(eyeRightY < middle && eyeLeftY < middle,
                    $"eyes at {eyeRightY:0.000}/{eyeLeftY:0.000} are not above the box middle {middle:0.000}");
                Assert.True(mouthY > middle,
                    $"mouth at {mouthY:0.000} is not below the box middle {middle:0.000}");

                // The nose sits between the eyes horizontally and below them vertically.
                Assert.InRange(noseX, Math.Min(eyeRightX, eyeLeftX) - 0.02, Math.Max(eyeRightX, eyeLeftX) + 0.02);
                Assert.True(noseY > Math.Min(eyeRightY, eyeLeftY), "the nose is above both eyes");

                checked_++;
            }
        }

        Assert.True(checked_ > 0, "no faces to check");
        _output.WriteLine($"{checked_} faces checked");
    }

    [Fact]
    public async Task FindsSeveralFacesInACrowd()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);
        string[] streets = Album("*_street_*.jpg").Concat(Album("*_car_*.jpg")).ToArray();

        int best = 0;
        foreach (string photo in streets)
        {
            FaceResult result = await detector.DetectAsync(photo);
            _output.WriteLine($"{Path.GetFileName(photo),-58} {result.Faces.Count}");
            best = Math.Max(best, result.Faces.Count);
        }

        // At least one of these has a group of people in it. Finding exactly one face in every
        // crowd scene would mean the stride decoding only works at one scale.
        Assert.True(best >= 2, $"the busiest scene yielded only {best} face(s)");
    }

    [Fact]
    public async Task MissingModelIsAnOrdinaryStateNotACrash()
    {
        using var detector = new YuNetFaceDetector(Path.Combine(Path.GetTempPath(), "no-face-model.onnx"));

        Assert.False(detector.IsAvailable);
        FaceResult result = await detector.DetectAsync(Album("*.jpg")[0]);

        Assert.Empty(result.Faces);
        Assert.Equal("none", result.ModelVersion);
    }

    [Fact]
    public async Task DetectionIsFastEnoughToSweepALibrary()
    {
        if (ModelPath is null) { return; }

        using var detector = new YuNetFaceDetector(ModelPath);
        string[] photos = Album("*.jpg").Take(8).ToArray();

        await detector.DetectAsync(photos[0]);   // warm up the graph

        var watch = System.Diagnostics.Stopwatch.StartNew();
        foreach (string photo in photos)
        {
            await detector.DetectAsync(photo);
        }

        watch.Stop();
        double per = watch.ElapsedMilliseconds / (double)photos.Length;
        _output.WriteLine($"{per:0} ms per photograph");

        Assert.True(per < 600, $"{per:0} ms per photograph is too slow for a library sweep");
    }

    /// <summary>
    /// Sweeps the confidence threshold over the whole album and reports what each value costs.
    /// </summary>
    /// <remarks>
    /// Not an assertion about a number — it is the evidence behind
    /// <see cref="YuNetFaceDetector.DefaultConfidenceThreshold" />. The album is split into
    /// photographs that contain people and photographs that do not, so a sweep shows recall and
    /// false positives side by side rather than one in isolation. Re-run it if the model or the
    /// corpus changes; the guard below only catches the case where the shipped default has become
    /// strictly worse than a looser one.
    /// </remarks>
    [Fact]
    public async Task ThresholdIsCalibratedAgainstTheAlbum()
    {
        if (ModelPath is null) { return; }

        // "*face*" would also match nothing useful and "*_face_*" misses the duplicate, whose
        // name is "duplicate-of-face_..." — it is the same photograph of the same people.
        string[] people = Album("*face_*.jpg").Concat(Album("*_street_*.jpg")).ToArray();
        string[] noPeople = Album("*.jpg").Except(people)
                                          .Except(Album("*corrupt*.jpg"))
                                          .ToArray();

        _output.WriteLine("threshold   faces in people-photos   false positives");

        foreach (float threshold in new[] { 0.9f, 0.8f, 0.75f, 0.6f, 0.5f, 0.4f })
        {
            using var detector = new YuNetFaceDetector(ModelPath, confidenceThreshold: threshold);

            int found = 0;
            foreach (string photo in people)
            {
                found += (await detector.DetectAsync(photo)).Faces.Count;
            }

            var spurious = new List<string>();
            foreach (string photo in noPeople)
            {
                int n = (await detector.DetectAsync(photo)).Faces.Count;
                if (n > 0)
                {
                    spurious.Add($"{Path.GetFileName(photo)}x{n}");
                }
            }

            _output.WriteLine($"{threshold:0.00}        {found,3}                      "
                            + $"{spurious.Count,2}  {string.Join(" ", spurious)}");
        }
    }
}
