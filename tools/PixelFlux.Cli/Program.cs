using System.Globalization;
using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;
using PixelFlux.Ai.Compute;
using PixelFlux.Ai.Faces;
using PixelFlux.Ai.Segmentation;
using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Search;

namespace PixelFlux.Cli;

/// <summary>Headless driver for a PixelFlux library.</summary>
internal static class Program
{
    /// <summary>A bare carriage return, for rewriting the progress line in place.</summary>
    private const string CarriageReturn = "\r";

    private static string LibraryRoot =>
        Environment.GetEnvironmentVariable("PIXELFLUX_HOME")
        ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "PixelFlux");

    private static async Task<int> Main(string[] args)
    {
        // Windows consoles still default to a legacy codepage, which turns every non-ASCII
        // character in this output — star ratings, arrows, and a great many real filenames —
        // into question marks. Photo libraries are full of accented filenames; a tool that
        // cannot print them is not usable for looking at one.
        try
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
        }
        catch (IOException)
        {
            // Redirected to a pipe that rejects the change. Cosmetic; carry on.
        }

        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Usage();
            return args.Length == 0 ? 1 : 0;
        }

        string root = LibraryRoot;
        Directory.CreateDirectory(root);

        var database = new PhotoDatabase(Path.Combine(root, "library.db"));
        database.Migrate();

        var store = new PhotoStore(database);
        var vectors = new VectorIndex(database);
        var collections = new CollectionStore(database);
        var segments = new SegmentStore(database);
        var faces = new FaceStore(database);
        var derivatives = new DerivativeGenerator(Path.Combine(root, "cache"));

        try
        {
            return args[0] switch
            {
                "import" => await ImportAsync(store, derivatives, args),
                "search" => await SearchAsync(store, vectors, args),
                "sort" => await SortAsync(store, args),
                "facets" => await FacetsAsync(store),
                "stats" => await StatsAsync(store, root),
                "vocab" => await VocabAsync(store, args),
                "dupes" => await DupesAsync(store),
                "analyze" or "analyse" => await AnalyseAsync(store, segments, root, args),
                "objects" => await ObjectsAsync(segments),
                "faces" => await FacesAsync(store, faces, root, args),
                "facecheck" => await FaceCheckAsync(root, args),
                "name" => await NameAsync(faces, args),
                "who" => await WhoAsync(faces, args),
                "people" => await PeopleAsync(faces, args),
                "embed" => await EmbedAsync(store, vectors, root, args),
                "vsearch" => await VectorSearchAsync(store, vectors, root, args),
                "find" => await FindAsync(store, vectors, root, args),
                "describe" => await DescribeAsync(store, root, args),
                "pipeline" => await PipelineCommand.RunAsync(database, root, FindRepoRoot(), args),
                "accel" => AccelCommand.Run(root, FindRepoRoot()),
                "albums" => await AlbumsAsync(collections, args),
                _ => Unknown(args[0]),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            return 1;
        }
    }

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"unknown command '{command}'");
        Usage();
        return 1;
    }

    private static void Usage() => Console.WriteLine("""
        pixelflux — headless driver for a PixelFlux library

          import <folder>...            index every image under these folders
          search <text> [options]       search the library
          sort <order> [--limit N]      list photos in a given order
          facets                        browsable dimensions and their counts
          stats                         library summary
          vocab [prefix]                indexed vocabulary, or fuzzy suggestions for a prefix
          dupes                         near-duplicate groups
          analyze [--limit N]           run segmentation over photos waiting to be analysed
          objects                       detected objects and how many photos contain each
          faces [--limit N] [--reset]   sweep for faces, then list what is on the faces page
          facecheck <folder> [--accel cpu|gpu|auto]
                                        report faces per image; for vetting a test corpus
          name <face-id> <name>         say who a face is; omit the name to clear it
          name --list                   everybody who has been named
          who <face-id> [--at 0.40]     photos containing the person in that face
          people [--at 0.40]            faces collapsed to one entry per person
          embed [--limit N]             describe photos with CLIP so they can be searched by meaning
          vsearch <phrase> [--limit N]  search by meaning, not by words
          find <phrase>                 the app's own search: words and meaning blended
          describe [--limit N] [--redo]  write a description of each photo with the vision model
          accel                         what hardware models can run on
          pipeline status               what each analysis stage has left to do
          pipeline run [--limit N] [--gap S]
                                        work the queue: describe, segment, faces, embed — in order
          pipeline redo <stage>         mark one stage outstanding again for every photo
          pipeline reset                mark every stage outstanding again
          albums list
          albums new <name>
          albums add <name> <photo-id>...

        search options
          --camera <text>      --from <yyyy-mm-dd>    --to <yyyy-mm-dd>
          --folder <path>      --tag <tag>            --min-rating <0-5>
          --order <order>      --limit <n>
          --bbox <s,w,n,e>     bounding box in decimal degrees

        orders
          captured-desc captured-asc indexed-desc filename
          rating size camera folder shuffle

        The library lives at %LOCALAPPDATA%\\PixelFlux, or $PIXELFLUX_HOME if set.
        """);

    // ---------------------------------------------------------------------------- import

    private static async Task<int> ImportAsync(PhotoStore store, DerivativeGenerator derivatives, string[] args)
    {
        string[] folders = args.Skip(1).Where(a => !a.StartsWith('-')).ToArray();
        if (folders.Length == 0)
        {
            Console.Error.WriteLine("import needs at least one folder");
            return 1;
        }

        var ingestor = new LibraryIngestor(store, derivatives);

        // Progress on one rewritten line rather than a scrolling log: an import of 50,000 files
        // should not produce 50,000 lines of output.
        var lastReport = 0L;
        var progress = new Progress<IngestProgress>(p =>
        {
            long now = Environment.TickCount64;
            if (now - Interlocked.Read(ref lastReport) < 120 && p.Processed < p.Discovered)
            {
                return;
            }

            Interlocked.Exchange(ref lastReport, now);
            Console.Write($"\r  {p.Processed}/{p.Discovered}  +{p.Imported} new  "
                        + $"{p.Duplicates} known  {p.Failed} failed   ");
        });

        IngestResult result = await ingestor.ImportAsync(folders, progress);
        Console.WriteLine();
        Console.WriteLine($"imported {result.Imported}, {result.Duplicates} already known, "
                        + $"{result.Failed} failed, in {result.Elapsed.TotalSeconds:0.0}s");

        foreach ((string path, string reason) in result.Failures.Take(10))
        {
            Console.WriteLine($"  failed: {Path.GetFileName(path)} — {reason}");
        }

        return 0;
    }

    // ---------------------------------------------------------------------------- search

    private static async Task<int> SearchAsync(PhotoStore store, VectorIndex vectors, string[] args)
    {
        string text = string.Join(' ', args.Skip(1).TakeWhile(a => !a.StartsWith("--", StringComparison.Ordinal)));
        var flags = ParseFlags(args);

        var query = new PhotoQuery
        {
            Text = string.IsNullOrWhiteSpace(text) ? null : text,
            CameraModel = flags.GetValueOrDefault("camera"),
            SourceFolder = flags.GetValueOrDefault("folder"),
            From = ParseDate(flags.GetValueOrDefault("from")),
            To = ParseDate(flags.GetValueOrDefault("to")),
            Tags = flags.TryGetValue("tag", out string? tag) ? [tag] : [],
            MinRating = int.TryParse(flags.GetValueOrDefault("min-rating"), out int r) ? r : null,
            Bounds = ParseBox(flags.GetValueOrDefault("bbox")),
            Order = ParseOrder(flags.GetValueOrDefault("order")),
            Limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 25,
        };

        var engine = new SearchEngine(store, vectors);
        SearchResult result = await engine.SearchAsync(query);

        if (result.Corrections.Count > 0)
        {
            Console.WriteLine("did you mean: " + string.Join(", ",
                result.Corrections.Select(c => $"{c.Typed} -> {c.Used}")));
        }

        Console.WriteLine($"{result.TotalMatched} matched"
                        + (result.UsedSemantic ? " (semantic search contributed)" : string.Empty));
        Console.WriteLine();

        foreach (SearchHit hit in result.Hits)
        {
            PhotoRecord p = hit.Photo;
            string when = p.CapturedUtc.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
            string camera = p.Camera.Model ?? "—";
            string where = p.Location is { } loc
                ? $"{loc.Latitude:0.00},{loc.Longitude:0.00}"
                : "—";

            Console.WriteLine($"  {hit.Score,5:0.00}  {when}  {Truncate(p.FileName, 46),-46}  "
                            + $"{Truncate(camera, 22),-22}  {where}"
                            + (hit.Explanation is null ? string.Empty : $"   [{hit.Explanation}]"));
        }

        return 0;
    }

    // ---------------------------------------------------------------------------- sort

    private static async Task<int> SortAsync(PhotoStore store, string[] args)
    {
        var flags = ParseFlags(args);
        PhotoOrder order = ParseOrder(args.Length > 1 && !args[1].StartsWith('-') ? args[1] : null);
        int limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 20;

        IReadOnlyList<PhotoRecord> photos = await store.QueryAsync(new PhotoQuery
        {
            Order = order,
            Limit = limit,
            SourceFolder = flags.GetValueOrDefault("folder"),
        });

        Console.WriteLine($"order: {order}   showing {photos.Count}\n");

        foreach (PhotoRecord p in photos)
        {
            Console.WriteLine(
                $"  {p.CapturedUtc:yyyy-MM-dd}  {p.Rating}★  {p.FileSize / 1024,6}KB  "
                + $"{Truncate(p.Camera.Model ?? "—", 22),-22}  {Truncate(p.FileName, 48)}");
        }

        return 0;
    }

    // ---------------------------------------------------------------------------- facets

    private static async Task<int> FacetsAsync(PhotoStore store)
    {
        IReadOnlyDictionary<string, IReadOnlyList<(string Value, int Count)>> facets =
            await store.GetFacetsAsync();

        foreach ((string name, IReadOnlyList<(string Value, int Count)> values) in facets)
        {
            Console.WriteLine($"\n{name.ToUpperInvariant()}  ({values.Count} distinct)");
            foreach ((string value, int count) in values.Take(12))
            {
                Console.WriteLine($"  {count,5}  {Truncate(value, 66)}");
            }
        }

        return 0;
    }

    // ---------------------------------------------------------------------------- stats

    private static async Task<int> StatsAsync(PhotoStore store, string root)
    {
        IReadOnlyDictionary<ProcessingState, int> states = await store.GetStateCountsAsync();
        IReadOnlyList<TimeBucket> buckets = await store.GetTimeBucketsAsync();

        Console.WriteLine($"library    {root}");
        Console.WriteLine($"photos     {states.Values.Sum()}");

        foreach ((ProcessingState state, int count) in states.OrderBy(kv => kv.Key))
        {
            Console.WriteLine($"  {state,-12} {count}");
        }

        if (buckets.Count > 0)
        {
            Console.WriteLine($"span       {buckets[0].Start:yyyy-MM} to {buckets[^1].Start:yyyy-MM}"
                            + $"  ({buckets.Count} months with photos)");
            TimeBucket busiest = buckets.MaxBy(b => b.Count);
            Console.WriteLine($"busiest    {busiest.Start:yyyy-MM} with {busiest.Count}");
        }

        return 0;
    }

    // ---------------------------------------------------------------------------- vocab

    private static async Task<int> VocabAsync(PhotoStore store, string[] args)
    {
        IReadOnlyList<string> vocabulary = await store.GetVocabularyAsync();

        if (args.Length < 2)
        {
            Console.WriteLine($"{vocabulary.Count} terms indexed\n");
            foreach (string term in vocabulary.Take(40))
            {
                Console.WriteLine($"  {term}");
            }

            return 0;
        }

        string prefix = args[1];
        Console.WriteLine($"suggestions for \"{prefix}\":\n");

        foreach ((string term, double score) in FuzzyMatch.Suggest(prefix, vocabulary, limit: 12))
        {
            Console.WriteLine($"  {score:0.00}  {term}");
        }

        return 0;
    }

    // ---------------------------------------------------------------------------- dupes

    private static async Task<int> DupesAsync(PhotoStore store)
    {
        IReadOnlyList<IReadOnlyList<long>> groups = await store.FindNearDuplicateGroupsAsync();
        Console.WriteLine($"{groups.Count} near-duplicate groups\n");

        foreach (IReadOnlyList<long> group in groups)
        {
            Console.WriteLine($"  group of {group.Count}:");
            foreach (long id in group)
            {
                PhotoRecord? photo = await store.GetAsync(id);
                if (photo is not null)
                {
                    Console.WriteLine($"    {photo.PerceptualHash}  {Truncate(photo.FileName, 60)}");
                }
            }
        }

        return 0;
    }

    // ---------------------------------------------------------------------------- analyse

    private static async Task<int> AnalyseAsync(
        PhotoStore store, SegmentStore segments, string root, string[] args)
    {
        var flags = ParseFlags(args);
        int limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 1000;

        // The model is a loose file the user supplies rather than something shipped: it is
        // AGPL-licensed and 11 MB. Absent, this says so plainly instead of quietly doing nothing.
        string modelPath = flags.GetValueOrDefault("model")
            ?? Path.Combine(FindRepoRoot() ?? root, "models", "yolo11n-seg.onnx");

        using var segmenter = new YoloSegmenter(modelPath);
        if (!segmenter.IsAvailable)
        {
            Console.Error.WriteLine($"no segmentation model at {modelPath}");
            Console.Error.WriteLine("download yolo11n-seg.onnx from the Ultralytics assets release (AGPL-3.0)");
            return 1;
        }

        var worker = new SegmentationWorker(store, segments, segmenter, Path.Combine(root, "cache"));
        var lastReport = 0L;

        var progress = new Progress<AnalysisProgress>(p =>
        {
            long now = Environment.TickCount64;
            if (now - Interlocked.Read(ref lastReport) < 150 && p.Done < p.Total)
            {
                return;
            }

            Interlocked.Exchange(ref lastReport, now);
            string current = Truncate(p.Current ?? string.Empty, 44);
            Console.Write(CarriageReturn + $"  {p.Done}/{p.Total}  {p.Found} segments  {current,-44}");
        });

        Console.WriteLine($"model: {segmenter.ModelVersion}");
        (int analysed, int found) = await worker.RunAsync(limit, progress);

        Console.WriteLine();
        Console.WriteLine($"analysed {analysed} photos, found {found} segments");
        return 0;
    }

    private static async Task<int> ObjectsAsync(SegmentStore segments)
    {
        IReadOnlyList<(string Label, int Count)> facet = await segments.GetObjectFacetAsync();
        Console.WriteLine($"{facet.Count} distinct objects");
        Console.WriteLine();

        foreach ((string label, int count) in facet)
        {
            Console.WriteLine($"  {count,5}  {label}");
        }

        return 0;
    }

    // ----------------------------------------------------------------------------- faces

    /// <summary>
    /// Sweeps for faces and prints what the faces page would show.
    /// </summary>
    /// <remarks>
    /// One command for both because the page cannot be inspected any other way — it lives inside
    /// a WebView on a machine nobody can click on while it works. Running the sweep and then
    /// printing the same query the page issues is the closest thing to looking at it.
    /// </remarks>
    private static async Task<int> FacesAsync(
        PhotoStore store, FaceStore faces, string root, string[] args)
    {
        var flags = ParseFlags(args);
        int limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 1000;

        string modelPath = flags.GetValueOrDefault("model")
            ?? Path.Combine(FindRepoRoot() ?? root, "models", "face_yunet_2023mar.onnx");

        // Read straight from args, not from ParseFlags: that helper pairs each flag with the
        // token after it and so never sees a valueless flag in last position — which is exactly
        // where "--reset" gets typed.
        if (args.Contains("--reset", StringComparer.OrdinalIgnoreCase))
        {
            int cleared = await faces.ResetAsync();
            Console.WriteLine($"cleared {cleared} face rows");
        }

        using var detector = new YuNetFaceDetector(modelPath);
        if (!detector.IsAvailable)
        {
            Console.Error.WriteLine($"no face model at {modelPath}");
            Console.Error.WriteLine("download face_detection_yunet_2023mar.onnx from the OpenCV Model Zoo (Apache-2.0)");
            return 1;
        }

        // Recognition is optional: the model is 38 MB and the sweep is useful without it. When
        // it is absent the sweep still finds and crops every face, and only grouping goes away.
        string recognizerPath = flags.GetValueOrDefault("recognizer")
            ?? Path.Combine(FindRepoRoot() ?? root, "models", "face_sface_2021dec.onnx");

        using var recognizer = new SFaceRecognizer(recognizerPath);
        Console.WriteLine(recognizer.IsAvailable
            ? $"recognition: {recognizer.ModelVersion}"
            : $"recognition: off (no model at {recognizerPath})");

        var worker = new FaceWorker(store, faces, detector, Path.Combine(root, "cache"), recognizer);
        var lastReport = 0L;

        var progress = new Progress<FaceSweepProgress>(p =>
        {
            long now = Environment.TickCount64;
            if (now - Interlocked.Read(ref lastReport) < 150 && p.Done < p.Total)
            {
                return;
            }

            Interlocked.Exchange(ref lastReport, now);
            string current = Truncate(p.Current ?? string.Empty, 44);
            Console.Write(CarriageReturn + $"  {p.Done}/{p.Total}  {p.Found} faces  {current,-44}");
        });

        Console.WriteLine($"model: {detector.ModelVersion}");
        (int examined, int found) = await worker.RunAsync(limit, progress);

        Console.WriteLine();
        Console.WriteLine($"examined {examined} photos, found {found} faces");
        Console.WriteLine();

        (int total, int photographs) = await faces.CountAsync();
        (int embedded, _) = await faces.EmbeddingCoverageAsync();
        Console.WriteLine($"{total} faces across {photographs} photos, {embedded} comparable");
        Console.WriteLine();

        IReadOnlyList<FaceListing> listing = await faces.ListAsync(FaceOrder.Prominence, limit: 60);

        foreach (FaceListing item in listing)
        {
            PhotoFaceRecord face = item.Face;
            string crop = face.CropKey is null ? "no crop" : "crop ok";
            Console.WriteLine(
                $"  {face.Prominence:0.00}  conf {face.Confidence:0.00}  "
                + $"{face.AreaFraction * 100,5:0.0}%  roll {face.RollDegrees,6:+0.0;-0.0}  "
                + $"{crop}  {Truncate(item.PhotoFileName, 46)}");
        }

        return 0;
    }

    /// <summary>
    /// Reports what the face detector finds in every image under a folder.
    /// </summary>
    /// <remarks>
    /// Not a library command — a corpus-vetting tool. Building a same-person test set from a
    /// public photo archive means downloading candidates whose subject is asserted by a category
    /// name and nothing else, and category names lie: a category for a public figure holds
    /// protest placards, buildings, and oil paintings alongside photographs of the person.
    /// Running the detector over the staged candidates reduces a manual review of sixty images
    /// to a manual review of the dozen that actually contain one dominant face.
    /// </remarks>
    private static async Task<int> FaceCheckAsync(string root, string[] args)
    {
        string? folder = args.Skip(1).FirstOrDefault(a => !a.StartsWith('-'));
        if (folder is null || !Directory.Exists(folder))
        {
            Console.Error.WriteLine("facecheck needs a folder");
            return 1;
        }

        string modelPath = Path.Combine(FindRepoRoot() ?? root, "models", "face_yunet_2023mar.onnx");

        // --accel exists so the two paths can be compared on the same photographs. A per-run
        // microbenchmark says the graphics processor is twice as fast at this model; whether that
        // survives JPEG decode and file I/O is a different question and this is how it gets asked.
        var flags = ParseFlags(args);
        AcceleratorPreference preference =
            ComputeBackend.ParsePreference(flags.GetValueOrDefault("accel"));
        var compute = new ComputeBackend(
            Path.Combine(FindRepoRoot() ?? root, "models", ComputeBackend.ProviderFolderName),
            preference);

        Console.Error.WriteLine(
            $"accelerator: {preference.ToString().ToLowerInvariant()} " +
            $"-> yunet on {compute.PreferenceFor("face_yunet_2023mar").ToString().ToLowerInvariant()}");

        using var detector = new YuNetFaceDetector(modelPath, compute: compute);

        if (!detector.IsAvailable)
        {
            Console.Error.WriteLine($"no face model at {modelPath}");
            return 1;
        }

        string[] images = Directory
            .GetFiles(folder, "*.*", SearchOption.AllDirectories)
            .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                     || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToArray();

        // Tab-separated so the fetch script can read it back and prune. Deliberately not JSON:
        // the only consumer is a dozen lines of Python and a person reading the terminal.
        Console.WriteLine("file	faces	largest_area	confidence");

        foreach (string image in images)
        {
            FaceResult result;

            try
            {
                result = await detector.DetectAsync(image);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{image}	-1	0	0	{ex.GetType().Name}");
                continue;
            }

            double area = result.Faces.Count > 0 ? result.Faces.Max(f => f.AreaFraction) : 0;
            double confidence = result.Faces.Count > 0 ? result.Faces.Max(f => f.Confidence) : 0;

            Console.WriteLine(
                $"{image}	{result.Faces.Count}	{area.ToString("0.0000", CultureInfo.InvariantCulture)}"
                + $"	{confidence.ToString("0.00", CultureInfo.InvariantCulture)}");
        }

        return 0;
    }


    // ------------------------------------------------------------------------------- who

    /// <summary>
    /// Lists every face that looks like a given one.
    /// </summary>
    /// <remarks>
    /// The headless equivalent of clicking a face on the faces page. It exists so the grouping
    /// can be judged from the terminal — printing the similarity next to each filename makes it
    /// obvious whether a match is a confident one or a coin toss, which a wall of thumbnails
    /// does not.
    /// </remarks>

    /// <summary>Names a face, or lists everybody who has one.</summary>
    /// <remarks>
    /// The only way to attach a name until the faces page grows a field for it. Worth having on
    /// its own account: naming forty faces from a terminal is faster than clicking forty times,
    /// and it is how the reconciliation between faces and person outlines gets tested.
    /// </remarks>
    private static async Task<int> NameAsync(FaceStore faces, string[] args)
    {
        if (args.Contains("--list", StringComparer.OrdinalIgnoreCase))
        {
            IReadOnlyList<NamedPerson> people = await faces.ListNamedAsync();

            if (people.Count == 0)
            {
                Console.WriteLine("nobody has been named yet");
                return 0;
            }

            foreach (NamedPerson person in people)
            {
                Console.WriteLine(
                    $"  {person.PhotoCount,4} photos  {person.FaceCount,4} faces  #{person.Id,-4} {person.Name}");
            }

            return 0;
        }

        string[] rest = [.. args.Skip(1).Where(a => !a.StartsWith("--", StringComparison.Ordinal))];

        if (rest.Length == 0 || !long.TryParse(rest[0], out long faceId))
        {
            Console.Error.WriteLine("name needs a face id — try `pixelflux faces` to find one");
            return 1;
        }

        string? name = rest.Length > 1 ? string.Join(' ', rest.Skip(1)) : null;
        long? personId = await faces.NameFaceAsync(faceId, name);

        Console.WriteLine(personId is null
            ? $"face #{faceId} is nobody in particular again"
            : $"face #{faceId} is {name} (#{personId})");

        return 0;
    }
    private static async Task<int> WhoAsync(FaceStore faces, string[] args)
    {
        if (args.Length < 2 || !long.TryParse(args[1], out long faceId))
        {
            Console.Error.WriteLine("who needs a face id — run `pixelflux faces` to list them");
            return 1;
        }

        var flags = ParseFlags(args);
        double threshold = double.TryParse(flags.GetValueOrDefault("at"),
            NumberStyles.Float, CultureInfo.InvariantCulture, out double t)
            ? t
            : FaceGrouping.DefaultThreshold;

        IReadOnlyList<FaceMatch> matches = await faces.FindSimilarAsync(faceId, threshold);

        if (matches.Count == 0)
        {
            (int embedded, int total) = await faces.EmbeddingCoverageAsync();
            Console.Error.WriteLine(embedded == 0
                ? $"no faces are comparable ({total} detected). Re-run `pixelflux faces --reset` "
                  + "with the recognition model installed."
                : $"face {faceId} matched nothing at {threshold:0.00} — it may be too small to compare");
            return 1;
        }

        Console.WriteLine($"{matches.Count} faces at or above {threshold:0.00}, "
                        + $"in {matches.Select(m => m.Listing.Face.PhotoId).Distinct().Count()} photos");
        Console.WriteLine();

        foreach (FaceMatch match in matches)
        {
            string marker = match.Listing.Face.Id == faceId ? "*" : " ";
            Console.WriteLine(
                $" {marker} {match.Similarity:0.000}  #{match.Listing.Face.Id,-4}  "
                + $"{Truncate(match.Listing.PhotoFileName, 54)}");
        }

        return 0;
    }

    /// <summary>Lists the library's faces collapsed to one entry per person.</summary>
    /// <remarks>
    /// The headless form of the grouped faces wall. Printing the group sizes next to a
    /// representative filename is the quickest way to see whether the collapse is doing the
    /// right thing — one line per person, with the count, beats squinting at a grid of crops.
    /// </remarks>
    private static async Task<int> PeopleAsync(FaceStore faces, string[] args)
    {
        var flags = ParseFlags(args);
        double threshold = double.TryParse(flags.GetValueOrDefault("at"),
            NumberStyles.Float, CultureInfo.InvariantCulture, out double t)
            ? t
            : FaceGrouping.DefaultThreshold;

        IReadOnlyList<FaceGroup> groups = await faces.ListPeopleAsync(threshold);
        (int total, int photographs) = await faces.CountAsync();

        Console.WriteLine($"{groups.Count} people from {total} faces across {photographs} photos, "
                        + $"grouped at {threshold:0.00}");
        Console.WriteLine();

        foreach (FaceGroup group in groups.Take(40))
        {
            Console.WriteLine(
                $"  x{group.FaceCount,-3} in {group.PhotoCount,3} photos   #{group.Representative.Face.Id,-4}  "
                + $"{Truncate(group.Representative.PhotoFileName, 48)}");
        }

        return 0;
    }

    // ------------------------------------------------------------------ search by meaning

    /// <summary>Builds a CLIP embedder from the models directory, installed or not.</summary>
    private static ClipEmbedder OpenClip(string root)
    {
        string models = Path.Combine(FindRepoRoot() ?? root, "models");

        return new ClipEmbedder(
            Path.Combine(models, "clip_vision_model.onnx"),
            Path.Combine(models, "clip_text_model.onnx"),
            Path.Combine(models, "clip_vocab.json"),
            Path.Combine(models, "clip_merges.txt"));
    }

    /// <summary>Describes every photograph that has no vector yet.</summary>
    private static async Task<int> EmbedAsync(
        PhotoStore store, VectorIndex vectors, string root, string[] args)
    {
        var flags = ParseFlags(args);
        int limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 100000;

        using ClipEmbedder clip = OpenClip(root);

        if (!clip.IsAvailable)
        {
            Console.Error.WriteLine("CLIP is not installed.");
            Console.Error.WriteLine("Put clip_vision_model.onnx, clip_text_model.onnx, clip_vocab.json");
            Console.Error.WriteLine("and clip_merges.txt in models/ (Xenova/clip-vit-base-patch32, MIT).");
            return 1;
        }

        var worker = new EmbeddingWorker(store, vectors, clip, Path.Combine(root, "cache"));
        var lastReport = 0L;

        var progress = new Progress<EmbeddingProgress>(p =>
        {
            long now = Environment.TickCount64;
            if (now - Interlocked.Read(ref lastReport) < 150 && p.Done < p.Total)
            {
                return;
            }

            Interlocked.Exchange(ref lastReport, now);
            Console.Write(CarriageReturn + $"  {p.Done}/{p.Total}  {Truncate(p.Current ?? string.Empty, 48),-48}");
        });

        Console.WriteLine($"model: {clip.ModelVersion}, {clip.Dimensions} dimensions");
        int done = await worker.RunAsync(limit, progress);

        (int described, int total) = await vectors.CoverageAsync();
        Console.WriteLine();
        Console.WriteLine($"described {done} photos; {described} of {total} in the library now searchable by meaning");
        return 0;
    }

    /// <summary>
    /// Searches by meaning and prints the ranking.
    /// </summary>
    /// <remarks>
    /// Prints the similarity beside every hit, because that is the only way to tell a search
    /// that found the right thing from one that returned the least-wrong thing. CLIP always
    /// returns a ranking; the scores are what say whether to believe it.
    /// </remarks>
    private static async Task<int> VectorSearchAsync(
        PhotoStore store, VectorIndex vectors, string root, string[] args)
    {
        string phrase = string.Join(' ',
            args.Skip(1).TakeWhile(a => !a.StartsWith("--", StringComparison.Ordinal)));

        if (string.IsNullOrWhiteSpace(phrase))
        {
            Console.Error.WriteLine("vsearch needs something to look for");
            return 1;
        }

        var flags = ParseFlags(args);
        int limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 12;

        using ClipEmbedder clip = OpenClip(root);

        if (!clip.IsAvailable)
        {
            Console.Error.WriteLine("CLIP is not installed; run `pixelflux embed` for the setup notes");
            return 1;
        }

        float[]? query = await clip.EmbedQueryAsync(phrase);
        if (query is null)
        {
            Console.Error.WriteLine("could not turn that phrase into a vector");
            return 1;
        }

        // Learn which photographs are agreeable to everything, so they can be discounted. One
        // model run per reference phrase, then arithmetic — under a second on any library.
        var bank = new List<ReadOnlyMemory<float>>(ReferencePhrases.All.Count);
        foreach (string reference in ReferencePhrases.All)
        {
            if (await clip.EmbedTextAsync(reference) is { } v)
            {
                bank.Add(v);
            }
        }

        await vectors.CalibrateAsync(bank);

        IReadOnlyList<VectorHit> hits = await vectors.SearchAsync(query, limit);

        (int described, int total) = await vectors.CoverageAsync();
        Console.WriteLine($"\"{phrase}\" over {described} of {total} photos");
        Console.WriteLine();

        foreach (VectorHit hit in hits)
        {
            PhotoRecord? photo = await store.GetAsync(hit.PhotoId);
            Console.WriteLine($"  {hit.Similarity:0.000}  z{hit.Standout,5:+0.00;-0.00}  "
                            + $"{Truncate(photo?.FileName ?? "?", 52)}");
        }

        return 0;
    }

    /// <summary>
    /// Runs exactly the search the application runs.
    /// </summary>
    /// <remarks>
    /// Distinct from <c>vsearch</c>, which is raw embedding similarity with no floor and no
    /// blending. This goes through the search engine — spelling correction, the word index, the
    /// relevance floor — so what it prints is what the search box would show. Judging the
    /// feature on the raw ranking would be judging something the user never sees.
    /// </remarks>
    private static async Task<int> FindAsync(
        PhotoStore store, VectorIndex vectors, string root, string[] args)
    {
        string phrase = string.Join(' ',
            args.Skip(1).TakeWhile(a => !a.StartsWith("--", StringComparison.Ordinal)));

        if (string.IsNullOrWhiteSpace(phrase))
        {
            Console.Error.WriteLine("find needs something to look for");
            return 1;
        }

        using ClipEmbedder clip = OpenClip(root);
        var engine = new SearchEngine(store, vectors);

        float[]? query = null;

        if (clip.IsAvailable)
        {
            var bank = new List<ReadOnlyMemory<float>>(ReferencePhrases.All.Count);
            foreach (string reference in ReferencePhrases.All)
            {
                if (await clip.EmbedTextAsync(reference) is { } v)
                {
                    bank.Add(v);
                }
            }

            await vectors.CalibrateAsync(bank);
            query = await clip.EmbedQueryAsync(phrase);
        }

        SearchResult result = await engine.SearchAsync(
            new PhotoQuery { Text = phrase, Limit = 40 }, query);

        Console.WriteLine($"{result.Hits.Count} results"
                        + (result.UsedSemantic ? ", including matches by meaning" : string.Empty));

        foreach ((string typed, string used) in result.Corrections)
        {
            Console.WriteLine($"  spelling: {typed} -> {used}");
        }

        Console.WriteLine();

        foreach (SearchHit hit in result.Hits)
        {
            Console.WriteLine($"  {hit.Score:0.000}  {Truncate(hit.Photo.FileName, 56)}");
        }

        return 0;
    }

    // -------------------------------------------------------------- describing photographs

    /// <summary>
    /// Runs the vision-language model over the library, writing a description of each photograph.
    /// </summary>
    /// <remarks>
    /// Prints each description as it is written. This is the slowest thing the application does
    /// — seconds per photograph — and watching the first few is the only way to tell whether the
    /// output is worth the wait before committing half an hour to it.
    /// </remarks>
    private static async Task<int> DescribeAsync(PhotoStore store, string root, string[] args)
    {
        var flags = ParseFlags(args);
        int limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 100000;

        string modelDir = flags.GetValueOrDefault("model")
            ?? Path.Combine(FindRepoRoot() ?? root, "models", "qwen3vl");

        // --profile describes a handful of photographs and prints where the time actually went,
        // rather than describing the library. Added after an accelerator was chased for a stage
        // whose inference turned out to be under a tenth of its runtime.
        if (args.Contains("--profile", StringComparer.OrdinalIgnoreCase))
        {
            int edge = int.TryParse(flags.GetValueOrDefault("edge"), out int e) ? e : 672;
            int tokens = int.TryParse(flags.GetValueOrDefault("tokens"), out int t) ? t : 220;

            double penalty = double.TryParse(flags.GetValueOrDefault("reppen"),
                System.Globalization.NumberStyles.Float,
                CultureInfo.InvariantCulture, out double rp) ? rp : 1.0;

            using var tuned = new QwenVisionDescriber(
                modelDir, longEdge: edge, maximumTokens: tokens, repetitionPenalty: penalty);
            return await ProfileDescribeAsync(store, tuned, root, flags);
        }

        using var describer = new QwenVisionDescriber(modelDir);

        if (!describer.IsAvailable)
        {
            Console.Error.WriteLine($"no vision-language model at {modelDir}");
            Console.Error.WriteLine("download the ONNX Runtime GenAI build of");
            Console.Error.WriteLine("onnx-community/Qwen3-VL-2B-Instruct-ONNX (Apache-2.0) into that folder");
            return 1;
        }

        if (args.Contains("--redo", StringComparer.OrdinalIgnoreCase))
        {
            // Describing is expensive enough that redoing it has to be asked for explicitly.
            foreach (PhotoRecord photo in await store.QueryAsync(new PhotoQuery { Limit = 100000 }))
            {
                await store.SetDescriptionAsync(photo.Id, null, describer.ModelVersion);
            }

            Console.WriteLine("cleared every description");
        }

        var worker = new DescriptionWorker(store, describer, Path.Combine(root, "cache"));
        var watch = System.Diagnostics.Stopwatch.StartNew();

        var progress = new Progress<DescriptionProgress>(p =>
        {
            if (p.Latest is { } text)
            {
                Console.WriteLine($"  [{p.Done}/{p.Total}] {Truncate(p.Current ?? "", 40)}");
                Console.WriteLine($"      {Truncate(text.Replace('\n', ' '), 150)}");
            }
        });

        Console.WriteLine($"model: {describer.ModelVersion}");
        Console.WriteLine();

        int done = await worker.RunAsync(limit, progress);
        watch.Stop();

        (int described, int total) = await store.DescriptionCoverageAsync();

        Console.WriteLine();
        Console.WriteLine($"described {done} photos in {watch.Elapsed.TotalMinutes:0.0} min"
                        + (done > 0 ? $" ({watch.ElapsedMilliseconds / (double)done / 1000:0.0} s each)" : string.Empty));
        Console.WriteLine($"{described} of {total} photos now have a description");
        return 0;
    }

    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "models")))
            {
                return dir.FullName;
            }

            dir = dir.Parent;
        }

        return null;
    }

    // ---------------------------------------------------------------------------- albums


    /// <summary>Describes a few photographs and reports the prefill and decode split.</summary>
    private static async Task<int> ProfileDescribeAsync(
        PhotoStore store,
        QwenVisionDescriber describer,
        string root,
        Dictionary<string, string> flags)
    {
        if (!describer.IsAvailable)
        {
            Console.Error.WriteLine("no vision-language model installed");
            return 1;
        }

        int count = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : 3;

        IReadOnlyList<PhotoRecord> photos = await store.QueryAsync(
            new PhotoQuery { Limit = count });

        Console.WriteLine($"edge {describer.LongEdge}, token budget {describer.MaximumTokens}, repetition penalty {describer.RepetitionPenalty:F2}");
        Console.WriteLine($"{"file",-34} {"prefill",9} {"decode",9} {"tokens",7} {"ms/tok",7}");

        foreach (PhotoRecord photo in photos)
        {
            string source = photo.ProxyKey is { } proxy
                ? Path.Combine(root, "cache", proxy.Replace('/', Path.DirectorySeparatorChar))
                : photo.OriginalPath;

            if (!File.Exists(source))
            {
                source = photo.OriginalPath;
            }

            string? description = await describer.DescribeAsync(source);
            QwenVisionDescriber.DescribeTiming timing = describer.LastTiming;

            Console.WriteLine(
                $"{Truncate(photo.FileName, 34),-34} "
                + $"{timing.Prefill.TotalMilliseconds,9:F0} "
                + $"{timing.Decode.TotalMilliseconds,9:F0} "
                + $"{timing.Tokens,7} "
                + $"{timing.MillisecondsPerToken,7:F1}");

            if (flags.ContainsKey("show"))
            {
                Console.WriteLine($"    {description}");
                Console.WriteLine();
            }
        }

        return 0;
    }
    private static async Task<int> AlbumsAsync(CollectionStore collections, string[] args)
    {
        string action = args.Length > 1 ? args[1] : "list";

        switch (action)
        {
            case "new":
                if (args.Length < 3)
                {
                    Console.Error.WriteLine("albums new <name>");
                    return 1;
                }

                long id = await collections.CreateAlbumAsync(string.Join(' ', args.Skip(2)));
                Console.WriteLine($"created album {id}");
                return 0;

            case "add":
            {
                if (args.Length < 4)
                {
                    Console.Error.WriteLine("albums add <name> <photo-id>...");
                    return 1;
                }

                IReadOnlyList<PhotoCollection> all = await collections.ListAsync();
                PhotoCollection? album = all.FirstOrDefault(c =>
                    c.Name.Equals(args[2], StringComparison.OrdinalIgnoreCase));

                if (album is null)
                {
                    Console.Error.WriteLine($"no album named '{args[2]}'");
                    return 1;
                }

                long[] ids = args.Skip(3).Select(a => long.Parse(a, CultureInfo.InvariantCulture)).ToArray();
                int added = await collections.AddAsync(album.Id, ids);
                Console.WriteLine($"added {added} to {album.Name}");
                return 0;
            }

            default:
            {
                IReadOnlyList<PhotoCollection> all = await collections.ListAsync();
                Console.WriteLine($"{all.Count} collections\n");

                foreach (PhotoCollection c in all)
                {
                    string count = c.Count < 0 ? "smart" : $"{c.Count} photos";
                    Console.WriteLine($"  {c.Id,4}  {c.Name,-32}  {c.Kind,-6}  {count}");
                }

                return 0;
            }
        }
    }

    // ---------------------------------------------------------------------------- helpers

    private static Dictionary<string, string> ParseFlags(string[] args)
    {
        var flags = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal))
            {
                flags[args[i][2..]] = args[i + 1];
            }
        }

        return flags;
    }

    private static DateTimeOffset? ParseDate(string? value)
        => DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out DateTimeOffset parsed)
            ? parsed
            : null;

    private static (double South, double West, double North, double East)? ParseBox(string? value)
    {
        if (value is null)
        {
            return null;
        }

        string[] parts = value.Split(',');
        if (parts.Length != 4)
        {
            return null;
        }

        double[] numbers = parts
            .Select(p => double.TryParse(p, CultureInfo.InvariantCulture, out double d) ? d : double.NaN)
            .ToArray();

        return numbers.Any(double.IsNaN)
            ? null
            : (numbers[0], numbers[1], numbers[2], numbers[3]);
    }

    private static PhotoOrder ParseOrder(string? value) => value switch
    {
        "captured-asc" => PhotoOrder.CapturedAscending,
        "indexed-desc" => PhotoOrder.IndexedDescending,
        "filename" => PhotoOrder.FileName,
        "rating" => PhotoOrder.RatingDescending,
        "size" => PhotoOrder.FileSizeDescending,
        "camera" => PhotoOrder.Camera,
        "folder" => PhotoOrder.Folder,
        "shuffle" => PhotoOrder.Shuffle,
        _ => PhotoOrder.CapturedDescending,
    };

    private static string Truncate(string value, int max)
        => value.Length <= max ? value : value[..(max - 1)] + "…";
}
