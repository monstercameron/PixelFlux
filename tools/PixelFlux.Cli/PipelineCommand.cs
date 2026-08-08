using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using PixelFlux.Ai.Faces;
using PixelFlux.Ai.Pipeline;
using PixelFlux.Ai.Segmentation;
using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Index;
using PixelFlux.Core.Pipeline;
using PixelFlux.Core.Search;

namespace PixelFlux.Cli;

/// <summary>
/// The <c>pipeline</c> command: run, inspect and reset the analysis queue from a terminal.
/// </summary>
/// <remarks>
/// Headless on purpose. Analysing a library takes hours, and running it from the application means
/// the application has to stay open on a machine somebody is using for something else. It is also
/// the only way to watch the queue while the application is doing its own thing, which is how every
/// question about ordering and caching in this design was actually answered.
/// </remarks>
public static class PipelineCommand
{
    /// <summary>Runs the command.</summary>
    /// <param name="database">The migrated library database.</param>
    /// <param name="root">Library root, holding the derivative cache and the models folder.</param>
    /// <param name="repoRoot">Repository root, if the models live there instead.</param>
    /// <param name="args">The whole argument list, starting with <c>pipeline</c>.</param>
    /// <returns>A process exit code.</returns>
    public static async Task<int> RunAsync(
        PhotoDatabase database,
        string root,
        string? repoRoot,
        string[] args)
    {
        ArgumentNullException.ThrowIfNull(database);
        ArgumentNullException.ThrowIfNull(args);

        var jobs = new JobStore(database);
        string verb = args.Length > 1 ? args[1] : "status";

        return verb switch
        {
            "status" => await StatusAsync(database, jobs),
            "run" => await RunQueueAsync(database, jobs, root, repoRoot, args),
            "redo" => await RedoAsync(jobs, args),
            "reset" => await ResetAsync(jobs),
            "sweep" => await SweepAsync(database, root, repoRoot),
            _ => Unknown(verb),
        };
    }

    private static int Unknown(string verb)
    {
        Console.Error.WriteLine($"unknown pipeline verb '{verb}'");
        Console.Error.WriteLine("try: status, run, redo <stage>, reset");
        return 1;
    }

    private static async Task<int> StatusAsync(PhotoDatabase database, JobStore jobs)
    {
        await jobs.ReconcileAsync();
        await jobs.AdoptExistingWorkAsync();

        IReadOnlyList<StageStatus> stages = await jobs.StatusAsync();
        var cache = new StageCache(database);
        (int entries, long bytes) = await cache.SizeAsync();

        Console.WriteLine("stage      done  waiting  failed  stuck  skipped");
        foreach (StageStatus stage in stages)
        {
            Console.WriteLine(
                $"{stage.Stage.Slug(),-9} {stage.Done,5} {stage.Pending,8} " +
                $"{stage.Failed,7} {stage.Stuck,6} {stage.Skipped,8}");
        }

        Console.WriteLine();
        Console.WriteLine($"cache: {entries} results, {bytes / 1024.0 / 1024.0:F1} MB");

        var settings = new SettingsStore(database);
        PipelineSchedule schedule = PipelineSchedule.Parse(
            await settings.GetAsync(PipelineSchedule.SettingKey));

        Console.WriteLine(schedule.Mode switch
        {
            ScheduleMode.Off => "schedule: off",
            ScheduleMode.Window =>
                $"schedule: {schedule.Start:HH\\:mm}-{schedule.End:HH\\:mm}, " +
                $"{schedule.Gap.TotalSeconds:F0}s between photos",
            _ => $"schedule: always, {schedule.Gap.TotalSeconds:F0}s between photos",
        });

        return 0;
    }

    private static async Task<int> RedoAsync(JobStore jobs, string[] args)
    {
        if (args.Length < 3 || PipelineStages.FromSlug(args[2]) is not { } stage)
        {
            Console.Error.WriteLine("redo needs a stage: describe, segment, faces or embed");
            return 1;
        }

        int requeued = await jobs.RequeueAsync(stage);
        Console.WriteLine($"requeued {requeued} photos for {stage.Slug()}");
        return 0;
    }

    private static async Task<int> ResetAsync(JobStore jobs)
    {
        int requeued = await jobs.RequeueAsync(null);
        Console.WriteLine($"requeued {requeued} jobs");
        return 0;
    }

    private static async Task<int> SweepAsync(
        PhotoDatabase database,
        string root,
        string? repoRoot)
    {
        string models = Path.Combine(repoRoot ?? root, "models");

        using ClipEmbedder clip = new(
            Path.Combine(models, "clip_vision_model.onnx"),
            Path.Combine(models, "clip_text_model.onnx"),
            Path.Combine(models, "clip_vocab.json"),
            Path.Combine(models, "clip_merges.txt"));

        return await BlendSweep.RunAsync(
            new PhotoStore(database), clip, Path.Combine(root, "cache"));
    }

    private static async Task<int> RunQueueAsync(
        PhotoDatabase database,
        JobStore jobs,
        string root,
        string? repoRoot,
        string[] args)
    {
        Dictionary<string, string> flags = Flags(args);
        int? limit = int.TryParse(flags.GetValueOrDefault("limit"), out int n) ? n : null;
        TimeSpan gap = double.TryParse(flags.GetValueOrDefault("gap"), out double seconds)
            ? TimeSpan.FromSeconds(seconds)
            : TimeSpan.Zero;

        string models = Path.Combine(repoRoot ?? root, "models");
        string cacheRoot = Path.Combine(root, "cache");

        var photos = new PhotoStore(database);
        var vectors = new VectorIndex(database);
        var segments = new SegmentStore(database);
        var faceStore = new FaceStore(database);

        // Every model is opened whether or not it is installed. An absent one reports itself
        // unavailable, its stage is skipped, and the rest of the queue runs — which is the same
        // graceful-degradation rule the application follows, and the reason a partial install is
        // a slower library rather than a broken one.
        using var describer = new QwenVisionDescriber(Path.Combine(models, "qwen3vl"));
        using var segmenter = new YoloSegmenter(Path.Combine(models, "yolo11n-seg.onnx"));
        using var detector = new YuNetFaceDetector(
            Path.Combine(models, "face_yunet_2023mar.onnx"));
        using var recognizer = new SFaceRecognizer(
            Path.Combine(models, "face_sface_2021dec.onnx"));
        using ClipEmbedder clip = new(
            Path.Combine(models, "clip_vision_model.onnx"),
            Path.Combine(models, "clip_text_model.onnx"),
            Path.Combine(models, "clip_vocab.json"),
            Path.Combine(models, "clip_merges.txt"));

        var segmentWorker = new SegmentationWorker(photos, segments, segmenter, cacheRoot);
        var faceWorker = new FaceWorker(photos, faceStore, detector, cacheRoot, recognizer);

        IStageHandler[] handlers =
        [
            new DescribeHandler(photos, describer, cacheRoot),
            new SegmentHandler(segmentWorker, photos),
            new FacesHandler(faceWorker, photos),
            new EmbedHandler(photos, vectors, clip, cacheRoot),
        ];

        foreach (IStageHandler handler in handlers)
        {
            Console.WriteLine($"  {handler.Stage.Slug(),-9} {handler.ModelVersion ?? "not installed"}");
        }

        int added = await jobs.ReconcileAsync();
        int adopted = await jobs.AdoptExistingWorkAsync();
        int released = await jobs.ReleaseClaimsAsync();
        if (added > 0 || adopted > 0 || released > 0)
        {
            Console.WriteLine(
                $"queued {added} new jobs, adopted {adopted} already done, " +
                $"released {released} stale claims");
        }

        var runner = new PipelineRunner(
            jobs, new StageCache(database), handlers, NullLogger<PipelineRunner>.Instance);

        int computed = 0;
        int reused = 0;
        var progress = new Progress<PipelineTick>(tick =>
        {
            if (tick.Result == ItemResult.Computed)
            {
                computed++;
            }
            else if (tick.Result == ItemResult.Reused)
            {
                reused++;
            }

            Console.WriteLine(
                $"  {tick.Result.ToString().ToLowerInvariant(),-8} {tick.Stage?.Slug(),-9} " +
                $"{(int)tick.Elapsed.TotalMilliseconds,7} ms  {tick.FileName}");
        });

        using var stopping = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            // Ctrl-C should end the run at the next photograph, not abandon one halfway through
            // and leave its claim behind. The runner puts an interrupted claim back untouched.
            e.Cancel = true;
            stopping.Cancel();
        };

        Console.WriteLine();
        int handled;
        try
        {
            handled = await runner.DrainAsync(gap, limit, progress, stopping.Token);
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("stopped");
            return 0;
        }

        Console.WriteLine();
        Console.WriteLine($"{handled} items: {computed} computed, {reused} reused from cache");
        return 0;
    }

    private static Dictionary<string, string> Flags(string[] args)
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
}
