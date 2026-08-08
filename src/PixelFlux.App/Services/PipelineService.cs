using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using PixelFlux.Ai.Faces;
using PixelFlux.Ai.Pipeline;
using PixelFlux.Ai.Segmentation;
using PixelFlux.Ai.Semantic;
using PixelFlux.Core.Index;
using PixelFlux.Core.Pipeline;
using PixelFlux.Core.Search;
using PixelFlux.Ai.Compute;

namespace PixelFlux.App.Services;

/// <summary>What the analysis queue is doing right now, for the status bar.</summary>
/// <param name="Running">Whether a photograph is being worked on.</param>
/// <param name="Waiting">Photographs left across every stage.</param>
/// <param name="Stage">The stage in progress, if one is.</param>
/// <param name="FileName">The photograph in progress, if one is.</param>
/// <param name="NextOpening">When the schedule next allows work, if it is currently closed.</param>
public readonly record struct PipelineState(
    bool Running,
    int Waiting,
    PipelineStage? Stage,
    string? FileName,
    DateTime? NextOpening);

/// <summary>
/// Runs the analysis queue in the background, on a schedule, one photograph at a time.
/// </summary>
/// <remarks>
/// <para>
/// The loop is deliberately dull: ask the queue for one item, do it, wait, ask again. Everything
/// interesting — what runs before what, what has already been done, what can be reused — belongs to
/// the queue and the cache, not here. That is what lets the schedule change, the application close
/// mid-photograph, and an import arrive halfway through, all without this class knowing.
/// </para>
/// <para>
/// It never runs two photographs at once, and the pause between them is the throttle. On a
/// fanless machine that is the difference between analysis you can ignore and analysis you have to
/// stop working to wait out.
/// </para>
/// </remarks>
public sealed class PipelineService : IDisposable
{
    private readonly JobStore _jobs;
    private readonly StageCache _cache;
    private readonly SettingsStore _settings;
    private readonly PipelineRunner _runner;
    private readonly ComputeBackend _compute;
    private readonly ILogger<PipelineService> _log;

    private readonly SemaphoreSlim _oneAtATime = new(1, 1);
    private CancellationTokenSource? _loop;
    private PipelineState _state;
    private PipelineSchedule _schedule = PipelineSchedule.Default;

    /// <summary>Creates the service and its stage handlers.</summary>
    /// <param name="database">The library database.</param>
    /// <param name="photos">The photo index.</param>
    /// <param name="segments">The segmentation index.</param>
    /// <param name="faces">The face index.</param>
    /// <param name="vectors">The embedding index.</param>
    /// <param name="paths">Where the models and the derivative cache live.</param>
    /// <param name="compute">What hardware the models run on.</param>
    /// <param name="logger">Logger.</param>
    /// <remarks>
    /// Every model is constructed here and none is loaded. Each of these constructors does no more
    /// than check whether a file exists — the weights, 1.4 GB of them in the vision model's case,
    /// are opened by the first photograph that needs them. A stage whose file is missing reports
    /// itself unavailable and gets skipped, so a partial install is a library that analyses less
    /// rather than an application that will not start.
    /// </remarks>
    public PipelineService(
        PhotoDatabase database,
        PhotoStore photos,
        SegmentStore segments,
        FaceStore faces,
        VectorIndex vectors,
        LibraryPaths paths,
        ComputeBackend compute,
        ILogger<PipelineService> logger)
    {
        ArgumentNullException.ThrowIfNull(database);
        ArgumentNullException.ThrowIfNull(paths);

        _jobs = new JobStore(database);
        _cache = new StageCache(database);
        _settings = new SettingsStore(database);
        _log = logger ?? NullLogger<PipelineService>.Instance;
        _compute = compute;

        // Every model takes the same backend, which is what makes the accelerator one decision
        // rather than five. Each still gets its own answer out of it — the measurements say the
        // graphics processor helps face work and hurts CLIP and segmentation.
        var describer = new QwenVisionDescriber(paths.VisionModelDirectory);
        var segmenter = new YoloSegmenter(paths.SegmentationModelPath, compute: compute);
        var detector = new YuNetFaceDetector(paths.FaceModelPath, compute: compute);
        var recognizer = new SFaceRecognizer(paths.RecognitionModelPath, compute: compute);
        var clip = new ClipEmbedder(
            paths.ClipVisionModelPath,
            paths.ClipTextModelPath,
            paths.ClipVocabularyPath,
            paths.ClipMergesPath,
            compute: compute);

        IStageHandler[] handlers =
        [
            new DescribeHandler(photos, describer, paths.CacheRoot),
            new SegmentHandler(
                new SegmentationWorker(photos, segments, segmenter, paths.CacheRoot), photos),
            new FacesHandler(
                new FaceWorker(photos, faces, detector, paths.CacheRoot, recognizer), photos),
            new EmbedHandler(photos, vectors, clip, paths.CacheRoot),
        ];

        _runner = new PipelineRunner(
            _jobs, _cache, handlers, NullLogger<PipelineRunner>.Instance);
    }

    /// <summary>Raised whenever the queue's state changes, so a status readout can refresh.</summary>
    public event Action? Changed;

    /// <summary>What the queue is doing.</summary>
    public PipelineState State => _state;

    /// <summary>When the queue is allowed to run.</summary>
    public PipelineSchedule Schedule => _schedule;

    /// <summary>What hardware the runtime can see.</summary>
    public IReadOnlyList<AcceleratorDevice> Devices => _compute.Devices;

    /// <summary>Which hardware models are set to prefer.</summary>
    public AcceleratorPreference Accelerator => _compute.Preference;

    /// <summary>The stages that an accelerator is actually used for, given the current setting.</summary>
    /// <remarks>
    /// "Automatic" is not one answer, it is four — the measurements put face work on the graphics
    /// processor and leave describing, segmenting and indexing on the processor, because the
    /// graphics path is slower for those. Somebody who has selected Automatic has no way to know
    /// that, and a setting whose effect is unknowable is a setting people stop trusting.
    /// </remarks>
    public IReadOnlyList<PipelineStage> AcceleratedStages
    {
        get
        {
            if (!_compute.HasAccelerator)
            {
                return [];
            }

            // Keyed off the models each stage actually opens. Faces is the only stage that opens
            // two, and it counts if either one is accelerated.
            (PipelineStage Stage, string[] Models)[] models =
            [
                (PipelineStage.Describe, ["qwen3vl"]),
                (PipelineStage.Segment, ["yolo11n-seg"]),
                (PipelineStage.Faces, ["face_yunet", "face_sface"]),
                (PipelineStage.Embed, ["clip_vision", "clip_text"]),
            ];

            return
            [
                .. models
                    .Where(entry => entry.Models.Any(model =>
                        _compute.PreferenceFor(model) is AcceleratorPreference.Gpu
                                                     or AcceleratorPreference.Npu))
                    .Select(entry => entry.Stage),
            ];
        }
    }

    /// <summary>Records a new hardware preference.</summary>
    /// <param name="preference">What to prefer.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// Stored, not applied. A model decides its hardware when it opens its session and keeps that
    /// session for the life of the process, so changing this affects the next run of the
    /// application rather than the current one. Saying so in the interface is the honest option;
    /// tearing down and reopening five sessions underneath a running queue to avoid saying it
    /// is not.
    /// </remarks>
    public Task SetAcceleratorAsync(
        AcceleratorPreference preference,
        CancellationToken cancellationToken = default) =>
        _settings.SetAsync(
            ComputeBackend.SettingKey,
            preference.ToString().ToLowerInvariant(),
            cancellationToken);

    /// <summary>Reads the schedule, tidies the queue, and starts the loop.</summary>
    /// <param name="cancellationToken">Cancels the startup work.</param>
    /// <remarks>
    /// The tidying matters as much as the starting. Reconciling gives new photographs their rows,
    /// adopting recognises work the old per-stage sweeps already did — without which a library
    /// analysed before this existed would be analysed again from nothing — and releasing claims
    /// frees rows left running by a process that was closed mid-photograph.
    /// </remarks>
    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        _schedule = PipelineSchedule.Parse(
            await _settings.GetAsync(PipelineSchedule.SettingKey, cancellationToken)
                .ConfigureAwait(false));

        await _jobs.ReconcileAsync(cancellationToken).ConfigureAwait(false);
        await _jobs.AdoptExistingWorkAsync(cancellationToken).ConfigureAwait(false);
        await _jobs.ReleaseClaimsAsync(cancellationToken).ConfigureAwait(false);

        Restart();
    }

    /// <summary>Changes when the queue runs and saves the choice.</summary>
    /// <param name="schedule">The new schedule.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task SetScheduleAsync(
        PipelineSchedule schedule,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(schedule);

        _schedule = schedule;
        await _settings.SetAsync(
            PipelineSchedule.SettingKey, schedule.Serialise(), cancellationToken)
            .ConfigureAwait(false);

        Restart();
    }

    /// <summary>Works the queue now, whatever the schedule says.</summary>
    /// <param name="limit">Most photographs to handle, or null for all of them.</param>
    /// <param name="cancellationToken">Stops at the next photograph.</param>
    /// <returns>How many items were handled.</returns>
    /// <remarks>
    /// The manual override, for somebody who has just imported photographs and wants to see them
    /// analysed rather than wait for ten at night. It takes the same lock as the scheduled loop, so
    /// pressing it while the loop is mid-photograph waits rather than doubling up.
    /// </remarks>
    public async Task<int> RunNowAsync(
        int? limit = null,
        CancellationToken cancellationToken = default)
    {
        await _oneAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await _runner.DrainAsync(
                TimeSpan.Zero, limit, new Progress<PipelineTick>(Observe), cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _oneAtATime.Release();
            await RefreshAsync(CancellationToken.None).ConfigureAwait(false);
        }
    }

    /// <summary>How far along each stage is.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>One entry per stage, in run order.</returns>
    public Task<IReadOnlyList<StageStatus>> StatusAsync(
        CancellationToken cancellationToken = default) =>
        _jobs.StatusAsync(cancellationToken);

    /// <summary>What every stage has done to one photograph.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>One entry per stage, in run order.</returns>
    public Task<IReadOnlyList<(PipelineStage Stage, JobState State, string? Error)>>
        ForPhotoAsync(long photoId, CancellationToken cancellationToken = default) =>
        _jobs.ForPhotoAsync(photoId, cancellationToken);

    /// <summary>Gives another chance to work that ran out of retries.</summary>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many rows were given their retries back.</returns>
    public async Task<int> RetryStuckAsync(CancellationToken cancellationToken = default)
    {
        int retried = await _jobs.RetryStuckAsync(cancellationToken).ConfigureAwait(false);
        await RefreshAsync(cancellationToken).ConfigureAwait(false);
        return retried;
    }

    /// <summary>Re-reads how much is left and tells anybody watching.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    public async Task RefreshAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<StageStatus> stages = await _jobs.StatusAsync(cancellationToken)
            .ConfigureAwait(false);

        _state = _state with
        {
            Waiting = stages.Sum(stage => stage.Pending + stage.Failed),
            NextOpening = _schedule.NextOpening(DateTime.Now),
        };

        Changed?.Invoke();
    }

    /// <summary>Stops the loop.</summary>
    public void Dispose()
    {
        _loop?.Cancel();
        _loop?.Dispose();
        _loop = null;
        _oneAtATime.Dispose();
    }

    private void Restart()
    {
        _loop?.Cancel();
        _loop?.Dispose();
        _loop = new CancellationTokenSource();

        if (_schedule.Mode == ScheduleMode.Off)
        {
            _ = RefreshAsync();
            return;
        }

        _ = LoopAsync(_loop.Token);
    }

    private async Task LoopAsync(CancellationToken cancellationToken)
    {
        // A closed window is checked once a minute rather than continuously. The queue is measured
        // in hours; being up to sixty seconds late to a window that lasts until morning is not a
        // cost worth a busier loop.
        TimeSpan closedPoll = TimeSpan.FromMinutes(1);

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                if (!_schedule.IsOpenAt(DateTime.Now))
                {
                    await RefreshAsync(cancellationToken).ConfigureAwait(false);
                    await Task.Delay(closedPoll, cancellationToken).ConfigureAwait(false);
                    continue;
                }

                await _oneAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
                PipelineTick tick;

                try
                {
                    tick = await _runner.TickAsync(cancellationToken).ConfigureAwait(false);
                }
                finally
                {
                    _oneAtATime.Release();
                }

                Observe(tick);

                if (tick.Result == ItemResult.Idle)
                {
                    // Nothing to do. Wait as though the window were closed — an import will add
                    // work, and finding it a minute later than it appeared is not worth a loop
                    // that never sleeps.
                    _state = _state with { Running = false, Stage = null, FileName = null };
                    await RefreshAsync(cancellationToken).ConfigureAwait(false);
                    await Task.Delay(closedPoll, cancellationToken).ConfigureAwait(false);
                    continue;
                }

                await RefreshAsync(cancellationToken).ConfigureAwait(false);

                if (_schedule.Gap > TimeSpan.Zero)
                {
                    await Task.Delay(_schedule.Gap, cancellationToken).ConfigureAwait(false);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Stopping is how this loop is meant to end.
        }
        catch (Exception error)
        {
            // The runner already swallows per-photograph failures, so anything reaching here is
            // the loop itself breaking. Log it and stop rather than spin: a loop that throws every
            // iteration would fill the log and the processor with the same message.
            _log.LogError(error, "The analysis queue stopped.");
        }
        finally
        {
            _state = _state with { Running = false, Stage = null, FileName = null };
            Changed?.Invoke();
        }
    }

    private void Observe(PipelineTick tick)
    {
        _state = _state with
        {
            Running = tick.Result is ItemResult.Computed or ItemResult.Reused,
            Stage = tick.Stage,
            FileName = tick.FileName,
        };

        Changed?.Invoke();
    }
}
