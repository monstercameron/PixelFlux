using Microsoft.Extensions.Logging;

namespace PixelFlux.Core.Pipeline;

/// <summary>What happened to one item.</summary>
public enum ItemResult
{
    /// <summary>The model ran.</summary>
    Computed = 0,

    /// <summary>A previous result for the same image and model was reused.</summary>
    Reused = 1,

    /// <summary>The stage has no model installed, or the file is gone.</summary>
    Skipped = 2,

    /// <summary>The stage threw. It will be retried until the attempt limit.</summary>
    Failed = 3,

    /// <summary>There was nothing to do.</summary>
    Idle = 4,
}

/// <summary>One item's worth of progress, for a status readout.</summary>
/// <param name="Result">What happened.</param>
/// <param name="Stage">Which stage ran, if one did.</param>
/// <param name="FileName">Which photograph, if one was picked up.</param>
/// <param name="Elapsed">How long it took.</param>
public readonly record struct PipelineTick(
    ItemResult Result,
    PipelineStage? Stage,
    string? FileName,
    TimeSpan Elapsed);

/// <summary>
/// Works through the analysis queue one photograph at a time.
/// </summary>
/// <remarks>
/// <para>
/// One at a time is a decision, not a simplification. Every stage here is already using as much of
/// the processor as it can — the vision model was measured fastest at six threads and slower at
/// eighteen — so running two photographs at once does not halve the wall clock, it doubles the
/// contention and makes the machine unpleasant to use. The throughput knob that actually works is
/// the gap between items, not the number of them.
/// </para>
/// <para>
/// The runner holds no queue state of its own. Every tick re-asks the database what is next, which
/// is what makes the schedule, a photograph imported mid-run, and closing the application halfway
/// through all behave sensibly without any of them being special cases.
/// </para>
/// </remarks>
public sealed class PipelineRunner
{
    private readonly JobStore _jobs;
    private readonly StageCache _cache;
    private readonly IReadOnlyDictionary<PipelineStage, IStageHandler> _handlers;
    private readonly ILogger<PipelineRunner> _log;

    /// <summary>Creates a runner over a queue, a cache and a set of stage handlers.</summary>
    /// <param name="jobs">The queue.</param>
    /// <param name="cache">Where results are remembered between runs.</param>
    /// <param name="handlers">
    /// One handler per stage. A stage with no handler is skipped, which is how a build without the
    /// AI project still runs the queue rather than failing to start.
    /// </param>
    /// <param name="log">Where progress and failures go.</param>
    public PipelineRunner(
        JobStore jobs,
        StageCache cache,
        IEnumerable<IStageHandler> handlers,
        ILogger<PipelineRunner> log)
    {
        ArgumentNullException.ThrowIfNull(jobs);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(handlers);
        ArgumentNullException.ThrowIfNull(log);

        _jobs = jobs;
        _cache = cache;
        _log = log;
        _handlers = handlers.ToDictionary(handler => handler.Stage);
    }

    /// <summary>Does one item of work.</summary>
    /// <param name="cancellationToken">Cancels the item.</param>
    /// <returns>What happened, including <see cref="ItemResult.Idle"/> when the queue is empty.</returns>
    /// <remarks>
    /// A single tick rather than a loop, so the thing that decides how often to call it — the
    /// schedule, a progress dialog, a test — is not also the thing doing the work. It never throws
    /// for a failure inside a stage: that is recorded on the job row and reported as a result,
    /// because one unreadable photograph must not stop the other nine thousand.
    /// </remarks>
    public async Task<PipelineTick> TickAsync(CancellationToken cancellationToken = default)
    {
        long started = System.Diagnostics.Stopwatch.GetTimestamp();

        PipelineJob? claimed = await _jobs.ClaimNextAsync(cancellationToken).ConfigureAwait(false);
        if (claimed is not { } job)
        {
            return new PipelineTick(ItemResult.Idle, null, null, TimeSpan.Zero);
        }

        TimeSpan Elapsed() => System.Diagnostics.Stopwatch.GetElapsedTime(started);

        if (!_handlers.TryGetValue(job.Stage, out IStageHandler? handler) ||
            handler.ModelVersion is not { } model)
        {
            await _jobs.SkipAsync(job.PhotoId, job.Stage, "No model installed for this stage.",
                cancellationToken).ConfigureAwait(false);
            return new PipelineTick(ItemResult.Skipped, job.Stage, job.FileName, Elapsed());
        }

        try
        {
            string? cached = await _cache
                .GetAsync(job.ContentHash, job.Stage, model, cancellationToken)
                .ConfigureAwait(false);

            if (cached is not null)
            {
                await handler.ApplyAsync(job.PhotoId, cached, cancellationToken)
                    .ConfigureAwait(false);
                await _jobs.CompleteAsync(job.PhotoId, job.Stage, model, cancellationToken)
                    .ConfigureAwait(false);

                _log.LogDebug("Reused {Stage} for {File}.", job.Stage, job.FileName);
                return new PipelineTick(ItemResult.Reused, job.Stage, job.FileName, Elapsed());
            }

            string? payload = await handler.ExecuteAsync(job.PhotoId, cancellationToken)
                .ConfigureAwait(false);

            if (payload is null)
            {
                // The handler declined — usually a file that has moved out from under the index.
                // Not a failure: there is nothing to retry and nothing anybody can fix from here.
                await _jobs.SkipAsync(job.PhotoId, job.Stage, "Nothing to analyse.",
                    cancellationToken).ConfigureAwait(false);
                return new PipelineTick(ItemResult.Skipped, job.Stage, job.FileName, Elapsed());
            }

            await _cache.PutAsync(job.ContentHash, job.Stage, model, payload, cancellationToken)
                .ConfigureAwait(false);
            await _jobs.CompleteAsync(job.PhotoId, job.Stage, model, cancellationToken)
                .ConfigureAwait(false);

            _log.LogInformation("Ran {Stage} on {File} in {Ms} ms.",
                job.Stage, job.FileName, (int)Elapsed().TotalMilliseconds);
            return new PipelineTick(ItemResult.Computed, job.Stage, job.FileName, Elapsed());
        }
        catch (OperationCanceledException)
        {
            // Being asked to stop is not the photograph's fault. Put the claim back untouched so
            // the next run picks it up with its attempt count intact.
            await _jobs.RequeueOneAsync(job.PhotoId, job.Stage, job.Attempts, CancellationToken.None)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception error)
        {
            await _jobs.FailAsync(job.PhotoId, job.Stage, job.Attempts, error.Message,
                CancellationToken.None).ConfigureAwait(false);
            _log.LogWarning(error, "{Stage} failed on {File}.", job.Stage, job.FileName);
            return new PipelineTick(ItemResult.Failed, job.Stage, job.FileName, Elapsed());
        }
    }

    /// <summary>Works through the queue until it is empty, pausing between items.</summary>
    /// <param name="gap">How long to wait after each photograph.</param>
    /// <param name="limit">Stop after this many items, or null to run the queue dry.</param>
    /// <param name="progress">Called after every item.</param>
    /// <param name="cancellationToken">Stops the run at the next item boundary.</param>
    /// <returns>How many items were handled.</returns>
    /// <remarks>
    /// The pause is after the work, not before, so pressing run does something immediately. A tick
    /// that found nothing to do returns rather than waiting — deciding when to look again belongs
    /// to the caller that knows the schedule.
    /// </remarks>
    public async Task<int> DrainAsync(
        TimeSpan gap,
        int? limit = null,
        IProgress<PipelineTick>? progress = null,
        CancellationToken cancellationToken = default)
    {
        int handled = 0;

        while (limit is null || handled < limit)
        {
            PipelineTick tick = await TickAsync(cancellationToken).ConfigureAwait(false);
            if (tick.Result == ItemResult.Idle)
            {
                break;
            }

            handled++;
            progress?.Report(tick);

            if (gap > TimeSpan.Zero)
            {
                await Task.Delay(gap, cancellationToken).ConfigureAwait(false);
            }
        }

        return handled;
    }
}
