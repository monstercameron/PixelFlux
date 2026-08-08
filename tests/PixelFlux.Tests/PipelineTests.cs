using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using PixelFlux.Core.Pipeline;
using Microsoft.Extensions.Logging.Abstractions;

namespace PixelFlux.Tests;

/// <summary>
/// The analysis queue: ordering, retries, caching and the schedule.
///
/// No models here. Every stage is a fake that records what it was asked to do, because the
/// questions worth asking of a queue — does it run stages in order, does it stop retrying a broken
/// photograph, does the same image cost the model twice — are all answerable without spending
/// sixteen seconds a photograph to ask them.
/// </summary>
public sealed class PipelineTests : IAsyncLifetime
{
    private string _workDir = string.Empty;
    private PhotoDatabase _database = null!;
    private PhotoStore _store = null!;
    private JobStore _jobs = null!;
    private StageCache _cache = null!;

    public Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-pipeline-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_workDir);

        _database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        _database.Migrate();
        _store = new PhotoStore(_database);
        _jobs = new JobStore(_database);
        _cache = new StageCache(_database);

        return Task.CompletedTask;
    }

    public Task DisposeAsync()
    {
        try
        {
            Directory.Delete(_workDir, recursive: true);
        }
        catch (IOException)
        {
            // A model file still mapped, or a slow virus scanner. The directory is in the
            // temporary folder and its removal is not what is under test.
        }

        return Task.CompletedTask;
    }

    /// <summary>Records every call, and can be told to throw.</summary>
    private sealed class FakeStage(PipelineStage stage, string? model = "v1") : IStageHandler
    {
        public PipelineStage Stage { get; } = stage;

        public string? ModelVersion { get; set; } = model;

        public List<long> Executed { get; } = [];

        public List<long> Applied { get; } = [];

        public int ThrowTimes { get; set; }

        public Task ApplyAsync(long photoId, string payload, CancellationToken cancellationToken)
        {
            Applied.Add(photoId);
            return Task.CompletedTask;
        }

        public Task<string?> ExecuteAsync(long photoId, CancellationToken cancellationToken)
        {
            if (ThrowTimes > 0)
            {
                ThrowTimes--;
                throw new InvalidOperationException("stage failed on purpose");
            }

            Executed.Add(photoId);
            return Task.FromResult<string?>($"result for {photoId}");
        }
    }

    private async Task<long> AddPhotoAsync(string name, string contentHash)
    {
        var photo = new PhotoRecord
        {
            FileName = name,
            OriginalPath = Path.Combine(_workDir, name),
            ContentHash = contentHash,
            // Distinct per photograph, because two rows sharing one is a near-duplicate as far as
            // the rest of the library is concerned and that is not what these tests are about.
            PerceptualHash = contentHash.PadRight(16, '0')[..16],
            MimeType = "image/jpeg",
        };

        (long id, _) = await _store.UpsertAsync(photo);
        return id;
    }

    private PipelineRunner Runner(params IStageHandler[] handlers) =>
        new(_jobs, _cache, handlers, NullLogger<PipelineRunner>.Instance);

    [Fact]
    public async Task EveryStageRunsBeforeTheNextStageStarts()
    {
        // Three photographs, two stages. Stage-first ordering means all three are described
        // before any is segmented — a part-analysed library should be shallow across the whole
        // collection, not deep on a couple of photographs.
        for (int i = 0; i < 3; i++)
        {
            await AddPhotoAsync($"photo{i}.jpg", $"hash{i}");
        }

        await _jobs.ReconcileAsync();

        var order = new List<PipelineStage>();
        var describe = new FakeStage(PipelineStage.Describe);
        var segment = new FakeStage(PipelineStage.Segment);
        PipelineRunner runner = Runner(describe, segment);

        for (int i = 0; i < 12; i++)
        {
            PipelineTick tick = await runner.TickAsync();
            if (tick.Result == ItemResult.Idle)
            {
                break;
            }

            if (tick.Stage is { } stage)
            {
                order.Add(stage);
            }
        }

        List<PipelineStage> describes = [.. order.Where(s => s == PipelineStage.Describe)];
        Assert.Equal(3, describes.Count);

        int firstSegment = order.IndexOf(PipelineStage.Segment);
        int lastDescribe = order.LastIndexOf(PipelineStage.Describe);
        Assert.True(
            lastDescribe < firstSegment,
            $"segmentation started at {firstSegment} before describing finished at {lastDescribe}");
    }

    [Fact]
    public async Task AnImageAnalysedBeforeIsNotAnalysedAgainUnderANewId()
    {
        // What the content-hash key actually buys, established by this test failing first.
        //
        // It is NOT "the same file imported twice", which was the claim in the first draft of
        // this design: photos.content_hash is unique, so importing the same bytes twice produces
        // one row, and there is no second analysis to save. What it buys is that a photograph's
        // ANALYSIS outlives its ROW. Remove a photograph and put it back — reorganising folders
        // does this, and so does rebuilding the index — and it comes back with a new id and the
        // same bytes. Without the cache that is sixteen seconds of vision model per photograph to
        // recompute an answer that cannot have changed.
        long before = await AddPhotoAsync("photo.jpg", "the-same-bytes");
        await _jobs.ReconcileAsync();

        var describe = new FakeStage(PipelineStage.Describe);
        Assert.Equal(ItemResult.Computed, (await Runner(describe).TickAsync()).Result);
        Assert.Single(describe.Executed);

        // Straight to SQL: the library has no delete yet, and the point of this test is what the
        // cache does when a row goes away, not how it goes away.
        await using (Microsoft.Data.Sqlite.SqliteConnection connection = _database.Open())
        {
            await using Microsoft.Data.Sqlite.SqliteCommand remove = connection.CreateCommand();
            remove.CommandText = "DELETE FROM photos WHERE id = $id;";
            remove.Parameters.AddWithValue("$id", before);
            await remove.ExecuteNonQueryAsync();
        }

        long after = await AddPhotoAsync("photo.jpg", "the-same-bytes");
        Assert.NotEqual(before, after);

        await _jobs.ReconcileAsync();

        PipelineTick again = await Runner(describe).TickAsync();

        Assert.Equal(ItemResult.Reused, again.Result);
        Assert.Single(describe.Executed);          // the model did not run a second time
        Assert.Equal([after], describe.Applied);   // and the new row still got the result
    }

    [Fact]
    public async Task ABrokenPhotographStopsBeingRetried()
    {
        await AddPhotoAsync("broken.jpg", "broken");
        await _jobs.ReconcileAsync();

        var describe = new FakeStage(PipelineStage.Describe) { ThrowTimes = 99 };
        PipelineRunner runner = Runner(describe);

        for (int i = 0; i < 10; i++)
        {
            await runner.TickAsync();
        }

        IReadOnlyList<StageStatus> status = await _jobs.StatusAsync();
        StageStatus stage = status.Single(s => s.Stage == PipelineStage.Describe);

        Assert.Equal(1, stage.Stuck);
        Assert.Equal(0, stage.Pending);
        Assert.False(stage.HasWork);
    }

    [Fact]
    public async Task AStageThatCannotRunDoesNotBlockTheOnesAfterIt()
    {
        // No vision model installed. Describing is skipped, and the stages that do not depend on
        // it must still run — a missing optional model should cost one feature, not all of them.
        await AddPhotoAsync("photo.jpg", "hash");
        await _jobs.ReconcileAsync();

        var describe = new FakeStage(PipelineStage.Describe, model: null);
        var segment = new FakeStage(PipelineStage.Segment);
        PipelineRunner runner = Runner(describe, segment);

        PipelineTick skipped = await runner.TickAsync();
        PipelineTick ran = await runner.TickAsync();

        Assert.Equal(ItemResult.Skipped, skipped.Result);
        Assert.Equal(ItemResult.Computed, ran.Result);
        Assert.Single(segment.Executed);
    }

    [Fact]
    public async Task AStuckStageDoesNotStrandTheStagesAfterIt()
    {
        // The same rule as a skipped stage, for the harder case: a stage that failed its way out
        // of retries. Out-of-retries has to count as terminal, or one unreadable photograph would
        // park itself at the head of its own queue forever.
        await AddPhotoAsync("broken.jpg", "broken");
        await _jobs.ReconcileAsync();

        var describe = new FakeStage(PipelineStage.Describe) { ThrowTimes = 99 };
        var segment = new FakeStage(PipelineStage.Segment);
        PipelineRunner runner = Runner(describe, segment);

        await runner.DrainAsync(TimeSpan.Zero, limit: 20);

        Assert.Single(segment.Executed);
    }

    [Fact]
    public async Task WorkTheOldSweepsDidIsNotRepeated()
    {
        long described = await AddPhotoAsync("described.jpg", "a");
        await AddPhotoAsync("fresh.jpg", "b");

        await _store.SetDescriptionAsync(described, "A photograph of something.", "old-model");
        await _jobs.ReconcileAsync();

        int adopted = await _jobs.AdoptExistingWorkAsync();
        Assert.Equal(1, adopted);

        var describe = new FakeStage(PipelineStage.Describe);
        await Runner(describe).DrainAsync(TimeSpan.Zero, limit: 10);

        // Only the photograph that had no description was described.
        Assert.Single(describe.Executed);
        Assert.DoesNotContain(described, describe.Executed);
    }

    [Fact]
    public async Task AClaimLeftByACrashIsReleasedButALiveOneIsNot()
    {
        await AddPhotoAsync("photo.jpg", "hash");
        await _jobs.ReconcileAsync();

        PipelineJob? claimed = await _jobs.ClaimNextAsync();
        Assert.NotNull(claimed);

        // A second runner must not be able to take the same item.
        Assert.Null(await _jobs.ClaimNextAsync());

        // Nor may it steal the claim by "cleaning up" after a process that is still working. This
        // is the case that matters when the application and the command line are both open: the
        // second one to start would otherwise hand itself a photograph the first was mid-way
        // through, and both would write a result for it.
        Assert.Equal(0, await _jobs.ReleaseClaimsAsync());
        Assert.Null(await _jobs.ClaimNextAsync());

        // Age the claim past the threshold, as a crashed process's row would be.
        await using (Microsoft.Data.Sqlite.SqliteConnection connection = _database.Open())
        {
            await using Microsoft.Data.Sqlite.SqliteCommand age = connection.CreateCommand();
            age.CommandText = "UPDATE photo_jobs SET updated_at = $then WHERE state = 'running';";
            age.Parameters.AddWithValue(
                "$then",
                DateTimeOffset.UtcNow
                    .Subtract(JobStore.ClaimIsStaleAfter)
                    .AddMinutes(-1)
                    .ToString("O"));
            await age.ExecuteNonQueryAsync();
        }

        Assert.Equal(1, await _jobs.ReleaseClaimsAsync());
        Assert.NotNull(await _jobs.ClaimNextAsync());
    }

    [Fact]
    public async Task OnePhotographCanExplainItself()
    {
        // The question the viewer asks: not "how is the library doing" but "why does this one have
        // no description". Waiting, tried-and-failed, and out-of-retries are three different
        // answers and the caller has to be able to tell them apart.
        long photoId = await AddPhotoAsync("broken.jpg", "broken");
        await _jobs.ReconcileAsync();

        IReadOnlyList<(PipelineStage Stage, JobState State, string? Error)> fresh =
            await _jobs.ForPhotoAsync(photoId);

        Assert.Equal(PipelineStages.InOrder.Count, fresh.Count);
        Assert.All(fresh, entry => Assert.Equal(JobState.Pending, entry.State));

        // Run it out of retries.
        var describe = new FakeStage(PipelineStage.Describe) { ThrowTimes = 99 };
        await Runner(describe).DrainAsync(TimeSpan.Zero, limit: 10);

        IReadOnlyList<(PipelineStage Stage, JobState State, string? Error)> after =
            await _jobs.ForPhotoAsync(photoId);

        (PipelineStage Stage, JobState State, string? Error) read =
            after.Single(entry => entry.Stage == PipelineStage.Describe);

        // Reported as Skipped rather than Failed: the retry budget is spent, so "will be tried
        // again" would be a lie. The stored row still says failed; the distinction is the
        // attempt count, and resolving it here is why callers do not have to.
        Assert.Equal(JobState.Skipped, read.State);
        Assert.False(string.IsNullOrWhiteSpace(read.Error));
    }

    [Fact]
    public async Task RetryingStuckWorkDoesNotRedoWorkThatSucceeded()
    {
        // The distinction the button depends on. Somebody clearing one broken file must not
        // thereby re-analyse a library that took nine hours.
        long broken = await AddPhotoAsync("broken.jpg", "broken");
        long fine = await AddPhotoAsync("fine.jpg", "fine");
        await _jobs.ReconcileAsync();

        // One photograph fails forever; the runner reaches the other because out-of-retries is
        // terminal and does not block the queue.
        var describe = new FailFor(PipelineStage.Describe, broken);
        await Runner(describe).DrainAsync(TimeSpan.Zero, limit: 20);

        Assert.Equal([fine], describe.Executed);

        int retried = await _jobs.RetryStuckAsync();
        Assert.Equal(1, retried);

        await Runner(describe).DrainAsync(TimeSpan.Zero, limit: 20);

        // The broken one was tried again; the good one was not touched a second time.
        Assert.Equal(1, describe.Executed.Count(id => id == fine));

        // Six, not four: retrying restores the whole budget rather than granting one more go.
        // That is the right shape — somebody pressing "try again" has usually just fixed the
        // cause, and a single attempt against a transient fault would strand the file again on
        // the first hiccup — but it does mean a genuinely dead file costs three attempts per
        // press rather than one.
        Assert.Equal(
            PipelineStages.MaximumAttempts * 2,
            describe.Attempts.Count(id => id == broken));
    }

    /// <summary>Fails for one photograph and succeeds for every other.</summary>
    private sealed class FailFor(PipelineStage stage, long failing) : IStageHandler
    {
        public PipelineStage Stage { get; } = stage;

        public string? ModelVersion => "v1";

        public List<long> Executed { get; } = [];

        public List<long> Attempts { get; } = [];

        public Task ApplyAsync(long photoId, string payload, CancellationToken cancellationToken) =>
            Task.CompletedTask;

        public Task<string?> ExecuteAsync(long photoId, CancellationToken cancellationToken)
        {
            Attempts.Add(photoId);

            if (photoId == failing)
            {
                throw new InvalidOperationException("this one is broken");
            }

            Executed.Add(photoId);
            return Task.FromResult<string?>("ok");
        }
    }

    [Theory]
    // A window that crosses midnight is the normal case for this feature, not the edge case.
    [InlineData(22, 6, 23, true)]
    [InlineData(22, 6, 3, true)]
    [InlineData(22, 6, 5, true)]
    [InlineData(22, 6, 6, false)]
    [InlineData(22, 6, 12, false)]
    [InlineData(22, 6, 21, false)]
    // And one that does not.
    [InlineData(9, 17, 12, true)]
    [InlineData(9, 17, 8, false)]
    [InlineData(9, 17, 17, false)]
    public void TheWindowIncludesTheRightHours(int start, int end, int hour, bool expected)
    {
        var schedule = new PipelineSchedule(
            ScheduleMode.Window,
            new TimeOnly(start, 0),
            new TimeOnly(end, 0),
            TimeSpan.FromSeconds(20));

        var moment = new DateTime(2026, 8, 8, hour, 30, 0, DateTimeKind.Local);
        Assert.Equal(expected, schedule.IsOpenAt(moment));
    }

    [Fact]
    public void AScheduleSurvivesBeingStoredAndReadBack()
    {
        var original = new PipelineSchedule(
            ScheduleMode.Window,
            new TimeOnly(22, 30),
            new TimeOnly(6, 15),
            TimeSpan.FromSeconds(45));

        Assert.Equal(original, PipelineSchedule.Parse(original.Serialise()));
    }

    [Fact]
    public void AnUnreadableScheduleFallsBackRatherThanThrowing()
    {
        // A setting written by a newer build should cost the user their schedule, not their
        // photographs.
        Assert.Equal(PipelineSchedule.Default, PipelineSchedule.Parse("nonsense"));
        Assert.Equal(PipelineSchedule.Default, PipelineSchedule.Parse(null));
        Assert.Equal(PipelineSchedule.Default, PipelineSchedule.Parse(string.Empty));
    }

    [Fact]
    public async Task ChangingTheModelMakesTheWorkOutstandingAgain()
    {
        await AddPhotoAsync("photo.jpg", "hash");
        await _jobs.ReconcileAsync();

        var describe = new FakeStage(PipelineStage.Describe, "v1");
        await Runner(describe).DrainAsync(TimeSpan.Zero, limit: 5);
        Assert.Single(describe.Executed);

        // A better model arrives. Requeueing only what is stale leaves nothing else disturbed.
        describe.ModelVersion = "v2";
        Assert.Equal(1, await _jobs.RequeueAsync(PipelineStage.Describe, onlyStale: "v2"));

        await Runner(describe).DrainAsync(TimeSpan.Zero, limit: 5);
        Assert.Equal(2, describe.Executed.Count);
    }
}
