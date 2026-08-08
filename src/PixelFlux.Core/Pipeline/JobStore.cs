using Microsoft.Data.Sqlite;
using PixelFlux.Core.Index;

namespace PixelFlux.Core.Pipeline;

/// <summary>One unit of work: a photograph and the stage that is next for it.</summary>
/// <param name="PhotoId">The photograph.</param>
/// <param name="Stage">What to do to it.</param>
/// <param name="ContentHash">
/// The photograph's content hash, carried on the job so the runner can look in the cache without
/// a second query. It is also what lets an analysis outlive the row it was computed for.
/// </param>
/// <param name="FileName">Filename, for the progress readout.</param>
/// <param name="Attempts">How many times this stage has already been tried and thrown.</param>
public sealed record PipelineJob(
    long PhotoId,
    PipelineStage Stage,
    string ContentHash,
    string FileName,
    int Attempts);

/// <summary>How far along one stage is across the whole library.</summary>
/// <param name="Stage">The stage.</param>
/// <param name="Done">Photographs finished.</param>
/// <param name="Pending">Photographs waiting.</param>
/// <param name="Failed">Photographs that threw and are still within the retry limit.</param>
/// <param name="Stuck">Photographs that threw and have run out of retries.</param>
/// <param name="Skipped">Photographs deliberately not run — usually a model that is not installed.</param>
public sealed record StageStatus(
    PipelineStage Stage,
    int Done,
    int Pending,
    int Failed,
    int Stuck,
    int Skipped)
{
    /// <summary>Photographs this stage knows about.</summary>
    public int Total => Done + Pending + Failed + Stuck + Skipped;

    /// <summary>How much of the stage is finished, from 0 to 1. An empty stage reads as finished.</summary>
    public double Fraction => Total == 0 ? 1 : (double)(Done + Skipped) / Total;

    /// <summary>Whether anything is left that the runner would pick up.</summary>
    public bool HasWork => Pending > 0 || Failed > 0;
}

/// <summary>
/// The analysis queue: which photographs still need which stage, and in what order.
/// </summary>
/// <remarks>
/// <para>
/// This is deliberately a plain table rather than an in-memory queue. Analysis takes hours and the
/// application is closed several times during them, so the queue has to be the kind of thing that
/// survives being switched off mid-item — which means the durable record is the queue, not a
/// projection of one. It also means the queue can be read: "what is left" is a query anybody can
/// run, and a photograph that is stuck says so on its own row.
/// </para>
/// <para>
/// One writer. Claiming marks a row <see cref="JobState.Running"/> inside a transaction, so a
/// second runner cannot take the same item, but nothing here tries to be a distributed scheduler.
/// A photo library is one machine at a time, and the cost of pretending otherwise is a lease
/// column, a clock, and a class of bug that only appears on somebody else's computer.
/// </para>
/// </remarks>
public sealed class JobStore
{
    private readonly PhotoDatabase _database;

    /// <summary>Creates a queue over a migrated database.</summary>
    /// <param name="database">The database handle.</param>
    public JobStore(PhotoDatabase database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
    }

    /// <summary>Gives every photograph a row for every stage, leaving existing rows alone.</summary>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many rows were added.</returns>
    /// <remarks>
    /// Called after an import and on startup. Reconciling rather than tracking means a photograph
    /// added by some path that forgot to enqueue it still gets analysed, and adding a new stage to
    /// <see cref="PipelineStages.InOrder"/> is enough to have the whole library queued for it —
    /// no migration, no backfill script.
    /// </remarks>
    public async Task<int> ReconcileAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        int added = 0;
        foreach (PipelineStage stage in PipelineStages.InOrder)
        {
            await using SqliteCommand insert = connection.CreateCommand();
            insert.Transaction = tx;
            insert.CommandText = """
                INSERT INTO photo_jobs (photo_id, stage, ord, state, attempts, updated_at)
                SELECT p.id, $stage, $ord, 'pending', 0, $now
                FROM photos p
                WHERE NOT EXISTS (
                    SELECT 1 FROM photo_jobs j
                    WHERE j.photo_id = p.id AND j.stage = $stage
                );
                """;
            insert.Parameters.AddWithValue("$stage", stage.Slug());
            insert.Parameters.AddWithValue("$ord", (int)stage);
            insert.Parameters.AddWithValue("$now", Now());
            added += await insert.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        return added;
    }

    /// <summary>Marks work the old per-stage sweeps already did as done.</summary>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many rows were adopted.</returns>
    /// <remarks>
    /// <para>
    /// Before this queue existed, each analysis swept the library on its own and recorded its
    /// progress in whatever it happened to write — a description column, the presence of segment
    /// rows, a sweep marker. A library analysed under that arrangement is fully analysed; it just
    /// has no queue rows saying so, and without this it would spend hours recomputing answers it
    /// already has.
    /// </para>
    /// <para>
    /// Evidence-based rather than optimistic: a stage is adopted only where the thing it produces
    /// is actually present, and the model recorded is the one that produced it, so a stage whose
    /// model has since been upgraded stays outstanding. Embedding is deliberately absent — the
    /// vectors it used to write are image-only, and the caption-blended ones that replace them are
    /// a different recipe under a different version. Those genuinely do have to be recomputed.
    /// </para>
    /// </remarks>
    public async Task<int> AdoptExistingWorkAsync(CancellationToken cancellationToken = default)
    {
        (PipelineStage Stage, string Sql)[] evidence =
        [
            // 'legacy' rather than the photo's model_version, which is not the describer's: that
            // column is written by whichever analysis ran last, so on a segmented library it holds
            // the segmenter. Recording a version this cannot actually establish would be worse
            // than recording none — a wrong provenance is acted on, an honest placeholder is not.
            (PipelineStage.Describe, """
                SELECT id, 'legacy' FROM photos
                WHERE ai_description IS NOT NULL AND TRIM(ai_description) <> ''
                """),
            (PipelineStage.Segment, """
                SELECT p.id, COALESCE(MAX(s.model), 'legacy')
                FROM photos p JOIN photo_segments s ON s.photo_id = p.id
                GROUP BY p.id
                """),
            (PipelineStage.Faces, """
                SELECT p.id, COALESCE(MAX(f.model), 'legacy')
                FROM photos p JOIN face_sweeps f ON f.photo_id = p.id
                GROUP BY p.id
                """),
        ];

        int adopted = 0;

        foreach ((PipelineStage stage, string sql) in evidence)
        {
            await using SqliteConnection connection = _database.Open();
            var found = new List<(long PhotoId, string Model)>();

            await using (SqliteCommand read = connection.CreateCommand())
            {
                read.CommandText = sql;

                try
                {
                    await using SqliteDataReader reader =
                        await read.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
                    while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
                    {
                        found.Add((reader.GetInt64(0), reader.GetString(1)));
                    }
                }
                catch (SqliteException)
                {
                    // face_sweeps is created lazily by the face store rather than by a migration,
                    // so on a library that has never been swept the table is simply not there.
                    // Nothing to adopt is a normal answer, not a failure.
                    continue;
                }
            }

            await using SqliteTransaction tx = (SqliteTransaction)await connection
                .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

            foreach ((long photoId, string model) in found)
            {
                await using SqliteCommand update = connection.CreateCommand();
                update.Transaction = tx;
                // Only rows still waiting. A stage somebody has since requeued on purpose, or one
                // that failed, should stay as it is — adopting those would silently undo a
                // deliberate decision.
                update.CommandText = """
                    UPDATE photo_jobs
                    SET state = 'done', model = $model, error = NULL, updated_at = $now
                    WHERE photo_id = $photo AND stage = $stage AND state = 'pending'
                      AND attempts = 0;
                    """;
                update.Parameters.AddWithValue("$photo", photoId);
                update.Parameters.AddWithValue("$stage", stage.Slug());
                update.Parameters.AddWithValue("$model", model);
                update.Parameters.AddWithValue("$now", Now());
                adopted += await update.ExecuteNonQueryAsync(cancellationToken)
                    .ConfigureAwait(false);
            }

            await tx.CommitAsync(cancellationToken).ConfigureAwait(false);
        }

        return adopted;
    }

    /// <summary>Takes the next item of work, marking it as running.</summary>
    /// <param name="cancellationToken">Cancels the claim.</param>
    /// <returns>The job, or null when there is nothing runnable.</returns>
    /// <remarks>
    /// <para>
    /// The ordering rule lives in the <c>NOT EXISTS</c> clause: a stage is only offered once every
    /// earlier stage for that same photograph has reached a terminal state. Terminal includes
    /// skipped and out-of-retries, not just done — otherwise one unreadable file would park itself
    /// at the head of its own queue and that photograph would never reach any later stage.
    /// </para>
    /// <para>
    /// Work goes stage-first, not photograph-first: every photograph gets described before any
    /// gets segmented. On a library that is only part-way through, that produces a shallow front
    /// across the whole collection rather than a handful of perfect photographs and a thousand
    /// untouched ones, and searching a library where everything has a description beats searching
    /// one where a tenth of it has everything.
    /// </para>
    /// </remarks>
    public async Task<PipelineJob?> ClaimNextAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteTransaction tx = (SqliteTransaction)await connection
            .BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        await using SqliteCommand pick = connection.CreateCommand();
        pick.Transaction = tx;
        pick.CommandText = """
            SELECT j.photo_id, j.stage, j.attempts, p.content_hash, p.file_name
            FROM photo_jobs j
            JOIN photos p ON p.id = j.photo_id
            WHERE j.state IN ('pending', 'failed')
              AND j.attempts < $max
              AND NOT EXISTS (
                  SELECT 1 FROM photo_jobs earlier
                  WHERE earlier.photo_id = j.photo_id
                    AND earlier.ord < j.ord
                    AND (earlier.state IN ('pending', 'running')
                         OR (earlier.state = 'failed' AND earlier.attempts < $max))
              )
            ORDER BY j.ord, j.attempts, j.photo_id
            LIMIT 1;
            """;
        pick.Parameters.AddWithValue("$max", PipelineStages.MaximumAttempts);

        long photoId;
        PipelineStage stage;
        int attempts;
        string hash;
        string name;

        await using (SqliteDataReader reader =
            await pick.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false))
        {
            if (!await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                return null;
            }

            photoId = reader.GetInt64(0);
            PipelineStage? parsed = PipelineStages.FromSlug(reader.GetString(1));
            if (parsed is null)
            {
                return null;
            }

            stage = parsed.Value;
            attempts = reader.GetInt32(2);
            hash = reader.GetString(3);
            name = reader.GetString(4);
        }

        await MarkAsync(connection, tx, photoId, stage, JobState.Running, null, attempts, null,
            cancellationToken).ConfigureAwait(false);
        await tx.CommitAsync(cancellationToken).ConfigureAwait(false);

        return new PipelineJob(photoId, stage, hash, name, attempts);
    }

    /// <summary>Records that a stage finished.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="stage">The stage.</param>
    /// <param name="model">
    /// The model version that produced the result. Stored so that installing a better model makes
    /// the row outstanding again without anything having to remember which rows to touch.
    /// </param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task CompleteAsync(
        long photoId,
        PipelineStage stage,
        string model,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await MarkAsync(connection, null, photoId, stage, JobState.Done, model, 0, null,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Records that a stage threw, and counts the attempt.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="stage">The stage.</param>
    /// <param name="attempts">Attempts before this one; the row is stored with one more.</param>
    /// <param name="error">What went wrong, kept so a stuck photograph can explain itself.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task FailAsync(
        long photoId,
        PipelineStage stage,
        int attempts,
        string error,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await MarkAsync(connection, null, photoId, stage, JobState.Failed, null, attempts + 1,
            Trim(error), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Records that a stage was deliberately not run.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="stage">The stage.</param>
    /// <param name="reason">Why, in a few words — usually that the model is not installed.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task SkipAsync(
        long photoId,
        PipelineStage stage,
        string reason,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await MarkAsync(connection, null, photoId, stage, JobState.Skipped, null, 0, Trim(reason),
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Puts one claimed row back without counting an attempt.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="stage">The stage.</param>
    /// <param name="attempts">The attempt count to keep. Defaults to leaving it where it was.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// For a run that was cancelled rather than one that went wrong — closing the application, the
    /// schedule window ending. The distinction matters because attempts are a budget for a
    /// photograph that might be broken, and spending it on a photograph that was merely
    /// interrupted would eventually strand a perfectly good file.
    /// </remarks>
    public async Task RequeueOneAsync(
        long photoId,
        PipelineStage stage,
        int attempts = 0,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await MarkAsync(connection, null, photoId, stage, JobState.Pending, null, attempts, null,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>How long a claim is presumed live before it is treated as abandoned.</summary>
    /// <remarks>
    /// The slowest stage takes about eleven seconds a photograph, so anything still claimed after
    /// fifteen minutes is not being worked on. The margin is deliberately enormous: releasing a
    /// claim that is genuinely live is the expensive mistake, because two runners would then do
    /// the same photograph at once and race to write its result.
    /// </remarks>
    public static TimeSpan ClaimIsStaleAfter { get; } = TimeSpan.FromMinutes(15);

    /// <summary>Puts abandoned claims back, so a run interrupted by a crash resumes cleanly.</summary>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many rows were released.</returns>
    /// <remarks>
    /// <para>
    /// Called on startup. A row left <see cref="JobState.Running"/> usually means a process died
    /// holding it; without this it would stay claimed forever and, because it may be an earlier
    /// stage for its photograph, would block every later stage for that photograph too. The
    /// attempt is not counted — being killed is not the photograph's fault.
    /// </para>
    /// <para>
    /// <b>Only claims older than <see cref="ClaimIsStaleAfter"/>.</b> This used to release every
    /// running row unconditionally, which is correct when nothing else is running and wrong the
    /// moment something is. Opening the application while a command-line run is working, or
    /// starting a command-line run while the application is open, would hand the second runner a
    /// photograph the first was still analysing — both would write results for it, and the loser
    /// of that race would have its work silently overwritten. An age test costs nothing and makes
    /// the two safe to use together.
    /// </para>
    /// </remarks>
    public async Task<int> ReleaseClaimsAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            UPDATE photo_jobs SET state = 'pending', updated_at = $now
            WHERE state = 'running' AND updated_at < $cutoff;
            """;
        command.Parameters.AddWithValue("$now", Now());
        command.Parameters.AddWithValue(
            "$cutoff", DateTimeOffset.UtcNow.Subtract(ClaimIsStaleAfter).ToString("O"));
        return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Marks work outstanding again.</summary>
    /// <param name="stage">The stage to requeue, or null for all of them.</param>
    /// <param name="onlyStale">
    /// When set, only rows whose recorded model differs from this one are requeued — the "a better
    /// model is installed, redo the work it would improve" case. When null, everything is requeued.
    /// </param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many rows were requeued.</returns>
    public async Task<int> RequeueAsync(
        PipelineStage? stage,
        string? onlyStale = null,
        CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = $"""
            UPDATE photo_jobs
            SET state = 'pending', attempts = 0, error = NULL, updated_at = $now
            WHERE (($stage IS NULL) OR stage = $stage)
              AND (($model IS NULL) OR model IS NULL OR model <> $model);
            """;
        command.Parameters.AddWithValue("$now", Now());
        command.Parameters.AddWithValue("$stage", (object?)stage?.Slug() ?? DBNull.Value);
        command.Parameters.AddWithValue("$model", (object?)onlyStale ?? DBNull.Value);
        return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>What every stage has done to one photograph.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>One entry per stage the photograph has a row for, in run order.</returns>
    /// <remarks>
    /// For the question a person actually asks in front of a single photograph: not "how is the
    /// library doing" but "why does this one have no description". Library-wide progress cannot
    /// answer that, and the difference between waiting, failed and no-model-installed is the whole
    /// answer.
    /// </remarks>
    public async Task<IReadOnlyList<(PipelineStage Stage, JobState State, string? Error)>>
        ForPhotoAsync(long photoId, CancellationToken cancellationToken = default)
    {
        var found = new List<(PipelineStage, JobState, string?)>();

        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            SELECT stage, state, error, attempts FROM photo_jobs
            WHERE photo_id = $photo ORDER BY ord;
            """;
        command.Parameters.AddWithValue("$photo", photoId);

        await using SqliteDataReader reader =
            await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            if (PipelineStages.FromSlug(reader.GetString(0)) is not { } stage)
            {
                continue;
            }

            JobState state = PipelineStages.StateFromSlug(reader.GetString(1));
            string? error = reader.IsDBNull(2) ? null : reader.GetString(2);

            // A failure that has used its retries is a different thing to tell somebody than one
            // that will be tried again tonight, and the row does not distinguish them on its own.
            if (state == JobState.Failed && reader.GetInt32(3) >= PipelineStages.MaximumAttempts)
            {
                state = JobState.Skipped;
            }

            found.Add((stage, state, error));
        }

        return found;
    }

    /// <summary>Gives another chance to work that ran out of retries.</summary>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>How many rows were given their retries back.</returns>
    /// <remarks>
    /// <para>
    /// Only the rows that are actually stuck — not the whole stage. A photograph that failed three
    /// times is usually a file that will fail a fourth, so this is not something to do
    /// automatically; but the reasons it failed are often outside the library and fixable. A drive
    /// that was not mounted, a file half-written by whatever was copying it, a model that was not
    /// installed at the time. Once the cause is gone the queue has no way to notice, because
    /// out-of-retries is deliberately permanent.
    /// </para>
    /// <para>
    /// Distinct from <see cref="RequeueAsync"/>, which redoes work that succeeded. Somebody
    /// clearing a handful of failures should not thereby re-analyse the entire library.
    /// </para>
    /// </remarks>
    public async Task<int> RetryStuckAsync(CancellationToken cancellationToken = default)
    {
        await using SqliteConnection connection = _database.Open();
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = """
            UPDATE photo_jobs
            SET state = 'pending', attempts = 0, error = NULL, updated_at = $now
            WHERE state = 'failed' AND attempts >= $max;
            """;
        command.Parameters.AddWithValue("$now", Now());
        command.Parameters.AddWithValue("$max", PipelineStages.MaximumAttempts);
        return await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>How far along every stage is.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>One entry per stage, in run order, including stages with nothing to do.</returns>
    public async Task<IReadOnlyList<StageStatus>> StatusAsync(
        CancellationToken cancellationToken = default)
    {
        var counts = new Dictionary<(PipelineStage Stage, JobState State), int>();
        var stuck = new Dictionary<PipelineStage, int>();

        await using (SqliteConnection connection = _database.Open())
        {
            await using SqliteCommand command = connection.CreateCommand();
            // Out-of-retries is split out here rather than stored as a fifth state, because it is
            // not a different outcome — it is the same failure with the retry budget spent. Making
            // it a state would mean writing it from two places and getting it wrong from one.
            command.CommandText = """
                SELECT stage,
                       state,
                       CASE WHEN state = 'failed' AND attempts >= $max THEN 1 ELSE 0 END AS spent,
                       COUNT(*)
                FROM photo_jobs
                GROUP BY stage, state, spent;
                """;
            command.Parameters.AddWithValue("$max", PipelineStages.MaximumAttempts);

            await using SqliteDataReader reader =
                await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                PipelineStage? stage = PipelineStages.FromSlug(reader.GetString(0));
                if (stage is null)
                {
                    continue;
                }

                JobState state = PipelineStages.StateFromSlug(reader.GetString(1));
                int count = reader.GetInt32(3);

                if (reader.GetInt32(2) == 1)
                {
                    stuck[stage.Value] = stuck.GetValueOrDefault(stage.Value) + count;
                    continue;
                }

                (PipelineStage Stage, JobState State) key = (stage.Value, state);
                counts[key] = counts.GetValueOrDefault(key) + count;
            }
        }

        return
        [
            .. PipelineStages.InOrder.Select(stage => new StageStatus(
                stage,
                Done: counts.GetValueOrDefault((stage, JobState.Done)),
                // A row claimed by the runner right now is still outstanding as far as anybody
                // reading a progress bar is concerned.
                Pending: counts.GetValueOrDefault((stage, JobState.Pending))
                         + counts.GetValueOrDefault((stage, JobState.Running)),
                Failed: counts.GetValueOrDefault((stage, JobState.Failed)),
                Stuck: stuck.GetValueOrDefault(stage),
                Skipped: counts.GetValueOrDefault((stage, JobState.Skipped)))),
        ];
    }

    private static async Task MarkAsync(
        SqliteConnection connection,
        SqliteTransaction? tx,
        long photoId,
        PipelineStage stage,
        JobState state,
        string? model,
        int attempts,
        string? error,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = tx;
        // Upsert rather than update. A stage can be completed for a photograph the reconcile has
        // not reached yet — an import that analyses as it goes, a single-photo redo from the
        // viewer — and losing that result because no row existed to update would be silent.
        command.CommandText = """
            INSERT INTO photo_jobs (photo_id, stage, ord, state, model, attempts, error, updated_at)
            VALUES ($photo, $stage, $ord, $state, $model, $attempts, $error, $now)
            ON CONFLICT(photo_id, stage) DO UPDATE SET
                state      = excluded.state,
                model      = excluded.model,
                attempts   = excluded.attempts,
                error      = excluded.error,
                updated_at = excluded.updated_at;
            """;
        command.Parameters.AddWithValue("$photo", photoId);
        command.Parameters.AddWithValue("$stage", stage.Slug());
        command.Parameters.AddWithValue("$ord", (int)stage);
        command.Parameters.AddWithValue("$state", state.Slug());
        command.Parameters.AddWithValue("$model", (object?)model ?? DBNull.Value);
        command.Parameters.AddWithValue("$attempts", attempts);
        command.Parameters.AddWithValue("$error", (object?)error ?? DBNull.Value);
        command.Parameters.AddWithValue("$now", Now());
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static string Now() => DateTimeOffset.UtcNow.ToString("O");

    /// <summary>Keeps a stack trace from becoming the largest thing in the database.</summary>
    private static string Trim(string text) =>
        text.Length <= 500 ? text : text[..500];
}
