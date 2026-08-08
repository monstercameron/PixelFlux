namespace PixelFlux.Core.Pipeline;

/// <summary>
/// The analyses PixelFlux runs over a photograph, in the order it runs them.
/// </summary>
/// <remarks>
/// <para>
/// The numbers are the order, and the order is load-bearing rather than decorative. A stage never
/// starts for a photograph until every earlier stage has finished with it, so the sequence here is
/// the sequence on the machine.
/// </para>
/// <para>
/// Why this sequence. <see cref="Describe"/> is first because its output is an input: the search
/// vector is measurably better when the encoder can read a sentence about the picture as well as
/// look at it, so the description has to exist before <see cref="Embed"/> runs. It is also the
/// slowest stage by an order of magnitude, and putting the slow stage first means a photograph
/// that has been through the queue at all has been through the part that matters most.
/// <see cref="Segment"/> and <see cref="Faces"/> depend on nothing and could run in any order;
/// they sit in the middle because they are what makes a photograph browsable, and a person
/// watching the queue would rather see objects and faces appear than a vector they cannot see.
/// <see cref="Embed"/> is last because it is the one stage that reads another stage's work.
/// </para>
/// </remarks>
public enum PipelineStage
{
    /// <summary>A vision model looks at the photograph and writes a paragraph about it.</summary>
    Describe = 0,

    /// <summary>A segmenter finds objects and outlines them.</summary>
    Segment = 1,

    /// <summary>A detector finds faces and measures each one.</summary>
    Faces = 2,

    /// <summary>An encoder turns the photograph, and its description, into a search vector.</summary>
    Embed = 3,
}

/// <summary>Where a photograph has got to in one stage.</summary>
public enum JobState
{
    /// <summary>Not done yet. The runner will pick it up.</summary>
    Pending = 0,

    /// <summary>Claimed by a runner right now.</summary>
    Running = 1,

    /// <summary>Finished. Will not run again unless the model version changes.</summary>
    Done = 2,

    /// <summary>
    /// Tried and threw. Retried on later passes until <see cref="PipelineStages.MaximumAttempts"/>,
    /// then left alone.
    /// </summary>
    Failed = 3,

    /// <summary>
    /// Deliberately not run — the stage has no model installed, or the file cannot be read.
    /// Distinct from <see cref="Failed"/> because nothing is wrong and nobody should be told
    /// anything is.
    /// </summary>
    Skipped = 4,
}

/// <summary>Facts about the stages that both the queue and the runner need.</summary>
public static class PipelineStages
{
    /// <summary>
    /// How many times a stage is retried before the queue stops offering it.
    /// </summary>
    /// <remarks>
    /// Three, because the two reasons a stage fails have different shapes and this number has to
    /// serve both. Transient reasons — a file locked by whatever wrote it, memory briefly gone —
    /// clear within a retry or two. Permanent reasons — a truncated JPEG, a format the decoder
    /// does not know — never clear, and without a cap the runner would spend the library's whole
    /// life rediscovering the same broken file. The failure is kept on the row either way, so a
    /// stopped job is visible rather than merely absent.
    /// </remarks>
    public const int MaximumAttempts = 3;

    /// <summary>Every stage, in the order the runner works through them.</summary>
    public static IReadOnlyList<PipelineStage> InOrder { get; } =
    [
        PipelineStage.Describe,
        PipelineStage.Segment,
        PipelineStage.Faces,
        PipelineStage.Embed,
    ];

    /// <summary>The name a stage is stored and logged under.</summary>
    /// <param name="stage">The stage.</param>
    /// <returns>A short lowercase token, stable across releases.</returns>
    /// <remarks>
    /// Spelled out rather than <c>ToString().ToLower()</c>, because these strings are in the
    /// database. Renaming the enum member should be a refactor, not a silent migration that
    /// orphans every row written under the old spelling.
    /// </remarks>
    public static string Slug(this PipelineStage stage) => stage switch
    {
        PipelineStage.Describe => "describe",
        PipelineStage.Segment => "segment",
        PipelineStage.Faces => "faces",
        PipelineStage.Embed => "embed",
        _ => throw new ArgumentOutOfRangeException(nameof(stage)),
    };

    /// <summary>The stage a stored slug refers to.</summary>
    /// <param name="slug">A value previously returned by <see cref="Slug(PipelineStage)"/>.</param>
    /// <returns>The stage, or null if the slug belongs to a version that no longer exists.</returns>
    public static PipelineStage? FromSlug(string slug) => slug switch
    {
        "describe" => PipelineStage.Describe,
        "segment" => PipelineStage.Segment,
        "faces" => PipelineStage.Faces,
        "embed" => PipelineStage.Embed,
        _ => null,
    };

    /// <summary>The name a stored state is written under.</summary>
    /// <param name="state">The state.</param>
    /// <returns>A short lowercase token, stable across releases.</returns>
    public static string Slug(this JobState state) => state switch
    {
        JobState.Pending => "pending",
        JobState.Running => "running",
        JobState.Done => "done",
        JobState.Failed => "failed",
        JobState.Skipped => "skipped",
        _ => throw new ArgumentOutOfRangeException(nameof(state)),
    };

    /// <summary>The state a stored slug refers to.</summary>
    /// <param name="slug">A value previously returned by <see cref="Slug(JobState)"/>.</param>
    /// <returns>The state; unknown slugs read as <see cref="JobState.Pending"/> so a row written
    /// by a newer build is retried rather than treated as finished.</returns>
    public static JobState StateFromSlug(string slug) => slug switch
    {
        "running" => JobState.Running,
        "done" => JobState.Done,
        "failed" => JobState.Failed,
        "skipped" => JobState.Skipped,
        _ => JobState.Pending,
    };
}
