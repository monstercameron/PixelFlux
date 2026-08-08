namespace PixelFlux.Core.Pipeline;

/// <summary>
/// Does one stage's work for one photograph.
/// </summary>
/// <remarks>
/// <para>
/// The split between the two methods is the whole reason the cache works. <see cref="ExecuteAsync"/>
/// does the expensive thing and hands back a serialised result; <see cref="ApplyAsync"/> takes that
/// same serialised result and writes it into the library. When a photograph has been analysed
/// before — the same bytes, the same model — only the second half runs, and a sixteen-second stage
/// becomes a millisecond one.
/// </para>
/// <para>
/// Which means the two must agree. Whatever <see cref="ExecuteAsync"/> returns has to be enough for
/// <see cref="ApplyAsync"/> to reproduce every database write the stage made, or a cache hit will
/// quietly produce a less complete photograph than a miss — the kind of bug that only shows up on a
/// second import, months later. The contract is: execute writes nothing that apply could not.
/// </para>
/// </remarks>
public interface IStageHandler
{
    /// <summary>Which stage this handles.</summary>
    PipelineStage Stage { get; }

    /// <summary>
    /// The model version in use, or null when the stage cannot run at all.
    /// </summary>
    /// <remarks>
    /// Null is not an error, it is an absence: no model file installed, no optional dependency
    /// present. The runner marks such work skipped and moves on, which is what keeps a partial
    /// installation from filling the queue with failures nobody can act on. The string is also the
    /// cache key and the staleness check, so it has to change whenever the output would.
    /// </remarks>
    string? ModelVersion { get; }

    /// <summary>Writes a previously computed result into the library.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="payload">A payload this handler produced earlier, from the cache.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    Task ApplyAsync(long photoId, string payload, CancellationToken cancellationToken);

    /// <summary>Runs the model and writes the result into the library.</summary>
    /// <param name="photoId">The photograph.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>
    /// The result, serialised, for the cache — or null when there is nothing worth caching, such
    /// as a photograph whose file could not be found.
    /// </returns>
    Task<string?> ExecuteAsync(long photoId, CancellationToken cancellationToken);
}
