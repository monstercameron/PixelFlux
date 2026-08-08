namespace PixelFlux.Tests;

/// <summary>
/// Groups every test class that runs a neural network, so they never run at the same time.
/// </summary>
/// <remarks>
/// <para>
/// xUnit puts each test class in its own collection and runs collections in parallel. That is
/// right for tests that touch a temp directory and wrong for tests that saturate the CPU: ONNX
/// Runtime already spreads one inference across half the cores, so two classes doing it at once
/// simply halve each other's throughput.
/// </para>
/// <para>
/// The correctness assertions do not care. The timing assertions do, and they failed exactly
/// this way — both passing alone at around 65 ms per photograph, both failing together — which
/// is the signature of contention rather than a regression. Naming the shared resource is the
/// honest fix; loosening the thresholds until the flake stops would have thrown away the only
/// test that would notice inference getting slower.
/// </para>
/// </remarks>
[CollectionDefinition(Inference.Name, DisableParallelization = true)]
public sealed class InferenceCollection
{
}

/// <summary>The name of the inference collection.</summary>
public static class Inference
{
    /// <summary>Collection name, referenced by <c>[Collection]</c> on each inference test class.</summary>
    public const string Name = "inference";
}
