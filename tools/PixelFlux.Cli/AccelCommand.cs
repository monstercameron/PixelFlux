using PixelFlux.Ai.Compute;

namespace PixelFlux.Cli;

/// <summary>
/// The <c>accel</c> command: what hardware this machine can run models on.
/// </summary>
/// <remarks>
/// Exists because "is the accelerator working" is otherwise unanswerable. A model that silently
/// fell back to the processor behaves identically to one that did not, only slower, and slower is
/// exactly what somebody installing an accelerator cannot distinguish from normal.
/// </remarks>
public static class AccelCommand
{
    /// <summary>Prints the available devices and where providers are loaded from.</summary>
    /// <param name="root">Library root.</param>
    /// <param name="repoRoot">Repository root, if the models live there instead.</param>
    /// <returns>A process exit code.</returns>
    public static int Run(string root, string? repoRoot)
    {
        string models = Path.Combine(repoRoot ?? root, "models");
        string providers = Path.Combine(models, ComputeBackend.ProviderFolderName);

        var backend = new ComputeBackend(providers);
        backend.Probe();

        Console.WriteLine($"provider folder: {providers}");
        Console.WriteLine(Directory.Exists(providers)
            ? $"  {backend.RegisteredProviders.Count} loaded" +
              (backend.RegisteredProviders.Count > 0
                  ? ": " + string.Join(", ", backend.RegisteredProviders)
                  : string.Empty)
            : "  (does not exist — drop execution provider DLLs here)");

        Console.WriteLine();
        Console.WriteLine("devices the runtime can target:");

        if (backend.Devices.Count == 0)
        {
            Console.WriteLine("  none reported; everything runs on the processor");
        }

        foreach (AcceleratorDevice device in backend.Devices)
        {
            Console.WriteLine(
                $"  {device.Kind.ToString().ToUpperInvariant(),-4} {device.Provider,-32} " +
                $"{device.Vendor ?? string.Empty}");
        }

        Console.WriteLine();
        Console.WriteLine(backend.HasAccelerator
            ? "an accelerator is available"
            : "processor only");

        return 0;
    }
}
