using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;

namespace PixelFlux.Ai.Compute;

/// <summary>Which kind of hardware a model should prefer.</summary>
public enum AcceleratorPreference
{
    /// <summary>Let the runtime decide from what is installed.</summary>
    Auto = 0,

    /// <summary>Processor only, ignoring any accelerator that is present.</summary>
    Cpu = 1,

    /// <summary>Prefer a graphics processor.</summary>
    Gpu = 2,

    /// <summary>Prefer a neural processor.</summary>
    Npu = 3,
}

/// <summary>One piece of hardware the runtime can target.</summary>
/// <param name="Provider">The execution provider that owns it, for example <c>QNNExecutionProvider</c>.</param>
/// <param name="Kind">Processor, graphics or neural.</param>
/// <param name="Vendor">Who makes it, when the runtime says.</param>
/// <param name="Description">A human-readable name for the device.</param>
public sealed record AcceleratorDevice(
    string Provider,
    AcceleratorPreference Kind,
    string? Vendor,
    string Description);

/// <summary>
/// Decides what hardware every model in PixelFlux runs on, and is the seam other silicon plugs into.
/// </summary>
/// <remarks>
/// <para>
/// <b>The problem this exists to solve.</b> Until now each model opened its own session with its own
/// hand-written options, four times, always on the processor. Adding a graphics or neural path meant
/// editing four files identically and getting one of them subtly wrong. Worse, the obvious way to add
/// one — swap <c>Microsoft.ML.OnnxRuntime</c> for <c>.DirectML</c> or <c>.Qnn</c> — does not work
/// here: those packages are pinned at 1.24.4 while the base runtime is at 1.28.0, and the vision
/// model's GenAI layer requires 1.28 or newer. Choosing an accelerator that way would mean giving up
/// the vision model entirely.
/// </para>
/// <para>
/// <b>What makes it work anyway.</b> Runtime 1.28 can load an execution provider from a plain DLL at
/// run time — <c>RegisterExecutionProviderLibrary</c> — and then be told to prefer a class of device
/// rather than a named provider. So a vendor's provider ships as a file in a folder rather than as a
/// package reference, nothing is pinned, and a machine with a Qualcomm neural processor and a machine
/// with an AMD one run the same build. That is the whole design: this class finds those files, loads
/// them, and states a preference.
/// </para>
/// <para>
/// <b>It degrades rather than fails.</b> A missing provider folder, a provider DLL built for a
/// different runtime, a device that turns out not to work — each of those leaves the processor path
/// exactly as it was. An accelerator is an optimisation, and an optimisation that can stop a photo
/// library from opening is not one.
/// </para>
/// </remarks>
public sealed class ComputeBackend
{
    /// <summary>Folder name, beside the models, that provider libraries are loaded from.</summary>
    /// <remarks>
    /// A folder rather than a configured list of paths. Installing an accelerator should be copying
    /// a file somewhere obvious, because the person doing it is holding a DLL a vendor gave them and
    /// has no interest in learning this application's configuration format.
    /// </remarks>
    public const string ProviderFolderName = "providers";

    /// <summary>The setting the preference is stored under.</summary>
    public const string SettingKey = "compute.accelerator";

    /// <summary>
    /// What each model prefers when nobody has said otherwise, measured on this hardware.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>A graphics processor is not uniformly faster, and assuming it is would have made this
    /// library slower.</b> Measured on a Snapdragon X2, ten runs each, against the DirectML
    /// provider Windows ML ships:
    /// </para>
    /// <code>
    /// model                CPU (6 threads)   DirectML
    /// clip_vision              23.4 ms        51.7 ms    2.2x slower
    /// yolo11n-seg              45.8 ms        99.3 ms    2.2x slower
    /// face_yunet                9.7 ms         4.9 ms    2.0x faster
    /// face_sface               15.2 ms         7.9 ms    1.9x faster
    /// </code>
    /// <para>
    /// It is not about model size — YuNet and YOLO take the same 640x640 input and land on
    /// opposite sides. It is operator coverage: the runtime warns that some nodes could not be
    /// assigned to the preferred provider, and every node it hands back to the processor costs a
    /// round trip across the bus. A model whose graph the provider covers wins; one it covers
    /// patchily loses twice over.
    /// </para>
    /// <para>
    /// Which is why the preference is per model rather than one switch. A single "use the
    /// accelerator" setting would have been half a speed-up and half a regression, and the
    /// regression would have been in the two stages that run most often.
    /// </para>
    /// </remarks>
    public static IReadOnlyDictionary<string, AcceleratorPreference> MeasuredDefaults { get; } =
        new Dictionary<string, AcceleratorPreference>(StringComparer.OrdinalIgnoreCase)
        {
            ["face_yunet"] = AcceleratorPreference.Gpu,
            ["face_sface"] = AcceleratorPreference.Gpu,
            ["clip_vision"] = AcceleratorPreference.Cpu,
            ["clip_text"] = AcceleratorPreference.Cpu,
            ["yolo11n-seg"] = AcceleratorPreference.Cpu,
        };

    private readonly ILogger _log;
    private readonly List<AcceleratorDevice> _devices = [];
    private readonly List<string> _registered = [];
    private bool _probed;

    /// <summary>Creates a backend that loads provider libraries from a folder.</summary>
    /// <param name="providerFolder">
    /// Where provider DLLs live. Missing is normal and means processor only.
    /// </param>
    /// <param name="preference">Which kind of hardware to prefer.</param>
    /// <param name="logger">Logger.</param>
    public ComputeBackend(
        string? providerFolder = null,
        AcceleratorPreference preference = AcceleratorPreference.Auto,
        ILogger? logger = null)
    {
        ProviderFolder = providerFolder;
        Preference = preference;
        _log = logger ?? NullLogger.Instance;
    }

    /// <summary>Where provider libraries are loaded from, if anywhere.</summary>
    public string? ProviderFolder { get; }

    /// <summary>Which kind of hardware models prefer.</summary>
    public AcceleratorPreference Preference { get; }

    /// <summary>Everything the runtime can target, after probing.</summary>
    public IReadOnlyList<AcceleratorDevice> Devices
    {
        get
        {
            Probe();
            return _devices;
        }
    }

    /// <summary>Provider libraries that were successfully loaded from the folder.</summary>
    public IReadOnlyList<string> RegisteredProviders
    {
        get
        {
            Probe();
            return _registered;
        }
    }

    /// <summary>Whether anything other than the processor is available.</summary>
    public bool HasAccelerator =>
        Devices.Any(device => device.Kind is AcceleratorPreference.Gpu or AcceleratorPreference.Npu);

    /// <summary>What a named model should run on.</summary>
    /// <param name="modelKey">
    /// The model's file stem, such as <c>face_yunet_2023mar</c>. Matched by prefix so a version
    /// bump in the filename does not silently lose the measurement that goes with it.
    /// </param>
    /// <returns>The preference to use for that model.</returns>
    /// <remarks>
    /// An explicit <see cref="AcceleratorPreference.Cpu"/> or a named accelerator set by the user
    /// applies to everything and overrides the measurements — somebody who has turned the
    /// accelerator off means it. <see cref="AcceleratorPreference.Auto"/> is where the per-model
    /// measurements are consulted, because auto means "do the right thing" and the right thing is
    /// different for each of these models.
    /// </remarks>
    public AcceleratorPreference PreferenceFor(string modelKey)
    {
        if (Preference != AcceleratorPreference.Auto)
        {
            return Preference;
        }

        foreach ((string key, AcceleratorPreference measured) in MeasuredDefaults)
        {
            if (modelKey.StartsWith(key, StringComparison.OrdinalIgnoreCase))
            {
                return measured;
            }
        }

        // Unmeasured. The processor, because it is the one that is known to work at a known speed,
        // and a model nobody has benchmarked is exactly the one an accelerator might halve or
        // double without anybody noticing which.
        return AcceleratorPreference.Cpu;
    }

    /// <summary>Builds session options for one model.</summary>
    /// <param name="intraOpThreads">
    /// Threads for processor work. Ignored by an accelerator, and still needed: even a model running
    /// on a neural processor does some of its graph on the processor.
    /// </param>
    /// <param name="modelKey">
    /// The model's file stem, so its measured preference applies. Null uses the global preference,
    /// which is what a caller with no particular model — a benchmark, a probe — wants.
    /// </param>
    /// <returns>Options the caller owns and must dispose.</returns>
    /// <remarks>
    /// The single place session options are made. Every model in the application comes through here,
    /// which is what makes "run everything on the neural processor" one setting rather than four
    /// edits — and what makes it impossible for three models to be accelerated and the fourth to be
    /// quietly left behind.
    /// </remarks>
    public SessionOptions CreateSessionOptions(int intraOpThreads, string? modelKey = null)
    {
        Probe();

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = Math.Max(1, intraOpThreads),
        };

        AcceleratorPreference wanted = modelKey is null ? Preference : PreferenceFor(modelKey);

        if (wanted == AcceleratorPreference.Cpu || _devices.Count == 0)
        {
            return options;
        }

        try
        {
            // A policy, not a provider name. Naming a provider means this code has to know that
            // Qualcomm's is called QNN and Microsoft's is called DML and AMD's is called VitisAI,
            // and has to be edited for the next one. Stating what is wanted — a neural processor,
            // whatever brand — is the same sentence on every machine.
            options.SetEpSelectionPolicy(Policy(wanted));
        }
        catch (Exception error) when (error is OnnxRuntimeException or EntryPointNotFoundException)
        {
            // An older runtime, or one built without the selection API. The options are already
            // valid and processor-bound, which is the right answer rather than a failure.
            _log.LogDebug(error, "This runtime cannot select an accelerator by policy.");
        }

        return options;
    }

    /// <summary>Loads provider libraries and lists what the runtime can see.</summary>
    /// <remarks>
    /// Runs once. Registration is process-wide and permanent — the runtime has no way to load a
    /// provider twice harmlessly — so this is the one place it may happen, and it must happen before
    /// any session is created.
    /// </remarks>
    public void Probe()
    {
        if (_probed)
        {
            return;
        }

        _probed = true;

        RegisterProviderLibraries();
        ReadDevices();
    }

    private void RegisterProviderLibraries()
    {
        if (ProviderFolder is null || !Directory.Exists(ProviderFolder))
        {
            return;
        }

        foreach (string library in Directory.EnumerateFiles(ProviderFolder, "*.dll")
                     .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
        {
            string name = Path.GetFileNameWithoutExtension(library);

            try
            {
                OrtEnv.Instance().RegisterExecutionProviderLibrary(name, library);
                _registered.Add(name);
                _log.LogInformation("Loaded execution provider {Name}.", name);
            }
            catch (Exception error) when (error is OnnxRuntimeException
                                              or DllNotFoundException
                                              or BadImageFormatException
                                              or EntryPointNotFoundException)
            {
                // Overwhelmingly this is a provider built against a different runtime version, and
                // the person who dropped it in the folder needs to be told which file was refused
                // rather than left with an accelerator that silently did nothing.
                _log.LogWarning("Execution provider {Name} was refused: {Reason}",
                    name, error.Message);
            }
        }
    }

    private void ReadDevices()
    {
        try
        {
            foreach (OrtEpDevice device in OrtEnv.Instance().GetEpDevices())
            {
                _devices.Add(Describe(device));
            }
        }
        catch (Exception error) when (error is OnnxRuntimeException or EntryPointNotFoundException)
        {
            _log.LogDebug(error, "This runtime cannot enumerate devices.");
        }
    }

    private static AcceleratorDevice Describe(OrtEpDevice device)
    {
        string provider = device.EpName;
        string vendor = device.EpVendor;

        // The runtime reports a hardware device with a type; map it onto the three kinds a person
        // choosing in a settings panel actually distinguishes between.
        AcceleratorPreference kind = device.HardwareDevice.Type switch
        {
            OrtHardwareDeviceType.GPU => AcceleratorPreference.Gpu,
            OrtHardwareDeviceType.NPU => AcceleratorPreference.Npu,
            _ => AcceleratorPreference.Cpu,
        };

        return new AcceleratorDevice(
            provider,
            kind,
            string.IsNullOrWhiteSpace(vendor) ? null : vendor,
            $"{provider} ({kind.ToString().ToUpperInvariant()})");
    }

    private static ExecutionProviderDevicePolicy Policy(AcceleratorPreference preference) =>
        preference switch
        {
            AcceleratorPreference.Gpu => ExecutionProviderDevicePolicy.PREFER_GPU,
            AcceleratorPreference.Npu => ExecutionProviderDevicePolicy.PREFER_NPU,
            _ => ExecutionProviderDevicePolicy.DEFAULT,
        };

    /// <summary>Reads a stored preference.</summary>
    /// <param name="stored">A value previously written, or null.</param>
    /// <returns>The preference; anything unrecognised reads as <see cref="AcceleratorPreference.Auto"/>.</returns>
    public static AcceleratorPreference ParsePreference(string? stored) => stored?.ToLowerInvariant() switch
    {
        "cpu" => AcceleratorPreference.Cpu,
        "gpu" => AcceleratorPreference.Gpu,
        "npu" => AcceleratorPreference.Npu,
        _ => AcceleratorPreference.Auto,
    };
}
