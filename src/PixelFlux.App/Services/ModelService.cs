using Microsoft.Extensions.Logging;
using PixelFlux.Core.Setup;

namespace PixelFlux.App.Services;

/// <summary>
/// Owns model installation: what is present, what is downloading, and how far along.
/// </summary>
/// <remarks>
/// <para>
/// A singleton holding one download at a time. Two models fetched at once would halve each
/// other's bandwidth, make the progress bars lie about which is finishing, and — on the vision
/// model's 1.4 GB — turn one long wait into two longer ones. One at a time is also the only
/// arrangement where "stop" has an obvious meaning.
/// </para>
/// <para>
/// The <see cref="HttpClient"/> lives here for the lifetime of the application rather than being
/// created per download, which is the documented way to avoid exhausting sockets. Its timeout is
/// deliberately long: a gigabyte on a slow connection is a legitimate half-hour, and the default
/// hundred seconds would abandon it.
/// </para>
/// </remarks>
public sealed class ModelService : IDisposable
{
    private readonly HttpClient _http;
    private readonly ModelDownloader _downloader;
    private readonly LibraryPaths _paths;
    private readonly ILogger<ModelService> _log;

    private CancellationTokenSource? _cancelling;
    private DownloadProgress _progress;

    /// <summary>Creates the service.</summary>
    /// <param name="paths">Where models are kept.</param>
    /// <param name="logger">Logger.</param>
    public ModelService(LibraryPaths paths, ILogger<ModelService> logger)
    {
        ArgumentNullException.ThrowIfNull(paths);

        _paths = paths;
        _log = logger;

        _http = new HttpClient
        {
            // Per-request, not per-byte: a large model on a slow line is a legitimately long
            // request and the default would give up on it. A stalled connection is caught by the
            // read failing, not by this.
            Timeout = TimeSpan.FromHours(2),
        };

        _http.DefaultRequestHeaders.UserAgent.ParseAdd("PixelFlux/1.0");
        _downloader = new ModelDownloader(_http, logger);
    }

    /// <summary>Raised whenever progress or installed state changes.</summary>
    public event Action? Changed;

    /// <summary>Whether a download is running.</summary>
    public bool Busy => _cancelling is not null;

    /// <summary>The model being fetched, if any.</summary>
    public string? Current => Busy ? _progress.Model : null;

    /// <summary>The file being fetched, for the line under the bar.</summary>
    public string CurrentFile => _progress.File;

    /// <summary>How far the current model has got, from 0 to 1.</summary>
    public double Fraction => _progress.Fraction;

    /// <summary>The last failure, or null.</summary>
    public string? Error { get; private set; }

    /// <summary>Where models are written.</summary>
    public string ModelsRoot => _paths.ModelsRoot;

    /// <summary>Whether every catalogued model is present.</summary>
    public bool AllInstalled => ModelCatalog.All.All(IsInstalled);

    /// <summary>Whether this is a fresh install with nothing to work with.</summary>
    public bool NothingInstalled => ModelCatalog.All.All(model => !IsInstalled(model));

    /// <summary>How much is still to download.</summary>
    public long RemainingBytes =>
        ModelCatalog.All.Where(model => !IsInstalled(model)).Sum(model => model.Bytes);

    /// <summary>Whether one model is fully present.</summary>
    /// <param name="model">The model.</param>
    /// <returns>True when every file is there and the right length.</returns>
    public bool IsInstalled(CatalogueModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        // Asked of every place the loader looks, not just the directory downloads land in. A
        // developer checkout keeps its models in the repository folder, and checking only the
        // per-user one offered to re-download two gigabytes that were already present.
        return model.Files.All(file =>
        {
            string? found = _paths.FindExistingModel(file.RelativePath);
            return found is not null && new FileInfo(found).Length == file.Bytes;
        });
    }

    /// <summary>Downloads one model.</summary>
    /// <param name="model">The model.</param>
    public async Task InstallAsync(CatalogueModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (Busy)
        {
            return;
        }

        await RunAsync([model]).ConfigureAwait(false);
    }

    /// <summary>Downloads everything not yet present, largest last.</summary>
    /// <remarks>
    /// Smallest first, deliberately. The face models are 37 MB and turn on a whole page; the
    /// vision model is 1.4 GB. Ordering this way means somebody who changes their mind ten minutes
    /// in keeps the most capability per minute spent.
    /// </remarks>
    public async Task InstallAllAsync()
    {
        if (Busy)
        {
            return;
        }

        await RunAsync(
            [.. ModelCatalog.All.Where(model => !IsInstalled(model)).OrderBy(model => model.Bytes)])
            .ConfigureAwait(false);
    }

    /// <summary>Stops the current download at the next chunk.</summary>
    public void Cancel() => _cancelling?.Cancel();

    /// <summary>Disposes the client.</summary>
    public void Dispose()
    {
        _cancelling?.Cancel();
        _cancelling?.Dispose();
        _http.Dispose();
    }

    private async Task RunAsync(IReadOnlyList<CatalogueModel> models)
    {
        Error = null;
        _cancelling = new CancellationTokenSource();
        Changed?.Invoke();

        var progress = new Progress<DownloadProgress>(update =>
        {
            _progress = update;
            Changed?.Invoke();
        });

        try
        {
            foreach (CatalogueModel model in models)
            {
                _progress = new DownloadProgress(model.Id, string.Empty, 0, model.Bytes);
                Changed?.Invoke();

                await _downloader.InstallAsync(
                    model, _paths.ModelsRoot, progress, _cancelling.Token).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            // Somebody pressed stop. Whatever finished is kept; the part-file is already gone.
            _log.LogInformation("Model download stopped.");
        }
        catch (ModelDownloadException failure)
        {
            // Shown as written: the downloader phrases these for a person, not for a log.
            Error = failure.Message;
            _log.LogWarning(failure, "Model download failed.");
        }
        finally
        {
            _cancelling?.Dispose();
            _cancelling = null;
            _progress = default;
            Changed?.Invoke();
        }
    }
}
