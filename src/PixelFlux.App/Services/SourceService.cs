using Microsoft.Extensions.Logging;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Pipeline;

namespace PixelFlux.App.Services;

/// <summary>One folder the library imports from, and what is known about it.</summary>
/// <param name="Path">The folder.</param>
/// <param name="Exists">Whether it is reachable right now.</param>
public sealed record SourceFolder(string Path, bool Exists);

/// <summary>
/// The folders a library imports from.
/// </summary>
/// <remarks>
/// <para>
/// Until this existed the import button was wired to a hard-coded path inside the repository —
/// a development shortcut that worked on exactly one machine. Choosing folders is the first thing
/// anybody does with a photo manager, so it is also the first thing that has to be real.
/// </para>
/// <para>
/// The list is a library setting rather than a per-machine one, so it travels with the
/// photographs. A folder that does not exist is kept and shown as missing rather than dropped:
/// an unplugged drive is the usual reason, and silently forgetting somebody's archive because it
/// was offline on the wrong Tuesday would be much worse than a greyed-out row.
/// </para>
/// </remarks>
public sealed class SourceService
{
    private readonly SettingsStore _settings;
    private readonly LibraryService _library;
    private readonly IFolderChooser _chooser;
    private readonly ILogger<SourceService> _log;

    /// <summary>Creates the service.</summary>
    /// <param name="settings">Where the list is kept.</param>
    /// <param name="library">Used to run the import.</param>
    /// <param name="chooser">Shows the folder picker.</param>
    /// <param name="logger">Logger.</param>
    public SourceService(
        SettingsStore settings,
        LibraryService library,
        IFolderChooser chooser,
        ILogger<SourceService> logger)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(library);
        ArgumentNullException.ThrowIfNull(chooser);

        _settings = settings;
        _library = library;
        _chooser = chooser;
        _log = logger;
    }

    /// <summary>Raised when the list changes.</summary>
    public event Action? Changed;

    /// <summary>Whether an import is running.</summary>
    public bool Scanning { get; private set; }

    /// <summary>The line of progress text, while scanning.</summary>
    public string? Progress { get; private set; }

    /// <summary>The folders, with whether each is currently reachable.</summary>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>One entry per folder, in the order they were added.</returns>
    public async Task<IReadOnlyList<SourceFolder>> ListAsync(
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<string> stored = SourceFolders.Parse(
            await _settings.GetAsync(SourceFolders.SettingKey, cancellationToken)
                .ConfigureAwait(false));

        return [.. stored.Select(path => new SourceFolder(path, Directory.Exists(path)))];
    }

    /// <summary>Shows a picker and adds whatever was chosen.</summary>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <returns>The folder that was added, or null if the picker was dismissed or it was already covered.</returns>
    public async Task<string?> AddAsync(CancellationToken cancellationToken = default)
    {
        string? chosen = await _chooser.ChooseAsync(cancellationToken).ConfigureAwait(false);

        if (string.IsNullOrWhiteSpace(chosen))
        {
            return null;
        }

        IReadOnlyList<string> before = SourceFolders.Parse(
            await _settings.GetAsync(SourceFolders.SettingKey, cancellationToken)
                .ConfigureAwait(false));

        IReadOnlyList<string> after = SourceFolders.Add(before, chosen);

        if (after.Count == before.Count && after.SequenceEqual(before))
        {
            // Already inside a folder being watched. Nothing to save and nothing to report as an
            // error: the answer to "is this folder included" is yes.
            return null;
        }

        await SaveAsync(after, cancellationToken).ConfigureAwait(false);
        return SourceFolders.Normalise(chosen);
    }

    /// <summary>Stops importing from a folder. Photographs already imported stay.</summary>
    /// <param name="folder">The folder to drop.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    /// <remarks>
    /// Deliberately does not remove the photographs. Somebody tidying their list of folders has
    /// not asked to delete half their library, and the originals are untouched on disk either way.
    /// </remarks>
    public async Task RemoveAsync(string folder, CancellationToken cancellationToken = default)
    {
        IReadOnlyList<string> before = SourceFolders.Parse(
            await _settings.GetAsync(SourceFolders.SettingKey, cancellationToken)
                .ConfigureAwait(false));

        await SaveAsync(SourceFolders.Remove(before, folder), cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>Imports from every folder that is currently reachable.</summary>
    /// <param name="cancellationToken">Stops the scan.</param>
    /// <returns>What the import found.</returns>
    /// <remarks>
    /// Safe to run as often as you like: ingestion keys on content hash, so a photograph already
    /// in the library is recognised and skipped rather than duplicated. That is what makes "scan
    /// again" the right verb for both "I added a folder" and "I put new pictures in one".
    /// </remarks>
    public async Task<IngestResult?> ScanAsync(CancellationToken cancellationToken = default)
    {
        if (Scanning)
        {
            return null;
        }

        string[] folders =
        [
            .. (await ListAsync(cancellationToken).ConfigureAwait(false))
                .Where(source => source.Exists)
                .Select(source => source.Path),
        ];

        if (folders.Length == 0)
        {
            return null;
        }

        Scanning = true;
        Progress = null;
        Changed?.Invoke();

        try
        {
            var progress = new Progress<IngestProgress>(update =>
            {
                Progress = $"{update.Processed}/{update.Discovered}  {update.Current}";
                Changed?.Invoke();
            });

            IngestResult result = await _library
                .ImportAsync(folders, progress, cancellationToken).ConfigureAwait(false);

            _log.LogInformation(
                "Scanned {Count} source folders: {Imported} new, {Known} already known.",
                folders.Length, result.Imported, result.Duplicates);

            return result;
        }
        finally
        {
            Scanning = false;
            Progress = null;
            Changed?.Invoke();
        }
    }

    private async Task SaveAsync(IReadOnlyList<string> folders, CancellationToken cancellationToken)
    {
        await _settings.SetAsync(
            SourceFolders.SettingKey, SourceFolders.Serialise(folders), cancellationToken)
            .ConfigureAwait(false);

        Changed?.Invoke();
    }
}
