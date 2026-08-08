using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using PixelFlux.Core.Geo;
using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Model;
using SixLabors.ImageSharp;

namespace PixelFlux.Core.Ingest;

/// <summary>Progress from an in-flight import.</summary>
/// <param name="Discovered">Files found so far.</param>
/// <param name="Processed">Files finished, successfully or not.</param>
/// <param name="Imported">New photos added to the library.</param>
/// <param name="Duplicates">Files skipped because their content hash was already known.</param>
/// <param name="Failed">Files that could not be read or decoded.</param>
/// <param name="Current">Name of the file being worked on, for the status line.</param>
public readonly record struct IngestProgress(
    int Discovered,
    int Processed,
    int Imported,
    int Duplicates,
    int Failed,
    string? Current);

/// <summary>The outcome of a completed import.</summary>
/// <param name="Discovered">Files found.</param>
/// <param name="Imported">New photos added.</param>
/// <param name="Duplicates">Files already in the library.</param>
/// <param name="Failed">Files that could not be read.</param>
/// <param name="Elapsed">Wall-clock duration.</param>
/// <param name="Failures">Per-file failure reasons, for the import report.</param>
public sealed record IngestResult(
    int Discovered,
    int Imported,
    int Duplicates,
    int Failed,
    TimeSpan Elapsed,
    IReadOnlyList<(string Path, string Reason)> Failures);

/// <summary>
/// Walks folders, reads each image once, and writes it into the local index.
/// </summary>
/// <remarks>
/// <para>
/// The pipeline per file is: hash the bytes → check whether the library already has that hash →
/// decode once to produce dimensions, perceptual hash, thumbnail, and proxy → read EXIF →
/// insert. The early hash check is what makes re-importing a folder nearly free, and it is
/// checked <em>before</em> decoding because decoding is two orders of magnitude more expensive
/// than hashing.
/// </para>
/// <para>
/// <b>Nothing here modifies an original.</b> Originals are read-only to PixelFlux; every
/// derived artefact goes to the cache directory. That is a hard rule, not a default — a photo
/// manager that can corrupt the only copy of a photograph is not worth using.
/// </para>
/// <para>
/// <b>A file that fails is recorded, not fatal.</b> Real libraries contain truncated downloads,
/// zero-byte placeholders, and files with the wrong extension. Each one is logged with a reason
/// and the walk continues; an import that aborts on the first bad file would never finish on a
/// library large enough to need this application.
/// </para>
/// </remarks>
public sealed class LibraryIngestor
{
    // Extensions worth opening. Checked before touching the file because a photo folder is full
    // of Thumbs.db, .picasa.ini, and sidecar files, and hashing those wastes real time.
    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".jpe", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif",
        ".heic", ".heif",   // metadata-only on builds without a HEIF decoder; see below
    };

    private static readonly Dictionary<string, string> MimeByExtension = new(StringComparer.OrdinalIgnoreCase)
    {
        [".jpg"] = "image/jpeg", [".jpeg"] = "image/jpeg", [".jpe"] = "image/jpeg",
        [".png"] = "image/png",
        [".webp"] = "image/webp",
        [".tif"] = "image/tiff", [".tiff"] = "image/tiff",
        [".bmp"] = "image/bmp",
        [".gif"] = "image/gif",
        [".heic"] = "image/heic", [".heif"] = "image/heif",
    };

    private readonly PhotoStore _store;
    private readonly DerivativeGenerator _derivatives;
    private readonly ILogger<LibraryIngestor> _log;

    /// <summary>Creates an ingestor.</summary>
    /// <param name="store">Where indexed photos are written.</param>
    /// <param name="derivatives">Generator for thumbnails and proxies.</param>
    /// <param name="logger">Optional logger.</param>
    public LibraryIngestor(
        PhotoStore store,
        DerivativeGenerator derivatives,
        ILogger<LibraryIngestor>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(derivatives);

        _store = store;
        _derivatives = derivatives;
        _log = logger ?? NullLogger<LibraryIngestor>.Instance;
    }

    /// <summary>
    /// Maximum files decoded at once.
    /// </summary>
    /// <remarks>
    /// Bounded rather than unbounded because each in-flight decode holds a full-resolution
    /// bitmap in memory — a 45-megapixel file is around 180 MB as RGBA. Sixteen concurrent
    /// decodes of large files would be several gigabytes of pressure for no throughput gain,
    /// since the work is already CPU-saturated well below that. Capped at 6 regardless of core
    /// count for the same reason.
    /// </remarks>
    private static int MaxConcurrency => Math.Clamp(Environment.ProcessorCount / 2, 2, 6);

    /// <summary>
    /// Imports every supported image under a set of folders.
    /// </summary>
    /// <param name="folders">Root folders to walk, recursively.</param>
    /// <param name="progress">Optional progress sink, reported per completed file.</param>
    /// <param name="cancellationToken">Cancels the import. Already-imported photos are kept.</param>
    /// <returns>A summary of what happened.</returns>
    public async Task<IngestResult> ImportAsync(
        IReadOnlyList<string> folders,
        IProgress<IngestProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(folders);

        long startedAt = Environment.TickCount64;
        List<string> files = Discover(folders);

        int processed = 0, imported = 0, duplicates = 0, failed = 0;
        var failures = new ConcurrentBag<(string Path, string Reason)>();

        await Parallel.ForEachAsync(
            files,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = MaxConcurrency,
                CancellationToken = cancellationToken,
            },
            async (path, token) =>
            {
                ImportOutcome outcome;
                try
                {
                    outcome = await ImportOneAsync(path, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    // Deliberately broad. This loop is the boundary between a messy filesystem
                    // and a clean index, and every exception type that reaches it means the
                    // same thing: this one file is not importable, the other 49,999 are.
                    _log.LogWarning(ex, "Could not import {Path}", path);
                    outcome = ImportOutcome.Failure(ex.Message);
                }

                switch (outcome.Kind)
                {
                    case ImportKind.Imported:
                        Interlocked.Increment(ref imported);
                        break;
                    case ImportKind.Duplicate:
                        Interlocked.Increment(ref duplicates);
                        break;
                    default:
                        Interlocked.Increment(ref failed);
                        failures.Add((path, outcome.Reason ?? "unknown"));
                        break;
                }

                int done = Interlocked.Increment(ref processed);
                progress?.Report(new IngestProgress(
                    files.Count, done,
                    Volatile.Read(ref imported),
                    Volatile.Read(ref duplicates),
                    Volatile.Read(ref failed),
                    Path.GetFileName(path)));
            }).ConfigureAwait(false);

        return new IngestResult(
            files.Count, imported, duplicates, failed,
            TimeSpan.FromMilliseconds(Environment.TickCount64 - startedAt),
            failures.ToArray());
    }

    private List<string> Discover(IReadOnlyList<string> folders)
    {
        var files = new List<string>();

        foreach (string folder in folders)
        {
            if (!System.IO.Directory.Exists(folder))
            {
                _log.LogWarning("Import folder does not exist: {Folder}", folder);
                continue;
            }

            try
            {
                // IgnoreInaccessible keeps a single permission-denied subfolder — a system
                // directory, another user's profile — from aborting the whole walk.
                files.AddRange(System.IO.Directory
                    .EnumerateFiles(folder, "*", new EnumerationOptions
                    {
                        RecurseSubdirectories = true,
                        IgnoreInaccessible = true,
                        AttributesToSkip = FileAttributes.System | FileAttributes.Temporary,
                    })
                    .Where(f => SupportedExtensions.Contains(Path.GetExtension(f))));
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _log.LogWarning(ex, "Could not walk {Folder}", folder);
            }
        }

        return files;
    }

    private enum ImportKind { Imported, Duplicate, Failed }

    private readonly record struct ImportOutcome(ImportKind Kind, string? Reason)
    {
        public static ImportOutcome Imported() => new(ImportKind.Imported, null);
        public static ImportOutcome Duplicate() => new(ImportKind.Duplicate, null);
        public static ImportOutcome Failure(string reason) => new(ImportKind.Failed, reason);
    }

    private async Task<ImportOutcome> ImportOneAsync(string path, CancellationToken cancellationToken)
    {
        var info = new FileInfo(path);
        if (!info.Exists || info.Length == 0)
        {
            return ImportOutcome.Failure("empty or missing file");
        }

        // Hash first. It is cheap relative to a decode, and it answers "do we already have
        // this?" without paying for the expensive part.
        string contentHash = await ImageHashing.ComputeContentHashAsync(path, cancellationToken)
            .ConfigureAwait(false);

        ExifData exif = ExifExtractor.Read(path);

        DerivativeResult derivatives;
        try
        {
            derivatives = await _derivatives.GenerateAsync(path, contentHash, cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is UnknownImageFormatException or InvalidImageContentException
                                      or NotSupportedException or ImageFormatException)
        {
            // The file has a picture extension but cannot be decoded: truncated, or a format
            // this build has no decoder for (HEIC on a stock ImageSharp being the common case).
            //
            // It is still indexed. A HEIC that only yields EXIF still has a date, a camera, and
            // a GPS fix — it belongs on the timeline and in place searches, marked Unreadable so
            // the UI shows a placeholder instead of a broken thumbnail. Dropping it would make
            // an iPhone library look mysteriously incomplete.
            return await IndexUndecodableAsync(info, contentHash, exif, ex.Message, cancellationToken)
                .ConfigureAwait(false);
        }

        DateTimeOffset modified = new(info.LastWriteTimeUtc, TimeSpan.Zero);
        var record = new PhotoRecord
        {
            ContentHash = contentHash,
            PerceptualHash = derivatives.PerceptualHash,
            OriginalPath = path,
            FileName = info.Name,
            SourceFolder = info.DirectoryName ?? string.Empty,
            MimeType = MimeFor(path),
            Width = derivatives.Width,
            Height = derivatives.Height,
            FileSize = info.Length,
            CapturedUtc = exif.CapturedUtc ?? modified,
            CaptureTimeIsExact = exif.CapturedUtc is not null,
            FileModifiedUtc = modified,
            IndexedUtc = DateTimeOffset.UtcNow,
            Camera = exif.Camera,
            Location = exif.Location,
            Place = ResolvePlace(exif.Location),
            Orientation = exif.Orientation,
            ThumbnailKey = DerivativeGenerator.ThumbnailKey(contentHash),
            ProxyKey = DerivativeGenerator.ProxyKey(contentHash),
            State = ProcessingState.Pending,
            Rating = exif.Rating,

            // A five-star rating is imported as a favourite.
            //
            // No image format records "favourite" — it is not an EXIF concept. Five stars is the
            // closest thing a file carries to that intent, and it is what Explorer, Lightroom,
            // and Photos all treat as the top of the scale. Mapping it means a freshly imported
            // library arrives with the favourites filter already meaningful instead of showing
            // an empty screen until the user has clicked fifty hearts. It is a starting point,
            // not a lock: favouriting is user metadata from that moment on and is never
            // overwritten by a later re-import.
            IsFavourite = exif.Rating >= 5,
        };

        // Keywords another tool embedded are imported as File-sourced tags — useful immediately,
        // and distinguishable later from anything the model or the user adds.
        PhotoTag[] tags = exif.Keywords
            .Select(k => new PhotoTag(k.ToLowerInvariant(), 1.0, MetadataSource.File))
            .ToArray();

        (long _, bool inserted) = await _store.UpsertAsync(record, tags, cancellationToken).ConfigureAwait(false);
        return inserted ? ImportOutcome.Imported() : ImportOutcome.Duplicate();
    }

    private async Task<ImportOutcome> IndexUndecodableAsync(
        FileInfo info,
        string contentHash,
        ExifData exif,
        string reason,
        CancellationToken cancellationToken)
    {
        DateTimeOffset modified = new(info.LastWriteTimeUtc, TimeSpan.Zero);

        var record = new PhotoRecord
        {
            ContentHash = contentHash,
            // No pixels means no perceptual hash. Zeroes are used rather than null so the
            // column stays NOT NULL; the all-zero value is excluded from duplicate detection
            // by the state check, since otherwise every undecodable file would cluster together.
            PerceptualHash = "0000000000000000",
            OriginalPath = info.FullName,
            FileName = info.Name,
            SourceFolder = info.DirectoryName ?? string.Empty,
            MimeType = MimeFor(info.FullName),
            FileSize = info.Length,
            CapturedUtc = exif.CapturedUtc ?? modified,
            CaptureTimeIsExact = exif.CapturedUtc is not null,
            FileModifiedUtc = modified,
            IndexedUtc = DateTimeOffset.UtcNow,
            Camera = exif.Camera,
            Location = exif.Location,
            Place = ResolvePlace(exif.Location),
            Orientation = exif.Orientation,
            State = ProcessingState.Unreadable,
            StateDetail = reason,
        };

        (long _, bool inserted) = await _store.UpsertAsync(record, null, cancellationToken).ConfigureAwait(false);
        return inserted ? ImportOutcome.Imported() : ImportOutcome.Duplicate();
    }

    /// <summary>
    /// Turns a GPS fix into a city and country, once, at import.
    /// </summary>
    /// <remarks>
    /// Resolved here rather than at render time because it is a nearest-neighbour search over
    /// 34,000 places, and a gallery that ran one per thumbnail per repaint would spend more time
    /// geocoding than drawing. The result is denormalised into the row, so the facet, the search
    /// index, and the viewer all read the same three columns.
    /// </remarks>
    private static PlaceName? ResolvePlace(GeoPoint? location)
    {
        if (location is not { } fix)
        {
            return null;
        }

        ResolvedPlace? resolved = Gazetteer.Instance.Resolve(fix.Latitude, fix.Longitude);
        return resolved is { } place
            ? new PlaceName(place.City, place.Country, place.CountryCode, place.Label)
            : null;
    }

    private static string MimeFor(string path)
        => MimeByExtension.TryGetValue(Path.GetExtension(path), out string? mime)
            ? mime
            : "application/octet-stream";
}
