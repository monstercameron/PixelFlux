namespace PixelFlux.Core.Model;

/// <summary>
/// Where a piece of metadata came from, and therefore which value wins when two disagree.
/// </summary>
/// <remarks>
/// Provenance is a first-class column throughout the schema rather than an afterthought.
/// A model can be re-run and will overwrite its own previous output without warning; a caption
/// the user typed must survive that. Any code that writes metadata states its source, and the
/// resolution rule is uniform: <see cref="User"/> beats <see cref="Ai"/> beats
/// <see cref="File"/>.
/// </remarks>
public enum MetadataSource
{
    /// <summary>Read out of the file itself — EXIF, XMP, IPTC, or the filesystem.</summary>
    File = 0,

    /// <summary>Produced by a local model. Freely replaceable when models are re-run.</summary>
    Ai = 1,

    /// <summary>Entered by a person. Never overwritten by ingestion or by AI processing.</summary>
    User = 2,
}

/// <summary>
/// Where a photo sits in the AI processing pipeline.
/// </summary>
public enum ProcessingState
{
    /// <summary>Indexed and waiting to be analysed.</summary>
    Pending = 0,

    /// <summary>Claimed by a worker. Carries the claiming device in <see cref="PhotoRecord.StateDetail"/>.</summary>
    Processing = 1,

    /// <summary>Analysed successfully by the model version in <see cref="PhotoRecord.ModelVersion"/>.</summary>
    Complete = 2,

    /// <summary>
    /// Analysis failed. <see cref="PhotoRecord.StateDetail"/> holds the reason.
    /// A failed photo stays in the library and stays browsable — only its AI metadata is missing.
    /// </summary>
    Failed = 3,

    /// <summary>
    /// The file could not be decoded at all: truncated, or not really an image.
    /// Distinct from <see cref="Failed"/> because retrying the model will never help.
    /// </summary>
    Unreadable = 4,
}

/// <summary>A GPS fix expressed as somewhere a person would recognise.</summary>
/// <param name="City">Nearest populated place.</param>
/// <param name="Country">Country name in English.</param>
/// <param name="CountryCode">ISO-3166 alpha-2 code.</param>
/// <param name="Label">
/// How to write it in the interface — <c>"Ely, United Kingdom"</c>, or <c>"near Ely, ..."</c>
/// when the fix is some way from the town, or just the country when it is far from anywhere.
/// The qualifier is stored rather than recomputed so that every surface says the same thing.
/// </param>
public readonly record struct PlaceName(string City, string Country, string CountryCode, string Label);

/// <summary>A latitude/longitude/altitude fix read from EXIF GPS tags.</summary>
/// <param name="Latitude">Signed decimal degrees; positive is north.</param>
/// <param name="Longitude">Signed decimal degrees; positive is east.</param>
/// <param name="Altitude">Metres above sea level, or <see langword="null"/> if not recorded.</param>
public readonly record struct GeoPoint(double Latitude, double Longitude, double? Altitude);

/// <summary>Camera and exposure settings read from EXIF.</summary>
/// <remarks>
/// Grouped into its own type because these travel together, are frequently all absent (a
/// screenshot, an export, a scan), and are the basis of the "taken with my Canon" class of
/// search. Every member is nullable: a partial EXIF block is normal, not an error.
/// </remarks>
public sealed record CameraInfo
{
    /// <summary>Manufacturer, for example <c>Canon</c> or <c>NIKON CORPORATION</c>.</summary>
    public string? Make { get; init; }

    /// <summary>Body model, for example <c>Canon EOS R6 Mark II</c>.</summary>
    public string? Model { get; init; }

    /// <summary>Lens model string, where the body records one.</summary>
    public string? Lens { get; init; }

    /// <summary>ISO sensitivity.</summary>
    public int? Iso { get; init; }

    /// <summary>Aperture as an f-number, for example 2.8.</summary>
    public double? FNumber { get; init; }

    /// <summary>Exposure time in seconds, for example 0.004 for 1/250.</summary>
    public double? ExposureSeconds { get; init; }

    /// <summary>Focal length in millimetres.</summary>
    public double? FocalLengthMm { get; init; }

    /// <summary>True when no field carries a value, which is the common case for non-camera images.</summary>
    public bool IsEmpty =>
        Make is null && Model is null && Lens is null &&
        Iso is null && FNumber is null && ExposureSeconds is null && FocalLengthMm is null;
}

/// <summary>
/// One image in the library: the row the whole application is built around.
/// </summary>
/// <remarks>
/// <para>
/// Deliberately a flat record rather than an object graph. The gallery loads thousands of these
/// at a time, search filters over their columns directly, and the sync layer diffs them field by
/// field — all three get simpler when there is nothing to traverse. Tags, detected objects, and
/// embeddings are the exceptions and live in their own tables, because they are one-to-many.
/// </para>
/// <para>
/// The two hashes serve different questions and both are needed.
/// <see cref="ContentHash"/> answers "is this the same file?" and is the identity of the photo.
/// <see cref="PerceptualHash"/> answers "does this look like that?" and is what finds the
/// re-exported copy, the burst sequence, and the same shot at two resolutions.
/// </para>
/// </remarks>
public sealed record PhotoRecord
{
    /// <summary>Local row id. Not stable across devices — <see cref="ContentHash"/> is the portable identity.</summary>
    public long Id { get; init; }

    /// <summary>Lowercase hex SHA-256 of the file bytes. Unique across the library.</summary>
    public required string ContentHash { get; init; }

    /// <summary>16-character hex difference hash of the decoded pixels. See <c>ImageHashing</c>.</summary>
    public required string PerceptualHash { get; init; }

    /// <summary>Absolute path to the original file, which PixelFlux reads but never modifies.</summary>
    public required string OriginalPath { get; init; }

    /// <summary>Filename with extension, denormalised out of the path so search can match on it.</summary>
    public required string FileName { get; init; }

    /// <summary>
    /// Directory the photo was imported from, denormalised out of <see cref="OriginalPath"/>.
    /// </summary>
    /// <remarks>
    /// Stored rather than derived on read because folders are a browsing axis, not a detail:
    /// people organise photographs into directories and then expect to navigate by them, and
    /// the sidebar groups by this on every load.
    /// </remarks>
    public string SourceFolder { get; init; } = string.Empty;

    /// <summary>Detected MIME type, for example <c>image/jpeg</c>.</summary>
    public required string MimeType { get; init; }

    /// <summary>Pixel width of the original, after orientation is applied.</summary>
    public int Width { get; init; }

    /// <summary>Pixel height of the original, after orientation is applied.</summary>
    public int Height { get; init; }

    /// <summary>File size in bytes.</summary>
    public long FileSize { get; init; }

    /// <summary>
    /// When the photo was taken, from EXIF <c>DateTimeOriginal</c>, or the file's write time
    /// when there is no EXIF.
    /// </summary>
    /// <remarks>
    /// This is the date the entire product sorts and filters by, so it must always have a value —
    /// falling back to the file timestamp is far better than leaving a third of a real library
    /// undateable. <see cref="CaptureTimeIsExact"/> records which of the two it was.
    /// </remarks>
    public DateTimeOffset CapturedUtc { get; init; }

    /// <summary>
    /// Whether <see cref="CapturedUtc"/> came from EXIF (<see langword="true"/>) or was inferred
    /// from the filesystem (<see langword="false"/>). Surfaced in the UI so a date the app
    /// guessed never looks like one the camera recorded.
    /// </summary>
    public bool CaptureTimeIsExact { get; init; }

    /// <summary>Filesystem last-write time, in UTC.</summary>
    public DateTimeOffset FileModifiedUtc { get; init; }

    /// <summary>When this row was first written to the local index.</summary>
    public DateTimeOffset IndexedUtc { get; init; }

    /// <summary>Camera and exposure settings, or an empty instance when the file carries none.</summary>
    public CameraInfo Camera { get; init; } = new();

    /// <summary>GPS fix, when the file records one.</summary>
    public GeoPoint? Location { get; init; }

    /// <summary>
    /// Where the GPS fix actually is, resolved against the offline gazetteer at ingest time.
    /// </summary>
    /// <remarks>
    /// Coordinates tell a person nothing. This is the pair of names they would actually use —
    /// "Ely", "United Kingdom" — and it is what the interface shows, what the place facet groups
    /// by, and what a search for a country matches. Null when the photo has no GPS fix, or when
    /// the fix could not be resolved.
    /// </remarks>
    public PlaceName? Place { get; init; }

    /// <summary>
    /// EXIF orientation tag (1-8) as found in the file.
    /// </summary>
    /// <remarks>
    /// Retained even though derivatives are written upright, because exporting a copy needs to
    /// know whether the original was rotated in metadata rather than in pixels.
    /// </remarks>
    public int Orientation { get; init; } = 1;

    /// <summary>Store-relative key of the cached thumbnail, or null if not generated yet.</summary>
    public string? ThumbnailKey { get; init; }

    /// <summary>Store-relative key of the cached display proxy, or null if not generated yet.</summary>
    public string? ProxyKey { get; init; }

    /// <summary>Position in the AI processing pipeline.</summary>
    public ProcessingState State { get; init; } = ProcessingState.Pending;

    /// <summary>Free-text detail for <see cref="State"/>: the claiming device, or a failure reason.</summary>
    public string? StateDetail { get; init; }

    /// <summary>
    /// Identifier of the model set that produced this photo's AI metadata, for example
    /// <c>qwen3.6-27b/2026-08</c>. Lets a later version re-queue only what it would improve.
    /// </summary>
    public string? ModelVersion { get; init; }

    /// <summary>One-line caption generated by the vision model.</summary>
    public string? AiCaption { get; init; }

    /// <summary>Longer description generated by the vision model.</summary>
    public string? AiDescription { get; init; }

    /// <summary>Title typed by the user. Takes precedence over <see cref="AiCaption"/> everywhere.</summary>
    public string? UserTitle { get; init; }

    /// <summary>Notes typed by the user.</summary>
    public string? UserNotes { get; init; }

    /// <summary>Star rating 0-5, user-set. Zero means unrated.</summary>
    public int Rating { get; init; }

    /// <summary>Whether the user marked this a favourite.</summary>
    public bool IsFavourite { get; init; }

    /// <summary>
    /// Monotonically increasing local revision, bumped on every change to this row.
    /// The sync layer uses it to decide what still needs publishing.
    /// </summary>
    public long Revision { get; init; }

    /// <summary>
    /// What to show as this photo's headline: the user's title if they set one, else the model's
    /// caption, else the filename. Never empty, so the UI never has to decide.
    /// </summary>
    public string DisplayTitle =>
        !string.IsNullOrWhiteSpace(UserTitle) ? UserTitle!
        : !string.IsNullOrWhiteSpace(AiCaption) ? AiCaption!
        : FileName;

    /// <summary>Aspect ratio (width / height), or 1 when the dimensions are unknown.</summary>
    public double AspectRatio => Height > 0 ? (double)Width / Height : 1d;
}

/// <summary>A single tag attached to a photo, carrying its provenance and confidence.</summary>
/// <param name="Tag">The normalised tag text, lowercase, for example <c>beach</c>.</param>
/// <param name="Confidence">
/// Model confidence 0-1. Always 1.0 for <see cref="MetadataSource.User"/> tags — a person who
/// typed a tag is not uncertain about it.
/// </param>
/// <param name="Source">Where the tag came from.</param>
public readonly record struct PhotoTag(string Tag, double Confidence, MetadataSource Source);

/// <summary>An object located in a photo by the detection model.</summary>
/// <param name="Label">The detected class, for example <c>dog</c>.</param>
/// <param name="Confidence">Detector confidence, 0-1.</param>
/// <param name="X">Left edge as a fraction of image width, 0-1.</param>
/// <param name="Y">Top edge as a fraction of image height, 0-1.</param>
/// <param name="Width">Box width as a fraction of image width, 0-1.</param>
/// <param name="Height">Box height as a fraction of image height, 0-1.</param>
/// <remarks>
/// Boxes are stored as fractions rather than pixels so they stay valid against the thumbnail,
/// the proxy, and the original without a conversion step at every use.
/// </remarks>
public readonly record struct DetectedObject(
    string Label,
    double Confidence,
    double X,
    double Y,
    double Width,
    double Height);
