using System.Globalization;
using MetadataExtractor;
using MetadataExtractor.Formats.Exif;
using PixelFlux.Core.Model;
using Directory = MetadataExtractor.Directory;

namespace PixelFlux.Core.Imaging;

/// <summary>The metadata read out of a file, before anything is inferred or defaulted.</summary>
/// <param name="Camera">Camera and exposure settings; empty when the file has none.</param>
/// <param name="Location">GPS fix, when present.</param>
/// <param name="CapturedUtc">
/// EXIF capture time, or <see langword="null"/> when the file does not record one. Left null
/// rather than defaulted so the caller decides the fallback and can record that it did.
/// </param>
/// <param name="Orientation">EXIF orientation tag 1-8; 1 when absent.</param>
/// <param name="Keywords">IPTC/XMP/XP keywords already embedded in the file by another tool.</param>
/// <param name="Rating">
/// Star rating 0-5 recorded in the file, or 0 when none is present.
/// </param>
public readonly record struct ExifData(
    CameraInfo Camera,
    GeoPoint? Location,
    DateTimeOffset? CapturedUtc,
    int Orientation,
    IReadOnlyList<string> Keywords,
    int Rating);

/// <summary>
/// Reads embedded metadata from image files.
/// </summary>
/// <remarks>
/// <para>
/// Kept separate from pixel decoding for a practical reason: the two have different format
/// coverage. MetadataExtractor reads EXIF out of containers ImageSharp cannot decode at all,
/// HEIC being the one that matters most on a library full of iPhone photos. Splitting them means
/// a HEIC can be indexed, dated, and placed on the map even on a build with no HEIC decoder.
/// </para>
/// <para>
/// Every read here is defensive. Metadata in the wild is routinely malformed — truncated IFDs,
/// impossible dates, GPS at exactly 0,0 from a camera that never got a fix — and none of it is
/// worth failing an import over. Bad values are dropped and the photo is still indexed.
/// </para>
/// </remarks>
public static class ExifExtractor
{
    /// <summary>Reads what the file records, returning empty values for anything absent or malformed.</summary>
    /// <param name="path">Absolute path to the image file.</param>
    /// <returns>The extracted metadata. Never throws for a merely unreadable or odd file.</returns>
    public static ExifData Read(string path)
    {
        try
        {
            IReadOnlyList<Directory> directories = ImageMetadataReader.ReadMetadata(path);
            return new ExifData(
                ReadCamera(directories),
                ReadLocation(directories),
                ReadCaptureTime(directories),
                ReadOrientation(directories),
                ReadKeywords(directories),
                ReadRating(directories));
        }
        catch (Exception ex) when (ex is ImageProcessingException or IOException or UnauthorizedAccessException)
        {
            // No metadata is a perfectly ordinary state — screenshots, exports, and scans all
            // land here — and so is a corrupt metadata block on an otherwise fine image.
            return new ExifData(new CameraInfo(), null, null, 1, [], 0);
        }
    }

    private static CameraInfo ReadCamera(IReadOnlyList<Directory> directories)
    {
        ExifIfd0Directory? ifd0 = directories.OfType<ExifIfd0Directory>().FirstOrDefault();
        ExifSubIfdDirectory? sub = directories.OfType<ExifSubIfdDirectory>().FirstOrDefault();

        return new CameraInfo
        {
            Make = Clean(ifd0?.GetDescription(ExifDirectoryBase.TagMake)),
            Model = Clean(ifd0?.GetDescription(ExifDirectoryBase.TagModel)),
            Lens = Clean(sub?.GetDescription(ExifDirectoryBase.TagLensModel)),
            Iso = TryInt(sub, ExifDirectoryBase.TagIsoEquivalent),
            FNumber = TryDouble(sub, ExifDirectoryBase.TagFNumber),
            ExposureSeconds = TryDouble(sub, ExifDirectoryBase.TagExposureTime),
            FocalLengthMm = TryDouble(sub, ExifDirectoryBase.TagFocalLength),
        };
    }

    private static GeoPoint? ReadLocation(IReadOnlyList<Directory> directories)
    {
        GpsDirectory? gps = directories.OfType<GpsDirectory>().FirstOrDefault();
        if (gps is null)
        {
            return null;
        }

        MetadataExtractor.GeoLocation? location = gps.GetGeoLocation();

        // IsZero screens out the 0,0 "null island" fix that cameras emit when GPS is enabled
        // but never locked on. Treating that as a real location drops photos in the Gulf of
        // Guinea and quietly poisons any place-based search.
        if (location is null || location.IsZero)
        {
            return null;
        }

        double? altitude = null;
        if (gps.TryGetRational(GpsDirectory.TagAltitude, out MetadataExtractor.Rational raw))
        {
            // TagAltitudeRef: 0 = above sea level, 1 = below.
            int reference = gps.TryGetInt32(GpsDirectory.TagAltitudeRef, out int r) ? r : 0;
            altitude = reference == 1 ? -raw.ToDouble() : raw.ToDouble();
        }

        return new GeoPoint(location.Latitude, location.Longitude, altitude);
    }

    private static DateTimeOffset? ReadCaptureTime(IReadOnlyList<Directory> directories)
    {
        // Preference order matters. DateTimeOriginal is when the shutter fired. DateTimeDigitized
        // is when it was scanned or imported, which for a digitised negative can be decades
        // later. DateTime (IFD0) is last because editors rewrite it on save.
        ExifSubIfdDirectory? sub = directories.OfType<ExifSubIfdDirectory>().FirstOrDefault();
        if (sub is not null)
        {
            if (sub.TryGetDateTime(ExifDirectoryBase.TagDateTimeOriginal, out DateTime original))
            {
                return Normalise(original);
            }

            if (sub.TryGetDateTime(ExifDirectoryBase.TagDateTimeDigitized, out DateTime digitized))
            {
                return Normalise(digitized);
            }
        }

        ExifIfd0Directory? ifd0 = directories.OfType<ExifIfd0Directory>().FirstOrDefault();
        if (ifd0 is not null && ifd0.TryGetDateTime(ExifDirectoryBase.TagDateTime, out DateTime fallback))
        {
            return Normalise(fallback);
        }

        return null;
    }

    /// <summary>
    /// Converts an EXIF timestamp to UTC and rejects values that cannot be real.
    /// </summary>
    /// <remarks>
    /// EXIF timestamps carry no timezone, so they are treated as UTC rather than as local time.
    /// That is a deliberate choice: interpreting them as local would make the same photo change
    /// date depending on which device imported it, and a library that reorders itself when you
    /// travel is worse than one that is uniformly offset by a few hours.
    /// <para>
    /// The range check catches the classic corrupt-EXIF outputs (year 0000, year 2153) that
    /// would otherwise sit at the far ends of the timeline and stretch the time rail flat.
    /// </para>
    /// </remarks>
    private static DateTimeOffset? Normalise(DateTime value)
    {
        if (value.Year < 1826 || value > DateTime.UtcNow.AddDays(2))
        {
            return null; // 1826 is the oldest surviving photograph; anything earlier is corrupt.
        }

        return new DateTimeOffset(DateTime.SpecifyKind(value, DateTimeKind.Utc));
    }

    private static int ReadOrientation(IReadOnlyList<Directory> directories)
    {
        ExifIfd0Directory? ifd0 = directories.OfType<ExifIfd0Directory>().FirstOrDefault();
        return ifd0 is not null && ifd0.TryGetInt32(ExifDirectoryBase.TagOrientation, out int value)
               && value is >= 1 and <= 8
            ? value
            : 1;
    }

    private static IReadOnlyList<string> ReadKeywords(IReadOnlyList<Directory> directories)
    {
        // Keywords another tool already wrote are treated as File-sourced metadata: better than
        // nothing, not as authoritative as something the user typed in PixelFlux.
        var found = new List<string>();

        foreach (Directory directory in directories)
        {
            foreach (Tag tag in directory.Tags)
            {
                if (!tag.Name.Contains("Keyword", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                string? description = tag.Description;
                if (string.IsNullOrWhiteSpace(description))
                {
                    continue;
                }

                found.AddRange(description
                    .Split([';', ',', '|'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Where(k => k.Length is > 1 and < 64));
            }
        }

        return found.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
    }

    /// <summary>
    /// Reads a star rating out of the file, 0-5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Two tags carry this and they disagree surprisingly often. <c>Rating</c> (0x4746) is the
    /// 0-5 integer Windows Explorer shows and Adobe writes; <c>RatingPercent</c> (0x4749) is the
    /// same value as 0-100 and is what some tools update while leaving the first stale.
    /// </para>
    /// <para>
    /// The integer wins when present because it is the one a person set directly, and the
    /// percentage is used only as a fallback. Out-of-range values are clamped rather than
    /// rejected — a file claiming 7 stars is a tool bug, not a reason to lose the fact that the
    /// user rated it highly.
    /// </para>
    /// </remarks>
    private static int ReadRating(IReadOnlyList<Directory> directories)
    {
        foreach (Directory directory in directories)
        {
            if (directory.TryGetInt32(ExifDirectoryBase.TagRating, out int rating) && rating > 0)
            {
                return Math.Clamp(rating, 0, 5);
            }
        }

        foreach (Directory directory in directories)
        {
            if (directory.TryGetInt32(ExifDirectoryBase.TagRatingPercent, out int percent) && percent > 0)
            {
                return Math.Clamp((int)Math.Round(percent / 20.0), 0, 5);
            }
        }

        return 0;
    }

    private static string? Clean(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        // Camera firmware pads strings with NULs and trailing spaces; left in, they break both
        // exact-match search and grouping ("Canon\0" and "Canon" become two cameras).
        string trimmed = value.Trim().Trim('\0').Trim();
        return trimmed.Length == 0 ? null : trimmed;
    }

    private static int? TryInt(Directory? directory, int tag)
        => directory is not null && directory.TryGetInt32(tag, out int value) ? value : null;

    private static double? TryDouble(Directory? directory, int tag)
    {
        if (directory is null)
        {
            return null;
        }

        if (directory.TryGetRational(tag, out MetadataExtractor.Rational rational) &&
            rational.Denominator != 0)
        {
            return rational.ToDouble();
        }

        return directory.TryGetDouble(tag, out double value) &&
               !double.IsNaN(value) && !double.IsInfinity(value)
            ? value
            : null;
    }

    /// <summary>Formats an exposure time the way a camera would print it: <c>1/250</c>, or <c>2.5s</c>.</summary>
    /// <param name="seconds">Exposure time in seconds.</param>
    /// <returns>A display string, or <see langword="null"/> when there is nothing to show.</returns>
    public static string? FormatShutter(double? seconds)
    {
        if (seconds is null or <= 0)
        {
            return null;
        }

        return seconds >= 1
            ? string.Create(CultureInfo.InvariantCulture, $"{seconds.Value:0.#}s")
            : string.Create(CultureInfo.InvariantCulture, $"1/{Math.Round(1 / seconds.Value)}");
    }
}
