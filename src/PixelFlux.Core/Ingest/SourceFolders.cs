namespace PixelFlux.Core.Ingest;

/// <summary>
/// The folders a library imports from, and the rules for keeping that list sane.
/// </summary>
/// <remarks>
/// <para>
/// A list of paths looks like it needs no logic until somebody adds <c>D:\Photos\2019</c> to a
/// library that already watches <c>D:\Photos</c>. Nothing breaks — importing is idempotent, so the
/// photographs are simply found twice and deduplicated by content hash — but every scan then costs
/// twice as long over that subtree for no result, and the settings screen shows two entries that
/// are really one. Both of those are the kind of thing a person notices months later and cannot
/// explain.
/// </para>
/// <para>
/// So the list is kept flat: adding a folder inside one already watched changes nothing, and
/// adding a folder that contains watched ones absorbs them. That is the entire content of this
/// class, and it is separate from anything that touches a disk so it can be tested by naming
/// paths.
/// </para>
/// </remarks>
public static class SourceFolders
{
    /// <summary>The setting the list is stored under.</summary>
    public const string SettingKey = "library.sources";

    /// <summary>Reads a stored list.</summary>
    /// <param name="stored">A value written by <see cref="Serialise"/>, or null.</param>
    /// <returns>The folders, in the order they were added.</returns>
    /// <remarks>
    /// One path per line rather than JSON. A path cannot contain a newline on any platform this
    /// runs on, the file is read by people debugging as often as by code, and a malformed line
    /// costs one folder rather than the whole list.
    /// </remarks>
    public static IReadOnlyList<string> Parse(string? stored) =>
        string.IsNullOrWhiteSpace(stored)
            ? []
            : [.. stored
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];

    /// <summary>Renders a list for storage.</summary>
    /// <param name="folders">The folders.</param>
    /// <returns>One path per line.</returns>
    public static string Serialise(IEnumerable<string> folders) =>
        string.Join('\n', folders);

    /// <summary>Adds a folder, keeping the list flat and free of duplicates.</summary>
    /// <param name="existing">The current list.</param>
    /// <param name="folder">The folder to add.</param>
    /// <returns>The new list, which may be unchanged.</returns>
    /// <remarks>
    /// Three outcomes, in order of how often they happen: the folder is new and is appended; it is
    /// already covered by something watched and the list is returned untouched; or it contains
    /// folders already watched, which are removed in its favour.
    /// </remarks>
    public static IReadOnlyList<string> Add(IReadOnlyList<string> existing, string folder)
    {
        ArgumentNullException.ThrowIfNull(existing);

        if (string.IsNullOrWhiteSpace(folder))
        {
            return existing;
        }

        string added = Normalise(folder);

        if (existing.Any(current => Covers(Normalise(current), added)))
        {
            return existing;
        }

        return [.. existing.Where(current => !Covers(added, Normalise(current))), added];
    }

    /// <summary>Removes a folder.</summary>
    /// <param name="existing">The current list.</param>
    /// <param name="folder">The folder to drop.</param>
    /// <returns>The new list.</returns>
    public static IReadOnlyList<string> Remove(IReadOnlyList<string> existing, string folder)
    {
        ArgumentNullException.ThrowIfNull(existing);

        string target = Normalise(folder);
        return [.. existing.Where(current => !PathsEqual(Normalise(current), target))];
    }

    /// <summary>
    /// Trims a path to the form used for comparison: absolute, no trailing separator.
    /// </summary>
    /// <param name="folder">Any path.</param>
    /// <returns>The comparable form.</returns>
    /// <remarks>
    /// <c>D:\Photos</c> and <c>D:\Photos\</c> are the same folder, and one of them arrives from a
    /// picker while the other is typed. Comparing them as strings without this makes the list grow
    /// a duplicate that looks identical on screen.
    /// </remarks>
    public static string Normalise(string folder)
    {
        if (string.IsNullOrWhiteSpace(folder))
        {
            return string.Empty;
        }

        string full = Path.GetFullPath(folder.Trim());

        // A drive root keeps its separator: "D:" is not a folder, "D:\" is.
        return full.Length > 3 ? full.TrimEnd(Path.DirectorySeparatorChar) : full;
    }

    /// <summary>Whether one folder contains another, or is the same folder.</summary>
    /// <param name="outer">The possible parent, normalised.</param>
    /// <param name="inner">The possible child, normalised.</param>
    /// <returns>True when scanning <paramref name="outer"/> would also cover
    /// <paramref name="inner"/>.</returns>
    /// <remarks>
    /// The separator on the end of the prefix is what stops <c>D:\Photos</c> from appearing to
    /// contain <c>D:\PhotosOld</c> — a bug that would silently drop a real source folder from the
    /// list the moment somebody added its similarly named neighbour.
    /// </remarks>
    public static bool Covers(string outer, string inner)
    {
        if (PathsEqual(outer, inner))
        {
            return true;
        }

        string prefix = outer.EndsWith(Path.DirectorySeparatorChar)
            ? outer
            : outer + Path.DirectorySeparatorChar;

        return inner.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
    }

    private static bool PathsEqual(string a, string b) =>
        // Case-insensitive because this runs on Windows, where it is the filesystem's own rule and
        // a picker will happily return a different casing than the one that was typed.
        string.Equals(a, b, StringComparison.OrdinalIgnoreCase);
}
