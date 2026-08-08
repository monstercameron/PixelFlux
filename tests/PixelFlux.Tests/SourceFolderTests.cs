using PixelFlux.Core.Ingest;

namespace PixelFlux.Tests;

/// <summary>
/// Keeping the list of import folders flat.
///
/// Pure string work, so these cost nothing to run — and the interesting cases are the ones where a
/// naive implementation quietly does the wrong thing months after anybody would connect it to this
/// code.
/// </summary>
public sealed class SourceFolderTests
{
    private static string P(params string[] parts) =>
        Path.GetFullPath(Path.Combine([Path.GetTempPath(), .. parts]));

    [Fact]
    public void AFolderIsAddedOnce()
    {
        IReadOnlyList<string> list = SourceFolders.Add([], P("Photos"));
        list = SourceFolders.Add(list, P("Photos"));

        Assert.Single(list);
    }

    [Fact]
    public void ATrailingSeparatorIsTheSameFolder()
    {
        // One of these comes from a picker and the other from somebody typing. Treating them as
        // different puts two identical-looking rows in the settings list.
        IReadOnlyList<string> list = SourceFolders.Add([], P("Photos"));
        list = SourceFolders.Add(list, P("Photos") + Path.DirectorySeparatorChar);

        Assert.Single(list);
    }

    [Fact]
    public void AddingASubfolderOfAWatchedFolderChangesNothing()
    {
        // Importing is idempotent, so this would not corrupt anything — it would just scan the
        // same subtree twice for ever and show two entries that are really one.
        IReadOnlyList<string> list = SourceFolders.Add([], P("Photos"));
        list = SourceFolders.Add(list, P("Photos", "2019"));

        Assert.Equal([SourceFolders.Normalise(P("Photos"))], list);
    }

    [Fact]
    public void AddingAParentAbsorbsTheFoldersInsideIt()
    {
        IReadOnlyList<string> list = SourceFolders.Add([], P("Photos", "2019"));
        list = SourceFolders.Add(list, P("Photos", "2020"));
        list = SourceFolders.Add(list, P("Photos"));

        Assert.Equal([SourceFolders.Normalise(P("Photos"))], list);
    }

    [Fact]
    public void ASimilarlyNamedNeighbourIsNotAbsorbed()
    {
        // The bug this exists to prevent: without a separator on the prefix, "Photos" appears to
        // contain "PhotosOld", and adding one silently deletes the other from the list.
        IReadOnlyList<string> list = SourceFolders.Add([], P("PhotosOld"));
        list = SourceFolders.Add(list, P("Photos"));

        Assert.Equal(2, list.Count);
        Assert.Contains(SourceFolders.Normalise(P("PhotosOld")), list);
        Assert.Contains(SourceFolders.Normalise(P("Photos")), list);
    }

    [Fact]
    public void CaseDoesNotCreateADuplicateOnWindows()
    {
        IReadOnlyList<string> list = SourceFolders.Add([], P("Photos"));
        list = SourceFolders.Add(list, P("Photos").ToUpperInvariant());

        Assert.Single(list);
    }

    [Fact]
    public void RemovingTakesTheFolderOutAndLeavesTheRest()
    {
        IReadOnlyList<string> list = SourceFolders.Add([], P("Photos"));
        list = SourceFolders.Add(list, P("Scans"));
        list = SourceFolders.Remove(list, P("Photos"));

        Assert.Equal([SourceFolders.Normalise(P("Scans"))], list);
    }

    [Fact]
    public void TheListSurvivesBeingStoredAndReadBack()
    {
        IReadOnlyList<string> list = SourceFolders.Add([], P("Photos"));
        list = SourceFolders.Add(list, P("Scans"));

        Assert.Equal(list, SourceFolders.Parse(SourceFolders.Serialise(list)));
    }

    [Fact]
    public void NothingStoredIsAnEmptyList()
    {
        Assert.Empty(SourceFolders.Parse(null));
        Assert.Empty(SourceFolders.Parse(string.Empty));
        Assert.Empty(SourceFolders.Parse("   "));
    }
}
