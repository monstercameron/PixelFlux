using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Ingest;
using PixelFlux.Core.Model;

namespace PixelFlux.Tests;

/// <summary>
/// Corrections a person makes to what the model said.
///
/// The model calls a red sports car a truck. Being able to say otherwise is only worth anything
/// if the correction sticks: it has to change what the photograph is called, change what finds
/// it, and survive a better model being run over the library later. Those three are what these
/// tests are about.
/// </summary>
public sealed class ManualLabelTests : IAsyncLifetime
{
    private string _workDir = string.Empty;
    private PhotoStore _store = null!;
    private SegmentStore _segments = null!;

    private static string AlbumPath
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "testdata", "album")))
            {
                dir = dir.Parent;
            }

            return dir is null
                ? throw new DirectoryNotFoundException("Could not locate testdata/album.")
                : Path.Combine(dir.FullName, "testdata", "album");
        }
    }

    public async Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-labels", Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(_workDir);

        var database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        database.Migrate();

        _store = new PhotoStore(database);
        _segments = new SegmentStore(database);

        var ingestor = new LibraryIngestor(_store, new DerivativeGenerator(Path.Combine(_workDir, "cache")));
        await ingestor.ImportAsync([AlbumPath]);
    }

    public Task DisposeAsync()
    {
        try
        {
            Directory.Delete(_workDir, recursive: true);
        }
        catch (IOException)
        {
            // A temp directory nobody deleted is not a failing test.
        }

        return Task.CompletedTask;
    }

    private async Task<PhotoRecord> AnyPhotoAsync()
        => (await _store.QueryAsync(new PhotoQuery { Limit = 1 }))[0];

    private static PhotoSegmentRecord Segment(
        long photoId, string label, double x = 0.2, double y = 0.2, double size = 0.4) =>
        new(0, photoId, label, 0.9, x, y, size, size, size * size, 0.6, null, "yolo-test");

    // ------------------------------------------------------------------- segment labels

    [Fact]
    public async Task ACorrectionChangesWhatTheRegionIsCalled()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _segments.ReplaceAsync(photo.Id, [Segment(photo.Id, "truck")], "yolo-test");

        PhotoSegmentRecord before = (await _segments.GetAsync(photo.Id))[0];
        Assert.Equal("truck", before.DisplayLabel);
        Assert.False(before.IsCorrected);

        Assert.True(await _segments.SetUserLabelAsync(before.Id, "car"));

        PhotoSegmentRecord after = (await _segments.GetAsync(photo.Id))[0];

        // The person's word is what the interface shows, and what the colour is derived from —
        // a correction that only changed a tooltip would not be worth making.
        Assert.Equal("car", after.DisplayLabel);
        Assert.True(after.IsCorrected);
        Assert.Equal(PhotoSegmentRecord.HueFor("car"), after.Hue);

        // The model's own answer is kept alongside, not overwritten. It is a different kind of
        // claim, and a re-run needs to be able to say whether it now agrees.
        Assert.Equal("truck", after.Label);
    }

    [Fact]
    public async Task ClearingACorrectionRestoresTheModelsWord()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _segments.ReplaceAsync(photo.Id, [Segment(photo.Id, "bed")], "yolo-test");

        long id = (await _segments.GetAsync(photo.Id))[0].Id;
        await _segments.SetUserLabelAsync(id, "cardboard box");
        Assert.Equal("cardboard box", (await _segments.GetAsync(photo.Id))[0].DisplayLabel);

        await _segments.SetUserLabelAsync(id, null);

        PhotoSegmentRecord reverted = (await _segments.GetAsync(photo.Id))[0];
        Assert.Equal("bed", reverted.DisplayLabel);
        Assert.False(reverted.IsCorrected);
    }

    [Fact]
    public async Task ABlankCorrectionIsTreatedAsClearingIt()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _segments.ReplaceAsync(photo.Id, [Segment(photo.Id, "bed")], "yolo-test");

        long id = (await _segments.GetAsync(photo.Id))[0].Id;
        await _segments.SetUserLabelAsync(id, "   ");

        // Not a region named with three spaces: a region you could neither read nor search for.
        Assert.False((await _segments.GetAsync(photo.Id))[0].IsCorrected);
    }

    [Fact]
    public async Task ACorrectionSurvivesABetterModelRunningOverIt()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _segments.ReplaceAsync(photo.Id, [
            Segment(photo.Id, "truck", 0.20, 0.20, 0.40),
            Segment(photo.Id, "chair", 0.70, 0.10, 0.15)], "yolo-test");

        IReadOnlyList<PhotoSegmentRecord> first = await _segments.GetAsync(photo.Id);
        long truckId = first.Single(g => g.Label == "truck").Id;
        await _segments.SetUserLabelAsync(truckId, "car");

        // A newer model runs. Every row is replaced, the ids change, the boxes shift slightly,
        // and it still thinks the car is a truck.
        await _segments.ReplaceAsync(photo.Id, [
            Segment(photo.Id, "truck", 0.21, 0.19, 0.41),
            Segment(photo.Id, "chair", 0.70, 0.10, 0.15),
            Segment(photo.Id, "dog", 0.05, 0.60, 0.20)], "yolo-better");

        IReadOnlyList<PhotoSegmentRecord> second = await _segments.GetAsync(photo.Id);

        PhotoSegmentRecord carried = Assert.Single(second, g => g.IsCorrected);
        Assert.Equal("car", carried.DisplayLabel);
        Assert.Equal("yolo-better", carried.Model);

        // Only the region it was made about. The chair and the newly-found dog are untouched.
        Assert.Equal(3, second.Count);
        Assert.Contains(second, g => g.DisplayLabel == "chair" && !g.IsCorrected);
        Assert.Contains(second, g => g.DisplayLabel == "dog" && !g.IsCorrected);
    }

    [Fact]
    public async Task ACorrectionIsDroppedWhenTheRegionItDescribedIsGone()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _segments.ReplaceAsync(photo.Id, [Segment(photo.Id, "truck", 0.20, 0.20, 0.40)], "yolo-test");

        long id = (await _segments.GetAsync(photo.Id))[0].Id;
        await _segments.SetUserLabelAsync(id, "car");

        // The new run finds one thing, in a completely different part of the frame. Inheriting
        // the name would put "car" on whatever that is, which is worse than losing the
        // correction: it would be the application inventing a claim the user never made.
        await _segments.ReplaceAsync(photo.Id, [Segment(photo.Id, "bird", 0.80, 0.80, 0.15)], "yolo-better");

        PhotoSegmentRecord only = Assert.Single(await _segments.GetAsync(photo.Id));
        Assert.Equal("bird", only.DisplayLabel);
        Assert.False(only.IsCorrected);
    }

    [Fact]
    public async Task OneCorrectionCannotClaimTwoRegions()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _segments.ReplaceAsync(photo.Id, [Segment(photo.Id, "truck", 0.20, 0.20, 0.40)], "yolo-test");

        long id = (await _segments.GetAsync(photo.Id))[0].Id;
        await _segments.SetUserLabelAsync(id, "car");

        // The new run splits it into two heavily overlapping boxes. Exactly one may inherit.
        await _segments.ReplaceAsync(photo.Id, [
            Segment(photo.Id, "truck", 0.20, 0.20, 0.40),
            Segment(photo.Id, "truck", 0.21, 0.21, 0.40)], "yolo-better");

        Assert.Single(await _segments.GetAsync(photo.Id), g => g.IsCorrected);
    }

    [Fact]
    public async Task TheObjectFilterAndFacetFollowTheCorrection()
    {
        IReadOnlyList<PhotoRecord> photos = await _store.QueryAsync(new PhotoQuery { Limit = 2 });

        await _segments.ReplaceAsync(photos[0].Id, [Segment(photos[0].Id, "truck")], "yolo-test");
        await _segments.ReplaceAsync(photos[1].Id, [Segment(photos[1].Id, "car")], "yolo-test");

        long id = (await _segments.GetAsync(photos[0].Id))[0].Id;
        await _segments.SetUserLabelAsync(id, "car");

        // Clicking "car" in the sidebar has to return the photograph the user themselves said
        // was a car. Filtering on the raw model label would silently leave it out.
        IReadOnlyList<PhotoRecord> found = await _store.QueryAsync(new PhotoQuery { Object = "car" });
        Assert.Equal(2, found.Count);

        Assert.Empty(await _store.QueryAsync(new PhotoQuery { Object = "truck" }));

        IReadOnlyList<(string Label, int Count)> facet = await _segments.GetObjectFacetAsync();
        Assert.Equal(2, facet.Single(f => f.Label == "car").Count);
        Assert.DoesNotContain(facet, f => f.Label == "truck");
    }

    // ------------------------------------------------------------------------ user tags

    [Fact]
    public async Task AUserTagIsAddedRemovedAndSearchable()
    {
        PhotoRecord photo = await AnyPhotoAsync();

        Assert.True(await _store.AddUserTagAsync(photo.Id, "  Grandad's Boat  "));

        PhotoTag tag = Assert.Single(await _store.GetTagsAsync(photo.Id),
            t => t.Source == MetadataSource.User);

        // Trimmed and lower-cased: a library where "Beach", "beach" and "beach " are three tags
        // has a nonsense facet list and half its searches miss.
        Assert.Equal("grandad's boat", tag.Tag);

        IReadOnlyList<PhotoRecord> found = await _store.QueryAsync(new PhotoQuery { Text = "grandad" });
        Assert.Contains(found, p => p.Id == photo.Id);

        Assert.True(await _store.RemoveUserTagAsync(photo.Id, "grandad's boat"));
        Assert.DoesNotContain(await _store.GetTagsAsync(photo.Id), t => t.Source == MetadataSource.User);
    }

    [Fact]
    public async Task AnEmptyTagIsRefused()
    {
        PhotoRecord photo = await AnyPhotoAsync();

        Assert.False(await _store.AddUserTagAsync(photo.Id, "   "));
        Assert.DoesNotContain(await _store.GetTagsAsync(photo.Id), t => t.Source == MetadataSource.User);
    }

    [Fact]
    public async Task AModelRerunDoesNotDisturbTheUsersOwnKeywords()
    {
        PhotoRecord photo = await AnyPhotoAsync();

        await _store.AddUserTagAsync(photo.Id, "holiday");
        await _store.AddTagsAsync(photo.Id, [new PhotoTag("dog", 1.0, MetadataSource.Ai)]);

        // A second analysis replaces the AI set wholesale. The user's word must be untouched —
        // it is the one piece of metadata in the library that cannot be recomputed.
        await _store.AddTagsAsync(photo.Id, [new PhotoTag("cat", 1.0, MetadataSource.Ai)]);

        IReadOnlyList<PhotoTag> tags = await _store.GetTagsAsync(photo.Id);

        Assert.Contains(tags, t => t.Tag == "holiday" && t.Source == MetadataSource.User);
        Assert.Contains(tags, t => t.Tag == "cat");
        Assert.DoesNotContain(tags, t => t.Tag == "dog");
    }

    [Fact]
    public async Task AUserCannotDeleteWhatTheModelFound()
    {
        PhotoRecord photo = await AnyPhotoAsync();
        await _store.AddTagsAsync(photo.Id, [new PhotoTag("dog", 1.0, MetadataSource.Ai)]);

        // Refused rather than half-done. A re-run would write it straight back, so a deletion
        // that appeared to work and then undid itself would look like a bug in the application.
        Assert.False(await _store.RemoveUserTagAsync(photo.Id, "dog"));
        Assert.Contains(await _store.GetTagsAsync(photo.Id), t => t.Tag == "dog");
    }
}
