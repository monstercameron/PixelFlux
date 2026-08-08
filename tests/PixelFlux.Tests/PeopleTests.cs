using PixelFlux.Core.Index;
using PixelFlux.Core.Model;

namespace PixelFlux.Tests;

/// <summary>
/// Naming faces.
///
/// No detector here. The questions are about what happens to a name once it exists — does it
/// survive the detector running again, does the same name spelled differently split somebody in
/// two — and none of those need a model to answer.
/// </summary>
public sealed class PeopleTests : IAsyncLifetime
{
    private string _workDir = string.Empty;
    private PhotoDatabase _database = null!;
    private PhotoStore _photos = null!;
    private FaceStore _faces = null!;

    public Task InitializeAsync()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "pixelflux-people-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_workDir);

        _database = new PhotoDatabase(Path.Combine(_workDir, "library.db"));
        _database.Migrate();
        _photos = new PhotoStore(_database);
        _faces = new FaceStore(_database);

        return Task.CompletedTask;
    }

    public Task DisposeAsync()
    {
        try
        {
            Directory.Delete(_workDir, recursive: true);
        }
        catch (IOException)
        {
            // Temporary directory; its removal is not what is under test.
        }

        return Task.CompletedTask;
    }

    private async Task<long> AddPhotoAsync(string name)
    {
        (long id, _) = await _photos.UpsertAsync(new PhotoRecord
        {
            FileName = name,
            OriginalPath = Path.Combine(_workDir, name),
            ContentHash = name + "-hash",
            PerceptualHash = name.PadRight(16, '0')[..16],
            MimeType = "image/jpeg",
        });

        return id;
    }

    private static PhotoFaceRecord Face(
        long photoId,
        double x,
        double y,
        double size = 0.2,
        long? personId = null) =>
        new(0, photoId, 0.95, x, y, size, size, size * size, 0,
            "0,0 0,0 0,0 0,0 0,0", null, "yunet", null, null, personId);

    [Fact]
    public async Task ANameSurvivesTheDetectorRunningAgain()
    {
        // The property the whole design turns on. A sweep replaces every face for a photograph, so
        // without deliberate carry-over a better model — or pressing "try again" on a file that
        // failed — would silently erase every name anybody had typed.
        long photoId = await AddPhotoAsync("group.jpg");

        await _faces.ReplaceAsync(photoId, [Face(photoId, 0.10, 0.10), Face(photoId, 0.60, 0.10)]);

        IReadOnlyList<PhotoFaceRecord> found = await _faces.GetAsync(photoId);
        long namedFace = found.Single(f => Math.Abs(f.X - 0.10) < 0.001).Id;

        await _faces.NameFaceAsync(namedFace, "Pisey");

        // A second sweep finds the same two people, very slightly differently placed — which is
        // what actually happens when a detector runs twice.
        await _faces.ReplaceAsync(photoId, [Face(photoId, 0.11, 0.105), Face(photoId, 0.61, 0.10)]);

        IReadOnlyList<PhotoFaceRecord> after = await _faces.GetAsync(photoId);

        PhotoFaceRecord moved = after.Single(f => Math.Abs(f.X - 0.11) < 0.001);
        PhotoFaceRecord other = after.Single(f => Math.Abs(f.X - 0.61) < 0.001);

        Assert.True(moved.IsNamed);
        Assert.Equal("Pisey", moved.PersonName);

        // And it did not spread to the person standing next to her.
        Assert.False(other.IsNamed);
    }

    [Fact]
    public async Task ANameDoesNotJumpToADifferentFace()
    {
        // The failure that matters more than losing a name: putting the wrong name on somebody.
        // Here the named face is gone from the second sweep entirely and only a distant face
        // remains, so the right outcome is no name at all.
        long photoId = await AddPhotoAsync("group.jpg");

        await _faces.ReplaceAsync(photoId, [Face(photoId, 0.05, 0.05)]);
        IReadOnlyList<PhotoFaceRecord> found = await _faces.GetAsync(photoId);
        await _faces.NameFaceAsync(found[0].Id, "Pisey");

        await _faces.ReplaceAsync(photoId, [Face(photoId, 0.70, 0.70)]);

        IReadOnlyList<PhotoFaceRecord> after = await _faces.GetAsync(photoId);
        Assert.False(after[0].IsNamed);
    }

    [Fact]
    public async Task TheSameNameSpelledDifferentlyIsOnePerson()
    {
        // Otherwise a collection silently splits in half with no visible cause.
        long first = await AddPhotoAsync("one.jpg");
        long second = await AddPhotoAsync("two.jpg");

        await _faces.ReplaceAsync(first, [Face(first, 0.1, 0.1)]);
        await _faces.ReplaceAsync(second, [Face(second, 0.1, 0.1)]);

        long a = (await _faces.GetAsync(first))[0].Id;
        long b = (await _faces.GetAsync(second))[0].Id;

        long? one = await _faces.NameFaceAsync(a, "Mum");
        long? two = await _faces.NameFaceAsync(b, "mum");

        Assert.Equal(one, two);

        IReadOnlyList<NamedPerson> people = await _faces.ListNamedAsync();
        NamedPerson only = Assert.Single(people);
        Assert.Equal(2, only.PhotoCount);
    }

    [Fact]
    public async Task ANameCanBeClearedAndAPersonRenamed()
    {
        long photoId = await AddPhotoAsync("one.jpg");
        await _faces.ReplaceAsync(photoId, [Face(photoId, 0.1, 0.1)]);
        long faceId = (await _faces.GetAsync(photoId))[0].Id;

        await _faces.NameFaceAsync(faceId, "Pisey");
        Assert.True(await _faces.RenamePersonAsync(
            (await _faces.ListNamedAsync())[0].Id, "Pisey Chan"));

        Assert.Equal("Pisey Chan", (await _faces.GetAsync(photoId))[0].PersonName);

        // Clearing leaves the person in existence — they may still be on other photographs — but
        // detaches this face from them.
        Assert.Null(await _faces.NameFaceAsync(faceId, "   "));
        Assert.False((await _faces.GetAsync(photoId))[0].IsNamed);
    }

    [Fact]
    public async Task RenamingOntoAnExistingNameIsRefusedRatherThanMerging()
    {
        // Merging two people is a decision with consequences the store cannot judge, so it says no
        // and leaves it to the caller.
        long photoId = await AddPhotoAsync("two.jpg");
        await _faces.ReplaceAsync(photoId, [Face(photoId, 0.1, 0.1), Face(photoId, 0.6, 0.1)]);

        IReadOnlyList<PhotoFaceRecord> found = await _faces.GetAsync(photoId);
        await _faces.NameFaceAsync(found[0].Id, "Ana");
        await _faces.NameFaceAsync(found[1].Id, "Bea");

        IReadOnlyList<NamedPerson> people = await _faces.ListNamedAsync();
        long ana = people.Single(p => p.Name == "Ana").Id;

        Assert.False(await _faces.RenamePersonAsync(ana, "Bea"));
        Assert.Equal(2, (await _faces.ListNamedAsync()).Count);
    }

    [Fact]
    public async Task ConfirmingASuggestionNamesEveryFaceAtOnce()
    {
        long photoId = await AddPhotoAsync("many.jpg");
        await _faces.ReplaceAsync(photoId,
        [
            Face(photoId, 0.1, 0.1),
            Face(photoId, 0.4, 0.1),
            Face(photoId, 0.7, 0.1),
        ]);

        long[] ids = [.. (await _faces.GetAsync(photoId)).Select(f => f.Id)];
        await _faces.NameFacesAsync(ids, "Pisey");

        Assert.All(await _faces.GetAsync(photoId), face => Assert.True(face.IsNamed));
        Assert.Equal(3, (await _faces.ListNamedAsync())[0].FaceCount);
    }
}
