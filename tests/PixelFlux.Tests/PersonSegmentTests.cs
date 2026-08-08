using PixelFlux.Core.Model;

namespace PixelFlux.Tests;

/// <summary>
/// Matching named faces to the person outlines they belong to.
///
/// Pure arithmetic on rectangles, so these are the cheap tests — and the ones worth having, since
/// every interesting case is a group photograph where outlines overlap and the wrong answer puts
/// somebody's name on somebody else.
/// </summary>
public sealed class PersonSegmentTests
{
    private static PhotoSegmentRecord Segment(
        long id,
        string label,
        double x,
        double y,
        double w,
        double h,
        string? userLabel = null) =>
        new(id, 1, label, 0.9, x, y, w, h, w * h, 0.5, null, "yolo", userLabel);

    private static PhotoFaceRecord Face(
        double x,
        double y,
        double size,
        string? person = null) =>
        new(1, 1, 0.95, x, y, size, size, size * size, 0, "0,0", null, "yunet",
            null, null, person is null ? null : 7, person);

    [Fact]
    public void AFaceInsideAPersonNamesThatPerson()
    {
        // A body from head to knee, with the face where a face goes.
        PhotoSegmentRecord person = Segment(10, "person", 0.30, 0.10, 0.30, 0.70);
        PhotoFaceRecord face = Face(0.38, 0.14, 0.10, "Pisey");

        IReadOnlyDictionary<long, string> named =
            PersonSegments.Reconcile([person], [face]);

        Assert.Equal("Pisey", named[10]);
    }

    [Fact]
    public void OnlyPeopleAreNamed()
    {
        // A face can sit inside a chair's outline in a photograph of somebody sitting down. The
        // chair is still a chair.
        PhotoSegmentRecord chair = Segment(11, "chair", 0.30, 0.10, 0.30, 0.70);
        PhotoFaceRecord face = Face(0.38, 0.14, 0.10, "Pisey");

        Assert.Empty(PersonSegments.Reconcile([chair], [face]));
    }

    [Fact]
    public void AnUnnamedFaceNamesNothing()
    {
        PhotoSegmentRecord person = Segment(10, "person", 0.30, 0.10, 0.30, 0.70);

        Assert.Empty(PersonSegments.Reconcile([person], [Face(0.38, 0.14, 0.10)]));
    }

    [Fact]
    public void ACorrectedLabelIsNeverOverruled()
    {
        // Somebody has already said this outline is "Dad". Deriving a name from a face would
        // either agree with them pointlessly or contradict them, and the second is unforgivable.
        PhotoSegmentRecord person = Segment(10, "person", 0.30, 0.10, 0.30, 0.70, userLabel: "Dad");
        PhotoFaceRecord face = Face(0.38, 0.14, 0.10, "Pisey");

        Assert.Empty(PersonSegments.Reconcile([person], [face]));
    }

    [Fact]
    public void AFaceThatOnlyClipsAnOutlineIsNotClaimedByIt()
    {
        // The face of somebody standing behind, whose head overlaps the edge of the person in
        // front. Refusing is the right answer.
        PhotoSegmentRecord person = Segment(10, "person", 0.30, 0.10, 0.30, 0.70);
        PhotoFaceRecord face = Face(0.24, 0.14, 0.10, "Pisey");   // mostly outside on the left

        Assert.Empty(PersonSegments.Reconcile([person], [face]));
    }

    [Fact]
    public void TwoPeopleSideBySideKeepTheirOwnNames()
    {
        PhotoSegmentRecord left = Segment(10, "person", 0.05, 0.10, 0.30, 0.70);
        PhotoSegmentRecord right = Segment(11, "person", 0.55, 0.10, 0.30, 0.70);

        PhotoFaceRecord onLeft = Face(0.13, 0.14, 0.10, "Ana");
        PhotoFaceRecord onRight = Face(0.63, 0.14, 0.10, "Bea");

        IReadOnlyDictionary<long, string> named =
            PersonSegments.Reconcile([left, right], [onLeft, onRight]);

        Assert.Equal("Ana", named[10]);
        Assert.Equal("Bea", named[11]);
    }

    [Fact]
    public void OneFaceInsideTwoOverlappingOutlinesPicksTheTighterOne()
    {
        // The case that makes greedy matching necessary. A wide outline behind swallows the space
        // occupied by a narrow one in front; the face belongs to the body that fits it.
        PhotoSegmentRecord wide = Segment(10, "person", 0.00, 0.05, 0.90, 0.90);
        PhotoSegmentRecord narrow = Segment(11, "person", 0.30, 0.10, 0.25, 0.70);

        PhotoFaceRecord face = Face(0.38, 0.14, 0.10, "Pisey");

        IReadOnlyDictionary<long, string> named =
            PersonSegments.Reconcile([wide, narrow], [face]);

        Assert.Equal("Pisey", named[11]);
        Assert.False(named.ContainsKey(10));
    }

    [Fact]
    public void OneNameIsNeverSpreadAcrossSeveralBodies()
    {
        // Three overlapping outlines, one named face. Exactly one of them may claim it.
        PhotoSegmentRecord a = Segment(10, "person", 0.20, 0.05, 0.50, 0.90);
        PhotoSegmentRecord b = Segment(11, "person", 0.25, 0.08, 0.45, 0.85);
        PhotoSegmentRecord c = Segment(12, "person", 0.30, 0.10, 0.40, 0.80);

        IReadOnlyDictionary<long, string> named =
            PersonSegments.Reconcile([a, b, c], [Face(0.40, 0.14, 0.10, "Pisey")]);

        Assert.Single(named);
    }
}
