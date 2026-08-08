namespace PixelFlux.Core.Model;

/// <summary>
/// Works out which segmented person is which named face.
/// </summary>
/// <remarks>
/// <para>
/// <b>Two models that have never heard of each other.</b> The segmenter outlines a <c>person</c>
/// and knows nothing about who they are; the face detector finds a face and, once somebody has
/// typed a name, knows exactly who they are and nothing about the body attached to them. Both are
/// drawn on the same photograph, so the overlay says "person" directly above a face the library
/// can name. Reconciling them is arithmetic on two rectangles.
/// </para>
/// <para>
/// <b>Computed, never stored.</b> Either layer can be re-run at any time — a better segmenter, a
/// re-swept face, a name typed a second ago — and a stored link would be stale the moment either
/// happened. This is a pure function over whatever the two layers currently say, in the same
/// spirit as face grouping: a derived opinion is recomputed, not persisted.
/// </para>
/// <para>
/// <b>It only ever adds a name to a person.</b> Not to a chair, not to a dog, and never over a
/// label a human typed. A name appearing on the wrong outline is worse than no name at all, so
/// every rule below is written to abstain rather than guess.
/// </para>
/// </remarks>
public static class PersonSegments
{
    /// <summary>The segmenter's class for a human being.</summary>
    /// <remarks>
    /// Matched against <see cref="PhotoSegmentRecord.Label"/> — the model's own word — rather than
    /// the display label. A person who has renamed a segment to "Dad" has already said who it is;
    /// re-deriving it from a face would either agree pointlessly or overrule them.
    /// </remarks>
    public const string PersonLabel = "person";

    /// <summary>
    /// How much of a face box must fall inside a person's outline to belong to them.
    /// </summary>
    /// <remarks>
    /// Containment of the face, not overlap between the two boxes: a face is a small fraction of a
    /// whole body, so intersection-over-union would be near zero for a perfect match and is the
    /// wrong measure entirely. Three quarters allows for a detector that clips an ear or a
    /// segmenter that cuts the top of a head, and still refuses a face that merely brushes the
    /// edge of somebody standing behind.
    /// </remarks>
    public const double MinimumContainment = 0.75;

    /// <summary>Matches named faces to the person outlines they sit inside.</summary>
    /// <param name="segments">What the segmenter found.</param>
    /// <param name="faces">What the face detector found, some of them named.</param>
    /// <returns>
    /// Segment id to person name, containing only the segments that could be resolved confidently.
    /// A segment absent from the result keeps whatever label it already had.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Greedy, best pair first, and each face and each segment is used once. In a group photograph
    /// person outlines overlap heavily and a face can sit inside two or three of them; taking the
    /// strongest containment first means the tightest, most certain pairing wins and the others
    /// are then competing for what is left. Assigning every candidate independently would put one
    /// name on three overlapping bodies.
    /// </para>
    /// <para>
    /// Ties break towards the smaller outline. Two boxes that both fully contain a face differ in
    /// that the smaller one is more likely to be the body that face is attached to — the larger is
    /// usually somebody in front, whose outline happens to swallow the space behind them.
    /// </para>
    /// </remarks>
    public static IReadOnlyDictionary<long, string> Reconcile(
        IReadOnlyList<PhotoSegmentRecord> segments,
        IReadOnlyList<PhotoFaceRecord> faces)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(faces);

        var named = new Dictionary<long, string>();

        List<PhotoSegmentRecord> people =
        [
            .. segments.Where(segment =>
                string.Equals(segment.Label, PersonLabel, StringComparison.OrdinalIgnoreCase)
                && !segment.IsCorrected),
        ];

        List<PhotoFaceRecord> candidates = [.. faces.Where(face => face.IsNamed)];

        if (people.Count == 0 || candidates.Count == 0)
        {
            return named;
        }

        var pairs = new List<(double Containment, double SegmentArea, long SegmentId, int Face)>();

        for (int s = 0; s < people.Count; s++)
        {
            PhotoSegmentRecord segment = people[s];

            for (int f = 0; f < candidates.Count; f++)
            {
                double containment = Containment(candidates[f], segment);

                if (containment >= MinimumContainment)
                {
                    pairs.Add((containment, segment.Width * segment.Height, segment.Id, f));
                }
            }
        }

        var usedFaces = new HashSet<int>();

        foreach ((double _, double _, long segmentId, int face) in pairs
                     .OrderByDescending(pair => pair.Containment)
                     .ThenBy(pair => pair.SegmentArea))
        {
            if (named.ContainsKey(segmentId) || !usedFaces.Add(face))
            {
                continue;
            }

            named[segmentId] = candidates[face].PersonName!;
        }

        return named;
    }

    /// <summary>What share of the face box lies inside the segment's box.</summary>
    private static double Containment(PhotoFaceRecord face, PhotoSegmentRecord segment)
    {
        double left = Math.Max(face.X, segment.X);
        double top = Math.Max(face.Y, segment.Y);
        double right = Math.Min(face.X + face.Width, segment.X + segment.Width);
        double bottom = Math.Min(face.Y + face.Height, segment.Y + segment.Height);

        if (right <= left || bottom <= top)
        {
            return 0;
        }

        double faceArea = face.Width * face.Height;
        return faceArea <= 0 ? 0 : (right - left) * (bottom - top) / faceArea;
    }
}
