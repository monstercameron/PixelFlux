namespace PixelFlux.Ai.Faces;

/// <summary>
/// How alike two faces have to be before PixelFlux will call them the same person.
/// </summary>
/// <remarks>
/// <para>
/// One number, in one place, because it is the whole product decision behind "find this person".
/// Set it too high and the feature finds two photographs out of eleven and looks broken. Set it
/// too low and it shows strangers, which is not the same feature with a bug in it — it is a
/// worse feature, and on faces specifically it is the kind of wrong that makes a person distrust
/// everything else the application claims.
/// </para>
/// <para>
/// Given that asymmetry, the rule is: never a stranger. Missing one photograph of somebody is a
/// disappointment the user can work around by clicking a different face of theirs. Being shown
/// somebody else's face under the heading "this person" is not.
/// </para>
/// </remarks>
public static class FaceGrouping
{
    /// <summary>
    /// Minimum cosine similarity for two faces to be treated as the same person.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measured against the labelled corpus in <c>testdata/people</c>, not taken from the model
    /// card — see <c>FaceRecognitionTests.ThresholdIsCalibratedAgainstLabelledPairs</c>, which
    /// prints the recall and the false matches at every candidate value and fails the build if
    /// any two different people score above this line.
    /// </para>
    /// <para>
    /// The measurement is unusually clear. Photographs of the same person score 0.75 to 0.97;
    /// the closest two different people score 0.256. Between those lies an empty band, and every
    /// threshold from 0.30 to 0.55 scores identically on the corpus: 19 of 20 same-person pairs
    /// found, no strangers. So this is not a knife-edge tuned to a number — it is the middle of
    /// a wide gap, 0.14 above the closest stranger and 0.35 below the same-person cluster.
    /// </para>
    /// <para>
    /// OpenCV publishes 0.363 for this model. Sitting slightly above it costs nothing measurable
    /// here and buys margin, which is the right side to err on given the rule above.
    /// </para>
    /// <para>
    /// The one same-person pair no threshold recovers scores 0.173 — two photographs of one
    /// person years apart in very different light. That is a real limit of the model and worth
    /// stating plainly rather than tuning around: dropping low enough to catch it starts
    /// admitting strangers three pairs later.
    /// </para>
    /// </remarks>
    public const double DefaultThreshold = 0.40;

    /// <summary>
    /// A stricter setting, for someone who would rather miss a photograph than see a stranger.
    /// </summary>
    public const double StrictThreshold = 0.50;

    /// <summary>
    /// A looser setting, for finding a person across years and changes of appearance.
    /// </summary>
    /// <remarks>
    /// Offered, but never the default. It still admitted no strangers on the corpus, but it sits
    /// only 0.044 above the closest pair of different people — close enough that a larger
    /// library would eventually cross it, and a user who has enabled it has at least chosen to.
    /// </remarks>
    public const double LooseThreshold = 0.30;
}
