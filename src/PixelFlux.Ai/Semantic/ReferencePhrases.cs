namespace PixelFlux.Ai.Semantic;

/// <summary>
/// A spread of unrelated captions, used to work out which photographs are agreeable to everything.
/// </summary>
/// <remarks>
/// <para>
/// These are not search terms and are never shown to anybody. They exist to be averaged: scoring
/// every photograph against all of them gives a per-photograph baseline that has nothing to do
/// with any particular query, and subtracting that baseline is what stops a busy photograph from
/// coming top of every search. See <c>VectorIndex.CalibrateAsync</c> for why that is necessary.
/// </para>
/// <para>
/// What matters about the list is its spread, not its contents. It should cover the space of
/// things photographs are of — people, places, objects, weather, times of day, indoors and out —
/// without leaning towards any one of them, because a bank that over-represents a subject would
/// discount exactly the photographs of that subject. It is deliberately generic: nothing here is
/// chosen to suit a particular library, and it must stay that way if the correction is to mean
/// the same thing in everybody's.
/// </para>
/// <para>
/// Thirty-two is enough for the average to be stable and small enough that calibrating costs
/// well under a second.
/// </para>
/// </remarks>
public static class ReferencePhrases
{
    /// <summary>The reference bank.</summary>
    public static IReadOnlyList<string> All { get; } =
    [
        "a photo of a person",
        "a photo of a group of people",
        "a photo of a child",
        "a portrait of a face",
        "a photo of an animal",
        "a photo of a bird",
        "a photo of a plant",
        "a photo of a flower",
        "a photo of a tree",
        "a photo of food on a plate",
        "a photo of a drink",
        "a photo of a car",
        "a photo of a bicycle",
        "a photo of a boat",
        "a photo of a building",
        "a photo of a door",
        "a photo of a window",
        "a photo of a street",
        "a photo of a room indoors",
        "a photo of furniture",
        "a photo of a mountain",
        "a photo of the sea",
        "a photo of a field",
        "a photo of the sky",
        "a photo taken at night",
        "a photo taken in the snow",
        "a photo of a sign with writing on it",
        "a photo of a machine",
        "a photo of clothing",
        "a photo of a celebration",
        "a blurry photograph",
        "a black and white photograph",
    ];
}
