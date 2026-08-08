namespace PixelFlux.Core.Model;

/// <summary>
/// A labelled region of a photograph, as stored in the library.
/// </summary>
/// <param name="Id">Row id.</param>
/// <param name="PhotoId">The photograph this belongs to.</param>
/// <param name="Label">What the model called it, for example <c>dog</c>.</param>
/// <param name="Confidence">Model confidence, 0-1.</param>
/// <param name="X">Left edge of the bounding box as a fraction of image width.</param>
/// <param name="Y">Top edge as a fraction of image height.</param>
/// <param name="Width">Box width as a fraction of image width.</param>
/// <param name="Height">Box height as a fraction of image height.</param>
/// <param name="AreaFraction">Share of the whole frame the mask actually covers, 0-1.</param>
/// <param name="Prominence">
/// Blended area and confidence, 0-1 — how much this segment is "what the photo is of".
/// Stored rather than recomputed so the sidebar can order by it in SQL.
/// </param>
/// <param name="MaskKey">
/// Cache key of the greyscale PNG holding the mask, or null if no mask was written.
/// The mask covers the bounding box only, not the whole frame.
/// </param>
/// <param name="Model">Identifier of the model that produced this, so a re-run can replace it.</param>
/// <param name="UserLabel">
/// What a person called this region, when they disagreed with the model. Null means nobody has
/// said otherwise.
/// </param>
/// <remarks>
/// Distinct from <see cref="DetectedObject"/>, which is a plain box with no mask. Both exist
/// because they answer different questions: a box is enough to say "there is a dog in this
/// photo", and only a mask can say which pixels the dog occupies — which is what the overlay
/// draws and what makes an area figure meaningful.
/// </remarks>
public sealed record PhotoSegmentRecord(
    long Id,
    long PhotoId,
    string Label,
    double Confidence,
    double X,
    double Y,
    double Width,
    double Height,
    double AreaFraction,
    double Prominence,
    string? MaskKey,
    string Model,
    string? UserLabel = null)
{
    /// <summary>What to call this region: the person's word where there is one.</summary>
    /// <remarks>
    /// Everything user-facing reads this rather than <see cref="Label" /> — the chip, the colour,
    /// the object facet, the search index. A correction that only changed a tooltip would not be
    /// worth making.
    /// </remarks>
    public string DisplayLabel => string.IsNullOrWhiteSpace(UserLabel) ? Label : UserLabel;

    /// <summary>Whether a person has overridden the model here.</summary>
    public bool IsCorrected => !string.IsNullOrWhiteSpace(UserLabel);

    /// <summary>
    /// A stable colour for this label, as an HSL hue in degrees.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The design system otherwise allows two accent colours, and this is the one deliberate
    /// exception. In an overlay, colour is not decoration — it is the only thing distinguishing
    /// one segment from the next where two masks touch, and it is what ties a label chip to the
    /// region it names. Encoding identity is exactly the job colour should be doing.
    /// </para>
    /// <para>
    /// Derived from the label rather than the row id so that every dog in the library is the
    /// same colour across every photograph, and so the colour survives a re-run of the model.
    /// The golden-ratio step spreads consecutive hashes far apart on the wheel instead of
    /// clustering them.
    /// </para>
    /// </remarks>
    public int Hue => HueFor(DisplayLabel);

    /// <summary>The hue a given label always gets, wherever it appears.</summary>
    /// <param name="label">The label.</param>
    /// <returns>Hue in degrees, 0-359.</returns>
    /// <remarks>
    /// Static so the segmentation writer can bake the same colour into the mask image that the
    /// UI uses for the chip. If these two ever disagreed, a chip would point at a differently
    /// coloured region and the connection between them would be broken.
    /// </remarks>
    public static int HueFor(string label)
    {
        uint hash = 2166136261u;
        foreach (char c in label)
        {
            hash = (hash ^ c) * 16777619u;
        }

        return (int)(hash * 0.618033988749895 % 360);
    }
}
