using System.Globalization;

namespace PixelFlux.Core.Model;

/// <summary>
/// A face found in a photograph, as stored in the library.
/// </summary>
/// <param name="Id">Row id.</param>
/// <param name="PhotoId">The photograph this face was found in.</param>
/// <param name="Confidence">Detector confidence, 0-1.</param>
/// <param name="X">Left edge of the face box as a fraction of image width.</param>
/// <param name="Y">Top edge as a fraction of image height.</param>
/// <param name="Width">Box width as a fraction of image width.</param>
/// <param name="Height">Box height as a fraction of image height.</param>
/// <param name="AreaFraction">Share of the frame the face occupies, 0-1.</param>
/// <param name="RollDegrees">
/// Head tilt measured from the eye line. The crop is straightened by this, and it is kept so a
/// later pass can tell an upright face from one lying on its side without re-running detection.
/// </param>
/// <param name="Landmarks">
/// Ten fractions, as <c>x,y</c> pairs: right eye, left eye, nose, right mouth corner, left mouth
/// corner. Stored as text because nothing queries an individual eye — see the schema notes.
/// </param>
/// <param name="CropKey">
/// Cache key of the square, straightened crop shown on the faces page, or null if writing it
/// failed. Null is survivable: the page falls back to the photograph's thumbnail.
/// </param>
/// <param name="Model">Which detector produced this, so a re-run can replace it.</param>
/// <param name="Embedding">
/// A unit-length vector describing what this face looks like, or null when no recognition model
/// was installed or the face was too small to describe usefully. Two faces of the same person
/// have a high dot product; see <c>FaceGrouping</c> for how high.
/// </param>
/// <param name="EmbedModel">
/// <param name="PersonId">Who this is, when somebody has said so. Null until then.</param>
/// <param name="PersonName">
/// That person's name, carried alongside the id so a list of faces can be labelled without a
/// second query per row. Read-only here — the name lives in the people table.
/// </param>
/// Which recognition model produced <see cref="Embedding" />. Vectors from different models are
/// not comparable, and comparing them anyway yields confident nonsense rather than an error.
/// </param>
/// <remarks>
/// Faces are the most sensitive rows in a photo library, and everything about how they are
/// produced reflects that: detection runs locally on this machine, the crops are ordinary files
/// in the same cache as the thumbnails, and nothing here is ever uploaded, shared, or sent to a
/// model that is not on disk.
/// </remarks>
public sealed record PhotoFaceRecord(
    long Id,
    long PhotoId,
    double Confidence,
    double X,
    double Y,
    double Width,
    double Height,
    double AreaFraction,
    double RollDegrees,
    string Landmarks,
    string? CropKey,
    string Model,
    float[]? Embedding = null,
    string? EmbedModel = null,
    long? PersonId = null,
    string? PersonName = null)
{
    /// <summary>Whether this face can take part in "find this person".</summary>
    public bool IsComparable => Embedding is { Length: > 0 };

    /// <summary>Whether somebody has said who this is.</summary>
    /// <remarks>
    /// A named face outranks whatever the recognition model grouped it with. The model's opinion
    /// is recomputed on every page load; this was typed by a person and is not.
    /// </remarks>
    public bool IsNamed => PersonId is not null;

    /// <summary>
    /// How prominent this face is in its photograph, 0-1.
    /// </summary>
    /// <remarks>
    /// Area alone puts a confidently-detected background face below a barely-detected large one.
    /// Blending in confidence — the same shape the segmentation prominence uses — orders the
    /// faces page the way a person scanning it would expect: subjects first, bystanders after.
    /// The square root pulls small faces up off the floor, since a face is a small share of a
    /// frame even when it is unmistakably the subject.
    /// </remarks>
    public double Prominence => (Math.Sqrt(AreaFraction) * 0.7) + (Confidence * 0.3);

    /// <summary>Parses <see cref="Landmarks" /> back into points.</summary>
    /// <returns>Five points as fractions of the image, or an empty list if the text is unusable.</returns>
    /// <remarks>
    /// Returns empty rather than throwing on malformed text. A face whose landmarks cannot be
    /// read is still a face worth showing; refusing to render the page over it would not be.
    /// </remarks>
    public IReadOnlyList<(double X, double Y)> ParseLandmarks()
    {
        string[] parts = Landmarks.Split(',', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2 || parts.Length % 2 != 0)
        {
            return [];
        }

        var points = new List<(double, double)>(parts.Length / 2);

        for (int i = 0; i + 1 < parts.Length; i += 2)
        {
            if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out double x) ||
                !double.TryParse(parts[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out double y))
            {
                return [];
            }

            points.Add((x, y));
        }

        return points;
    }

    /// <summary>Formats points for the <see cref="Landmarks" /> column.</summary>
    /// <param name="points">The points, as fractions of the image.</param>
    /// <returns>Comma-separated invariant-culture text.</returns>
    /// <remarks>
    /// Invariant culture, explicitly. A machine with a comma decimal separator would otherwise
    /// write "0,5178,0,3910" — which parses, in the wrong places, and puts every landmark
    /// somewhere plausible and wrong.
    /// </remarks>
    public static string FormatLandmarks(IEnumerable<(double X, double Y)> points)
    {
        ArgumentNullException.ThrowIfNull(points);

        return string.Join(',', points.SelectMany(p => new[]
        {
            p.X.ToString("0.#####", CultureInfo.InvariantCulture),
            p.Y.ToString("0.#####", CultureInfo.InvariantCulture),
        }));
    }
}
