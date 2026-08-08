using System.Buffers.Binary;
using System.Globalization;
using System.Reflection;
using System.Text;

namespace PixelFlux.Core.Geo;

/// <summary>A GPS fix resolved to somewhere a person would recognise.</summary>
/// <param name="City">Nearest populated place, for example <c>Ely</c>.</param>
/// <param name="Country">Country name in English, for example <c>United Kingdom</c>.</param>
/// <param name="CountryCode">ISO-3166 alpha-2 code, for grouping and flags.</param>
/// <param name="DistanceKm">Great-circle distance from the fix to the city centre.</param>
public readonly record struct ResolvedPlace(string City, string Country, string CountryCode, double DistanceKm)
{
    /// <summary>
    /// How to write this place in the interface.
    /// </summary>
    /// <remarks>
    /// The distance qualifier is the honest part. A fix 2 km from Ely genuinely is Ely; a fix
    /// 80 km away is not, and labelling it "Ely" would be a confident lie about where a
    /// photograph was taken. Beyond roughly 150 km even the nearest city says little, so only
    /// the country is claimed — which is still true, and still better than a pair of numbers.
    /// </remarks>
    public string Label => DistanceKm switch
    {
        <= 20 => $"{City}, {Country}",
        <= 150 => $"near {City}, {Country}",
        _ => Country,
    };
}

/// <summary>
/// Offline reverse geocoding: turns a latitude and longitude into a city and country.
/// </summary>
/// <remarks>
/// <para>
/// The data ships inside the assembly. Every reverse-geocoding <em>service</em> is a network
/// call, and sending the coordinates of someone's photographs to a third party in order to
/// print a place name under them would quietly undo the entire premise of a local-first photo
/// manager. The table is built by <c>tools/build_gazetteer.py</c> from GeoNames (CC BY 4.0) and
/// is about 680 KB for 34,000 populated places.
/// </para>
/// <para>
/// <b>Lookup structure.</b> Cities are bucketed into one-degree cells at load time. A query
/// examines its own cell first, then expands ring by ring until the nearest candidate found is
/// closer than the nearest possible point in the next ring — at which point no further ring can
/// improve on it. In practice that is one or two rings, a few dozen distance calculations, and
/// it beats scanning all 34,000 by a factor of roughly a thousand.
/// </para>
/// <para>
/// Loading is lazy and thread-safe: the first resolution pays for it, and an application that
/// never opens a geotagged photo never spends the memory.
/// </para>
/// </remarks>
public sealed class Gazetteer
{
    private const string ResourceName = "PixelFlux.Core.Geo.gazetteer.bin";
    private const int GridDegrees = 1;

    private static readonly Lazy<Gazetteer> Shared = new(() => new Gazetteer(), isThreadSafe: true);

    /// <summary>The process-wide instance. The table is read-only, so one copy serves everything.</summary>
    public static Gazetteer Instance => Shared.Value;

    private readonly float[] _latitudes;
    private readonly float[] _longitudes;
    private readonly ushort[] _countryIndex;
    private readonly string[] _names;
    private readonly string[] _countryNames;
    private readonly string[] _countryCodes;

    // Cell key -> indices of the cities inside it. The key packs a latitude and longitude cell
    // into one int so the dictionary needs no tuple hashing on a hot path.
    private readonly Dictionary<int, int[]> _grid;

    private Gazetteer()
    {
        using Stream? stream = typeof(Gazetteer).Assembly
            .GetManifestResourceStream(ResourceName);

        if (stream is null)
        {
            // A build without the resource must degrade to "no place names", not crash the app
            // on the first geotagged photograph.
            _latitudes = [];
            _longitudes = [];
            _countryIndex = [];
            _names = [];
            _countryNames = [];
            _countryCodes = [];
            _grid = [];
            return;
        }

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

        if (!reader.ReadBytes(4).AsSpan().SequenceEqual("PFGZ"u8) || reader.ReadInt32() != 1)
        {
            throw new InvalidDataException("gazetteer.bin is not a version 1 PixelFlux gazetteer.");
        }

        int countryCount = reader.ReadInt32();
        _countryCodes = new string[countryCount];
        _countryNames = new string[countryCount];

        for (int i = 0; i < countryCount; i++)
        {
            _countryCodes[i] = Encoding.ASCII.GetString(reader.ReadBytes(2));
            _countryNames[i] = Encoding.UTF8.GetString(reader.ReadBytes(reader.ReadByte()));
        }

        int cityCount = reader.ReadInt32();
        _latitudes = new float[cityCount];
        _longitudes = new float[cityCount];
        _countryIndex = new ushort[cityCount];
        _names = new string[cityCount];

        var buckets = new Dictionary<int, List<int>>(cityCount / 8);

        for (int i = 0; i < cityCount; i++)
        {
            _latitudes[i] = reader.ReadSingle();
            _longitudes[i] = reader.ReadSingle();
            _countryIndex[i] = reader.ReadUInt16();
            _names[i] = Encoding.UTF8.GetString(reader.ReadBytes(reader.ReadByte()));

            int key = CellKey(CellOf(_latitudes[i]), CellOf(_longitudes[i]));
            if (!buckets.TryGetValue(key, out List<int>? bucket))
            {
                buckets[key] = bucket = [];
            }

            bucket.Add(i);
        }

        _grid = buckets.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray());
    }

    /// <summary>Number of places in the table. Zero means the resource is missing.</summary>
    public int PlaceCount => _names.Length;

    /// <summary>
    /// Finds the nearest populated place to a coordinate.
    /// </summary>
    /// <param name="latitude">Signed decimal degrees, positive north.</param>
    /// <param name="longitude">Signed decimal degrees, positive east.</param>
    /// <returns>The resolved place, or <see langword="null"/> if nothing is in range.</returns>
    public ResolvedPlace? Resolve(double latitude, double longitude)
    {
        if (_names.Length == 0 ||
            double.IsNaN(latitude) || double.IsNaN(longitude) ||
            latitude is < -90 or > 90 || longitude is < -180 or > 180)
        {
            return null;
        }

        int latCell = CellOf(latitude);
        int lonCell = CellOf(longitude);

        int best = -1;
        double bestKm = double.MaxValue;

        // Expand outward a ring at a time. Stop as soon as the closest point the next ring
        // could possibly contain is further away than what has already been found.
        for (int ring = 0; ring <= 180; ring++)
        {
            double ringFloorKm = (ring - 1) * 111.0;   // one degree of latitude is ~111 km
            if (best >= 0 && ringFloorKm > bestKm)
            {
                break;
            }

            foreach (int candidate in Ring(latCell, lonCell, ring))
            {
                double km = HaversineKm(latitude, longitude, _latitudes[candidate], _longitudes[candidate]);
                if (km < bestKm)
                {
                    bestKm = km;
                    best = candidate;
                }
            }
        }

        if (best < 0)
        {
            return null;
        }

        ushort country = _countryIndex[best];
        return new ResolvedPlace(_names[best], _countryNames[country], _countryCodes[country], bestKm);
    }

    /// <summary>Enumerates city indices in the square ring at a given radius of grid cells.</summary>
    private IEnumerable<int> Ring(int latCell, int lonCell, int ring)
    {
        for (int dLat = -ring; dLat <= ring; dLat++)
        {
            for (int dLon = -ring; dLon <= ring; dLon++)
            {
                // Only the perimeter: inner cells were covered by a previous, closer ring.
                if (ring > 0 && Math.Abs(dLat) != ring && Math.Abs(dLon) != ring)
                {
                    continue;
                }

                int lat = latCell + dLat;
                if (lat is < -90 or > 90)
                {
                    continue;
                }

                // Longitude wraps. Without this, a fix just east of the antimeridian would
                // never find the cities just west of it.
                int lon = lonCell + dLon;
                while (lon < -180) { lon += 360; }
                while (lon > 180) { lon -= 360; }

                if (_grid.TryGetValue(CellKey(lat, lon), out int[]? bucket))
                {
                    foreach (int index in bucket)
                    {
                        yield return index;
                    }
                }
            }
        }
    }

    private static int CellOf(double degrees) => (int)Math.Floor(degrees / GridDegrees);

    // Pack two cell coordinates into one int: latitude fits in 9 bits of range, longitude in 10.
    private static int CellKey(int latCell, int lonCell) => ((latCell + 90) << 10) | (lonCell + 180);

    /// <summary>Great-circle distance in kilometres.</summary>
    /// <remarks>
    /// Haversine on a spherical earth. Good to about 0.5% against the true ellipsoid, which is
    /// far finer than this is used for — the answer feeds a three-way choice between "in", "near",
    /// and "country only".
    /// </remarks>
    private static double HaversineKm(double lat1, double lon1, double lat2, double lon2)
    {
        const double earthRadiusKm = 6371.0;
        const double toRadians = Math.PI / 180.0;

        double dLat = (lat2 - lat1) * toRadians;
        double dLon = (lon2 - lon1) * toRadians;

        double a = (Math.Sin(dLat / 2) * Math.Sin(dLat / 2))
                 + (Math.Cos(lat1 * toRadians) * Math.Cos(lat2 * toRadians)
                    * Math.Sin(dLon / 2) * Math.Sin(dLon / 2));

        return earthRadiusKm * 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));
    }

    /// <summary>Formats a coordinate pair as degrees, for diagnostics and export only.</summary>
    /// <param name="latitude">Signed decimal degrees.</param>
    /// <param name="longitude">Signed decimal degrees.</param>
    /// <returns>A string such as <c>52.398°N 0.262°E</c>.</returns>
    /// <remarks>
    /// Deliberately not used anywhere in the interface. Raw coordinates tell a person nothing
    /// about where a photograph was taken; this exists for the export path and for debugging a
    /// resolution that looks wrong.
    /// </remarks>
    public static string FormatCoordinates(double latitude, double longitude)
        => string.Create(CultureInfo.InvariantCulture,
            $"{Math.Abs(latitude):0.000}°{(latitude >= 0 ? 'N' : 'S')} "
            + $"{Math.Abs(longitude):0.000}°{(longitude >= 0 ? 'E' : 'W')}");
}
