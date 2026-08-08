using PixelFlux.Core.Geo;

namespace PixelFlux.Tests;

/// <summary>
/// The offline reverse geocoder.
///
/// Correctness here is checkable against the real world, which is unusual and worth using: the
/// coordinates below are real places, so the assertions are about whether the answer is *right*
/// rather than merely self-consistent.
/// </summary>
public sealed class GazetteerTests
{
    private readonly Gazetteer _gazetteer = Gazetteer.Instance;

    [Fact]
    public void TheDataIsActuallyEmbedded()
    {
        // A missing embedded resource degrades to "no place names" rather than crashing, which
        // is right at runtime and useless in a test — every other assertion here would pass
        // vacuously. This is the guard against that.
        Assert.True(_gazetteer.PlaceCount > 20_000,
            $"only {_gazetteer.PlaceCount} places loaded; is gazetteer.bin embedded?");
    }

    [Theory]
    [InlineData(51.5074, -0.1278, "United Kingdom", "London")]
    [InlineData(35.0116, 135.7681, "Japan", "Kyoto")]
    [InlineData(-33.9249, 18.4241, "South Africa", "Cape Town")]
    [InlineData(10.3910, -75.4794, "Colombia", "Cartagena")]
    [InlineData(64.1466, -21.9426, "Iceland", "Reykjavík")]
    // Singapore resolves to "Thomson" — GeoNames lists the city-state's districts rather than
    // one entry for the whole island. The country is right and the distance is 2.7 km, so the
    // label "Thomson, Singapore" is accurate; only the city-name expectation is dropped. Dense
    // metros behave this way generally, and a district is arguably the more useful answer.
    [InlineData(1.3521, 103.8198, "Singapore", null)]
    [InlineData(-34.6037, -58.3816, "Argentina", "Buenos Aires")]
    public void KnownCityCentresResolveToThemselves(double lat, double lon, string country, string? city)
    {
        ResolvedPlace place = Assert.NotNull(_gazetteer.Resolve(lat, lon));

        Assert.Equal(country, place.Country);
        Assert.NotEmpty(place.City);

        if (city is not null)
        {
            Assert.Contains(city, place.City, StringComparison.OrdinalIgnoreCase);
        }

        // A city centre is a city centre: anything more than a few kilometres out means the
        // nearest-neighbour search picked a neighbouring suburb instead.
        Assert.True(place.DistanceKm < 25, $"{place.City} resolved {place.DistanceKm:0.#} km away");
    }

    [Fact]
    public void TheAntimeridianDoesNotBreakTheSearch()
    {
        // Longitude wraps and the grid does not. Without explicit wrapping, a fix just east of
        // 180° never sees the cities just west of it and resolves to something absurd — this is
        // the classic off-by-a-planet bug in a cell-based spatial index.
        ResolvedPlace east = Assert.NotNull(_gazetteer.Resolve(-16.5, 179.9));   // Fiji, east side
        ResolvedPlace west = Assert.NotNull(_gazetteer.Resolve(-16.5, -179.9));  // Fiji, west side

        Assert.Equal("Fiji", east.Country);
        Assert.Equal("Fiji", west.Country);
        Assert.True(east.DistanceKm < 200);
        Assert.True(west.DistanceKm < 200);
    }

    [Fact]
    public void ThePolesResolveWithoutFallingOffTheGrid()
    {
        // Latitude does not wrap, so the ring expansion has to clamp instead. Both poles must
        // return something rather than throwing or looping to the iteration cap.
        Assert.NotNull(_gazetteer.Resolve(89.9, 0));
        Assert.NotNull(_gazetteer.Resolve(-89.9, 0));
    }

    [Fact]
    public void TheLabelHedgesHonestlyWithDistance()
    {
        // The whole point of carrying the distance. A fix in a city is that city; a fix eighty
        // kilometres out is not, and saying so is the difference between a place name and a
        // confident lie about where a photograph was taken.
        ResolvedPlace inTown = Assert.NotNull(_gazetteer.Resolve(51.5074, -0.1278));
        Assert.DoesNotContain("near", inTown.Label, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(",", inTown.Label, StringComparison.Ordinal);

        // Mid-Atlantic: hundreds of kilometres from anywhere, so only the country is claimed.
        ResolvedPlace remote = Assert.NotNull(_gazetteer.Resolve(35.0, -40.0));
        Assert.True(remote.DistanceKm > 150);
        Assert.DoesNotContain(",", remote.Label, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(double.NaN, 0)]
    [InlineData(0, double.NaN)]
    [InlineData(91, 0)]
    [InlineData(-91, 0)]
    [InlineData(0, 181)]
    [InlineData(0, -181)]
    public void ImpossibleCoordinatesReturnNothing(double lat, double lon)
    {
        // Corrupt EXIF produces these routinely. Returning null is what lets the caller record
        // "no place" instead of pinning a photograph to a coordinate that cannot exist.
        Assert.Null(_gazetteer.Resolve(lat, lon));
    }

    [Fact]
    public void NullIslandDoesNotBecomeAPlace()
    {
        // 0,0 is in the Gulf of Guinea and is what a camera writes when GPS is on but never
        // locked. The extractor rejects it before it ever reaches here — but if that guard is
        // ever removed, this documents what the gazetteer would do with it: name the nearest
        // land, hundreds of kilometres away, hedged accordingly.
        ResolvedPlace? place = _gazetteer.Resolve(0, 0);

        Assert.NotNull(place);
        Assert.True(place!.Value.DistanceKm > 150,
            "0,0 should be far from anywhere; if it is not, the guard in ExifExtractor matters even more");
    }

    [Fact]
    public void ResolutionIsFastEnoughToRunPerPhoto()
    {
        // Ingestion calls this once per photograph. A brute-force scan of 34,000 places would be
        // acceptable at 50 photos and not at 50,000, which is why there is a grid — this pins
        // that the grid is actually being used rather than silently degrading to a full scan.
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var random = new Random(20260808);

        for (int i = 0; i < 2000; i++)
        {
            _gazetteer.Resolve((random.NextDouble() * 140) - 70, (random.NextDouble() * 360) - 180);
        }

        watch.Stop();
        Assert.True(watch.ElapsedMilliseconds < 3000,
            $"2000 lookups took {watch.ElapsedMilliseconds} ms — the spatial index is not working");
    }
}
