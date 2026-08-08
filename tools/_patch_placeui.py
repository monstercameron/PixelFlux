"""One-shot patch: switch the UI from coordinate cells to resolved city/country."""
import io

# ---------------------------------------------------------------- strings
p = 'src/PixelFlux.Core/Localisation/Strings.cs'
s = io.open(p, encoding='utf-8').read()
pairs = {
    'en': ('["filter.place"] = "Place",', '["filter.city"] = "City",\n            ["filter.country"] = "Country",'),
    'es': ('["filter.place"] = "Lugar",', '["filter.city"] = "Ciudad",\n            ["filter.country"] = "País",'),
    'ja': ('["filter.place"] = "場所",', '["filter.city"] = "都市",\n            ["filter.country"] = "国",'),
    'ar': ('["filter.place"] = "المكان",', '["filter.city"] = "المدينة",\n            ["filter.country"] = "البلد",'),
}
for code, (anchor, addition) in pairs.items():
    idx = s.index(f'["{code}"] = new(StringComparer.Ordinal)')
    at = s.index(anchor, idx)
    s = s[:at] + addition + '\n            ' + s[at:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)

# ---------------------------------------------------------------- filter panel
p = 'src/PixelFlux.App/Components/Layout/FilterPanel.razor'
s = io.open(p, encoding='utf-8').read()
s = s.replace(
    '        @RenderFacet(T["filter.place"], "place", Place, v => OnPlaceChanged.InvokeAsync(v), coordinates: true)',
    '        @RenderFacet(T["filter.country"], "country", Country, v => OnCountryChanged.InvokeAsync(v))\n'
    '        @RenderFacet(T["filter.city"], "city", City, v => OnCityChanged.InvokeAsync(v))')

s = s.replace('''    /// <summary>Currently selected place cell, if any.</summary>
    [Parameter] public string? Place { get; set; }''',
'''    /// <summary>Currently selected city, if any.</summary>
    [Parameter] public string? City { get; set; }

    /// <summary>Currently selected country, if any.</summary>
    [Parameter] public string? Country { get; set; }''')

s = s.replace('''    /// <summary>Raised when the place filter changes. Null clears it.</summary>
    [Parameter] public EventCallback<string?> OnPlaceChanged { get; set; }''',
'''    /// <summary>Raised when the city filter changes. Null clears it.</summary>
    [Parameter] public EventCallback<string?> OnCityChanged { get; set; }

    /// <summary>Raised when the country filter changes. Null clears it.</summary>
    [Parameter] public EventCallback<string?> OnCountryChanged { get; set; }''')

s = s.replace('''        CameraModel is not null || SourceFolder is not null || Tag is not null ||
        Place is not null || MinRating > 0 || FavouritesOnly;''',
'''        CameraModel is not null || SourceFolder is not null || Tag is not null ||
        City is not null || Country is not null || MinRating > 0 || FavouritesOnly;''')

# The coordinate formatter and its plumbing are gone: nothing renders raw degrees any more.
s = s.replace('''        bool shorten = false,
        bool coordinates = false) => builder =>''', '''        bool shorten = false) => builder =>''')
s = s.replace('''                string label = coordinates ? PrettyPlace(value)
                             : shorten ? ShortPath(value)
                             : value;''',
'''                string label = shorten ? ShortPath(value) : value;''')

start = s.index('    /// <summary>\n    /// Formats a one-degree place cell as coordinates with hemispheres.')
end = s.index('}', s.index('return string.Create(CultureInfo.InvariantCulture,')) + 1
end = s.index('\n', s.index('}', end)) + 1
s = s[:start] + s[end:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("filter panel switched to city/country")

# ---------------------------------------------------------------- gallery
p = 'src/PixelFlux.App/Components/Pages/Gallery.razor'
s = io.open(p, encoding='utf-8').read()
s = s.replace('''                         Place="_place"''', '''                         City="_city"
                         Country="_country"''')
s = s.replace('''                         OnPlaceChanged="v => Apply(() => _place = v)"''',
              '''                         OnCityChanged="v => Apply(() => _city = v)"
                         OnCountryChanged="v => Apply(() => _country = v)"''')
s = s.replace('    private string? _place;', '    private string? _city;\n    private string? _country;')
s = s.replace('''        (_tag is null ? 0 : 1) + (_place is null ? 0 : 1) + (_minRating > 0 ? 1 : 0) +''',
              '''        (_tag is null ? 0 : 1) + (_city is null ? 0 : 1) + (_country is null ? 0 : 1) +
        (_minRating > 0 ? 1 : 0) +''')
s = s.replace('''        if (_place is { } place)
        {
            yield return (place, () => _place = null);
        }''',
'''        if (_country is { } country)
        {
            yield return (country, () => _country = null);
        }

        if (_city is { } city)
        {
            yield return (city, () => _city = null);
        }''')
s = s.replace('''        Bounds = PlaceToBounds(_place),
        CollectionId = _albumId,''',
              '''        City = _city,
        Country = _country,
        CollectionId = _albumId,''')
s = s.replace('''        _tag = null;
        _place = null;''', '''        _tag = null;
        _city = null;
        _country = null;''')

start = s.index('    /// <summary>\n    /// Converts a one-degree place cell back into the bounding box it came from.')
end = s.index('        return (lat - 0.5, lon - 0.5, lat + 0.5, lon + 0.5);\n    }\n\n')
s = s[:start] + s[end + len('        return (lat - 0.5, lon - 0.5, lat + 0.5, lon + 0.5);\n    }\n\n'):]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("gallery switched to city/country")

# ---------------------------------------------------------------- viewer
p = 'src/PixelFlux.App/Components/Pages/Viewer.razor'
s = io.open(p, encoding='utf-8').read()
start = s.index('    private string Place\n    {')
end = s.index('\n    }\n', s.index('return string.Create(CultureInfo.InvariantCulture,', start)) + len('\n    }\n')
s = s[:start] + '''    /// <summary>
    /// Where the photograph was taken, as a place name.
    /// </summary>
    /// <remarks>
    /// Never coordinates. "52.398°N 0.262°E" is not an answer to "where was this taken" — it is
    /// the question restated in numbers. The name is resolved once at import against the
    /// gazetteer embedded in the application, so this costs nothing at render time and needs no
    /// network call.
    /// </remarks>
    private string Place => Photo.Place?.Label ?? "—";
''' + s[end:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("viewer switched to place names")
