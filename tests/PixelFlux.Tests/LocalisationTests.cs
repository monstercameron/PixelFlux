using System.Globalization;
using System.Text.RegularExpressions;
using PixelFlux.Core.Localisation;

namespace PixelFlux.Tests;

/// <summary>
/// Tests for the string catalogue and for the property that matters more than any single
/// translation: that no English has been left welded into the components.
///
/// Localisation rots silently. A developer adds a button, types the label inline because it is
/// faster, and nothing fails — the app looks fine in English and quietly ships a hardcoded word
/// to every other language. The last test in this file is the one that catches that, and it is
/// the reason the catalogue is a greppable C# dictionary rather than a set of .resx files.
/// </summary>
public sealed class LocalisationTests
{
    private static string AppRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null &&
                   !Directory.Exists(Path.Combine(dir.FullName, "src", "PixelFlux.App")))
            {
                dir = dir.Parent;
            }

            return dir is null
                ? throw new DirectoryNotFoundException("Could not locate src/PixelFlux.App.")
                : Path.Combine(dir.FullName, "src", "PixelFlux.App");
        }
    }

    [Fact]
    public void EveryLanguageTranslatesEveryKey()
    {
        IReadOnlyCollection<string> english = Strings.AllKeys;
        Assert.NotEmpty(english);

        var gaps = new List<string>();

        foreach (Strings.Language language in Strings.Available.Where(l => l.Code != "en"))
        {
            IReadOnlyCollection<string> translated = Strings.KeysFor(language.Code);
            string[] missing = english.Except(translated, StringComparer.Ordinal).ToArray();

            if (missing.Length > 0)
            {
                gaps.Add($"{language.Code} is missing {missing.Length}: {string.Join(", ", missing.Take(8))}");
            }
        }

        Assert.True(gaps.Count == 0, string.Join("\n", gaps));
    }

    [Fact]
    public void NoLanguageDefinesAKeyEnglishDoesNot()
    {
        // A key present only in a translation is dead weight at best, and at worst a rename that
        // was applied to one language and not the source.
        IReadOnlyCollection<string> english = Strings.AllKeys;

        foreach (Strings.Language language in Strings.Available.Where(l => l.Code != "en"))
        {
            string[] orphans = Strings.KeysFor(language.Code)
                .Except(english, StringComparer.Ordinal)
                .ToArray();

            Assert.True(orphans.Length == 0,
                $"{language.Code} defines keys English does not: {string.Join(", ", orphans)}");
        }
    }

    [Fact]
    public void PlaceholdersMatchAcrossLanguages()
    {
        // The failure this prevents is nasty and only shows up in the translated build: if
        // English says "{0} queued" and a translation says "{0} of {1} queued", Format throws
        // FormatException at runtime — in that language only, and usually in front of the user.
        var strings = new Strings();
        var placeholder = new Regex(@"\{(\d+)\}", RegexOptions.Compiled);

        foreach (string key in Strings.AllKeys)
        {
            strings.Use("en");
            var expected = placeholder.Matches(strings[key]).Select(m => m.Value).ToHashSet(StringComparer.Ordinal);

            foreach (Strings.Language language in Strings.Available.Where(l => l.Code != "en"))
            {
                strings.Use(language.Code);
                var actual = placeholder.Matches(strings[key]).Select(m => m.Value).ToHashSet(StringComparer.Ordinal);

                Assert.True(expected.SetEquals(actual),
                    $"'{key}' in {language.Code} has placeholders [{string.Join(",", actual.Order())}] "
                    + $"but English has [{string.Join(",", expected.Order())}]");
            }
        }
    }

    [Fact]
    public void MissingKeyFallsBackRatherThanReturningBlank()
    {
        var strings = new Strings();
        strings.Use("ar");

        // A key nothing defines returns the key itself — loud and greppable, never an empty
        // label that is invisible in testing and ships.
        Assert.Equal("no.such.key", strings["no.such.key"]);

        // And a real key always yields something non-empty in every language.
        foreach (Strings.Language language in Strings.Available)
        {
            strings.Use(language.Code);
            foreach (string key in Strings.AllKeys)
            {
                Assert.False(string.IsNullOrWhiteSpace(strings[key]), $"{language.Code}/{key} is blank");
            }
        }
    }

    [Fact]
    public void SelectingALanguageAlsoSwitchesTheCulture()
    {
        // Strings alone are not localisation. Japanese labels above US-formatted dates is the
        // uncanny halfway state this guards against.
        var strings = new Strings();

        strings.Use("ja");
        Assert.Equal("ja", CultureInfo.CurrentCulture.TwoLetterISOLanguageName);

        strings.Use("ar");
        Assert.Equal("ar", CultureInfo.CurrentCulture.TwoLetterISOLanguageName);
        Assert.True(strings.IsRightToLeft);

        strings.Use("en");
        Assert.False(strings.IsRightToLeft);
    }

    [Fact]
    public void RegionalVariantsResolveToTheirLanguage()
    {
        var strings = new Strings();

        // es-MX and es-ES must both land on Spanish rather than falling back to English.
        // Shipping every regional variant is how a localisation effort becomes unmaintainable.
        strings.Use("es-MX");
        Assert.Equal("es", strings.Current.Code);

        strings.Use("ja-JP");
        Assert.Equal("ja", strings.Current.Code);

        // An unknown language falls back to English rather than throwing.
        strings.Use("xx-YY");
        Assert.Equal("en", strings.Current.Code);
    }

    [Fact]
    public void NoHardcodedEnglishRemainsInTheComponents()
    {
        // The test that actually keeps the app translatable.
        //
        // Scans every .razor file for user-visible text that is not coming from the catalogue:
        // element text content, and the attributes a screen reader or tooltip reads. Anything it
        // finds is a string a translator can never reach.
        string components = Path.Combine(AppRoot, "Components");
        Assert.True(Directory.Exists(components));

        // Text between tags: >Some words<
        var textContent = new Regex(@">\s*([A-Z][A-Za-z][A-Za-z' ]{3,})\s*<", RegexOptions.Compiled);
        // Human-facing attributes with a literal value.
        var attributes = new Regex(
            @"(?:title|aria-label|placeholder|alt)\s*=\s*""([A-Z][A-Za-z][A-Za-z' ]{3,})""",
            RegexOptions.Compiled);

        var offences = new List<string>();

        foreach (string file in Directory.EnumerateFiles(components, "*.razor", SearchOption.AllDirectories))
        {
            string source = File.ReadAllText(file);

            // Strip Razor comments and the @code block: neither reaches the screen, and both are
            // full of prose that would otherwise flood this test with false positives.
            source = Regex.Replace(source, @"@\*.*?\*@", string.Empty, RegexOptions.Singleline);

            // Strip <kbd> contents. Those name physical keys, and the legend printed on the
            // hardware in front of the user does not change when the app language does — a
            // Spanish speaker on a US keyboard still presses the key marked "Enter". Localising
            // them would describe a keyboard the user does not have.
            source = Regex.Replace(source, @"<kbd>.*?</kbd>", "<kbd></kbd>", RegexOptions.Singleline);
            int code = source.IndexOf("@code {", StringComparison.Ordinal);
            if (code >= 0)
            {
                source = source[..code];
            }

            string name = Path.GetFileName(file);

            foreach (Match match in textContent.Matches(source).Concat(attributes.Matches(source)))
            {
                string text = match.Groups[1].Value.Trim();

                // The wordmark is a brand name and is deliberately not translated.
                if (text is "Pixel" or "Flux" or "PixelFlux")
                {
                    continue;
                }

                offences.Add($"{name}: \"{text}\"");
            }
        }

        Assert.True(offences.Count == 0,
            "hardcoded user-visible English found — move these into Strings:\n  "
            + string.Join("\n  ", offences));
    }
}
