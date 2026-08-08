using PixelFlux.App.Services;

namespace PixelFlux.App;

/// <summary>The single native page; everything above it is Blazor.</summary>
public partial class MainPage : ContentPage
{
    private readonly LibraryPaths _paths;

    /// <summary>Creates the page and wires the WebView to the derivative cache.</summary>
    /// <param name="paths">Local storage locations.</param>
    public MainPage(LibraryPaths paths)
    {
        _paths = paths;
        InitializeComponent();

        // Which page the window opens on.
        //
        // This exists because the application cannot be driven from the outside. Every surface
        // beyond the front page is reached by clicking, and clicking means moving the pointer on
        // a machine somebody is using — so a page that is only reachable by hand is a page that
        // ships without anyone having looked at it, which has already cost this project several
        // visual bugs. Setting PIXELFLUX_START=/faces opens straight onto that page, and a
        // screenshot can then be taken with no focus stolen and no pointer moved.
        //
        // Unset, which it is for every real user, this changes nothing.
        if (Environment.GetEnvironmentVariable("PIXELFLUX_START") is { Length: > 1 } start &&
            start.StartsWith('/'))
        {
            blazorWebView.StartPath = start;
        }

        blazorWebView.BlazorWebViewInitialized += OnWebViewInitialized;
    }

    /// <summary>
    /// Maps a virtual hostname onto the derivative cache directory so the gallery can load
    /// thumbnails as ordinary image URLs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the piece that makes the grid fast. A Blazor Hybrid page cannot reference
    /// <c>file://</c> paths, so the alternative would be sending every thumbnail through the
    /// JS bridge as a base64 data URI — for a screen of 40 thumbnails that is several megabytes
    /// of string marshalling per scroll, and the browser cannot cache any of it.
    /// </para>
    /// <para>
    /// With the mapping in place, <c>https://pixelflux.cache/thumb/3f/....jpg</c> is served by
    /// WebView2 directly off disk, with its normal HTTP caching, lazy loading, and decode
    /// scheduling all working as they would on a real web page.
    /// </para>
    /// <para>
    /// Access is <c>Allow</c>, not <c>DenyCors</c>. DenyCors was the original choice and it is
    /// the tighter one, but it also blocks any subresource the engine treats as cross-origin —
    /// which quietly broke the segmentation overlay while ordinary thumbnails kept working. The
    /// mapped host is a private name that only this WebView resolves, reachable from no other
    /// page and from nothing outside the process, so the practical exposure is nil.
    /// </para>
    /// </remarks>
    private void OnWebViewInitialized(object? sender, Microsoft.AspNetCore.Components.WebView.BlazorWebViewInitializedEventArgs e)
    {
#if WINDOWS
        e.WebView.CoreWebView2.SetVirtualHostNameToFolderMapping(
            LibraryService.CacheHost,
            _paths.CacheRoot,
            Microsoft.Web.WebView2.Core.CoreWebView2HostResourceAccessKind.Allow);

        Microsoft.Web.WebView2.Core.CoreWebView2Settings settings = e.WebView.CoreWebView2.Settings;

        // No page zoom. The photo viewer has its own zoom, and a pinch that drives both at once
        // leaves the whole interface at 140% with no obvious way back — there is no visible
        // browser chrome here to reset it from. The JavaScript side blocks Ctrl+wheel and the
        // Ctrl+plus/minus/zero shortcuts; these two close the routes it cannot reach.
        settings.IsPinchZoomEnabled = false;
        settings.IsZoomControlEnabled = false;

        // This is an application, not a document: no right-click menu offering "Save image as",
        // no swipe-to-navigate history gesture, no browser-supplied status popups.
        settings.AreDefaultContextMenusEnabled = false;
        settings.IsSwipeNavigationEnabled = false;
        settings.IsStatusBarEnabled = false;
#endif
    }
}
