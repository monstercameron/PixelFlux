namespace PixelFlux.App;

/// <summary>The MAUI application object.</summary>
public partial class App : Application
{
    private readonly IServiceProvider _services;

    /// <summary>Creates the application.</summary>
    /// <param name="services">The container, used to construct the main page with its dependencies.</param>
    public App(IServiceProvider services)
    {
        _services = services;
        InitializeComponent();
    }

    /// <inheritdoc />
    protected override Window CreateWindow(IActivationState? activationState)
    {
        // Resolved from the container rather than newed up, because MainPage needs the library
        // paths in order to point the WebView at the derivative cache.
        var page = ActivatorUtilities.CreateInstance<MainPage>(_services);

        return new Window(page)
        {
            Title = "PixelFlux",
            // A photo grid needs room. Opening at a laptop-sane size beats the MAUI default,
            // which is small enough that the contact sheet shows two columns.
            Width = 1440,
            Height = 900,
            MinimumWidth = 720,
            MinimumHeight = 520,
        };
    }
}
