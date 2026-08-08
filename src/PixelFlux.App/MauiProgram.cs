using Microsoft.Extensions.Logging;
using PixelFlux.App.Services;
using PixelFlux.Core.Imaging;
using PixelFlux.Core.Index;
using PixelFlux.Core.Search;
using PixelFlux.Core.Localisation;
using PixelFlux.Core.Pipeline;
using PixelFlux.Ai.Compute;

namespace PixelFlux.App;

/// <summary>Application entry point and composition root.</summary>
public static class MauiProgram
{
    /// <summary>Builds the MAUI application.</summary>
    /// <returns>The configured app.</returns>
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder.UseMauiApp<App>();

        builder.Services.AddMauiBlazorWebView();

        // Paths are resolved eagerly rather than through DI because the WebView's virtual host
        // mapping is configured in MainPage, before a scope exists to resolve from.
        LibraryPaths paths = LibraryPaths.Default();
        builder.Services.AddSingleton(paths);

        // Migration runs here, at startup, on the composition root's thread. Doing it lazily on
        // first query would put a schema upgrade inside a component's lifecycle method — and in
        // MAUI Blazor Hybrid an exception there reaches no crash channel at all, so a failed
        // migration would present as a blank window with empty logs.
        var database = new PhotoDatabase(paths.DatabasePath);
        database.Migrate();

        builder.Services.AddSingleton(database);
        builder.Services.AddSingleton(sp => new PhotoStore(sp.GetRequiredService<PhotoDatabase>()));
        builder.Services.AddSingleton(sp => new CollectionStore(sp.GetRequiredService<PhotoDatabase>()));
        builder.Services.AddSingleton(sp => new SegmentStore(sp.GetRequiredService<PhotoDatabase>()));
        builder.Services.AddSingleton(sp => new FaceStore(sp.GetRequiredService<PhotoDatabase>()));
        builder.Services.AddSingleton(sp => new VectorIndex(sp.GetRequiredService<PhotoDatabase>()));
        builder.Services.AddSingleton(_ => new DerivativeGenerator(paths.CacheRoot));

        // What hardware models run on. A singleton because registering an execution provider is a
        // process-wide, one-way act — the runtime has no way to load one twice harmlessly — so
        // there has to be exactly one thing that does it, and it has to do it before any model
        // opens a session.
        //
        // The stored preference is read synchronously, here, rather than awaited later. It is one
        // row, once, on the composition root's thread, and the alternative is worse: a model that
        // opens before the preference arrives would run the whole session on the wrong hardware
        // and nothing would say so.
        AcceleratorPreference accelerator = ComputeBackend.ParsePreference(
            new SettingsStore(database).GetAsync(ComputeBackend.SettingKey)
                .GetAwaiter().GetResult());

        builder.Services.AddSingleton(sp => new ComputeBackend(
            paths.ProviderDirectory,
            accelerator,
            sp.GetRequiredService<ILoggerFactory>().CreateLogger<ComputeBackend>()));
        // Model installation. A singleton because it owns one HttpClient for the life of the
        // application and allows exactly one download at a time.
        builder.Services.AddSingleton(sp => new SettingsStore(sp.GetRequiredService<PhotoDatabase>()));
        builder.Services.AddSingleton<ModelService>();
        builder.Services.AddSingleton<IFolderChooser, FolderChooser>();
        builder.Services.AddSingleton<SourceService>();
        builder.Services.AddSingleton<LibraryService>();

        // The analysis queue. A singleton because it owns a loop and a claim on the queue table,
        // and two of those would compete for the same photographs. Constructing it opens no model
        // and reads no photograph — it is started explicitly, once the window is up, so a slow
        // first tick cannot delay the library appearing.
        builder.Services.AddSingleton<PipelineService>();

        // A singleton, not scoped: the chosen language is application state, and a Blazor
        // Hybrid app has exactly one user. Scoping it per circuit would silently reset the
        // language on every navigation.
        builder.Services.AddSingleton<Strings>();

#if DEBUG
        builder.Services.AddBlazorWebViewDeveloperTools();
        builder.Logging.AddDebug();
#endif

        return builder.Build();
    }
}
