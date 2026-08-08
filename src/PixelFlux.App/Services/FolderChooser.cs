namespace PixelFlux.App.Services;

/// <summary>Asks the operating system for a folder.</summary>
/// <remarks>
/// An interface with one method because the implementation is unavoidably platform-specific and
/// everything above it should not be. It also means the folder list can be exercised without a
/// dialog appearing.
/// </remarks>
public interface IFolderChooser
{
    /// <summary>Shows a folder picker.</summary>
    /// <param name="cancellationToken">Cancels the wait, not the dialog.</param>
    /// <returns>The chosen folder, or null if the person dismissed it.</returns>
    Task<string?> ChooseAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// The Windows folder picker.
/// </summary>
/// <remarks>
/// <para>
/// MAUI has <c>FilePicker</c> but no folder picker, so this goes to the WinRT one underneath.
/// That has a catch worth knowing: a WinRT picker created from a desktop process has no window to
/// attach to and throws unless it is given one explicitly, which is what the interop call below
/// does. Without it the dialog does not appear and the exception surfaces somewhere unrelated.
/// </para>
/// <para>
/// A file type filter is required even for folders — the picker refuses to open with an empty one,
/// which is a documented quirk rather than anything meaningful about the selection.
/// </para>
/// </remarks>
public sealed class FolderChooser : IFolderChooser
{
    /// <inheritdoc/>
    public async Task<string?> ChooseAsync(CancellationToken cancellationToken = default)
    {
#if WINDOWS
        var picker = new Windows.Storage.Pickers.FolderPicker
        {
            SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary,
        };

        // Required, and ignored: the picker will not open without at least one entry.
        picker.FileTypeFilter.Add("*");

        nint handle = WindowHandle();

        if (handle == 0)
        {
            return null;
        }

        WinRT.Interop.InitializeWithWindow.Initialize(picker, handle);

        Windows.Storage.StorageFolder? folder = await picker.PickSingleFolderAsync()
            .AsTask(cancellationToken).ConfigureAwait(false);

        return folder?.Path;
#else
        await Task.CompletedTask.ConfigureAwait(false);
        return null;
#endif
    }

#if WINDOWS
    private static nint WindowHandle()
    {
        // The first window is the only window; PixelFlux is a single-window application. Reached
        // through the handler rather than held onto, because the window is created after the
        // service container and a captured reference would be null on the one occasion it matters.
        object? platform = Microsoft.Maui.Controls.Application.Current?
            .Windows.FirstOrDefault()?.Handler?.PlatformView;

        return platform is Microsoft.UI.Xaml.Window window
            ? WinRT.Interop.WindowNative.GetWindowHandle(window)
            : 0;
    }
#endif
}
