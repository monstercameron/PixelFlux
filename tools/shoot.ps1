<#
.SYNOPSIS
    Screenshot the PixelFlux window without stealing focus.

.DESCRIPTION
    Visual inspection of a MAUI Blazor Hybrid app cannot be done from logs: the renderer
    swallows component lifecycle exceptions, so "the process is alive" and "the window has a
    title" stay true while the page is blank. The only way to know the UI rendered is to look
    at the pixels.

    HOW THIS CAPTURES, AND WHY IT CHANGED
    -------------------------------------
    The first version called SetForegroundWindow and then CopyFromScreen. That is wrong twice
    over on a machine someone else is using:

      * It yanks focus away from whatever they are doing.
      * Windows frequently *refuses* the foreground change (a background process may not steal
        focus), and CopyFromScreen then happily photographs whatever is actually on top at those
        coordinates. Twice this returned a video the user was watching instead of the app —
        so the capture was not merely rude, it was silently returning the wrong window.

    PrintWindow asks the window to render itself into a bitmap. It needs no focus, works when
    the window is behind others, and cannot return someone else's pixels. PW_RENDERFULLCONTENT
    (0x2) is required for DWM-composited content such as a WebView2 surface; without it a
    Blazor Hybrid window captures as an empty frame.

.PARAMETER Out
    Path to write the PNG to.

.PARAMETER ProcessName
    Process to find. Defaults to PixelFlux.App.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ProcessName = 'PixelFlux.App'
)

Add-Type -AssemblyName System.Drawing

$sig = @'
using System;
using System.Runtime.InteropServices;
public static class Shot {
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr hWnd, int attr, out RECT val, int size);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    public const uint PW_RENDERFULLCONTENT = 0x2;
}
'@
if (-not ('Shot' -as [type])) { Add-Type -TypeDefinition $sig }

$proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } |
        Select-Object -First 1

if (-not $proc) { Write-Error "No $ProcessName window found."; exit 1 }
$h = $proc.MainWindowHandle

# A minimised window has nothing to render. Restoring is the one unavoidable intrusion, and it
# is skipped whenever the window is already showing.
if ([Shot]::IsIconic($h)) {
    [Shot]::ShowWindow($h, 9) | Out-Null   # SW_RESTORE
    Start-Sleep -Milliseconds 600
}

# Extended frame bounds, not GetWindowRect: the latter includes the invisible resize border and
# drop shadow, which would add ~8px of neighbouring desktop on three sides.
$r = New-Object Shot+RECT
if ([Shot]::DwmGetWindowAttribute($h, 9, [ref]$r, 16) -ne 0) { Write-Error 'DwmGetWindowAttribute failed.'; exit 1 }

$w = $r.Right - $r.Left
$ht = $r.Bottom - $r.Top
if ($w -le 0 -or $ht -le 0) { Write-Error "Window has no area ($w x $ht)."; exit 1 }

$bmp = New-Object System.Drawing.Bitmap $w, $ht
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
$ok = [Shot]::PrintWindow($h, $hdc, [Shot]::PW_RENDERFULLCONTENT)
$g.ReleaseHdc($hdc)
$g.Dispose()

if (-not $ok) { $bmp.Dispose(); Write-Error 'PrintWindow failed.'; exit 1 }

# A fully blank capture means PrintWindow returned an empty frame — report it rather than
# writing a black PNG that looks like a rendering bug in the app.
$probe = $bmp.LockBits(
    (New-Object System.Drawing.Rectangle 0, 0, $bmp.Width, $bmp.Height),
    [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
    [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$bytes = New-Object byte[] ($probe.Stride * 4)
[System.Runtime.InteropServices.Marshal]::Copy($probe.Scan0, $bytes, 0, $bytes.Length)
$bmp.UnlockBits($probe)
$distinct = ($bytes | Select-Object -Unique).Count

$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

if ($distinct -le 2) { Write-Warning 'capture looks blank - PrintWindow may not support this surface' }
"captured $w x $ht -> $Out"
