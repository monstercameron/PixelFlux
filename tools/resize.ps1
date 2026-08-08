<#
.SYNOPSIS
    Resize the PixelFlux window without activating it.

.DESCRIPTION
    Responsive layout cannot be judged from the stylesheet. The breakpoints in app.css were
    written before the top bar grew a view switch, three selectors and two status-bar actions,
    and the only way to know what that does at 900 pixels is to make the window 900 pixels and
    look at it.

    MoveWindow is used rather than anything that raises or focuses the window, for the same
    reason shoot.ps1 uses PrintWindow: this runs on a machine somebody else is working on, and
    a tool that steals focus is a tool that gets switched off.

.PARAMETER Width
    Client width in pixels.

.PARAMETER Height
    Client height in pixels. Defaults to keeping the current height.

.PARAMETER ProcessName
    Process to resize.
#>
param(
    [Parameter(Mandatory = $true)][int]$Width,
    [int]$Height = 0,
    [string]$ProcessName = 'PixelFlux.App'
)

$ErrorActionPreference = 'Stop'

Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class Win {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool MoveWindow(IntPtr hWnd, int x, int y, int w, int h, bool repaint);
}
'@

$proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } |
        Select-Object -First 1

if (-not $proc) {
    Write-Error "$ProcessName is not running with a window"
    exit 1
}

$handle = $proc.MainWindowHandle
$rect = New-Object Win+RECT
[void][Win]::GetWindowRect($handle, [ref]$rect)

$currentHeight = $rect.Bottom - $rect.Top
if ($Height -le 0) { $Height = $currentHeight }

[void][Win]::MoveWindow($handle, $rect.Left, $rect.Top, $Width, $Height, $true)

# The window manager may refuse a size below the window's own minimum; report what actually
# happened rather than what was asked for.
Start-Sleep -Milliseconds 400
[void][Win]::GetWindowRect($handle, [ref]$rect)
Write-Output ("resized to {0} x {1}" -f ($rect.Right - $rect.Left), ($rect.Bottom - $rect.Top))
