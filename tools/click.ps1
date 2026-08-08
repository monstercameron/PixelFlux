<#
.SYNOPSIS
    Click inside the PixelFlux window at coordinates relative to its visible bounds.

.DESCRIPTION
    Companion to shoot.ps1. Coordinates are given in the same space that shoot.ps1 captures,
    so you can read a position straight off a screenshot and click it without converting to
    screen space by hand.

.PARAMETER X
    Horizontal position within the window's visible bounds.

.PARAMETER Y
    Vertical position within the window's visible bounds.

.PARAMETER ProcessName
    Process to target. Defaults to PixelFlux.App.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$X,
    [Parameter(Mandatory = $true)][int]$Y,
    [string]$ProcessName = 'PixelFlux.App'
)

$sig = @'
using System;
using System.Runtime.InteropServices;
public static class Clicker {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, IntPtr e);
    [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr hWnd, int attr, out RECT val, int size);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    public const uint DOWN = 0x0002, UP = 0x0004;
}
'@
if (-not ('Clicker' -as [type])) { Add-Type -TypeDefinition $sig }

$proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $proc) { Write-Error "No $ProcessName window found."; exit 1 }

$h = $proc.MainWindowHandle
[Clicker]::SetForegroundWindow($h) | Out-Null
Start-Sleep -Milliseconds 400

$r = New-Object Clicker+RECT
[Clicker]::DwmGetWindowAttribute($h, 9, [ref]$r, 16) | Out-Null

$sx = $r.Left + $X
$sy = $r.Top + $Y

[Clicker]::SetCursorPos($sx, $sy) | Out-Null
Start-Sleep -Milliseconds 120
[Clicker]::mouse_event([Clicker]::DOWN, 0, 0, 0, [IntPtr]::Zero)
Start-Sleep -Milliseconds 60
[Clicker]::mouse_event([Clicker]::UP, 0, 0, 0, [IntPtr]::Zero)

"clicked window($X,$Y) -> screen($sx,$sy)"
