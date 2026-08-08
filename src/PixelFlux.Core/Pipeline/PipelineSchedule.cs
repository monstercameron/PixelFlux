using System.Globalization;

namespace PixelFlux.Core.Pipeline;

/// <summary>When the analysis queue is allowed to run.</summary>
public enum ScheduleMode
{
    /// <summary>Never, unless somebody presses run.</summary>
    Off = 0,

    /// <summary>Whenever the application is open.</summary>
    Always = 1,

    /// <summary>Only between two times of day.</summary>
    Window = 2,
}

/// <summary>
/// When the queue runs and how gently.
/// </summary>
/// <param name="Mode">Whether the queue runs, and on what terms.</param>
/// <param name="Start">Start of the daily window. Ignored unless <paramref name="Mode"/> is
/// <see cref="ScheduleMode.Window"/>.</param>
/// <param name="End">End of the daily window, exclusive. May be earlier than
/// <paramref name="Start"/>, which means the window crosses midnight.</param>
/// <param name="Gap">
/// How long to wait after finishing one photograph before starting the next.
/// </param>
/// <remarks>
/// <para>
/// Two knobs, and they answer different questions. The window answers "may this run now" — the
/// point of it being that overnight is when a laptop is plugged in, idle, and nobody minds the
/// fans. The gap answers "how much of the machine may it take while it does" — because even
/// during the window somebody might be at the keyboard, and a stage that saturates the processor
/// for sixteen seconds at a time is the difference between a background task and an obstruction.
/// </para>
/// <para>
/// A window that crosses midnight is the normal case, not the edge case: the useful window is
/// something like ten at night until six in the morning. Any comparison of the current time
/// against the window has to handle that, which is why it happens in one place here rather than at
/// every call site.
/// </para>
/// </remarks>
public sealed record PipelineSchedule(
    ScheduleMode Mode,
    TimeOnly Start,
    TimeOnly End,
    TimeSpan Gap)
{
    /// <summary>The setting this is stored under.</summary>
    public const string SettingKey = "pipeline.schedule";

    /// <summary>
    /// What a library uses before anybody chooses: run whenever the application is open, with a
    /// long pause between photographs.
    /// </summary>
    /// <remarks>
    /// Not off, because a library nobody has configured should still slowly become searchable, and
    /// a queue that does nothing until it is found in a settings page will never be found. Not
    /// flat out either — twenty seconds between photographs means roughly a hundred an hour, which
    /// finishes a normal library overnight while leaving the machine plainly usable.
    /// </remarks>
    public static PipelineSchedule Default { get; } = new(
        ScheduleMode.Always,
        new TimeOnly(22, 0),
        new TimeOnly(6, 0),
        TimeSpan.FromSeconds(20));

    /// <summary>Whether the queue may run at a given local time.</summary>
    /// <param name="localNow">The current local time.</param>
    /// <returns>True when work is allowed to start.</returns>
    public bool IsOpenAt(DateTime localNow)
    {
        if (Mode == ScheduleMode.Off)
        {
            return false;
        }

        if (Mode == ScheduleMode.Always)
        {
            return true;
        }

        TimeOnly now = TimeOnly.FromDateTime(localNow);

        // A window whose start and end are the same is the whole day, not an instant. The other
        // reading — a window nothing can ever fall inside — is never what somebody meant.
        if (Start == End)
        {
            return true;
        }

        return Start < End
            ? now >= Start && now < End
            : now >= Start || now < End;   // crosses midnight
    }

    /// <summary>When the window next opens.</summary>
    /// <param name="localNow">The current local time.</param>
    /// <returns>
    /// The next moment work would be allowed, or null when it is allowed already or never will be.
    /// </returns>
    public DateTime? NextOpening(DateTime localNow)
    {
        if (Mode != ScheduleMode.Window || IsOpenAt(localNow))
        {
            return null;
        }

        DateTime today = localNow.Date + Start.ToTimeSpan();
        return today > localNow ? today : today.AddDays(1);
    }

    /// <summary>Renders the schedule for storage.</summary>
    /// <returns>A single line that <see cref="Parse"/> reads back.</returns>
    public string Serialise() => string.Create(
        CultureInfo.InvariantCulture,
        $"{Mode.ToString().ToLowerInvariant()};{Start:HH\\:mm};{End:HH\\:mm};{(int)Gap.TotalSeconds}");

    /// <summary>Reads a stored schedule.</summary>
    /// <param name="stored">A value from <see cref="Serialise"/>, or null.</param>
    /// <returns>The schedule, or <see cref="Default"/> when the value is missing or unreadable.</returns>
    /// <remarks>
    /// Falls back rather than throwing. A setting written by a newer build, or corrupted, should
    /// cost the user their schedule — not the ability to open their photographs.
    /// </remarks>
    public static PipelineSchedule Parse(string? stored)
    {
        if (string.IsNullOrWhiteSpace(stored))
        {
            return Default;
        }

        string[] parts = stored.Split(';');
        if (parts.Length != 4)
        {
            return Default;
        }

        ScheduleMode mode = parts[0] switch
        {
            "off" => ScheduleMode.Off,
            "window" => ScheduleMode.Window,
            "always" => ScheduleMode.Always,
            _ => Default.Mode,
        };

        return new PipelineSchedule(
            mode,
            ReadTime(parts[1], Default.Start),
            ReadTime(parts[2], Default.End),
            int.TryParse(parts[3], CultureInfo.InvariantCulture, out int seconds) && seconds >= 0
                ? TimeSpan.FromSeconds(seconds)
                : Default.Gap);
    }

    private static TimeOnly ReadTime(string text, TimeOnly fallback) =>
        TimeOnly.TryParseExact(text, "HH\\:mm", CultureInfo.InvariantCulture,
            DateTimeStyles.None, out TimeOnly parsed)
            ? parsed
            : fallback;
}
