namespace PixelFlux.Storage;

/// <summary>
/// Metadata about a single object in an <see cref="IObjectStore"/>.
/// </summary>
/// <param name="Key">
/// The full object key, forward-slash separated and relative to the store root
/// (for example <c>jobs/pending/3f9c1a.json</c>).
/// </param>
/// <param name="Size">Size of the object body in bytes.</param>
/// <param name="LastModified">
/// When the object was last written, in UTC.
/// <para>
/// Treated as advisory rather than authoritative. Clock skew between devices is real, and a
/// sync client rewriting a file can move this timestamp without the content changing. It is
/// used for one purpose only — deciding that a job claim is old enough to be considered
/// abandoned — and that decision is deliberately given a generous margin.
/// </para>
/// </param>
/// <param name="ETag">
/// An opaque content version supplied by the store, or <see langword="null"/> if it does not
/// provide one. S3 sets this; the filesystem store leaves it null. Never parse it — the only
/// valid operation is equality against a previously observed value.
/// </param>
public readonly record struct ObjectEntry(
    string Key,
    long Size,
    DateTimeOffset LastModified,
    string? ETag = null)
{
    /// <summary>
    /// The final segment of <see cref="Key"/> — everything after the last <c>/</c>, or the whole
    /// key if it contains none. Convenient when a key encodes an identifier in its filename,
    /// which is how job and revision records are named.
    /// </summary>
    public string Name
    {
        get
        {
            int slash = Key.LastIndexOf('/');
            return slash < 0 ? Key : Key[(slash + 1)..];
        }
    }
}
