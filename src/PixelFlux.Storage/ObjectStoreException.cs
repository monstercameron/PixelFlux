namespace PixelFlux.Storage;

/// <summary>
/// Thrown when an object store operation fails for a reason the caller cannot resolve —
/// the bucket is unreachable, credentials were rejected, the sync folder is gone, the disk
/// is full.
/// </summary>
/// <remarks>
/// Deliberately <em>not</em> thrown for a missing object: absence is a normal state in this
/// system, so <see cref="IObjectStore.OpenReadAsync"/> and
/// <see cref="IObjectStore.StatAsync"/> return <see langword="null"/> and
/// <see cref="IObjectStore.DeleteAsync"/> succeeds. That split lets callers use ordinary
/// control flow for the common case and reserve <c>try</c>/<c>catch</c> for genuine faults,
/// which in practice means "the shared storage is down, stop the worker and surface a banner".
/// </remarks>
public sealed class ObjectStoreException : IOException
{
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">Human-readable description of the failure.</param>
    public ObjectStoreException(string message) : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying fault.</summary>
    /// <param name="message">Human-readable description of the failure.</param>
    /// <param name="innerException">The store-specific exception that caused this failure.</param>
    public ObjectStoreException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
