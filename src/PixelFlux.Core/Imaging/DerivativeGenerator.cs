using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;

namespace PixelFlux.Core.Imaging;

/// <summary>Sizes and quality for the cached derivatives PixelFlux generates per photo.</summary>
/// <remarks>
/// Two derivatives, each with one job. The <b>thumbnail</b> is what the gallery paints — there
/// may be four hundred on screen, so it is small enough that the grid scrolls at full rate. The
/// <b>proxy</b> is what the viewer paints — one at a time, big enough to look right on a 4K
/// panel, but still a fraction of a 45-megapixel original. The original is opened only when the
/// user zooms in or exports, which on a library of raw files is the difference between an app
/// that feels instant and one that stutters.
/// </remarks>
public sealed record DerivativeOptions
{
    /// <summary>Longest edge of the gallery thumbnail, in pixels.</summary>
    /// <remarks>
    /// 480 rather than the more obvious 256: on a 200% display a 256px thumbnail in a 240px
    /// cell is visibly soft, and users notice blurry thumbnails long before they notice disk use.
    /// </remarks>
    public int ThumbnailEdge { get; init; } = 480;

    /// <summary>JPEG quality for thumbnails, 1-100.</summary>
    public int ThumbnailQuality { get; init; } = 78;

    /// <summary>Longest edge of the display proxy, in pixels.</summary>
    public int ProxyEdge { get; init; } = 2560;

    /// <summary>JPEG quality for proxies, 1-100.</summary>
    public int ProxyQuality { get; init; } = 88;
}

/// <summary>The derivatives written for one photo, plus the facts learned while decoding it.</summary>
/// <param name="ThumbnailPath">Absolute path to the written thumbnail.</param>
/// <param name="ProxyPath">Absolute path to the written proxy.</param>
/// <param name="Width">Width of the original in pixels, after orientation was applied.</param>
/// <param name="Height">Height of the original in pixels, after orientation was applied.</param>
/// <param name="PerceptualHash">Difference hash computed from the decoded pixels.</param>
public readonly record struct DerivativeResult(
    string ThumbnailPath,
    string ProxyPath,
    int Width,
    int Height,
    string PerceptualHash);

/// <summary>
/// Decodes an original once and writes the cached thumbnail and proxy from it.
/// </summary>
/// <remarks>
/// The single-decode rule is the whole point of this class. Decoding a large JPEG is the most
/// expensive step in ingestion by a wide margin, so the thumbnail, the proxy, the dimensions,
/// and the perceptual hash are all produced from one decode. Splitting them across separate
/// helpers would be tidier to read and roughly three times slower to run.
/// </remarks>
public sealed class DerivativeGenerator
{
    private readonly string _cacheRoot;
    private readonly DerivativeOptions _options;

    /// <summary>Creates a generator writing into a cache directory.</summary>
    /// <param name="cacheRoot">Directory for generated derivatives. Created if missing.</param>
    /// <param name="options">Sizes and quality, or null for the defaults.</param>
    public DerivativeGenerator(string cacheRoot, DerivativeOptions? options = null)
    {
        _cacheRoot = Path.GetFullPath(cacheRoot);
        _options = options ?? new DerivativeOptions();
        System.IO.Directory.CreateDirectory(_cacheRoot);
    }

    /// <summary>Relative key of a photo's thumbnail, given its content hash.</summary>
    /// <param name="contentHash">The photo's SHA-256 content hash.</param>
    /// <returns>A store-relative key such as <c>thumb/3f/3f9c....jpg</c>.</returns>
    public static string ThumbnailKey(string contentHash) => Shard("thumb", contentHash);

    /// <summary>Relative key of a photo's proxy, given its content hash.</summary>
    /// <param name="contentHash">The photo's SHA-256 content hash.</param>
    /// <returns>A store-relative key such as <c>proxy/3f/3f9c....jpg</c>.</returns>
    public static string ProxyKey(string contentHash) => Shard("proxy", contentHash);

    /// <summary>Resolves a derivative key to an absolute path in this generator's cache.</summary>
    /// <param name="key">A key from <see cref="ThumbnailKey"/> or <see cref="ProxyKey"/>.</param>
    /// <returns>The absolute path.</returns>
    public string ResolvePath(string key) => Path.Combine(_cacheRoot, key.Replace('/', Path.DirectorySeparatorChar));

    // Two hex characters of the hash become a subdirectory: 256 buckets. Without this, a
    // 50,000-photo library puts 50,000 files in one folder, which Explorer, backup tools, and
    // some filesystems all handle badly.
    private static string Shard(string kind, string contentHash)
        => $"{kind}/{contentHash[..2]}/{contentHash}.jpg";

    /// <summary>
    /// Decodes an original and writes both derivatives.
    /// </summary>
    /// <param name="sourcePath">Absolute path to the original image.</param>
    /// <param name="contentHash">The photo's content hash, used to name the outputs.</param>
    /// <param name="cancellationToken">Cancels decoding and writing.</param>
    /// <returns>Paths to the derivatives plus the dimensions and perceptual hash.</returns>
    /// <exception cref="UnknownImageFormatException">The file is not a decodable image.</exception>
    /// <exception cref="InvalidImageContentException">The file is a known format but corrupt.</exception>
    public async Task<DerivativeResult> GenerateAsync(
        string sourcePath,
        string contentHash,
        CancellationToken cancellationToken = default)
    {
        using Image image = await Image.LoadAsync(sourcePath, cancellationToken).ConfigureAwait(false);

        // ImageSharp applies the EXIF orientation on load, so Width/Height here are already the
        // upright dimensions. Recording those rather than the stored ones means the gallery
        // never lays out a portrait photo in a landscape cell.
        int width = image.Width;
        int height = image.Height;

        string perceptualHash = ImageHashing.ComputePerceptualHash(image);

        string thumbKey = ThumbnailKey(contentHash);
        string proxyKey = ProxyKey(contentHash);
        string thumbPath = ResolvePath(thumbKey);
        string proxyPath = ResolvePath(proxyKey);

        // Proxy first, at the larger size, then downscale that same instance to thumbnail size.
        // Going large-to-small reuses the work; doing it the other way needs a second decode.
        await WriteResizedAsync(image, proxyPath, _options.ProxyEdge, _options.ProxyQuality, cancellationToken)
            .ConfigureAwait(false);
        await WriteResizedAsync(image, thumbPath, _options.ThumbnailEdge, _options.ThumbnailQuality, cancellationToken)
            .ConfigureAwait(false);

        return new DerivativeResult(thumbPath, proxyPath, width, height, perceptualHash);
    }

    private static async Task WriteResizedAsync(
        Image source,
        string destination,
        int longestEdge,
        int quality,
        CancellationToken cancellationToken)
    {
        System.IO.Directory.CreateDirectory(Path.GetDirectoryName(destination)!);

        using Image copy = source.Clone(ctx => ctx.Resize(new ResizeOptions
        {
            // Max, not Crop: a derivative must never change the framing of the photograph.
            // The gallery does its own cropping at paint time, where it is reversible.
            Mode = ResizeMode.Max,
            Size = new Size(longestEdge, longestEdge),
            Sampler = KnownResamplers.Lanczos3,
        }));

        // Metadata is stripped from derivatives. They are a display cache, the original keeps
        // the authoritative EXIF, and a cache that carries GPS coordinates is a quiet privacy
        // leak the moment someone shares a thumbnail.
        copy.Metadata.ExifProfile = null;
        copy.Metadata.XmpProfile = null;
        copy.Metadata.IptcProfile = null;

        var encoder = new JpegEncoder { Quality = quality, ColorType = JpegEncodingColor.YCbCrRatio420 };

        // Written to a temp name and moved into place: an interrupted ingest must not leave a
        // half-written thumbnail that later looks like a valid cache entry.
        //
        // The temp name carries a GUID rather than being `destination + ".tmp"`. Derivative
        // paths are keyed by content hash, so two byte-identical files — which every real
        // library has — produce the *same* destination, and with ingestion running six files in
        // parallel both workers raced for one temp path. That surfaced as
        // "the process cannot access the file ... .tmp", an error that looks like a permissions
        // problem and is actually a collision with yourself.
        string temp = $"{destination}.{Guid.NewGuid():n}.tmp";

        try
        {
            await copy.SaveAsync(temp, encoder, cancellationToken).ConfigureAwait(false);

            // Both racers write identical bytes, so whoever lands second simply overwrites an
            // equivalent file. No locking is needed — only distinct temp names.
            File.Move(temp, destination, overwrite: true);
        }
        catch
        {
            try
            {
                File.Delete(temp);
            }
            catch (IOException)
            {
                // Leaving a stray .tmp behind is harmless; it is never read.
            }

            throw;
        }
    }
}
