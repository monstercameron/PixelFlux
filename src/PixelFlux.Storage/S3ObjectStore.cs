using System.Net;
using System.Runtime.CompilerServices;
using Amazon.Runtime;
using Amazon.S3;
using Amazon.S3.Model;

namespace PixelFlux.Storage;

/// <summary>
/// Configuration for an <see cref="S3ObjectStore"/>.
/// </summary>
/// <remarks>
/// Written to be provider-neutral. PixelFlux is not an AWS application that happens to allow
/// alternatives — it is an S3-protocol application, and AWS is one of several backends people
/// will point it at (MinIO on a NAS, Cloudflare R2, Backblaze B2, Wasabi). The two settings
/// that make that work are <see cref="ServiceUrl"/> and <see cref="ForcePathStyle"/>.
/// </remarks>
public sealed class S3StoreOptions
{
    /// <summary>The bucket name. Required.</summary>
    public required string Bucket { get; init; }

    /// <summary>
    /// Optional key prefix, so a bucket can hold a PixelFlux library alongside other data
    /// (for example <c>pixelflux/</c>). A trailing slash is added if missing.
    /// </summary>
    public string Prefix { get; init; } = string.Empty;

    /// <summary>
    /// Endpoint of the S3-compatible service, for example <c>https://s3.us-west-002.backblazeb2.com</c>
    /// or <c>http://nas.local:9000</c>. Leave <see langword="null"/> to use real AWS S3, in
    /// which case <see cref="Region"/> selects the endpoint.
    /// </summary>
    public string? ServiceUrl { get; init; }

    /// <summary>AWS region name, used only when <see cref="ServiceUrl"/> is not set.</summary>
    public string Region { get; init; } = "us-east-1";

    /// <summary>
    /// Whether to address the bucket as a path segment (<c>host/bucket/key</c>) rather than as
    /// a subdomain (<c>bucket.host/key</c>).
    /// </summary>
    /// <remarks>
    /// Defaults to <see langword="true"/> because that is what self-hosted services need, and a
    /// self-hosted MinIO on a bare IP address simply cannot do virtual-host addressing. Real
    /// AWS accepts both.
    /// </remarks>
    public bool ForcePathStyle { get; init; } = true;

    /// <summary>Access key id. Leave null to use the ambient AWS credential chain.</summary>
    public string? AccessKey { get; init; }

    /// <summary>Secret access key. Leave null to use the ambient AWS credential chain.</summary>
    public string? SecretKey { get; init; }

    /// <summary>
    /// Whether the service honours <c>If-None-Match: *</c> on PUT, which is what makes
    /// <see cref="IObjectStore.TryCreateAsync"/> genuinely atomic.
    /// </summary>
    /// <remarks>
    /// AWS S3 has supported this since August 2024, and R2 and recent MinIO do too. When it is
    /// unavailable the store falls back to a check-then-write, which is <em>not</em> atomic —
    /// see the remarks on <see cref="S3ObjectStore.TryCreateAsync"/> for what that costs.
    /// </remarks>
    public bool SupportsConditionalPut { get; init; } = true;
}

/// <summary>
/// An <see cref="IObjectStore"/> backed by an S3-compatible bucket.
/// </summary>
/// <remarks>
/// The alternative to a synchronised folder. Useful when devices are not on the same cloud
/// drive, when the library is large enough that a consumer sync client struggles, or when a
/// machine that is not a desktop (a NAS, a always-on mini PC) should participate as a
/// processing worker.
/// </remarks>
public sealed class S3ObjectStore : IObjectStore, IDisposable
{
    private readonly IAmazonS3 _client;
    private readonly bool _ownsClient;
    private readonly S3StoreOptions _options;
    private readonly string _prefix;

    /// <summary>Creates a store from configuration, building the underlying S3 client.</summary>
    /// <param name="options">Bucket, endpoint, and credential configuration.</param>
    public S3ObjectStore(S3StoreOptions options)
        : this(options, BuildClient(options), ownsClient: true)
    {
    }

    /// <summary>Creates a store over a caller-supplied S3 client.</summary>
    /// <param name="options">Bucket and prefix configuration. Endpoint/credential fields are ignored.</param>
    /// <param name="client">The client to use.</param>
    /// <param name="ownsClient">
    /// Whether disposing this store should dispose <paramref name="client"/>. Pass
    /// <see langword="false"/> when the client is shared or container-managed.
    /// </param>
    public S3ObjectStore(S3StoreOptions options, IAmazonS3 client, bool ownsClient = false)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(client);

        if (string.IsNullOrWhiteSpace(options.Bucket))
        {
            throw new ArgumentException("A bucket name is required.", nameof(options));
        }

        _options = options;
        _client = client;
        _ownsClient = ownsClient;
        _prefix = NormalisePrefix(options.Prefix);
    }

    /// <inheritdoc />
    public string DisplayName => $"s3:{_options.Bucket}/{_prefix}".TrimEnd('/');

    private static IAmazonS3 BuildClient(S3StoreOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        var config = new AmazonS3Config
        {
            ForcePathStyle = options.ForcePathStyle,
        };

        if (!string.IsNullOrWhiteSpace(options.ServiceUrl))
        {
            // A custom endpoint still needs an authority region for SigV4 signing even though
            // no AWS region is involved; non-AWS services accept whatever is presented.
            config.ServiceURL = options.ServiceUrl;
            config.AuthenticationRegion = options.Region;
        }
        else
        {
            config.RegionEndpoint = Amazon.RegionEndpoint.GetBySystemName(options.Region);
        }

        return options.AccessKey is { Length: > 0 } && options.SecretKey is { Length: > 0 }
            ? new AmazonS3Client(new BasicAWSCredentials(options.AccessKey, options.SecretKey), config)
            : new AmazonS3Client(config);
    }

    private static string NormalisePrefix(string prefix)
    {
        if (string.IsNullOrWhiteSpace(prefix))
        {
            return string.Empty;
        }

        string trimmed = prefix.Trim('/');
        return trimmed.Length == 0 ? string.Empty : trimmed + "/";
    }

    /// <summary>Maps a store-relative key to a full bucket key.</summary>
    private string ToS3Key(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            throw new ArgumentException("An object key is required.", nameof(key));
        }

        return _prefix + key;
    }

    /// <summary>Maps a full bucket key back to a store-relative key.</summary>
    private string FromS3Key(string s3Key)
        => _prefix.Length > 0 && s3Key.StartsWith(_prefix, StringComparison.Ordinal)
            ? s3Key[_prefix.Length..]
            : s3Key;

    /// <inheritdoc />
    public async IAsyncEnumerable<ObjectEntry> ListAsync(
        string prefix,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prefix);

        var request = new ListObjectsV2Request
        {
            BucketName = _options.Bucket,
            Prefix = _prefix + prefix,
            MaxKeys = 1000,
        };

        do
        {
            ListObjectsV2Response response;
            try
            {
                response = await _client.ListObjectsV2Async(request, cancellationToken).ConfigureAwait(false);
            }
            catch (AmazonS3Exception ex)
            {
                throw new ObjectStoreException($"Could not list '{prefix}' in bucket '{_options.Bucket}'.", ex);
            }

            foreach (S3Object item in response.S3Objects ?? [])
            {
                yield return new ObjectEntry(
                    FromS3Key(item.Key),
                    item.Size ?? 0,
                    item.LastModified is { } modified ? new DateTimeOffset(modified, TimeSpan.Zero) : default,
                    item.ETag?.Trim('"'));
            }

            // Paginate. The caller may stop enumerating at any point, in which case the next
            // page is simply never requested — which is why this is a streaming iterator.
            request.ContinuationToken = response.NextContinuationToken;
        }
        while (!string.IsNullOrEmpty(request.ContinuationToken));
    }

    /// <inheritdoc />
    public async Task<Stream?> OpenReadAsync(string key, CancellationToken cancellationToken = default)
    {
        try
        {
            GetObjectResponse response = await _client
                .GetObjectAsync(_options.Bucket, ToS3Key(key), cancellationToken)
                .ConfigureAwait(false);

            // The response stream owns the HTTP connection; handing it to the caller transfers
            // that ownership, which the interface documents as "the caller must dispose".
            return response.ResponseStream;
        }
        catch (AmazonS3Exception ex) when (ex.StatusCode == HttpStatusCode.NotFound)
        {
            return null;
        }
        catch (AmazonS3Exception ex)
        {
            throw new ObjectStoreException($"Could not read '{key}' from bucket '{_options.Bucket}'.", ex);
        }
    }

    /// <inheritdoc />
    public async Task WriteAsync(string key, Stream content, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(content);

        try
        {
            // S3 PUT is atomic by protocol: readers see the old object until the new one is
            // fully committed. No temp-and-rename dance is needed, unlike the filesystem store.
            await _client.PutObjectAsync(
                new PutObjectRequest
                {
                    BucketName = _options.Bucket,
                    Key = ToS3Key(key),
                    InputStream = content,
                    AutoCloseStream = false,   // the caller owns the stream
                    DisablePayloadSigning = true,
                },
                cancellationToken).ConfigureAwait(false);
        }
        catch (AmazonS3Exception ex)
        {
            throw new ObjectStoreException($"Could not write '{key}' to bucket '{_options.Bucket}'.", ex);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Uses <c>If-None-Match: *</c>, which instructs the service to reject the PUT with
    /// <c>412 Precondition Failed</c> if any object already exists at the key. That gives a
    /// true compare-and-swap, and it is the strongest claim primitive available over the S3
    /// protocol.
    /// </para>
    /// <para>
    /// When <see cref="S3StoreOptions.SupportsConditionalPut"/> is <see langword="false"/> the
    /// implementation degrades to HEAD-then-PUT. That window is small but real: two workers can
    /// both observe the key as free and both write, and the second silently wins. The system
    /// tolerates this because jobs are idempotent — the cost is duplicated work, not a corrupt
    /// index — but a deployment that cares about wasted cycles should be pointed at a service
    /// that supports the conditional form.
    /// </para>
    /// </remarks>
    public async Task<bool> TryCreateAsync(
        string key,
        ReadOnlyMemory<byte> content,
        CancellationToken cancellationToken = default)
    {
        string s3Key = ToS3Key(key);

        if (!_options.SupportsConditionalPut &&
            await ExistsRawAsync(s3Key, cancellationToken).ConfigureAwait(false))
        {
            return false;
        }

        try
        {
            using var body = new MemoryStream(content.ToArray(), writable: false);
            var request = new PutObjectRequest
            {
                BucketName = _options.Bucket,
                Key = s3Key,
                InputStream = body,
                AutoCloseStream = false,
                DisablePayloadSigning = true,
            };

            if (_options.SupportsConditionalPut)
            {
                request.IfNoneMatch = "*";
            }

            await _client.PutObjectAsync(request, cancellationToken).ConfigureAwait(false);
            return true;
        }
        catch (AmazonS3Exception ex) when (
            ex.StatusCode == HttpStatusCode.PreconditionFailed ||
            ex.StatusCode == HttpStatusCode.Conflict)
        {
            // Someone else got there first. This is the expected losing path, not a fault.
            return false;
        }
        catch (AmazonS3Exception ex)
        {
            throw new ObjectStoreException($"Could not create '{key}' in bucket '{_options.Bucket}'.", ex);
        }
    }

    /// <inheritdoc />
    public async Task DeleteAsync(string key, CancellationToken cancellationToken = default)
    {
        try
        {
            // S3 DELETE on a missing key succeeds, which matches the interface contract exactly.
            await _client.DeleteObjectAsync(_options.Bucket, ToS3Key(key), cancellationToken)
                .ConfigureAwait(false);
        }
        catch (AmazonS3Exception ex) when (ex.StatusCode == HttpStatusCode.NotFound)
        {
            // Already gone.
        }
        catch (AmazonS3Exception ex)
        {
            throw new ObjectStoreException($"Could not delete '{key}' from bucket '{_options.Bucket}'.", ex);
        }
    }

    /// <inheritdoc />
    public async Task<ObjectEntry?> StatAsync(string key, CancellationToken cancellationToken = default)
    {
        try
        {
            GetObjectMetadataResponse meta = await _client
                .GetObjectMetadataAsync(_options.Bucket, ToS3Key(key), cancellationToken)
                .ConfigureAwait(false);

            return new ObjectEntry(
                key,
                meta.ContentLength,
                new DateTimeOffset(meta.LastModified ?? default, TimeSpan.Zero),
                meta.ETag?.Trim('"'));
        }
        catch (AmazonS3Exception ex) when (ex.StatusCode == HttpStatusCode.NotFound)
        {
            return null;
        }
        catch (AmazonS3Exception ex)
        {
            throw new ObjectStoreException($"Could not stat '{key}' in bucket '{_options.Bucket}'.", ex);
        }
    }

    private async Task<bool> ExistsRawAsync(string s3Key, CancellationToken cancellationToken)
    {
        try
        {
            await _client.GetObjectMetadataAsync(_options.Bucket, s3Key, cancellationToken)
                .ConfigureAwait(false);
            return true;
        }
        catch (AmazonS3Exception ex) when (ex.StatusCode == HttpStatusCode.NotFound)
        {
            return false;
        }
    }

    /// <summary>Disposes the underlying S3 client if this store created it.</summary>
    public void Dispose()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }
    }
}
