using PixelFlux.Storage;

namespace PixelFlux.Tests;

/// <summary>
/// Tests for the filesystem object store.
///
/// The focus is deliberately narrow: <see cref="IObjectStore.TryCreateAsync"/> and the
/// atomicity of writes. Everything in PixelFlux's distributed design — job claiming, stale
/// reclaim, revision publishing — assumes exactly one caller can win a create and that a
/// reader never sees a half-written file. If those two properties hold, the layers above are
/// debuggable; if they quietly do not, every failure above looks like a logic bug somewhere
/// else.
/// </summary>
public sealed class FileSystemObjectStoreTests : IDisposable
{
    private readonly string _root;
    private readonly FileSystemObjectStore _store;

    public FileSystemObjectStoreTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "pixelflux-tests", Guid.NewGuid().ToString("n"));
        _store = new FileSystemObjectStore(_root);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_root))
            {
                Directory.Delete(_root, recursive: true);
            }
        }
        catch (IOException)
        {
            // A leaked temp directory is not worth failing a test run over.
        }
    }

    private static byte[] Bytes(string s) => System.Text.Encoding.UTF8.GetBytes(s);

    [Fact]
    public async Task WriteThenRead_RoundTrips()
    {
        await _store.WriteAllBytesAsync("a/b/c.txt", Bytes("hello"));

        byte[]? read = await _store.ReadAllBytesAsync("a/b/c.txt");

        Assert.Equal("hello", System.Text.Encoding.UTF8.GetString(read!));
    }

    [Fact]
    public async Task OpenRead_ReturnsNull_ForMissingKey()
    {
        // Absence is an ordinary outcome, not an exception — another device may have completed
        // and cleaned up a job between our listing it and our reading it.
        Assert.Null(await _store.OpenReadAsync("nope/missing.json"));
        Assert.Null(await _store.StatAsync("nope/missing.json"));
        Assert.False(await _store.ExistsAsync("nope/missing.json"));
    }

    [Fact]
    public async Task Delete_IsIdempotent()
    {
        await _store.WriteAllBytesAsync("gone.txt", Bytes("x"));
        await _store.DeleteAsync("gone.txt");
        await _store.DeleteAsync("gone.txt");   // must not throw
        await _store.DeleteAsync("never-existed.txt");

        Assert.False(await _store.ExistsAsync("gone.txt"));
    }

    [Fact]
    public async Task TryCreate_SucceedsOnce_ThenFails()
    {
        Assert.True(await _store.TryCreateAsync("jobs/claimed/x.json", Bytes("first")));
        Assert.False(await _store.TryCreateAsync("jobs/claimed/x.json", Bytes("second")));

        // The loser must not have overwritten the winner's content.
        byte[]? content = await _store.ReadAllBytesAsync("jobs/claimed/x.json");
        Assert.Equal("first", System.Text.Encoding.UTF8.GetString(content!));
    }

    [Fact]
    public async Task TryCreate_UnderContention_ElectsExactlyOneWinner()
    {
        // The property the job queue actually depends on. Thirty-two threads race for one key;
        // if more than one is told it won, two devices will process the same image and — worse
        // — both will believe they hold the claim.
        const int racers = 32;
        const string key = "jobs/claimed/contended.json";

        using var gate = new Barrier(racers);
        var wins = 0;

        Task<bool>[] attempts = Enumerable.Range(0, racers).Select(i => Task.Run(async () =>
        {
            gate.SignalAndWait();                       // maximise real overlap
            bool won = await _store.TryCreateAsync(key, Bytes($"worker-{i}"));
            if (won)
            {
                Interlocked.Increment(ref wins);
            }

            return won;
        })).ToArray();

        bool[] results = await Task.WhenAll(attempts);

        Assert.Equal(1, wins);
        Assert.Single(results, r => r);
        Assert.True(await _store.ExistsAsync(key));
    }

    [Fact]
    public async Task Write_IsAtomic_ReaderNeverSeesPartialContent()
    {
        // A sync client watching this folder would happily upload a half-written file to every
        // other device. The store writes to a temp name and renames, so a concurrent reader
        // sees either the whole old object or the whole new one.
        const string key = "big.bin";
        byte[] first = Enumerable.Repeat((byte)'A', 512 * 1024).ToArray();
        byte[] second = Enumerable.Repeat((byte)'B', 512 * 1024).ToArray();

        await _store.WriteAllBytesAsync(key, first);

        using var stop = new CancellationTokenSource(TimeSpan.FromSeconds(3));
        var reader = Task.Run(async () =>
        {
            while (!stop.IsCancellationRequested)
            {
                byte[]? seen = await _store.ReadAllBytesAsync(key, stop.Token);
                if (seen is null)
                {
                    continue;
                }

                // Every read must be a homogeneous buffer of one letter at full length.
                // A mixed or short buffer means a partial write was observable.
                Assert.Equal(first.Length, seen.Length);
                Assert.True(
                    seen.All(b => b == (byte)'A') || seen.All(b => b == (byte)'B'),
                    "reader observed a partially written object");
            }
        }, stop.Token);

        for (int i = 0; i < 20; i++)
        {
            await _store.WriteAllBytesAsync(key, i % 2 == 0 ? second : first);
        }

        await stop.CancelAsync();
        try
        {
            await reader;
        }
        catch (OperationCanceledException)
        {
            // Expected: the reader loop is stopped by cancellation.
        }
    }

    [Fact]
    public async Task List_MatchesStringPrefix_NotDirectoryBoundary()
    {
        await _store.WriteAllBytesAsync("jobs/pending/a.json", Bytes("1"));
        await _store.WriteAllBytesAsync("jobs/pending/b.json", Bytes("2"));
        await _store.WriteAllBytesAsync("jobs/claimed/c.json", Bytes("3"));
        await _store.WriteAllBytesAsync("metadata/d.json", Bytes("4"));

        List<ObjectEntry> pending = await _store.ListAllAsync("jobs/pending/");
        List<ObjectEntry> allJobs = await _store.ListAllAsync("jobs/");
        List<ObjectEntry> everything = await _store.ListAllAsync("");

        Assert.Equal(2, pending.Count);
        Assert.Equal(3, allJobs.Count);
        Assert.Equal(4, everything.Count);
        Assert.Contains(pending, e => e.Name == "a.json");
        Assert.All(pending, e => Assert.StartsWith("jobs/pending/", e.Key, StringComparison.Ordinal));
    }

    [Fact]
    public async Task List_ExcludesInFlightTempFiles()
    {
        // A crash mid-write, or another process writing right now, leaves a .pfxtmp- file.
        // It must never be handed to a caller as though it were a real object.
        await _store.WriteAllBytesAsync("real.json", Bytes("{}"));
        Directory.CreateDirectory(_root);
        await File.WriteAllTextAsync(Path.Combine(_root, ".pfxtmp-abc123"), "half-written");

        List<ObjectEntry> listed = await _store.ListAllAsync("");

        Assert.Single(listed);
        Assert.Equal("real.json", listed[0].Key);
    }

    [Theory]
    [InlineData("../escape.txt")]
    [InlineData("a/../../escape.txt")]
    [InlineData("/absolute.txt")]
    public async Task Keys_ThatEscapeTheRoot_AreRejected(string key)
    {
        // Keys arrive from shared storage written by other machines, so they are untrusted.
        await Assert.ThrowsAsync<ArgumentException>(() => _store.WriteAllBytesAsync(key, Bytes("x")));
    }

    [Fact]
    public async Task JsonRoundTrip_AndCorruptJsonReadsAsNull()
    {
        var record = new SampleRecord("device-a", 42, "edit");
        await _store.WriteJsonAsync("metadata/revisions/r1.json", record);

        SampleRecord? back = await _store.ReadJsonAsync<SampleRecord>("metadata/revisions/r1.json");
        Assert.Equal(record, back);

        // A conflict-renamed or partially-synced file must not stop a device syncing the rest.
        await _store.WriteAllBytesAsync("metadata/revisions/r2.json", Bytes("{ not json"));
        Assert.Null(await _store.ReadJsonAsync<SampleRecord>("metadata/revisions/r2.json"));
    }

    private sealed record SampleRecord(string DeviceId, long Revision, string Kind);
}
