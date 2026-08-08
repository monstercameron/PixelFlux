namespace PixelFlux.App.Services;

/// <summary>
/// Where PixelFlux keeps its local state on this machine.
/// </summary>
/// <remarks>
/// <para>
/// Resolved once at startup and injected, rather than each component calling
/// <see cref="Environment.GetFolderPath(Environment.SpecialFolder)"/> for itself. Two reasons:
/// the paths are needed before the DI container is built (the WebView's virtual host mapping is
/// configured against the cache directory), and tests need to point the whole application at a
/// temp directory by constructing one of these.
/// </para>
/// <para>
/// The split between local app data and roaming is deliberate. Everything here is either
/// rebuildable from the originals (the index, the derivatives) or specific to this machine, so
/// none of it belongs in a roaming profile that follows the user between computers. What
/// <em>does</em> need to travel — user edits, ratings, AI metadata — travels as revision records
/// through shared storage, which is a different mechanism entirely.
/// </para>
/// </remarks>
public sealed class LibraryPaths
{
    /// <summary>Creates paths rooted under a directory.</summary>
    /// <param name="root">The PixelFlux data directory. Created along with its children.</param>
    public LibraryPaths(string root)
    {
        Root = Path.GetFullPath(root);
        DatabasePath = Path.Combine(Root, "library.db");
        CacheRoot = Path.Combine(Root, "cache");
        SharedStorageRoot = Path.Combine(Root, "shared");
        ModelsRoot = Path.Combine(Root, "models");

        Directory.CreateDirectory(Root);
        Directory.CreateDirectory(CacheRoot);
        Directory.CreateDirectory(SharedStorageRoot);
        Directory.CreateDirectory(ModelsRoot);
    }

    /// <summary>Creates paths under the per-machine application data directory.</summary>
    /// <returns>Paths rooted at <c>%LOCALAPPDATA%\PixelFlux</c> or the platform equivalent.</returns>
    public static LibraryPaths Default() => new(Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "PixelFlux"));

    /// <summary>The PixelFlux data directory.</summary>
    public string Root { get; }

    /// <summary>Absolute path to the SQLite index.</summary>
    public string DatabasePath { get; }

    /// <summary>Directory holding generated thumbnails and proxies.</summary>
    public string CacheRoot { get; }

    /// <summary>
    /// Default location of the shared bus when the user has not chosen one.
    /// </summary>
    /// <remarks>
    /// A local folder by default, which makes a single-machine install work with no setup at
    /// all — the job queue and revision log are real, they simply have one participant. Pointing
    /// this at a synced folder in Settings is what turns on multi-device operation, and nothing
    /// above this line has to change for that.
    /// </remarks>
    public string SharedStorageRoot { get; }

    /// <summary>Where downloaded ONNX models are kept.</summary>
    public string ModelsRoot { get; }

    /// <summary>The face detector, wherever it was found. May not exist.</summary>
    /// <remarks>
    /// YuNet, Apache 2.0, 232 KB. Small enough that it can reasonably ship with the application
    /// rather than being downloaded — unlike the segmentation model, which is AGPL and 11 MB.
    /// </remarks>
    public string FaceModelPath => FindModel("face_yunet_2023mar.onnx");

    /// <summary>The face recognition model, wherever it was found. May not exist.</summary>
    /// <remarks>
    /// SFace, Apache 2.0, 38 MB. Too large to ship inside the application, so it is a file the
    /// user supplies — and everything above it degrades quietly without it: faces are still
    /// found, cropped, and listed; only "find this person" disappears.
    /// </remarks>
    public string RecognitionModelPath => FindModel("face_sface_2021dec.onnx");

    /// <summary>CLIP's vision encoder, wherever it was found. May not exist.</summary>
    /// <remarks>
    /// CLIP ViT-B/32, MIT, exported by transformers.js. About 580 MB across the two encoders,
    /// which is why they are files the user supplies rather than something shipped.
    /// </remarks>
    public string ClipVisionModelPath => FindModel("clip_vision_model.onnx");

    /// <summary>CLIP's text encoder, wherever it was found. May not exist.</summary>
    public string ClipTextModelPath => FindModel("clip_text_model.onnx");

    /// <summary>CLIP's token table.</summary>
    public string ClipVocabularyPath => FindModel("clip_vocab.json");

    /// <summary>CLIP's byte-pair merge table.</summary>
    public string ClipMergesPath => FindModel("clip_merges.txt");

    /// <summary>Where execution provider libraries are looked for.</summary>
    /// <remarks>
    /// Beside the models rather than beside the executable, because a provider is the same kind of
    /// thing a model is: a large vendor-supplied file the user drops in, not something shipped.
    /// Missing is the normal case and means everything runs on the processor.
    /// </remarks>
    public string ProviderDirectory => FindModelDirectory("providers");

    /// <summary>
    /// The vision-language model directory, wherever it was found. May not exist.
    /// </summary>
    /// <remarks>
    /// A directory rather than a file: Qwen3-VL is four graphs, a tokenizer and a configuration
    /// that the runtime loads together. About 1.4 GB, so it is something the user installs.
    /// </remarks>
    public string VisionModelDirectory => FindModelDirectory("qwen3vl");

    /// <summary>The segmentation model, wherever it was found. May not exist.</summary>
    public string SegmentationModelPath => FindModel("yolo11n-seg.onnx");

    /// <summary>Locates a model directory, preferring the user's own copy.</summary>
    /// <param name="name">Directory name beneath a models folder.</param>
    /// <returns>The first place it was found, or the per-user location if found nowhere.</returns>
    private string FindModelDirectory(string name)
    {
        string perUser = Path.Combine(ModelsRoot, name);
        if (Directory.Exists(perUser))
        {
            return perUser;
        }

        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "models", name);
            if (Directory.Exists(candidate))
            {
                return candidate;
            }

            dir = dir.Parent;
        }

        return perUser;
    }

    /// <summary>
    /// Locates a model file, preferring the user's own copy.
    /// </summary>
    /// <param name="fileName">Bare file name of the model.</param>
    /// <returns>
    /// The first place it was found, or the per-user location if it was found nowhere — so an
    /// error message names the directory the user should put it in, not the last place searched.
    /// </returns>
    /// <remarks>
    /// Three places, in order: the per-user models directory, which is where a downloaded model
    /// lands and therefore wins; alongside the executable, which is where a shipped one lives;
    /// and a <c>models</c> directory in an ancestor of the executable, which is the repository
    /// checkout during development. The last exists only so a developer build behaves the same
    /// as an installed one without anyone having to copy files around.
    /// </remarks>
    private string FindModel(string fileName)
    {
        string perUser = Path.Combine(ModelsRoot, fileName);
        if (File.Exists(perUser))
        {
            return perUser;
        }

        string beside = Path.Combine(AppContext.BaseDirectory, "models", fileName);
        if (File.Exists(beside))
        {
            return beside;
        }

        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "models", fileName);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            dir = dir.Parent;
        }

        return perUser;
    }

    /// <summary>Finds a model file wherever it already exists, or reports that it does not.</summary>
    /// <param name="relativePath">
    /// Path beneath a models directory, using forward slashes — a bare filename, or
    /// <c>qwen3vl/text.onnx</c> for a model that is a folder of parts.
    /// </param>
    /// <returns>The full path if the file exists somewhere, otherwise null.</returns>
    /// <remarks>
    /// Exists so the setup dialog asks the same question the loader does. It first checked only
    /// the per-user directory and therefore offered to download two gigabytes that were already
    /// on the machine, sitting in the repository's own <c>models</c> folder where a developer
    /// build reads them from. Anything that decides whether a model is present has to search the
    /// same three places as <see cref="FindModel"/>, or it is answering a different question.
    /// </remarks>
    public string? FindExistingModel(string relativePath)
    {
        string relative = relativePath.Replace('/', Path.DirectorySeparatorChar);

        string perUser = Path.Combine(ModelsRoot, relative);
        if (File.Exists(perUser))
        {
            return perUser;
        }

        string beside = Path.Combine(AppContext.BaseDirectory, "models", relative);
        if (File.Exists(beside))
        {
            return beside;
        }

        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "models", relative);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            dir = dir.Parent;
        }

        return null;
    }

}
