namespace PixelFlux.Core.Setup;

/// <summary>One file a model is made of.</summary>
/// <param name="Url">Where to fetch it.</param>
/// <param name="RelativePath">
/// Where it lands beneath the models directory, using forward slashes. A model that is a folder of
/// parts — the vision model is thirteen files — puts them all under one subdirectory.
/// </param>
/// <param name="Bytes">
/// Exactly how large the file is, checked after download.
/// </param>
public sealed record ModelFile(string Url, string RelativePath, long Bytes);

/// <summary>How freely a model may be used.</summary>
public enum ModelTerms
{
    /// <summary>Permissive — Apache-2.0 or MIT. Use it in anything.</summary>
    Permissive = 0,

    /// <summary>
    /// Copyleft — AGPL-3.0. Fine for personal use; distributing or serving a modified build
    /// obliges you to publish source.
    /// </summary>
    Copyleft = 1,
}

/// <summary>A model PixelFlux can fetch, and what it is for.</summary>
/// <param name="Id">Stable identifier, used in settings and progress.</param>
/// <param name="Name">What it is called by its authors.</param>
/// <param name="Enables">The feature that appears once it is installed.</param>
/// <param name="Licence">Its licence, spelled as its authors spell it.</param>
/// <param name="Terms">Whether that licence carries obligations worth stopping for.</param>
/// <param name="Source">Human-readable home of the model, for a link.</param>
/// <param name="Files">Everything that must be present.</param>
public sealed record CatalogueModel(
    string Id,
    string Name,
    string Enables,
    string Licence,
    ModelTerms Terms,
    string Source,
    IReadOnlyList<ModelFile> Files)
{
    /// <summary>Total download in bytes.</summary>
    public long Bytes => Files.Sum(file => file.Bytes);

    /// <summary>Whether every file is present and the right size.</summary>
    /// <param name="modelsRoot">The models directory.</param>
    /// <returns>True when nothing needs downloading.</returns>
    /// <remarks>
    /// Size, not just presence. A download interrupted by a closed laptop leaves a file that
    /// exists and is useless, and "the model is installed but every photograph fails" is a much
    /// worse state to debug than "the model is missing".
    /// </remarks>
    public bool IsInstalled(string modelsRoot) =>
        Files.All(file =>
        {
            var path = new FileInfo(Path.Combine(
                modelsRoot, file.RelativePath.Replace('/', Path.DirectorySeparatorChar)));

            return path.Exists && path.Length == file.Bytes;
        });
}

/// <summary>
/// Every model PixelFlux knows how to fetch.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a downloader exists at all.</b> These files come to about two gigabytes and none of them
/// is ours to redistribute, so they cannot ship inside the application. Asking somebody to find
/// five projects on two hosting sites and put the right files in the right folder is a setup step
/// most people would abandon, and abandoning it means the whole point of the application never
/// switches on.
/// </para>
/// <para>
/// <b>This is the one time PixelFlux touches the network, and only when asked.</b> Nothing here
/// runs on a schedule, at startup, or in the background. Analysis, search and browsing make no
/// network calls at all — the WebView's content security policy forbids them — and these downloads
/// deliberately do not go through the WebView: they are ordinary HTTP from the application
/// process, driven by a button somebody pressed.
/// </para>
/// <para>
/// <b>Sizes are exact and verified after each download</b> rather than treated as an estimate.
/// They are not a substitute for a checksum: HTTPS covers transport integrity, and a size check
/// catches the failure that actually happens here, which is a truncated file from a dropped
/// connection. A wrong-but-complete file from a compromised host would pass, and that is a
/// limitation stated rather than papered over.
/// </para>
/// </remarks>
public static class ModelCatalog
{
    private const string OpenCvZoo =
        "https://github.com/opencv/opencv_zoo/raw/main/models";

    private const string Clip =
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main";

    private const string Qwen =
        "https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX/resolve/main"
        + "/onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32";

    /// <summary>Everything, in the order it is worth installing.</summary>
    /// <remarks>
    /// Ordered by value for the megabyte. Faces cost 37 MB and produce a whole page; the vision
    /// model costs 1.4 GB and is the one that makes search genuinely good. Somebody on a metered
    /// connection should be able to stop reading part-way down and still have a useful library.
    /// </remarks>
    public static IReadOnlyList<CatalogueModel> All { get; } =
    [
        new CatalogueModel(
            "faces",
            "YuNet + SFace",
            "Finding faces, and telling one person from another",
            "Apache-2.0",
            ModelTerms.Permissive,
            "https://github.com/opencv/opencv_zoo",
            [
                new ModelFile(
                    $"{OpenCvZoo}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    "face_yunet_2023mar.onnx",
                    232_589),
                new ModelFile(
                    $"{OpenCvZoo}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                    "face_sface_2021dec.onnx",
                    38_696_353),
            ]),

        new CatalogueModel(
            "segment",
            "YOLO11n-seg",
            "Outlining and labelling the objects in a photograph",
            "AGPL-3.0",
            ModelTerms.Copyleft,
            "https://github.com/ultralytics/ultralytics",
            [
                new ModelFile(
                    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.onnx",
                    "yolo11n-seg.onnx",
                    11_763_717),
            ]),

        new CatalogueModel(
            "search",
            "CLIP ViT-B/32",
            "Searching by meaning rather than by filename",
            "MIT",
            ModelTerms.Permissive,
            "https://huggingface.co/Xenova/clip-vit-base-patch32",
            [
                new ModelFile($"{Clip}/onnx/vision_model.onnx", "clip_vision_model.onnx", 351_685_709),
                new ModelFile($"{Clip}/onnx/text_model.onnx", "clip_text_model.onnx", 254_058_553),
                new ModelFile($"{Clip}/vocab.json", "clip_vocab.json", 862_328),
                new ModelFile($"{Clip}/merges.txt", "clip_merges.txt", 524_619),
            ]),

        new CatalogueModel(
            "describe",
            "Qwen3-VL-2B-Instruct",
            "Reading each photograph and writing a description of it",
            "Apache-2.0",
            ModelTerms.Permissive,
            "https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX",
            [
                new ModelFile($"{Qwen}/genai_config.json", "qwen3vl/genai_config.json", 2_798),
                new ModelFile($"{Qwen}/config.json", "qwen3vl/config.json", 1_594),
                new ModelFile($"{Qwen}/generation_config.json", "qwen3vl/generation_config.json", 255),
                new ModelFile($"{Qwen}/processor_config.json", "qwen3vl/processor_config.json", 1_604),
                new ModelFile($"{Qwen}/tokenizer_config.json", "qwen3vl/tokenizer_config.json", 694),
                new ModelFile($"{Qwen}/chat_template.jinja", "qwen3vl/chat_template.jinja", 5_412),
                new ModelFile($"{Qwen}/tokenizer.json", "qwen3vl/tokenizer.json", 12_180_127),
                new ModelFile($"{Qwen}/embedding.onnx", "qwen3vl/embedding.onnx", 2_804),
                new ModelFile($"{Qwen}/embedding.onnx.data", "qwen3vl/embedding.onnx.data", 165_347_328),
                new ModelFile($"{Qwen}/vision.onnx", "qwen3vl/vision.onnx", 2_937_960),
                new ModelFile($"{Qwen}/vision.onnx.data", "qwen3vl/vision.onnx.data", 182_714_368),
                new ModelFile($"{Qwen}/text.onnx", "qwen3vl/text.onnx", 1_049_186),
                new ModelFile($"{Qwen}/text.onnx.data", "qwen3vl/text.onnx.data", 1_075_838_976),
            ]),
    ];

    /// <summary>Total size of everything, for the "install all" line.</summary>
    public static long TotalBytes => All.Sum(model => model.Bytes);

    /// <summary>Whether anything at all has been installed.</summary>
    /// <param name="modelsRoot">The models directory.</param>
    /// <returns>True when no model is fully present, which is a first run.</returns>
    public static bool NothingInstalled(string modelsRoot) =>
        All.All(model => !model.IsInstalled(modelsRoot));
}
