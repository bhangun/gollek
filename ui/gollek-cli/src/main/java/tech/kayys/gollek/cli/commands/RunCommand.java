package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;
import tech.kayys.gollek.observability.ObservabilityManager;
import tech.kayys.gollek.metrics.MetricsRegistry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.arc.Arc;
import io.quarkus.runtime.Quarkus;
import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.model.download.DownloadProgressListener;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.local.GollekManifest;
import tech.kayys.gollek.model.repo.local.ManifestStore;
import tech.kayys.gollek.onnx.runner.OnnxModelDiagnostics;
import tech.kayys.gollek.onnx.runner.MossTtsOnnxRunner;
import tech.kayys.gollek.onnx.runner.PaddleOcrVlOnnxPlanner;
import tech.kayys.gollek.onnx.runner.PaddleOcrVlOnnxProbe;
import tech.kayys.gollek.sdk.model.ModelResolver;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.loader.ModelConfigLoader;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.Message;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Stream;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.SourceDataLine;
import tech.kayys.gollek.cli.chat.ChatUIRenderer;
import tech.kayys.gollek.cli.runtime.CliMetalRuntime;
import tech.kayys.gollek.cli.util.CliInferenceFeatures;
import tech.kayys.gollek.cli.util.ExternalModelFamilyPluginScope;
import tech.kayys.gollek.cli.util.ExternalPluginClasspath;
import tech.kayys.gollek.cli.util.PluginAvailabilityChecker;
import tech.kayys.gollek.sdk.route.RunnerRouteReportFields;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;
import tech.kayys.gollek.sdk.util.GollekHome;
import tech.kayys.suling.FlacLibraryCheck;
import tech.kayys.suling.audio.FfmpegAudioEncoder;
import tech.kayys.suling.audio.Suling;

/**
 * Run inference command using GollekSdk.
 * Usage: gollek run --model <model> --prompt <prompt> [--provider
 * litert|llamacpp|gemini] [--stream]
 */
@Dependent
@Unremovable
@Command(name = "run", description = "Run inference using a specified model")
public class RunCommand implements Runnable {
    static { tech.kayys.gollek.safetensor.engine.route.DirectSafetensorRoutePolicy.resolver = new CliLocalModelResolver(); }
    private static final String MOSS_AUDIO_TOKENIZER_REPOSITORY =
            "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX";
    private static final String MOSS_AUDIO_TOKENIZER_MODEL_SPEC =
            "hf:" + MOSS_AUDIO_TOKENIZER_REPOSITORY;
    private static final String MOSS_CODEC_META_FILE = "codec_browser_onnx_meta.json";
    private static final String ONNX_OPTIMIZED_CACHE_ENABLED_PROPERTY = "gollek.onnx.optimized-model-cache.enabled";
    private static final String ONNX_OPTIMIZED_CACHE_ENABLED_ENV = "GOLLEK_ONNX_OPTIMIZED_MODEL_CACHE";
    private static final String ONNX_OPTIMIZED_CACHE_DIR_PROPERTY = "gollek.onnx.optimized-model-cache.dir";
    private static final String ONNX_OPTIMIZED_CACHE_DIR_ENV = "GOLLEK_ONNX_OPTIMIZED_MODEL_CACHE_DIR";
    private static final String MOSS_SESSION_CACHE_ENABLED_PROPERTY = "gollek.onnx.moss-tts.session-cache.enabled";
    private static final String MOSS_SESSION_CACHE_ENABLED_ENV = "GOLLEK_ONNX_MOSS_TTS_SESSION_CACHE";
    private static final String MOSS_SESSION_CACHE_MAX_ENTRIES_PROPERTY = "gollek.onnx.moss-tts.session-cache.max-entries";
    private static final String MOSS_SESSION_CACHE_MAX_ENTRIES_ENV = "GOLLEK_ONNX_MOSS_TTS_SESSION_CACHE_MAX_ENTRIES";
    private static final String MOSS_ASSET_CACHE_ENABLED_PROPERTY = "gollek.onnx.moss-tts.asset-cache.enabled";
    private static final String MOSS_ASSET_CACHE_ENABLED_ENV = "GOLLEK_ONNX_MOSS_TTS_ASSET_CACHE";
    private static final String MOSS_ASSET_CACHE_MAX_ENTRIES_PROPERTY = "gollek.onnx.moss-tts.asset-cache.max-entries";
    private static final String MOSS_ASSET_CACHE_MAX_ENTRIES_ENV = "GOLLEK_ONNX_MOSS_TTS_ASSET_CACHE_MAX_ENTRIES";
    private static final String ONNX_LOG_LEVEL_PROPERTY = "gollek.onnx.log-level";
    private static final String ONNX_LOG_LEVEL_ENV = "GOLLEK_ONNX_LOG_LEVEL";
    private Map<String, Object> forcedExecutionRouteMetadata = Map.of();

    @ParentCommand
    GollekCommand parentCommand;

    @Inject
    GollekSdk sdk;
    @Inject
    PluginAvailabilityChecker pluginChecker;
    @Inject
    ChatUIRenderer uiRenderer;
    @Inject
    DirectInferenceEngine directInferenceEngine;
    @Inject
    Instance<HuggingFaceClient> huggingFaceClient;
    @Inject
    Instance<ManifestStore> manifestStore;
    @Inject
    Instance<MossTtsOnnxRunner> mossTtsWarmupRunner;

    @Option(names = { "-m", "--model" }, description = "Model ID for repository resolution (e.g., huggingface ID)")
    public String modelId;

    @Option(names = { "--no-banner" }, description = "Suppress header and footer output")
    public boolean noBanner;

    @Option(names = { "--modelFile" }, description = "Path to a local model file (.gguf, .tflite, .task, .litertlm)")
    public String modelFile;

    @Option(names = { "--modelDir" }, description = "Path to a local model directory (Safetensors)")
    public String modelDir;

    @Option(names = { "-p", "--prompt" }, description = "Input prompt")
    public String prompt;

    @Option(names = {
            "--provider" }, description = "Provider: native, litert, llamacpp, safetensor, libtorch(experimental), gemini, openai, anthropic, cerebras. Omit for auto-detection.", arity = "0..1", fallbackValue = "")
    String providerId;

    @Option(names = { "--runner" },
            description = "Runner route: auto, safetensor, gguf, litert, or hybrid. Omit for auto-detection.",
            arity = "0..1",
            fallbackValue = "auto")
    String runner;

    @Option(names = {
            "--import" }, description = "Import (move) the model file/dir into the gollek model repository (~/.gollek/models/)")
    boolean importModel;

    @Option(names = {
            "--copy" }, description = "Copy the model file/dir into the gollek model repository (~/.gollek/models/)")
    boolean copyModel;

    @Option(names = { "-s", "--stream" }, description = "Stream output", defaultValue = "true")
    boolean stream;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.2")
    double temperature;

    @Option(names = { "--top-p" }, description = "Top-p sampling", defaultValue = "0.9")
    double topP;

    @Option(names = { "--top-k" }, description = "Top-k sampling", defaultValue = "40")
    int topK;

    @Option(names = { "--repeat-penalty" }, description = "Repeat penalty", defaultValue = "1.1")
    double repeatPenalty;

    @Option(names = { "--json" }, description = "Enable JSON mode", defaultValue = "false")
    boolean jsonMode;

    @Option(names = {
            "--enable-json" }, description = "Emit OpenAI-compatible SSE JSON for streamed chunks", defaultValue = "false")
    boolean enableJsonSse;

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate", defaultValue = "256")
    int maxTokens;

    @Option(names = { "--mirostat" }, description = "Mirostat mode (0, 1, 2)", defaultValue = "0")
    int mirostat;

    @Option(names = { "--grammar" }, description = "GBNF grammar string")
    String grammar;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

    @Option(names = { "--tool-file", "--tools-file" }, split = ",", description = "JSON tool definition file(s); accepts OpenAI-style tools arrays or Gollek ToolDefinition JSON")
    List<String> toolFiles;

    @Option(names = { "--tool-choice" }, description = "Tool choice policy such as auto, none, required, or a JSON object")
    String toolChoice;

    @Option(names = { "--mcp-tool", "--mcp-tools" }, split = ",", description = "MCP tool name(s) to expose, optionally as server/tool or server:tool")
    List<String> mcpTools;

    @Option(names = { "--rag-context" }, split = "\\|", description = "Inline retrieval context block(s), separated with | when repeated in one argument")
    List<String> ragContexts;

    @Option(names = { "--rag-file", "--context-file" }, split = ",", description = "Text file(s) to inject as retrieval context")
    List<String> ragFiles;

    @Option(names = { "--rag-max-chars" }, description = "Maximum retrieval context characters to inject", defaultValue = "12000")
    int ragMaxChars;

    @Option(names = { "--embedding-model", "--embed-model" }, description = "Embedding model id/path to attach to embedding or RAG-aware providers")
    String embeddingModel;

    @Option(names = { "--session-id" }, description = "Persistent conversation session id for KV/prefix reuse across repeated run calls")
    String sessionId;

    @Option(names = { "--no-cache" }, description = "Bypass response cache")
    boolean noCache;

    @Option(names = { "--offline",
            "--local" }, description = "Force using existing models without checking for updates/downloads")
    boolean offline;

    @Option(names = {
            "--model-path" }, description = "Path to a custom model file (bypasses repository lookup, prefer --modelFile)")
    String modelPath;

    @Option(names = {
            "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    @Option(names = { "-b", "--branch" }, description = "HuggingFace branch/revision (e.g. main, fp16, onnx)")
    String branch;

    @Option(names = { "--force" }, description = "Force re-download and replace existing files", defaultValue = "false")
    boolean force;

    @Option(names = { "--format" }, description = "Preferred model format (e.g., safetensors, gguf, onnx)")
    String format;

    @Option(names = { "--pipeline", "--feature" }, description = "Explicit feature pipeline id for out-of-core model orchestration")
    String featurePipelineId;

    @Option(names = { "--direct" }, description = "Force use of native Safetensor direct engine", defaultValue = "false")
    boolean direct;

    @Option(names = { "--gguf-quant" }, description = "Specify quantization for GGUF conversion (e.g. Q4_K_M)")
    String ggufQuant;

    @Option(names = { "--gguf" }, description = "Force GGUF conversion and usage", defaultValue = "false")
    boolean forceGguf;

    @Option(names = { "--fallback" }, split = ",", description = "Comma-separated fallback model IDs or short IDs to try only if the primary model is incompatible")
    List<String> fallbackModelIds;

    @Option(names = { "--quantize" }, description = "Enable JIT quantization during inference (bnb, turbo, awq, gptq, autoround)")
    String quantizeStrategy;

    @Option(names = { "--quantize-bits" }, description = "Bit width for JIT quantization (default: 4)", defaultValue = "4")
    int quantizeBits;

    // Stable Diffusion specific parameters
    @Option(names = { "--seed" }, description = "Random seed for image generation and TTS (MOSS TTS default: 1234)")
    Long seed;

    @Option(names = { "--output", "-o" }, description = "Output image file path (default: output.png)")
    String outputPath;

    @Option(names = { "--audio-metadata-output", "--tts-metadata-output" }, description = "Write audio generation metadata JSON to this path")
    String audioMetadataOutputPath;

    @Option(names = { "--audio-metadata-sidecar", "--tts-metadata-sidecar" }, description = "Write audio generation metadata JSON beside the output file")
    boolean audioMetadataSidecar;

    @Option(names = { "--audio-metadata-replay", "--tts-replay" }, description = "Replay a TTS/audio generation from a Gollek audio metadata JSON sidecar")
    String audioMetadataReplayPath;

    @Option(names = { "--audio-metadata-verify", "--tts-verify" }, description = "Verify an audio file against a Gollek audio metadata JSON sidecar")
    String audioMetadataVerifyPath;

    @Option(names = { "--audio-metadata-refresh", "--tts-refresh-metadata" }, description = "Refresh a Gollek audio metadata JSON sidecar with current audio size and SHA-256")
    String audioMetadataRefreshPath;

    @Option(names = { "--image-metadata-output" }, description = "Write image generation metadata JSON to this path")
    String imageMetadataOutputPath;

    @Option(names = { "--image-metadata-sidecar" }, description = "Write image generation metadata JSON beside the output file")
    boolean imageMetadataSidecar;

    @Option(names = { "--image-metadata-verify" }, description = "Verify an image file against a Gollek image metadata JSON sidecar")
    String imageMetadataVerifyPath;

    @Option(names = { "--image-metadata-refresh" }, description = "Refresh a Gollek image metadata JSON sidecar with current image size and SHA-256")
    String imageMetadataRefreshPath;

    @Option(names = { "--image", "--input-image", "--vision-input" }, description = "Input image path for vision/OCR models")
    String inputImagePath;

    @Option(names = { "--images", "--input-images" }, split = ",", description = "Comma-separated input image paths for vision/OCR models")
    List<String> inputImagePaths;

    @Option(names = { "--ocr" }, description = "Use the default OCR task prompt when --prompt is omitted")
    boolean ocrMode;

    @Option(names = { "--ocr-image-tokens", "--onnx-image-tokens" }, description = "Limit image embeddings used by PaddleOCR-VL ONNX generation (default: 0 = all)")
    Integer ocrImageTokens;

    @Option(names = { "--ocr-raw", "--onnx-ocr-raw" }, description = "Print raw PaddleOCR-VL generated token text without location postprocessing")
    boolean ocrRawOutput;

    @Option(names = { "--ocr-json", "--onnx-ocr-json" }, description = "Print PaddleOCR-VL OCR result as structured JSON")
    boolean ocrJsonOutput;

    @Option(names = { "--ocr-json-output", "--ocr-metadata-output" }, description = "Write PaddleOCR-VL OCR result JSON to this path")
    String ocrJsonOutputPath;

    @Option(names = { "--ocr-overlay-output", "--ocr-boxes-output" }, description = "Write a PNG preview with PaddleOCR-VL OCR boxes drawn over the input image")
    String ocrOverlayOutputPath;

    @Option(names = { "--onnx-diagnostics", "--onnx-preflight" }, description = "Inspect a local ONNX model/repository and exit")
    boolean onnxDiagnostics;

    @Option(names = { "--onnx-vision-probe", "--ocr-vision-probe" }, description = "Run PaddleOCR-VL vision_encoder with --image and print output tensor shapes")
    boolean onnxVisionProbe;

    @Option(names = { "--onnx-pipeline-probe", "--ocr-pipeline-probe" }, description = "Run PaddleOCR-VL vision probe and inspect selected embedding/decoder graph IO")
    boolean onnxPipelineProbe;

    @Option(names = { "--onnx-probe-image-tokens", "--ocr-probe-image-tokens" }, description = "Image embeddings to feed into PaddleOCR decoder probe (default: 64, 0 = all)")
    Integer onnxProbeImageTokens;

    @Option(names = { "--onnx-probe-decode-tokens", "--ocr-probe-decode-tokens" }, description = "Greedy tokens to decode in the PaddleOCR pipeline probe (default: 8, 0 = prefill only)")
    Integer onnxProbeDecodeTokens;

    @Option(names = { "--onnx-variant", "--onnx-precision" }, description = "Preferred ONNX graph variant for multi-graph models (auto, q4, q8, full, uint8)")
    String onnxVariant;

    @Option(names = { "--voice", "--tts-voice" }, description = "TTS voice preset name/id for supported audio models")
    String ttsVoice;

    @Option(names = { "--voice-lang", "--voice-language", "--tts-voice-lang", "--tts-voice-language" }, description = "TTS voice language selector, such as en, jp, zh, english, japanese, or chinese")
    String ttsVoiceLanguage;

    @Option(names = { "--voice-gender", "--tts-voice-gender" }, description = "TTS voice gender selector, such as male/female. With --voice-lang, non-gender values are treated as voice preset terms.")
    String ttsVoiceGender;

    @Option(names = { "--voice-search", "--tts-voice-search" }, description = "Search/filter MOSS TTS voices when used with --list-voices")
    String ttsVoiceSearch;

    @Option(names = { "--tts-codec", "--codec" }, description = "MOSS TTS companion audio tokenizer path or local model id")
    String ttsCodec;

    @Option(names = { "--no-tts-auto-codec", "--no-tts-auto-pull-codec" }, description = "Disable automatic pull of the MOSS TTS companion audio tokenizer")
    boolean disableTtsAutoCodec;

    @Option(names = { "--list-voices", "--voices", "--tts-voices" }, description = "List TTS voice presets for the resolved model and exit")
    boolean listTtsVoices;

    @Option(names = { "--voices-json", "--tts-voices-json" }, description = "List MOSS TTS voice presets as JSON and exit")
    boolean listTtsVoicesJson;

    @Option(names = { "--tts-diagnostics", "--tts-preflight" }, description = "Inspect local MOSS TTS model, codec, voices, and audio encoders without generating audio")
    boolean ttsDiagnostics;

    @Option(names = { "--tts-warmup", "--onnx-warmup" }, description = "Warm MOSS TTS ONNX assets and sessions without generating audio")
    boolean ttsWarmup;

    @Option(names = { "--tts-cache-prune", "--onnx-cache-prune" }, description = "Preview removal of stale MOSS TTS optimized graph cache files")
    boolean ttsCachePrune;

    @Option(names = { "--tts-cache-prune-apply", "--onnx-cache-prune-apply" }, description = "Actually delete stale MOSS TTS optimized graph cache files")
    boolean ttsCachePruneApply;

    @Option(names = { "--tts-max-frames", "--max-frames" }, description = "Maximum TTS audio frames to generate")
    Integer ttsMaxFrames;

    @Option(names = { "--tts-max-seconds", "--max-seconds", "--tts-duration" }, description = "Maximum TTS audio duration in seconds")
    Double ttsMaxSeconds;

    @Option(names = { "--audio-format", "--tts-audio-format" }, description = "Audio output format for TTS (wav, flac, mp3; opus/aac require the future FFmpeg backend)")
    String audioFormat;

    @Option(names = { "--audio-quality", "--tts-audio-quality" }, description = "TTS audio quality preset (compact, speech, balanced, high, lossless)")
    String audioQuality;

    @Option(names = { "--audio-bitrate", "--audio-bitrate-kbps", "--tts-bitrate", "--tts-bitrate-kbps" }, description = "Encoded TTS audio bitrate in kbps for formats that support it, such as MP3")
    Integer audioBitrateKbps;

    @Option(names = { "--audio-channels", "--tts-audio-channels" }, description = "TTS audio channel mode (auto, mono, stereo/dual-mono, native)")
    String audioChannels;

    @Option(names = { "--tts-codec-decode", "--codec-decode" }, description = "MOSS TTS codec decode mode (auto, streaming, full)")
    String ttsCodecDecode;

    @Option(names = { "--random-seed", "--tts-random-seed" }, description = "Use a fresh random TTS seed when --seed is omitted")
    boolean randomTtsSeed;

    @Option(names = { "--no-audio-polish", "--no-tts-audio-polish" }, description = "Disable TTS PCM polish (DC removal, edge fades, and peak normalization)")
    boolean disableAudioPolish;

    @Option(names = { "--no-audio-normalize", "--no-tts-audio-normalize" }, description = "Disable TTS peak normalization while keeping other audio polish")
    boolean disableAudioNormalize;

    @Option(names = { "--audio-gain-db", "--tts-gain-db" }, description = "Extra TTS output gain in dB before encoding")
    Double audioGainDb;

    @Option(names = { "--audio-peak-db", "--audio-peak-dbfs", "--tts-peak-db" }, description = "TTS peak-normalization target in dBFS (default: -3)")
    Double audioPeakDb;

    @Option(names = { "--audio-fade-ms", "--tts-fade-ms" }, description = "TTS fade-in/fade-out duration in milliseconds")
    Double audioFadeMs;

    @Option(names = { "--audio-trim-silence", "--tts-trim-silence" }, description = "Trim leading/trailing low-level silence from TTS audio after decode")
    boolean audioTrimSilence;

    @Option(names = { "--audio-trim-db", "--audio-trim-dbfs", "--tts-trim-db" }, description = "Silence trim threshold in dBFS for TTS audio (default: -48)")
    Double audioTrimDbfs;

    @Option(names = { "--audio-trim-padding-ms", "--tts-trim-padding-ms" }, description = "Silence trim padding kept at each edge in milliseconds (default: 25)")
    Double audioTrimPaddingMs;

    @Option(names = { "--audio-stream-chunk-kb", "--tts-stream-chunk-kb" }, description = "Audio chunk size for streamed TTS responses in KiB (default: 64)")
    Integer audioStreamChunkKb;

    @Option(names = { "--tts-stream-progress-frames" }, description = "Emit a TTS streaming progress update every N generated frames")
    Integer ttsStreamProgressFrames;

    @Option(names = { "--no-tts-stream-progress" }, description = "Hide live TTS progress updates during streaming")
    boolean disableTtsStreamProgress;

    @Option(names = { "--tts-stream-pcm", "--audio-stream-pcm" }, description = "Emit decoded PCM chunks while streaming TTS")
    boolean ttsStreamPcm;

    @Option(names = { "--live-audio", "--tts-live-audio" }, description = "Play decoded TTS PCM chunks while the final audio file is being generated")
    boolean liveAudio;

    @Option(names = { "--no-low-latency-audio", "--no-tts-low-latency-audio" }, description = "Emit TTS PCM preview chunks after generation instead of during frame generation")
    boolean disableLowLatencyAudio;

    @Option(names = { "--flac-compression" }, description = "FLAC compression level for TTS output, 0 fastest through 8 smallest")
    Integer flacCompression;

    @Option(names = { "--steps" }, description = "Number of denoising steps (default: 20)")
    Integer steps;

    @Option(names = { "--guidance-scale", "--cfg" }, description = "Classifier-free guidance scale (default: 7.5)")
    Float guidanceScale;

    @Option(names = { "--width" }, description = "Output image width in pixels (default: 512, must be multiple of 64)")
    Integer width;

    @Option(names = {
            "--height" }, description = "Output image height in pixels (default: 512, must be multiple of 64)")
    Integer height;

    @Option(names = { "--ext" }, description = "Explicit output file extension (mp3, wav, flac, png, jpg, mp4). Overrides extension in --output if provided.")
    String extension;

    @Option(names = { "--quantize-kv" }, description = "Enable KV cache quantization (none, int8, int4, turbo)", defaultValue = "none")
    String quantizeKv;

    @Option(names = { "--plugin" }, description = "Explicit plugin/engine to use (e.g. llamacpp, java, bnb)")
    public String pluginId;

    @Option(names = {
            ExternalPluginClasspath.OPTION_PLUGIN_CLASSPATH,
            ExternalPluginClasspath.OPTION_EXTERNAL_PLUGIN_CLASSPATH },
            split = ",",
            description = ExternalPluginClasspath.MODEL_FAMILY_OPTION_DESCRIPTION)
    List<String> externalPluginClasspath = new ArrayList<>();

    @Option(names = {
            ExternalPluginClasspath.OPTION_PLUGIN_DIR,
            ExternalPluginClasspath.OPTION_EXTERNAL_PLUGIN_DIR },
            split = ",",
            description = ExternalPluginClasspath.PLUGIN_DIRECTORY_OPTION_DESCRIPTION)
    List<String> externalPluginDirectories = new ArrayList<>();

    @Option(names = { "--engine", "--gguf-engine" }, description = "GGUF engine mode: auto, java, llamacpp, benchmark")
    String ggufEngine;

    @Option(names = { "--backend" }, description = "GGUF backend to use for local fast path (metal or cpu)")
    String ggufBackend;

    @Option(names = { "--java-native", "--java" }, description = "Use the Java-native GGUF loader/probe path")
    boolean javaNativeGguf;

    @Option(names = { "--llamacpp", "--llama-cpp" }, description = "Use the llama.cpp GGUF engine")
    boolean llamaCppGguf;

    @Option(names = { "--benchmark", "--bench" }, description = "Compare Java-native GGUF probe with llama.cpp fallback")
    boolean benchmarkGguf;

    @Option(names = { "--profile", "--onnx-profile" },
            description = "Request runner profile metadata and print detailed runtime performance breakdowns")
    boolean runtimeProfile;

    @Option(names = {
            "--prefer-alternate-runtime" }, description = "Allow Gemma safetensor checkpoints to route to local LiteRT/GGUF companion artifacts")
    boolean preferAlternateRuntime;

    @Option(names = { "--route-report-json", "--route-report", "--dry-run-route" },
            description = "Resolve runner/provider/model routing, print JSON, and exit before inference")
    boolean routeReportJson;

    @Option(names = { "--route-report-allow-pull", "--dry-run-route-allow-pull" },
            description = "Allow route-report mode to resolve/pull missing models through repository providers")
    boolean routeReportAllowPull;

    @Option(names = { "--route-report-require-local", "--dry-run-route-require-local" },
            description = "Exit non-zero when route-report mode cannot resolve a local runnable model artifact")
    boolean routeReportRequireLocal;

    @Parameters(arity = "0..*", hidden = true)
    List<String> positionalArgs = new ArrayList<>();

    @Override
    public void run() {
        try (ExternalModelFamilyPluginScope ignored =
                ExternalModelFamilyPluginScope.attach(
                        externalPluginClasspath,
                        externalPluginDirectories,
                        RunCommand.class,
                        pluginChecker)) {
            if (parentCommand != null) {
                parentCommand.bootstrapInheritedEnvironment();
            }

            if (!repairOrRejectPositionalArgs()) {
                requestProcessExit(1);
                return;
            }
            if (audioMetadataRefreshPath != null && !audioMetadataRefreshPath.isBlank()) {
                requestProcessExit(refreshAudioMetadataSidecar() ? 0 : 1);
                return;
            }
            if (audioMetadataVerifyPath != null && !audioMetadataVerifyPath.isBlank()) {
                requestProcessExit(verifyAudioMetadataSidecar() ? 0 : 1);
                return;
            }
            if (imageMetadataRefreshPath != null && !imageMetadataRefreshPath.isBlank()) {
                requestProcessExit(refreshImageMetadataSidecar() ? 0 : 1);
                return;
            }
            if (imageMetadataVerifyPath != null && !imageMetadataVerifyPath.isBlank()) {
                requestProcessExit(verifyImageMetadataSidecar() ? 0 : 1);
                return;
            }
            if (!loadAudioMetadataReplayIfRequested()) {
                requestProcessExit(1);
                return;
            }
            if (!validateTtsAudioCliOptions()) {
                requestProcessExit(1);
                return;
            }

            providerId = normalizeRequestedProvider(providerId);
            boolean providerExplicit = providerId != null && !providerId.isBlank();
            boolean requestedProviderExplicit = providerExplicit;
            String requestedProviderId = providerId;
            String requestedFormat = format;
            RunnerRoutePolicy.Selection runnerSelection = RunnerRoutePolicy.select(
                    runner,
                    providerId,
                    format,
                    providerExplicit,
                    preferAlternateRuntime,
                    forceGguf);
            if (!runnerSelection.valid()) {
                System.err.println("Error: " + runnerSelection.error());
                requestProcessExit(1);
                return;
            }
            RunnerRouteReport runnerRouteReport = RunnerRouteReport.from(
                    runner,
                    requestedProviderId,
                    requestedProviderExplicit,
                    requestedFormat,
                    runnerSelection);
            providerId = runnerSelection.providerId();
            format = runnerSelection.format();
            preferAlternateRuntime = runnerSelection.preferAlternateRuntime();
            forceGguf = runnerSelection.forceGguf();
            providerExplicit = providerId != null && !providerId.isBlank();
            boolean preferOnnxResolution = shouldPreferOnnxResolution(modelId, format, providerId);
            if (preferOnnxResolution) {
                if (!providerExplicit) {
                    providerId = "onnx";
                }
                if (format == null || format.isBlank()) {
                    format = "onnx";
                }
            }
            ensureBuiltinProviderRegistration();

            if (!routeReportJson && tryCachedGemma4MobileQatLiteRtFastPath(requestedProviderExplicit, requestedProviderId)) {
                return;
            }

            if (!routeReportJson && tryStandaloneGgufFastPath()) {
                return;
            }

            // Check plugin availability first
            if (!pluginChecker.hasProviders() && !pluginChecker.hasRunnerPlugins()) {
                System.err.println(pluginChecker.getNoPluginsError());
                System.exit(1);
                return;
            }

            // Auto-detect and display kernel platform
            KernelPlatform detectedPlatform;
            try {
                detectedPlatform = sdk.getPlatformInfo();
            } catch (Throwable t) {
                System.err.println("CRITICAL: Platform detection failed: " + t.getMessage());
                return;
            }
            if (!routeReportJson && CliMetalRuntime.isMetal(detectedPlatform)) {
                ensureMetalRuntimeInitialized();
            }

            if (!routeReportJson && parentCommand != null && parentCommand.verbose) {
                System.out.println(ChatUIRenderer.CYAN + "Platform: " + detectedPlatform.getDisplayName() + ChatUIRenderer.RESET);
                if (detectedPlatform.isCpu()) {
                    System.out.println(ChatUIRenderer.YELLOW + "⚠️  Running on CPU (GPU acceleration not available)" + ChatUIRenderer.RESET);
                } else {
                    boolean isMetalPlatform = CliMetalRuntime.isMetal(detectedPlatform);
                    if (isMetalPlatform && !isMetalNativeRuntimeActive()) {
                        System.out.println(ChatUIRenderer.YELLOW
                                + "⚠️  Metal selected but native runtime is not active (CPU fallback likely)"
                                + ChatUIRenderer.RESET);
                    } else {
                        System.out.println(ChatUIRenderer.GREEN + "✓ GPU acceleration enabled" + ChatUIRenderer.RESET);
                    }
                }
                printMetalRuntimeStatus();
                System.out.println();
            }

            boolean isMetalPlatform = CliMetalRuntime.isMetal(detectedPlatform);
            if (!routeReportJson
                    && isMetalPlatform
                    && !isMetalNativeRuntimeActive()
                    && !allowCpuFallbackWhenMetalRequested()) {
                System.err.println("Error: Metal platform selected but native Metal runtime is not active.");
                System.err.println("Refusing CPU fallback for this run so performance behavior stays explicit.");
                System.err.println("Set GOLLEK_ALLOW_CPU_FALLBACK=true to override.");
                System.exit(1);
                return;
            }

            long startTime = System.currentTimeMillis();

            // Check if specific provider is requested but not available
            if (providerExplicit) {
                if (!pluginChecker.hasProvider(providerId)) {
                    System.err.println(pluginChecker.getProviderNotFoundError(providerId));
                    System.exit(1);
                    return;
                }
            }

            boolean customModelPathUsed = false;
            String finalLocalPath = null;
            String requestedModelRef = modelId;
            uiRenderer.setJsonMode(enableJsonSse || jsonMode || listTtsVoicesJson || ocrJsonOutput || routeReportJson);
            if (!quietRouteResolutionOutput()) {
                uiRenderer.printBanner();
            }

            if (isMcpProvider()) {
                if (!quietRouteResolutionOutput()) {
                    System.out.println("MCP provider selected; skipping local model lookup.");
                }
            } else if (modelFile != null && !modelFile.isBlank()) {
                // --- Resolve from --modelFile ---
                Path filePath = Paths.get(modelFile);
                if (!Files.exists(filePath)) {
                    System.err.println("Error: Model file not found: " + modelFile);
                    return;
                }

                if (importModel || copyModel) {
                    var res = sdk.importModel(filePath, importModel);
                    filePath = Paths.get(res.getLocalPath());
                    if (!quietRouteResolutionOutput()) {
                        System.out.println((importModel ? "Imported" : "Copied") + " model to: " + filePath.toAbsolutePath());
                    }
                }

                modelId = filePath.toAbsolutePath().toString();
                finalLocalPath = modelId;
                customModelPathUsed = true;

                String inferredFormat = formatForLocalModelPath(filePath);
                if ((format == null || format.isBlank()) && inferredFormat != null) {
                    format = inferredFormat;
                }
                if (!providerExplicit && (providerId == null || providerId.isBlank())) {
                    providerId = providerForFormat(inferredFormat);
                }
                if ("native".equals(providerId)) {
                    if (modelFile.endsWith(".litertlm") || modelFile.endsWith(".tflite") || modelFile.endsWith(".task")) {
                        providerId = "litert";
                    }
                }
                if (!quietRouteResolutionOutput()) {
                    System.out.println("Model path: " + filePath.toAbsolutePath());
                }

            } else if (modelDir != null && !modelDir.isBlank()) {
                // --- Resolve from --modelDir ---
                Path dirPath = Paths.get(modelDir);
                if (!Files.isDirectory(dirPath)) {
                    System.err.println("Error: Model directory not found: " + modelDir);
                    return;
                }

                if (importModel || copyModel) {
                    var res = sdk.importModel(dirPath, importModel);
                    dirPath = Paths.get(res.getLocalPath());
                    if (!quietRouteResolutionOutput()) {
                        System.out.println((importModel ? "Imported" : "Copied") + " model to: " + dirPath.toAbsolutePath());
                    }
                }

                modelId = dirPath.toAbsolutePath().toString();
                finalLocalPath = modelId;
                customModelPathUsed = true;
                if (!providerExplicit) {
                    providerId = providerForModelDirectory(dirPath);
                }
                if (format == null || format.isBlank()) {
                    format = formatForModelDirectory(dirPath);
                }
                if (!quietRouteResolutionOutput()) {
                    System.out.println("Model dir: " + dirPath.toAbsolutePath());
                }

            } else if (modelPath != null && !modelPath.isEmpty()) {
                Path customModelPath = Paths.get(modelPath);
                if (!Files.exists(customModelPath)) {
                    System.err.println("Error: Model file not found: " + modelPath);
                    return;
                }
                if (!quietRouteResolutionOutput()) {
                    System.out.println("Using model from: " + customModelPath.toAbsolutePath());
                }
                modelId = customModelPath.toAbsolutePath().toString();
                finalLocalPath = modelId;
                customModelPathUsed = true;
                String inferredFormat = formatForLocalModelPath(customModelPath);
                if ((format == null || format.isBlank()) && inferredFormat != null) {
                    format = inferredFormat;
                }
                if (!providerExplicit && (providerId == null || providerId.isBlank())) {
                    providerId = providerForFormat(inferredFormat);
                }
            }

            if (!customModelPathUsed && modelId != null && !modelId.isBlank()) {
                try {
                    String preferredIndexFormat = preferredLocalIndexFormat(modelId, format, providerId);
                    var indexed = preferredIndexFormat == null
                            ? LocalModelIndex.find(modelId)
                            : LocalModelIndex.find(modelId, preferredIndexFormat);
                    if (indexed.isPresent()) {
                        var entry = indexed.get();
                        if (!entry.enabled) {
                            System.err.println("\n\u001B[31mError: Model '" + entry.id + "' is disabled.\u001B[0m");
                            System.err.println("\u001B[33mUse 'gollek model enable " + entry.shortId + "' to enable it for inference.\u001B[0m\n");
                            return;
                        }
                        if (entry.path != null && !entry.path.isBlank() && Files.exists(Path.of(entry.path))) {
                            finalLocalPath = Path.of(entry.path).toAbsolutePath().toString();
                            modelId = finalLocalPath;
                            customModelPathUsed = true;
                            if (!providerExplicit) {
                                String inferred = inferProviderFromIndex(entry);
                                if (inferred != null && !inferred.isBlank()) {
                                    providerId = inferred;
                                }
                            }
                            if (format == null || format.isBlank()) {
                                format = inferFormatFromIndex(entry);
                            }
                            if (!quietRouteResolutionOutput() && !jsonMode && !ocrJsonOutput) {
                                System.out.println("Resolved local model index entry: " + finalLocalPath);
                            }
                        }
                    }
                } catch (Exception indexLookupFailure) {
                    // Fallback to normal SDK resolution flow.
                }
            }

            boolean routeReportRepositoryResolutionSkipped = false;
            if (!customModelPathUsed && (modelFile == null || modelFile.isBlank()) && (modelDir == null || modelDir.isBlank())
                    && (modelPath == null || modelPath.isEmpty())) {
                if (!shouldAllowRepositoryResolutionDuringRouteReport()) {
                    routeReportRepositoryResolutionSkipped = true;
                } else {
                    // Prepare model using SDK (this handles pulling, registration, and conversion).
                    try {
                        String quant = ggufQuant != null ? ggufQuant : (ggufOutType != null ? ggufOutType : "Q4_K_M");
                        var resolution = sdk.prepareModel(modelId, (String) format, pluginId, forceGguf, quant,
                                fallbackModelIds == null ? List.of() : fallbackModelIds,
                                (Consumer<PullProgress>) progress -> {
                            if (quietRouteResolutionOutput()) {
                                return;
                            }
                            if (progress.getTotal() > 0) {
                                System.out.printf("\r%s %s %3d%% (%d/%d MB)",
                                        ChatUIRenderer.CYAN + progress.getStatus() + ChatUIRenderer.RESET,
                                        progress.getProgressBar(20),
                                        progress.getPercentComplete(),
                                        progress.getCompleted() / 1024 / 1024,
                                        progress.getTotal() / 1024 / 1024);
                            } else {
                                System.out.print("\r" + ChatUIRenderer.CYAN + progress.getStatus() + ChatUIRenderer.RESET);
                            }
                        });
                        if (!quietRouteResolutionOutput()) {
                            System.out.println();
                        }

                        modelId = resolution.getModelId();
                        if (resolution.getLocalPath() != null) {
                            finalLocalPath = resolution.getLocalPath();
                            String displayPath = finalLocalPath;
                            String userHome = System.getProperty("user.home");
                            if (userHome != null && displayPath.startsWith(userHome)) {
                                displayPath = "~" + displayPath.substring(userHome.length());
                            }
                            if (!quietRouteResolutionOutput()) {
                                System.out.println("Model ready at: " + displayPath);
                            }
                            customModelPathUsed = true;
                        }
                        if (!customModelPathUsed) {
                            LocalModelIndex.Entry downloaded = autoPullHfRepositoryForRunIfNeeded(
                                    requestedModelRef != null ? requestedModelRef : modelId,
                                    preferredLocalIndexFormat(requestedModelRef, format, providerId));
                            if (downloaded != null && downloaded.path != null && Files.exists(Path.of(downloaded.path))) {
                                finalLocalPath = Path.of(downloaded.path).toAbsolutePath().toString();
                                modelId = finalLocalPath;
                                customModelPathUsed = true;
                                if (!providerExplicit) {
                                    String inferred = inferProviderFromIndex(downloaded);
                                    if (inferred != null && !inferred.isBlank()) {
                                        providerId = inferred;
                                    }
                                }
                                if (format == null || format.isBlank()) {
                                    format = inferFormatFromIndex(downloaded);
                                }
                                if (!quietRouteResolutionOutput()) {
                                    System.out.println("Model ready at: " + displayPath(Path.of(finalLocalPath)));
                                }
                            } else if (requiresLocalHfRepositoryResolution(requestedModelRef, providerId)) {
                                return;
                            }
                        }

                        if (resolution.getNotice() != null && !resolution.getNotice().isBlank()) {
                            String notice = resolution.getNotice();
                            String compatibilityIssue = detectSafetensorCompatibilityIssue(providerId, finalLocalPath);
                            boolean staleGemma4Notice = notice.contains("Gemma4 multimodal text checkpoints")
                                    && compatibilityIssue == null;
                            if (!staleGemma4Notice && !quietRouteResolutionOutput()) {
                                System.out.println(notice);
                            }
                        }

                        if (providerId == null || providerId.isBlank()) {
                            providerId = resolution.getProviderId();
                        }
                        if ((providerId == null || providerId.isBlank()) && resolution.getLocalPath() != null) {
                            providerId = null;
                            if (providerId != null) {
                                
                            }
                        }

                    } catch (Exception e) {
                        try {
                            LocalModelIndex.Entry downloaded = autoPullHfRepositoryForRunIfNeeded(
                                    requestedModelRef != null ? requestedModelRef : modelId,
                                    preferredLocalIndexFormat(requestedModelRef, format, providerId));
                            if (downloaded != null && downloaded.path != null && Files.exists(Path.of(downloaded.path))) {
                                finalLocalPath = Path.of(downloaded.path).toAbsolutePath().toString();
                                modelId = finalLocalPath;
                                customModelPathUsed = true;
                                if (!providerExplicit) {
                                    String inferred = inferProviderFromIndex(downloaded);
                                    if (inferred != null && !inferred.isBlank()) {
                                        providerId = inferred;
                                    }
                                }
                                if (format == null || format.isBlank()) {
                                    format = inferFormatFromIndex(downloaded);
                                }
                                if (!quietRouteResolutionOutput()) {
                                    System.out.println("Model ready at: " + displayPath(Path.of(finalLocalPath)));
                                }
                            }
                        } catch (Exception fallbackError) {
                            System.err.println("\nError: Failed to prepare model: " + fallbackError.getMessage());
                            return;
                        }
                        if (!customModelPathUsed) {
                            System.err.println("\nError: Failed to prepare model: " + e.getMessage());
                            return;
                        }
                    }
                }
            }

            if (!quietRouteResolutionOutput()) {
                tech.kayys.gollek.cli.util.QuantSuggestionDetector.suggestIfNeeded(
                        modelId, finalLocalPath, quantizeStrategy, false);
            }

            ProviderLocalPathResolution providerLocalPathResolution =
                    resolveProviderSpecificLocalPath(providerId, modelId, finalLocalPath);
            if (!providerLocalPathResolution.ok()) {
                return;
            }
            finalLocalPath = providerLocalPathResolution.localPath();
            DirectSafetensorRoutePolicy.AlternateRuntimeSelection gemma3Selection =
                    DirectSafetensorRoutePolicy.selectGemma3AlternateRuntime(
                            providerId, modelId, finalLocalPath, providerExplicit,
                            preferAlternateRuntime, this::isProviderActive);
            if (!quietRouteResolutionOutput() && gemma3Selection.hasNotice()) {
                System.out.println(gemma3Selection.notice());
            }
            if (gemma3Selection.selected()) {
                String previousProviderId = providerId;
                String previousFormat = format;
                providerId = gemma3Selection.provider();
                finalLocalPath = gemma3Selection.localPath();
                format = gemma3Selection.format();
                runnerRouteReport = runnerRouteReport.withRuntimeRedirect(
                        previousProviderId,
                        previousFormat,
                        providerId,
                        format,
                        gemma3Selection.reason(),
                        gemma3Selection.cacheHit(),
                        gemma3Selection.cacheKind());
                if ("gguf".equalsIgnoreCase(format)) {
                    modelId = finalLocalPath;
                }
            }
            DirectSafetensorRoutePolicy.AlternateRuntimeSelection gemma4MobileQatSelection =
                    DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                            providerId, modelId, finalLocalPath, providerExplicit,
                            preferAlternateRuntime, this::isRuntimeRouteActive);
            if (!quietRouteResolutionOutput() && gemma4MobileQatSelection.hasNotice()) {
                System.out.println(gemma4MobileQatSelection.notice());
            }
            if (gemma4MobileQatSelection.selected()) {
                String previousProviderId = providerId;
                String previousFormat = format;
                providerId = gemma4MobileQatSelection.provider();
                finalLocalPath = gemma4MobileQatSelection.localPath();
                format = gemma4MobileQatSelection.format();
                runnerRouteReport = runnerRouteReport.withRuntimeRedirect(
                        previousProviderId,
                        previousFormat,
                        providerId,
                        format,
                        gemma4MobileQatSelection.reason(),
                        gemma4MobileQatSelection.cacheHit(),
                        gemma4MobileQatSelection.cacheKind());
            }
            DirectSafetensorRoutePolicy.AlternateRuntimeSelection gemma4TextSelection =
                    DirectSafetensorRoutePolicy.selectGemma4TextAlternateRuntime(
                            providerId, modelId, finalLocalPath, providerExplicit,
                            preferAlternateRuntime, this::isRuntimeRouteActive);
            if (!quietRouteResolutionOutput() && gemma4TextSelection.hasNotice()) {
                System.out.println(gemma4TextSelection.notice());
            }
            if (gemma4TextSelection.selected()) {
                String previousProviderId = providerId;
                String previousFormat = format;
                providerId = gemma4TextSelection.provider();
                finalLocalPath = gemma4TextSelection.localPath();
                modelId = finalLocalPath;
                format = gemma4TextSelection.format();
                runnerRouteReport = runnerRouteReport.withRuntimeRedirect(
                        previousProviderId,
                        previousFormat,
                        providerId,
                        format,
                        gemma4TextSelection.reason(),
                        gemma4TextSelection.cacheHit(),
                        gemma4TextSelection.cacheKind());
            }
            DirectSafetensorRoutePolicy.AlternateRuntimeSelection communityTextGgufSelection =
                    DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                            providerId, modelId, finalLocalPath, providerExplicit,
                            preferAlternateRuntime, this::isRuntimeRouteActive);
            if (!quietRouteResolutionOutput() && communityTextGgufSelection.hasNotice()) {
                System.out.println(communityTextGgufSelection.notice());
            }
            if (communityTextGgufSelection.selected()) {
                String previousProviderId = providerId;
                String previousFormat = format;
                providerId = communityTextGgufSelection.provider();
                finalLocalPath = communityTextGgufSelection.localPath();
                modelId = finalLocalPath;
                format = communityTextGgufSelection.format();
                runnerRouteReport = runnerRouteReport.withRuntimeRedirect(
                        previousProviderId,
                        previousFormat,
                        providerId,
                        format,
                        communityTextGgufSelection.reason(),
                        communityTextGgufSelection.cacheHit(),
                        communityTextGgufSelection.cacheKind());
            }
            DirectSafetensorRoutePolicy.RouteValidation gemma3RouteValidation =
                    DirectSafetensorRoutePolicy.validateGemma3ExecutionRoute(providerId, modelId, finalLocalPath);
            if (!gemma3RouteValidation.allowed()) {
                gemma3RouteValidation.messages().forEach(System.err::println);
                return;
            }

            List<String> normalizedInputImages = normalizedInputImagePaths();
            if (routeReportJson) {
                String routeExecutionProviderId = routeExecutionProviderId(providerId, finalLocalPath);
                RunnerRouteReport effectiveRouteReport =
                        runnerRouteReport.withEffectiveRoute(routeExecutionProviderId, format);
                if (direct) {
                    routeExecutionProviderId = "safetensor";
                    effectiveRouteReport = effectiveRouteReport.withEffectiveRoute(routeExecutionProviderId, format);
                }
                effectiveRouteReport = applyCachedBenchmarkRouteProfile(
                        effectiveRouteReport,
                        requestedModelRef,
                        modelId,
                        finalLocalPath);
                RoutePreflightReport routePreflight = RoutePreflightReport.evaluate(
                        requestedModelRef,
                        modelId,
                        finalLocalPath,
                        routeExecutionProviderId,
                        format,
                        routeReportAllowPull,
                        routeReportRequireLocal);
                routePreflight = DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        routePreflight,
                        routeExecutionProviderId,
                        modelId,
                        finalLocalPath,
                        !normalizedInputImages.isEmpty(),
                        ocrMode);
                printJsonPayload(RouteReportPayloads.routeReportPayload(
                        effectiveRouteReport,
                        requestedModelRef,
                        modelId,
                        finalLocalPath,
                        routeExecutionProviderId,
                        format,
                        routeReportAllowPull,
                        routeReportRepositoryResolutionSkipped,
                        routePreflight));
                requestProcessExit(routePreflight.exitCode());
                return;
            }

            if (listTtsVoices || listTtsVoicesJson) {
                requestProcessExit(printTtsVoices(modelId, finalLocalPath));
                return;
            }

            if (ttsDiagnostics) {
                requestProcessExit(printTtsDiagnostics(modelId, finalLocalPath));
                return;
            }

            if (ttsWarmup) {
                if (!ensureMossTtsCodecAvailableIfNeeded(modelId, finalLocalPath)) {
                    requestProcessExit(1);
                    return;
                }
                requestProcessExit(warmupMossTts(modelId, finalLocalPath));
                return;
            }

            if (ttsCachePrune || ttsCachePruneApply) {
                if (!ensureMossTtsCodecAvailableIfNeeded(modelId, finalLocalPath)) {
                    requestProcessExit(1);
                    return;
                }
                requestProcessExit(pruneMossTtsOptimizedCache(modelId, finalLocalPath));
                return;
            }

            if (onnxDiagnostics) {
                int diagnosticsExitCode =
                        printOnnxDiagnostics(requestedModelRef != null ? requestedModelRef : modelId, finalLocalPath);
                if (onnxVisionProbe || onnxPipelineProbe) {
                    diagnosticsExitCode = Math.max(diagnosticsExitCode,
                            printOnnxVisionProbe(
                                    requestedModelRef != null ? requestedModelRef : modelId,
                                    finalLocalPath,
                                    onnxPipelineProbe));
                }
                requestProcessExit(diagnosticsExitCode);
                return;
            }

            if (onnxVisionProbe || onnxPipelineProbe) {
                requestProcessExit(printOnnxVisionProbe(
                        requestedModelRef != null ? requestedModelRef : modelId,
                        finalLocalPath,
                        onnxPipelineProbe));
                return;
            }

            if (!validateInputImages(normalizedInputImages)) {
                requestProcessExit(1);
                return;
            }
            if ((ocrMode || !normalizedInputImages.isEmpty()) && (prompt == null || prompt.isBlank())) {
                prompt = "Extract all text from the image.";
            }
            boolean structuredOcrOutput =
                    (jsonMode
                            || ocrJsonOutput
                            || (ocrJsonOutputPath != null && !ocrJsonOutputPath.isBlank())
                            || (ocrOverlayOutputPath != null && !ocrOverlayOutputPath.isBlank()))
                            && (ocrMode || !normalizedInputImages.isEmpty());
            boolean effectiveStream = stream && !structuredOcrOutput;

            boolean promptlessExtensionRun = allowsPromptlessExtensionRun(modelId, finalLocalPath);
            if ((prompt == null || prompt.isBlank()) && !promptlessExtensionRun) {
                System.err.println("Error: --prompt is required for inference.");
                System.err.println("Use --list-voices, --voices-json, --tts-diagnostics, --tts-warmup, --tts-cache-prune, or --onnx-diagnostics to inspect, warm, or maintain without generating output.");
                System.err.println("For OCR/vision models, pass --image /path/to/image.png or --ocr with an image.");
                requestProcessExit(1);
                return;
            }
            if ((prompt == null || prompt.isBlank()) && promptlessExtensionRun) {
                prompt = "";
            }

            if (tryStandaloneLiteRtFastPath(finalLocalPath)) {
                return;
            }

            if (tryStandaloneGgufFastPath(true)) {
                return;
            }

            if (!ensureMossTtsCodecAvailableIfNeeded(modelId, finalLocalPath)) {
                requestProcessExit(1);
                return;
            }

            String executionProviderId = routeExecutionProviderId(providerId, finalLocalPath);
            RunnerRouteReport effectiveRunnerRouteReport =
                    runnerRouteReport.withEffectiveRoute(executionProviderId, format);
            forcedExecutionRouteMetadata = effectiveRunnerRouteReport.toMetadata();

            if (direct) {
                providerId = "safetensor";
                
                executionProviderId = "safetensor";
                effectiveRunnerRouteReport = effectiveRunnerRouteReport.withEffectiveRoute(executionProviderId, format);
                forcedExecutionRouteMetadata = effectiveRunnerRouteReport.toMetadata();
            } else if (executionProviderId != null && !executionProviderId.isEmpty()) {
                
            }
            effectiveRunnerRouteReport = applyCachedBenchmarkRouteProfile(
                    effectiveRunnerRouteReport,
                    requestedModelRef,
                    modelId,
                    finalLocalPath);
            forcedExecutionRouteMetadata = effectiveRunnerRouteReport.toMetadata();

            String litertPreflightFailure = detectUnsupportedLiteRtPreflight(executionProviderId, finalLocalPath);
            if (litertPreflightFailure != null) {
                if (!enableJsonSse) {
                    uiRenderer.printModelInfo(modelId, providerId, format, null, false);
                    printQuantizationInfo();
                }
                System.err.println(litertPreflightFailure);
                requestProcessExit();
                return;
            }

            DirectSafetensorRoutePolicy.RouteValidation gemma4UnifiedMultimodalRouteValidation =
                    DirectSafetensorRoutePolicy.validateGemma4UnifiedMultimodalExecutionRoute(
                            executionProviderId,
                            modelId,
                            finalLocalPath,
                            !normalizedInputImages.isEmpty(),
                            ocrMode);
            if (!gemma4UnifiedMultimodalRouteValidation.allowed()) {
                if (!enableJsonSse) {
                    uiRenderer.printModelInfo(modelId, providerId, format, null, false);
                    printQuantizationInfo();
                }
                gemma4UnifiedMultimodalRouteValidation.messages().forEach(System.err::println);
                requestProcessExit(1);
                return;
            }

            DirectSafetensorRoutePolicy.RouteValidation gemma4UnifiedRouteValidation =
                    DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                            executionProviderId, modelId, finalLocalPath);
            if (!gemma4UnifiedRouteValidation.allowed()) {
                if (!enableJsonSse) {
                    uiRenderer.printModelInfo(modelId, providerId, format, null, false);
                    printQuantizationInfo();
                }
                gemma4UnifiedRouteValidation.messages().forEach(System.err::println);
                requestProcessExit(1);
                return;
            }
            
            if (executionProviderId != null && !ensureProviderHealthy(executionProviderId)) {
                if ("litert".equalsIgnoreCase(executionProviderId)
                        && finalLocalPath != null
                        && !finalLocalPath.isBlank()
                        && tryStandaloneLiteRtExecution(finalLocalPath, startTime)) {
                    return;
                }
                return;
            }
            
            printCompatibilityHintBeforeInference();

            String requestModelTarget = shouldUseLocalModelPath(executionProviderId, finalLocalPath)
                    ? finalLocalPath
                    : modelId;
            List<tech.kayys.gollek.spi.tool.ToolDefinition> requestedTools =
                    CliInferenceFeatures.loadTools(toolFiles, mcpTools);
            Object requestedToolChoice = CliInferenceFeatures.parseToolChoice(toolChoice);
            CliInferenceFeatures.RagContext ragContext =
                    CliInferenceFeatures.loadRagContext(ragContexts, ragFiles, ragMaxChars, prompt);

            InferenceRequest.Builder requestBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .model(requestModelTarget)
                    .prompt(prompt)
                    .temperature(temperature)
                    .topP(topP)
                    .topK(topK)
                    .repeatPenalty(repeatPenalty)
                    .jsonMode(jsonMode)
                    .maxTokens(maxTokens)
                    .plugin(pluginId)
                    .streaming(effectiveStream);

            if (sessionId != null && !sessionId.isBlank()) {
                requestBuilder.sessionId(sessionId.trim());
            }

            if (customModelPathUsed && finalLocalPath != null) {
                requestBuilder.parameter("model_path", finalLocalPath);
            }
            if (requestedModelRef != null && !requestedModelRef.isBlank()) {
                requestBuilder.parameter("requested_model_ref", requestedModelRef);
            }
            if (onnxVariant != null && !onnxVariant.isBlank()) {
                requestBuilder.parameter("onnx_variant", PaddleOcrVlOnnxPlanner.normalizeVariant(onnxVariant));
            }

            if (mirostat > 0) {
                requestBuilder.mirostat(mirostat);
            }
            if (grammar != null && !grammar.isEmpty()) {
                requestBuilder.grammar(grammar);
            }

            if (quantizeStrategy != null && !quantizeStrategy.isBlank()) {
                requestBuilder.parameter("quantize_strategy", quantizeStrategy);
                requestBuilder.parameter("quantize_bits", quantizeBits);
            }

            if (quantizeKv != null && !quantizeKv.equalsIgnoreCase("none")) {
                requestBuilder.parameter("kv_cache_quant", quantizeKv);
            }

            if (runtimeProfile) {
                requestBuilder.parameter("profile", true);
                requestBuilder.parameter("onnx_profile", true);
                requestBuilder.metadata("profile", true);
                requestBuilder.metadata("onnx_profile", true);
            }

            if (seed != null) requestBuilder.parameter("seed", seed);
            if (steps != null) requestBuilder.parameter("steps", steps);
            if (format != null) requestBuilder.parameter("format", format);
            requestBuilder.metadata(effectiveRunnerRouteReport.toMetadata());
            if (runner != null && !runner.isBlank()) {
                requestBuilder.parameter("runner", effectiveRunnerRouteReport.normalizedRunner());
                requestBuilder.metadata("runner", effectiveRunnerRouteReport.normalizedRunner());
            }
            if (featurePipelineId != null && !featurePipelineId.isBlank()) {
                String pipelineId = featurePipelineId.trim();
                requestBuilder.parameter("pipeline", pipelineId);
                requestBuilder.parameter("feature", pipelineId);
                requestBuilder.parameter("feature_pipeline", true);
            }
            if (guidanceScale != null) requestBuilder.parameter("guidance_scale", guidanceScale);
            if (outputPath != null) requestBuilder.parameter("output_path", outputPath);
            if (width != null) requestBuilder.parameter("width", width);
            if (height != null) requestBuilder.parameter("height", height);
            if (!normalizedInputImages.isEmpty()) {
                requestBuilder.parameter("image_path", normalizedInputImages.get(0));
                requestBuilder.parameter("input_image", normalizedInputImages.get(0));
                requestBuilder.parameter("vision_input_path", normalizedInputImages.get(0));
                requestBuilder.parameter("image_paths", normalizedInputImages);
                requestBuilder.parameter("vision_input_paths", normalizedInputImages);
            }
            if (ocrMode) {
                requestBuilder.parameter("task", "ocr");
                requestBuilder.parameter("ocr", true);
            } else if (!normalizedInputImages.isEmpty()) {
                requestBuilder.parameter("task", "vision");
            }
            if (ocrImageTokens != null) {
                requestBuilder.parameter("ocr_image_tokens", Math.max(0, ocrImageTokens));
            }
            if (ocrRawOutput) {
                requestBuilder.parameter("ocr_raw_output", true);
            }
            String effectiveTtsVoice = effectiveTtsVoiceSelector();
            if (effectiveTtsVoice != null && !effectiveTtsVoice.isBlank()) requestBuilder.parameter("voice", effectiveTtsVoice);
            if (ttsCodec != null && !ttsCodec.isBlank()) requestBuilder.parameter("tts_codec", resolveTtsCodecReference(ttsCodec));
            if (ttsMaxFrames != null) requestBuilder.parameter("tts_max_frames", ttsMaxFrames);
            if (ttsMaxSeconds != null) requestBuilder.parameter("tts_max_seconds", ttsMaxSeconds);
            AudioQualityPreset audioPreset = audioQualityPreset();
            String effectiveAudioFormat = effectiveAudioFormat(audioPreset);
            Integer effectiveAudioBitrateKbps = effectiveAudioBitrateKbps(audioPreset, effectiveAudioFormat);
            Integer effectiveFlacCompression = effectiveFlacCompression(audioPreset, effectiveAudioFormat);
            if (audioQuality != null && !audioQuality.isBlank()) {
                requestBuilder.parameter("audio_quality", audioQuality.trim().toLowerCase(Locale.ROOT));
            }
            if (effectiveAudioFormat != null && !effectiveAudioFormat.isBlank()) {
                requestBuilder.parameter("audio_format", effectiveAudioFormat);
            }
            if (effectiveAudioBitrateKbps != null) {
                requestBuilder.parameter("audio_bitrate_kbps", effectiveAudioBitrateKbps);
            }
            if (audioChannels != null && !audioChannels.isBlank()) {
                requestBuilder.parameter("audio_channels", audioChannels.toLowerCase(Locale.ROOT));
            }
            if (ttsCodecDecode != null && !ttsCodecDecode.isBlank()) {
                requestBuilder.parameter("tts_codec_decode", ttsCodecDecode.toLowerCase(Locale.ROOT));
            }
            if (randomTtsSeed) {
                requestBuilder.parameter("random_seed", true);
            }
            if (disableAudioPolish) {
                requestBuilder.parameter("audio_polish", false);
            }
            if (disableAudioNormalize) {
                requestBuilder.parameter("audio_normalize", false);
            }
            if (audioGainDb != null) {
                requestBuilder.parameter("audio_gain_db", audioGainDb);
            }
            if (audioPeakDb != null) {
                requestBuilder.parameter("audio_peak_db", audioPeakDb);
            }
            if (audioFadeMs != null) {
                requestBuilder.parameter("audio_fade_ms", audioFadeMs);
            }
            if (audioTrimSilence) {
                requestBuilder.parameter("audio_trim_silence", true);
            }
            if (audioTrimDbfs != null) {
                requestBuilder.parameter("audio_trim_threshold_dbfs", audioTrimDbfs);
            }
            if (audioTrimPaddingMs != null) {
                requestBuilder.parameter("audio_trim_padding_ms", audioTrimPaddingMs);
            }
            if (audioStreamChunkKb != null) {
                requestBuilder.parameter("audio_stream_chunk_kb", audioStreamChunkKb);
            }
            if (ttsStreamProgressFrames != null) {
                requestBuilder.parameter("tts_stream_progress_frames", ttsStreamProgressFrames);
            }
            if (disableTtsStreamProgress) {
                requestBuilder.parameter("tts_stream_progress", false);
            }
            if (ttsStreamPcm || liveAudio) {
                requestBuilder.parameter("tts_stream_pcm", true);
            }
            if (liveAudio) {
                requestBuilder.parameter("tts_live_audio", true);
            }
            if (disableLowLatencyAudio) {
                requestBuilder.parameter("tts_low_latency_audio", false);
            }
            if (effectiveFlacCompression != null) {
                requestBuilder.parameter("flac_compression", effectiveFlacCompression);
            }

            if (extension != null && !extension.isBlank()) {
                requestBuilder.parameter("output_format", extension.toLowerCase());
            }

            CliInferenceFeatures.applyTools(requestBuilder, requestedTools, requestedToolChoice);
            CliInferenceFeatures.applyRagMetadata(requestBuilder, ragContext, embeddingModel);

            boolean libtorchSafetensorCliBridge = "libtorch".equalsIgnoreCase(providerId)
                    && "safetensor".equalsIgnoreCase(executionProviderId);
            String effectiveSystemPrompt = effectiveRunSystemPrompt(executionProviderId, finalLocalPath);
            if (effectiveSystemPrompt != null && !effectiveSystemPrompt.isEmpty()) {
                requestBuilder.message(Message.system(effectiveSystemPrompt));
            }
            String ragSystemPrompt = CliInferenceFeatures.ragSystemPrompt(ragContext);
            if (ragSystemPrompt != null && !ragSystemPrompt.isBlank()) {
                requestBuilder.message(Message.system(ragSystemPrompt));
            }
            requestBuilder.message(Message.user(prompt));

            if (executionProviderId != null && !executionProviderId.isEmpty()) {
                requestBuilder.preferredProvider(executionProviderId);
            }

            if (libtorchSafetensorCliBridge) {
                requestBuilder.metadata("requested_provider", "libtorch");
                requestBuilder.metadata("effective_provider", "safetensor");
                requestBuilder.metadata("provider_bridge_mode", "cli_libtorch_to_safetensor");
                requestBuilder.metadata("provider_bridge_reason", "raw_safetensor_checkpoint");
            }

            requestBuilder.cacheBypass(noCache);

            InferenceRequest request = requestBuilder.build();

            if (!quietRouteResolutionOutput() && !enableJsonSse) {
                uiRenderer.printModelInfo(modelId, providerId, format, null, false);
                printQuantizationInfo();
                if (libtorchSafetensorCliBridge) {
                    forcedExecutionRouteMetadata = bridgeExecutionRouteMetadata();
                    forcedExecutionRouteMetadata =
                            mergeExecutionRouteMetadata(effectiveRunnerRouteReport.toMetadata(),
                                    forcedExecutionRouteMetadata);
                    printExecutionRouteInfo(forcedExecutionRouteMetadata);
                } else {
                    forcedExecutionRouteMetadata = effectiveRunnerRouteReport.toMetadata();
                }
            }

            if (shouldUseDirectSafetensorRunPath(providerId, finalLocalPath)) {
                DirectSafetensorRunProfile directProfile =
                        directSafetensorRunProfile(executionProviderId, finalLocalPath);
                boolean useCliDirectSystemPrompt =
                        DirectSafetensorRoutePolicy.shouldForwardSystemPromptToDirectPath(systemPrompt, directProfile);
                String explicitDirectSystemPrompt = useCliDirectSystemPrompt
                        ? effectiveSystemPrompt
                        : null;
                InferenceResponse response = runDirectSafetensorCompletion(finalLocalPath, prompt,
                        explicitDirectSystemPrompt);
                printResponse(response, startTime);
                requestProcessExit();
            } else if (effectiveStream && shouldUseDirectLiteRtStreamPath(executionProviderId, finalLocalPath)) {
                streamDirectWithProvider("litert", request, startTime);
                requestProcessExit();
            } else if (effectiveStream) {
                java.io.ByteArrayOutputStream imageBuffer = new java.io.ByteArrayOutputStream();
                StreamingAudioOutput audioOutput = new StreamingAudioOutput();
                LivePcmAudioSink liveAudioSink = new LivePcmAudioSink(liveAudio);
                CountDownLatch latch = new CountDownLatch(1);
                java.util.concurrent.atomic.AtomicReference<java.util.Map<String, Object>> metricsRef = new java.util.concurrent.atomic.AtomicReference<>();
                java.util.concurrent.atomic.AtomicBoolean streamFailed = new java.util.concurrent.atomic.AtomicBoolean(false);
                java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(0);
                java.util.concurrent.atomic.AtomicLong firstTokenTime = new java.util.concurrent.atomic.AtomicLong(0);
                java.util.concurrent.atomic.AtomicLong lastTokenTime = new java.util.concurrent.atomic.AtomicLong(0);
                long streamStartTime = System.currentTimeMillis();
                String streamFinalLocalPath = finalLocalPath;
                java.util.concurrent.atomic.AtomicBoolean llamaFallbackAttempted = new java.util.concurrent.atomic.AtomicBoolean(false);

                // Helper to create a stream subscriber (used for primary and fallback)
                java.util.function.Consumer<io.smallrye.mutiny.Multi<tech.kayys.gollek.spi.inference.StreamingInferenceChunk>> subscribeToStream = (multi) -> {
                    multi.subscribe().with(
                            chunk -> {
                                if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                    metricsRef.set(chunk.metadata());
                                }

                                if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.IMAGE) {
                                    if (chunk.imageDeltaBase64() != null) {
                                        try {
                                            byte[] decoded = java.util.Base64.getDecoder().decode(chunk.imageDeltaBase64());
                                            imageBuffer.write(decoded);
                                        } catch (Exception imageDecodeFailure) {}
                                    }
                                    return;
                                }

                                if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.AUDIO) {
                                    handleAudioChunk(chunk, audioOutput, liveAudioSink);
                                    return;
                                }

                                String delta = chunk.getDelta();
                                if (delta != null) {
                                    boolean progressDelta = delta.startsWith("[") && delta.contains("]");
                                    if (!progressDelta && !delta.isEmpty()) {
                                        long now = System.currentTimeMillis();
                                        firstTokenTime.compareAndSet(0, now);
                                        lastTokenTime.set(now);
                                    }
                                    if (progressDelta) {
                                        if (!enableJsonSse) {
                                            System.out.print("\r" + ChatUIRenderer.CYAN + delta + ChatUIRenderer.RESET + "  ");
                                            System.out.flush();
                                        }
                                    } else if (enableJsonSse) {
                                        printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                                    } else {
                                        System.out.print(delta);
                                        System.out.flush();
                                    }
                                    tokenCount.incrementAndGet();
                                }
                            },
                            error -> {
                                // Attempt fallback to llama.cpp for streaming if available and not yet tried
                                try {
                                    if (!llamaFallbackAttempted.get() && pluginChecker != null && pluginChecker.hasProvider("llamacpp")) {
                                        llamaFallbackAttempted.set(true);
                                        System.err.println("Warning: stream provider failed: " + (error == null ? "unknown" : error.getMessage()));
                                        System.err.println("Attempting streaming fallback to provider 'llamacpp' (llama.cpp)...");
                                        try {  } catch (Throwable ignore) {}
                                        // start fallback stream
                                        try {
                                            java.util.concurrent.atomic.AtomicBoolean fallbackStreamFailed = new java.util.concurrent.atomic.AtomicBoolean(false);
                                            sdk.streamCompletion(request).subscribe().with(
                                                    chunk -> {
                                                        if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                                            metricsRef.set(chunk.metadata());
                                                        }
                                                        if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.IMAGE) {
                                                            if (chunk.imageDeltaBase64() != null) {
                                                                try {
                                                                    byte[] decoded = java.util.Base64.getDecoder().decode(chunk.imageDeltaBase64());
                                                                    imageBuffer.write(decoded);
                                                                } catch (Exception imageDecodeFailure) {}
                                                            }
                                                            return;
                                                        }
                                                        if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.AUDIO) {
                                                            handleAudioChunk(chunk, audioOutput, liveAudioSink);
                                                            return;
                                                        }
                                                        String delta = chunk.getDelta();
                                                        if (delta != null) {
                                                            boolean progressDelta = delta.startsWith("[") && delta.contains("]");
                                                            if (!progressDelta && !delta.isEmpty()) {
                                                                long now = System.currentTimeMillis();
                                                                firstTokenTime.compareAndSet(0, now);
                                                                lastTokenTime.set(now);
                                                            }
                                                            if (progressDelta) {
                                                                if (!enableJsonSse) {
                                                                    System.out.print("\r" + ChatUIRenderer.CYAN + delta + ChatUIRenderer.RESET + "  ");
                                                                    System.out.flush();
                                                                }
                                                            } else if (enableJsonSse) {
                                                                printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                                                            } else {
                                                                System.out.print(delta);
                                                                System.out.flush();
                                                            }
                                                            tokenCount.incrementAndGet();
                                                        }
                                                    },
                                                    fallbackError -> {
                                                        // fallback failed -> close sinks and report
                                                        liveAudioSink.close();
                                                        audioOutput.closeQuietly();
                                                        fallbackStreamFailed.set(true);
                                                        uiRenderer.printError(fallbackError == null ? "Fallback stream failed" : fallbackError.getMessage(), false);
                                                        printProviderHintFromError(fallbackError);
                                                        latch.countDown();
                                                    },
                                                    () -> {
                                                        liveAudioSink.close();
                                                        long duration = observedStreamDurationMillis(streamStartTime, System.currentTimeMillis(), lastTokenTime);
                                                        handleOutputs(imageBuffer, audioOutput, metricsRef.get());
                                                        double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                                                        Double ttftMs = ttftMillis(metricsRef.get(), streamStartTime, firstTokenTime);
                                                        Map<String, Object> streamMetrics = observedStreamMetrics(effectiveExecutionRouteMetadata(metricsRef.get()), tokenCount.get(), duration, ttftMs);
                                                        recordRouteBenchmarkProfile(request.getModel(), streamFinalLocalPath, streamMetrics, tokenCount.get(), duration, audioOutput.hasOutput());
                                                        if (enableJsonSse) {
                                                            printOpenAiSseFinal(request.getRequestId(), request.getModel());
                                                        } else {
                                                            printCompletionStatsForOutput(tokenCount.get(), duration, tps, ttftMs, streamMetrics, audioOutput.hasOutput());
                                                        }
                                                        latch.countDown();
                                                    });
                                        } catch (Throwable fallbackStartErr) {
                                            liveAudioSink.close();
                                            audioOutput.closeQuietly();
                                            streamFailed.set(true);
                                            uiRenderer.printError(fallbackStartErr.getMessage(), false);
                                            printProviderHintFromError(fallbackStartErr);
                                            latch.countDown();
                                        }
                                        return;
                                    }
                                } catch (Exception probeErr) {
                                    // ignore probing errors, fall through to failure closing
                                }

                                // No fallback available or fallback already attempted: close resources and finish
                                liveAudioSink.close();
                                audioOutput.closeQuietly();
                                streamFailed.set(true);
                                uiRenderer.printError(error == null ? "Stream failed" : error.getMessage(), false);
                                printProviderHintFromError(error);
                                latch.countDown();
                            },
                            () -> {
                                liveAudioSink.close();
                                long duration = observedStreamDurationMillis(
                                        streamStartTime, System.currentTimeMillis(), lastTokenTime);
                                handleOutputs(imageBuffer, audioOutput, metricsRef.get());
                                double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                                Double ttftMs = ttftMillis(metricsRef.get(), streamStartTime, firstTokenTime);
                                Map<String, Object> streamMetrics = observedStreamMetrics(
                                        effectiveExecutionRouteMetadata(metricsRef.get()),
                                        tokenCount.get(),
                                        duration,
                                        ttftMs);
                                recordRouteBenchmarkProfile(
                                        request.getModel(),
                                        streamFinalLocalPath,
                                        streamMetrics,
                                        tokenCount.get(),
                                        duration,
                                        audioOutput.hasOutput());
                                if (enableJsonSse) {
                                    printOpenAiSseFinal(request.getRequestId(), request.getModel());
                                } else {
                                    printCompletionStatsForOutput(
                                            tokenCount.get(),
                                            duration,
                                            tps,
                                            ttftMs,
                                            streamMetrics,
                                            audioOutput.hasOutput());
                                }
                                latch.countDown();
                            });
                    };

                // Start primary stream
                subscribeToStream.accept(sdk.streamCompletion(request));

                try {
                    latch.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    liveAudioSink.close();
                    audioOutput.closeQuietly();
                    streamFailed.set(true);
                }
                requestProcessExit(streamFailed.get() ? 1 : 0);
            } else {
                try {
                    InferenceResponse response = sdk.createCompletion(request);
                    printResponse(response, startTime);
                    requestProcessExit();
                } catch (Throwable createEx) {
                    // Attempt fallback to local llama.cpp provider if available
                    if (tryFallbackToLlamaCpp(createEx, request, startTime)) {
                        return;
                    }
                    throw createEx;
                }
            }

        } catch (Throwable e) {
            System.err.println("\n[FATAL] RunCommand failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private boolean tryFallbackToLlamaCpp(Throwable cause, InferenceRequest request, long startTime) {
        try {
            if (pluginChecker == null) return false;
            if (!pluginChecker.hasProvider("llamacpp")) return false;
        } catch (Exception e) {
            return false;
        }

        System.err.println("Warning: primary provider failed: " + (cause == null ? "unknown" : cause.getMessage()));
        System.err.println("Attempting fallback to provider 'llamacpp' (llama.cpp)...");
        try {
            try {
                
            } catch (Throwable ignore) {}
            InferenceResponse fallbackResponse = sdk.createCompletion(request);
            printResponse(fallbackResponse, startTime);
            requestProcessExit();
            return true;
        } catch (Throwable fallbackErr) {
            System.err.println("Fallback to 'llamacpp' failed: " + fallbackErr.getMessage());
            return false;
        }
    }

    private void printMetalRuntimeStatus() {
        var status = CliMetalRuntime.status();
        if (status.errorMessage() != null) {
            System.out.println(ChatUIRenderer.YELLOW + "⚠️  Unable to verify Metal runtime status: " + status.errorMessage() + ChatUIRenderer.RESET);
        } else if (status.active()) {
            System.out.println(ChatUIRenderer.GREEN + "✓ Metal native backend active: " + status.deviceName() + ChatUIRenderer.RESET);
        } else {
            System.out.println(ChatUIRenderer.YELLOW + "⚠️  Metal native backend NOT active (device: " + status.deviceName() + "), using CPU fallback path" + ChatUIRenderer.RESET);
        }
    }

    private boolean isMetalNativeRuntimeActive() {
        return CliMetalRuntime.isNativeActive();
    }

    private void ensureMetalRuntimeInitialized() {
        CliMetalRuntime.initialize();
    }

    private boolean repairOrRejectPositionalArgs() {
        if (positionalArgs == null || positionalArgs.isEmpty()) {
            return true;
        }

        if (positionalArgs.size() == 1
                && outputPath == null
                && ttsVoice != null
                && ttsVoice.contains("--output")) {
            String repairedVoice = ttsVoice.substring(0, ttsVoice.indexOf("--output")).trim();
            if (!repairedVoice.isBlank()) {
                outputPath = positionalArgs.get(0);
                ttsVoice = repairedVoice;
                positionalArgs = List.of();
                System.err.println("Warning: detected a missing space before --output; interpreting as "
                        + "--voice \"" + repairedVoice + "\" --output " + outputPath);
                return true;
            }
        }

        System.err.println("Error: unexpected positional argument"
                + (positionalArgs.size() == 1 ? "" : "s")
                + ": " + String.join(" ", positionalArgs));
        if (ttsVoice != null && ttsVoice.contains("--output")) {
            System.err.println("Hint: add a space before --output, for example:");
            System.err.println("  --voice \"" + ttsVoice.substring(0, ttsVoice.indexOf("--output")).trim()
                    + "\" --output " + positionalArgs.get(0));
        }
        return false;
    }

    private boolean verifyAudioMetadataSidecar() {
        try {
            Path sidecarPath = Path.of(audioMetadataVerifyPath);
            if (!Files.isRegularFile(sidecarPath)) {
                System.err.println("Error: audio metadata file not found: " + audioMetadataVerifyPath);
                return false;
            }
            JsonNode root = new ObjectMapper().readTree(sidecarPath.toFile());
            if (root == null || !root.isObject()) {
                System.err.println("Error: invalid audio metadata file: " + audioMetadataVerifyPath);
                return false;
            }
            String expectedSha256 = jsonText(root, "output_sha256");
            if (expectedSha256 == null || expectedSha256.isBlank()) {
                System.err.println("Error: metadata does not contain output_sha256; regenerate the sidecar with a newer Gollek build.");
                return false;
            }

            Path audioPath = outputPathForMetadataVerification(sidecarPath, root, "audio");
            if (!Files.isRegularFile(audioPath)) {
                System.err.println("Error: audio output not found: " + audioPath);
                return false;
            }

            long actualBytes = Files.size(audioPath);
            Long expectedBytes = parseOptionalReplayLong(jsonText(root, "output_size_bytes"));
            if (expectedBytes != null && expectedBytes != actualBytes) {
                System.err.println("Audio metadata verification failed: byte size mismatch.");
                System.err.println("  expected: " + expectedBytes);
                System.err.println("  actual:   " + actualBytes);
                return false;
            }

            String actualSha256 = sha256Hex(audioPath);
            if (!expectedSha256.equalsIgnoreCase(actualSha256)) {
                System.err.println("Audio metadata verification failed: SHA-256 mismatch.");
                System.err.println("  expected: " + expectedSha256);
                System.err.println("  actual:   " + actualSha256);
                return false;
            }

            System.out.println(ChatUIRenderer.GREEN + "✓ Audio metadata verified: " + ChatUIRenderer.RESET
                    + audioPath.toAbsolutePath().normalize());
            System.out.println(ChatUIRenderer.DIM + "Bytes: " + actualBytes
                    + ", SHA-256: " + actualSha256
                    + ChatUIRenderer.RESET);
            return true;
        } catch (Exception e) {
            System.err.println("Error: failed to verify audio metadata: " + e.getMessage());
            return false;
        }
    }

    private boolean refreshAudioMetadataSidecar() {
        try {
            Path sidecarPath = Path.of(audioMetadataRefreshPath);
            if (!Files.isRegularFile(sidecarPath)) {
                System.err.println("Error: audio metadata file not found: " + audioMetadataRefreshPath);
                return false;
            }
            ObjectMapper mapper = new ObjectMapper();
            JsonNode rootNode = mapper.readTree(sidecarPath.toFile());
            if (rootNode == null || !rootNode.isObject()) {
                System.err.println("Error: invalid audio metadata file: " + audioMetadataRefreshPath);
                return false;
            }

            com.fasterxml.jackson.databind.node.ObjectNode root =
                    (com.fasterxml.jackson.databind.node.ObjectNode) rootNode;
            Path audioPath = outputPathForMetadataVerification(sidecarPath, root, "audio");
            if (!Files.isRegularFile(audioPath)) {
                System.err.println("Error: audio output not found: " + audioPath);
                return false;
            }

            long sizeBytes = Files.size(audioPath);
            String sha256 = sha256Hex(audioPath);
            root.put("schema_version", 2);
            root.put("refreshed_at", Instant.now().toString());
            root.put("output_path", audioPath.toAbsolutePath().normalize().toString());
            root.put("output_size_bytes", sizeBytes);
            root.put("output_sha256", sha256);

            String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(root);
            Files.writeString(sidecarPath, json);
            System.out.println(ChatUIRenderer.GREEN + "✓ Audio metadata refreshed: " + ChatUIRenderer.RESET
                    + sidecarPath.toAbsolutePath().normalize());
            System.out.println(ChatUIRenderer.DIM + "Output: "
                    + audioPath.toAbsolutePath().normalize()
                    + ", bytes: " + sizeBytes
                    + ", SHA-256: " + sha256
                    + ChatUIRenderer.RESET);
            return true;
        } catch (Exception e) {
            System.err.println("Error: failed to refresh audio metadata: " + e.getMessage());
            return false;
        }
    }

    private boolean verifyImageMetadataSidecar() {
        try {
            Path sidecarPath = Path.of(imageMetadataVerifyPath);
            if (!Files.isRegularFile(sidecarPath)) {
                System.err.println("Error: image metadata file not found: " + imageMetadataVerifyPath);
                return false;
            }
            JsonNode root = new ObjectMapper().readTree(sidecarPath.toFile());
            if (root == null || !root.isObject()) {
                System.err.println("Error: invalid image metadata file: " + imageMetadataVerifyPath);
                return false;
            }
            String expectedSha256 = jsonText(root, "output_sha256");
            if (expectedSha256 == null || expectedSha256.isBlank()) {
                System.err.println("Error: metadata does not contain output_sha256; refresh the sidecar with --image-metadata-refresh.");
                return false;
            }

            Path imagePath = outputPathForMetadataVerification(sidecarPath, root, "image");
            if (!Files.isRegularFile(imagePath)) {
                System.err.println("Error: image output not found: " + imagePath);
                return false;
            }

            long actualBytes = Files.size(imagePath);
            Long expectedBytes = parseOptionalReplayLong(jsonText(root, "output_size_bytes"));
            if (expectedBytes != null && expectedBytes != actualBytes) {
                System.err.println("Image metadata verification failed: byte size mismatch.");
                System.err.println("  expected: " + expectedBytes);
                System.err.println("  actual:   " + actualBytes);
                return false;
            }

            String actualSha256 = sha256Hex(imagePath);
            if (!expectedSha256.equalsIgnoreCase(actualSha256)) {
                System.err.println("Image metadata verification failed: SHA-256 mismatch.");
                System.err.println("  expected: " + expectedSha256);
                System.err.println("  actual:   " + actualSha256);
                return false;
            }

            System.out.println(ChatUIRenderer.GREEN + "✓ Image metadata verified: " + ChatUIRenderer.RESET
                    + imagePath.toAbsolutePath().normalize());
            System.out.println(ChatUIRenderer.DIM + "Bytes: " + actualBytes
                    + ", SHA-256: " + actualSha256
                    + ChatUIRenderer.RESET);
            return true;
        } catch (Exception e) {
            System.err.println("Error: failed to verify image metadata: " + e.getMessage());
            return false;
        }
    }

    private boolean refreshImageMetadataSidecar() {
        try {
            Path sidecarPath = Path.of(imageMetadataRefreshPath);
            if (!Files.isRegularFile(sidecarPath)) {
                System.err.println("Error: image metadata file not found: " + imageMetadataRefreshPath);
                return false;
            }
            ObjectMapper mapper = new ObjectMapper();
            JsonNode rootNode = mapper.readTree(sidecarPath.toFile());
            if (rootNode == null || !rootNode.isObject()) {
                System.err.println("Error: invalid image metadata file: " + imageMetadataRefreshPath);
                return false;
            }

            com.fasterxml.jackson.databind.node.ObjectNode root =
                    (com.fasterxml.jackson.databind.node.ObjectNode) rootNode;
            Path imagePath = outputPathForMetadataVerification(sidecarPath, root, "image");
            if (!Files.isRegularFile(imagePath)) {
                System.err.println("Error: image output not found: " + imagePath);
                return false;
            }

            long sizeBytes = Files.size(imagePath);
            String sha256 = sha256Hex(imagePath);
            root.put("type", "gollek_image_metadata");
            root.put("schema_version", 1);
            root.put("refreshed_at", Instant.now().toString());
            root.put("output_path", imagePath.toAbsolutePath().normalize().toString());
            root.put("output_size_bytes", sizeBytes);
            root.put("output_sha256", sha256);

            String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(root);
            Files.writeString(sidecarPath, json);
            System.out.println(ChatUIRenderer.GREEN + "✓ Image metadata refreshed: " + ChatUIRenderer.RESET
                    + sidecarPath.toAbsolutePath().normalize());
            System.out.println(ChatUIRenderer.DIM + "Output: "
                    + imagePath.toAbsolutePath().normalize()
                    + ", bytes: " + sizeBytes
                    + ", SHA-256: " + sha256
                    + ChatUIRenderer.RESET);
            return true;
        } catch (Exception e) {
            System.err.println("Error: failed to refresh image metadata: " + e.getMessage());
            return false;
        }
    }

    private Path outputPathForMetadataVerification(Path sidecarPath, JsonNode root, String mediaLabel) {
        if (outputPath != null && !outputPath.isBlank()) {
            return Path.of(outputPath).toAbsolutePath().normalize();
        }
        String storedOutputPath = jsonText(root, "output_path");
        if (storedOutputPath == null || storedOutputPath.isBlank()) {
            throw new IllegalArgumentException("metadata does not contain output_path; pass --output <"
                    + mediaLabel + "-file> to verify manually");
        }
        Path audioPath = Path.of(storedOutputPath);
        if (!audioPath.isAbsolute()) {
            Path parent = sidecarPath.toAbsolutePath().normalize().getParent();
            audioPath = (parent == null ? audioPath : parent.resolve(audioPath)).normalize();
        }
        return audioPath.toAbsolutePath().normalize();
    }

    private boolean loadAudioMetadataReplayIfRequested() {
        if (audioMetadataReplayPath == null || audioMetadataReplayPath.isBlank()) {
            return true;
        }
        try {
            Path sidecarPath = Path.of(audioMetadataReplayPath);
            if (!Files.isRegularFile(sidecarPath)) {
                System.err.println("Error: audio metadata replay file not found: " + audioMetadataReplayPath);
                return false;
            }
            JsonNode root = new ObjectMapper().readTree(sidecarPath.toFile());
            if (root == null || !root.isObject()) {
                System.err.println("Error: invalid audio metadata replay file: " + audioMetadataReplayPath);
                return false;
            }

            boolean outputExplicit = outputPath != null && !outputPath.isBlank();
            String replayOutputPath = null;
            JsonNode replayArgs = root.get("replay_args");
            if (replayArgs != null && replayArgs.isArray()) {
                replayOutputPath = applyAudioMetadataReplayArgs(replayArgs);
            }
            String payloadOutputPath = applyAudioMetadataReplayPayload(root);
            if (replayOutputPath == null || replayOutputPath.isBlank()) {
                replayOutputPath = payloadOutputPath;
            }
            if (!outputExplicit && (outputPath == null || outputPath.isBlank())) {
                outputPath = defaultAudioReplayOutputPath(replayOutputPath, audioFormat);
            }
            if (!ttsVoicesJsonOutput()) {
                System.out.println("Loaded TTS replay metadata: " + sidecarPath.toAbsolutePath().normalize());
            }
            return true;
        } catch (Exception e) {
            System.err.println("Error: failed to load audio metadata replay file: " + e.getMessage());
            return false;
        }
    }

    private String applyAudioMetadataReplayArgs(JsonNode replayArgs) {
        String replayOutputPath = null;
        for (int i = 0; i < replayArgs.size(); i++) {
            String flag = jsonNodeText(replayArgs.get(i));
            if (flag == null || !flag.startsWith("--")) {
                continue;
            }
            switch (flag) {
                case "--model" -> {
                    applyReplayModelTarget(replayArgValue(replayArgs, ++i, flag));
                }
                case "--modelDir" -> {
                    if (modelDir == null || modelDir.isBlank()) {
                        modelDir = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--modelFile" -> {
                    if (modelFile == null || modelFile.isBlank()) {
                        modelFile = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--model-path" -> {
                    if (modelPath == null || modelPath.isBlank()) {
                        modelPath = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--provider" -> {
                    if (providerId == null || providerId.isBlank()) {
                        providerId = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--runner" -> {
                    if (runner == null || runner.isBlank()) {
                        runner = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--prompt" -> {
                    if (prompt == null || prompt.isBlank()) {
                        prompt = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--voice", "--tts-voice" -> {
                    if (ttsVoice == null || ttsVoice.isBlank()) {
                        ttsVoice = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--audio-format", "--tts-audio-format" -> {
                    if (audioFormat == null || audioFormat.isBlank()) {
                        audioFormat = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--seed" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (seed == null && !randomTtsSeed) {
                        seed = parseReplayLong(flag, value);
                    }
                }
                case "--output", "-o" -> {
                    replayOutputPath = replayArgValue(replayArgs, ++i, flag);
                }
                case "--audio-quality", "--tts-audio-quality" -> {
                    if (audioQuality == null || audioQuality.isBlank()) {
                        audioQuality = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--audio-bitrate", "--audio-bitrate-kbps", "--tts-bitrate", "--tts-bitrate-kbps" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (audioBitrateKbps == null) {
                        audioBitrateKbps = parseReplayInt(flag, value);
                    }
                }
                case "--audio-channels", "--tts-audio-channels" -> {
                    if (audioChannels == null || audioChannels.isBlank()) {
                        audioChannels = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--tts-codec-decode", "--codec-decode" -> {
                    if (ttsCodecDecode == null || ttsCodecDecode.isBlank()) {
                        ttsCodecDecode = replayArgValue(replayArgs, ++i, flag);
                    } else {
                        i++;
                    }
                }
                case "--tts-max-frames", "--max-frames" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (ttsMaxFrames == null) {
                        ttsMaxFrames = parseReplayInt(flag, value);
                    }
                }
                case "--tts-max-seconds", "--max-seconds", "--tts-duration" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (ttsMaxSeconds == null) {
                        ttsMaxSeconds = parseReplayDouble(flag, value);
                    }
                }
                case "--audio-gain-db", "--tts-gain-db" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (audioGainDb == null) {
                        audioGainDb = parseReplayDouble(flag, value);
                    }
                }
                case "--audio-peak-db", "--audio-peak-dbfs", "--tts-peak-db" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (audioPeakDb == null) {
                        audioPeakDb = parseReplayDouble(flag, value);
                    }
                }
                case "--audio-fade-ms", "--tts-fade-ms" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (audioFadeMs == null) {
                        audioFadeMs = parseReplayDouble(flag, value);
                    }
                }
                case "--audio-trim-db", "--audio-trim-dbfs", "--tts-trim-db" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (audioTrimDbfs == null) {
                        audioTrimDbfs = parseReplayDouble(flag, value);
                    }
                }
                case "--audio-trim-padding-ms", "--tts-trim-padding-ms" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (audioTrimPaddingMs == null) {
                        audioTrimPaddingMs = parseReplayDouble(flag, value);
                    }
                }
                case "--flac-compression" -> {
                    String value = replayArgValue(replayArgs, ++i, flag);
                    if (flacCompression == null) {
                        flacCompression = parseReplayInt(flag, value);
                    }
                }
                case "--no-audio-polish", "--no-tts-audio-polish" -> disableAudioPolish = true;
                case "--no-audio-normalize", "--no-tts-audio-normalize" -> disableAudioNormalize = true;
                case "--audio-trim-silence", "--tts-trim-silence" -> audioTrimSilence = true;
                default -> {
                    if (replayFlagTakesValue(flag) && i + 1 < replayArgs.size()) {
                        i++;
                    }
                }
            }
        }
        return replayOutputPath;
    }

    private String applyAudioMetadataReplayPayload(JsonNode root) {
        applyReplayModelTarget(jsonText(root, "model"));
        if (providerId == null || providerId.isBlank()) {
            providerId = jsonText(root, "provider");
        }
        if (runner == null || runner.isBlank()) {
            runner = jsonText(root, "runner");
        }
        if (prompt == null || prompt.isBlank()) {
            prompt = jsonText(root, "prompt");
        }
        if (ttsVoice == null || ttsVoice.isBlank()) {
            ttsVoice = jsonText(root, "voice_selector");
        }
        if (audioFormat == null || audioFormat.isBlank()) {
            audioFormat = jsonText(root, "audio_format");
        }

        JsonNode metadata = root.get("metadata");
        if (metadata != null && metadata.isObject()) {
            if (seed == null && !randomTtsSeed) {
                String metadataSeed = jsonText(metadata, "tts_seed");
                if (metadataSeed != null && !metadataSeed.isBlank()) {
                    seed = parseReplayLong("metadata.tts_seed", metadataSeed);
                }
            }
            if (audioFormat == null || audioFormat.isBlank()) {
                audioFormat = jsonText(metadata, "audio_format");
            }
            if (audioBitrateKbps == null) {
                String bitrate = jsonText(metadata, "audio_bitrate_kbps");
                if (bitrate != null && !bitrate.isBlank()) {
                    audioBitrateKbps = parseReplayInt("metadata.audio_bitrate_kbps", bitrate);
                }
            }
        }
        return jsonText(root, "output_path");
    }

    private void applyReplayModelTarget(String value) {
        if (value == null || value.isBlank()
                || (modelId != null && !modelId.isBlank())
                || (modelDir != null && !modelDir.isBlank())
                || (modelFile != null && !modelFile.isBlank())
                || (modelPath != null && !modelPath.isBlank())) {
            return;
        }
        try {
            Path path = Path.of(value);
            if (Files.isDirectory(path)) {
                modelDir = value;
                return;
            }
            if (Files.isRegularFile(path)) {
                modelFile = value;
                return;
            }
        } catch (Exception ignored) {
        }
        modelId = value;
    }

    private static boolean replayFlagTakesValue(String flag) {
        return Set.of(
                "--model", "--modelDir", "--modelFile", "--model-path", "--provider", "--runner", "--prompt",
                "--voice", "--tts-voice", "--audio-format", "--tts-audio-format", "--seed", "--output", "-o",
                "--audio-quality", "--tts-audio-quality", "--audio-bitrate", "--audio-bitrate-kbps",
                "--tts-bitrate", "--tts-bitrate-kbps", "--audio-channels", "--tts-audio-channels",
                "--tts-codec-decode", "--codec-decode", "--tts-max-frames", "--max-frames",
                "--tts-max-seconds", "--max-seconds", "--tts-duration", "--audio-gain-db", "--tts-gain-db",
                "--audio-peak-db", "--audio-peak-dbfs", "--tts-peak-db", "--audio-fade-ms", "--tts-fade-ms",
                "--audio-trim-db", "--audio-trim-dbfs", "--tts-trim-db", "--audio-trim-padding-ms",
                "--tts-trim-padding-ms", "--flac-compression")
                .contains(flag);
    }

    private static String replayArgValue(JsonNode replayArgs, int index, String flag) {
        if (index >= replayArgs.size()) {
            throw new IllegalArgumentException("missing value for " + flag);
        }
        String value = jsonNodeText(replayArgs.get(index));
        if (value == null) {
            throw new IllegalArgumentException("missing value for " + flag);
        }
        return value;
    }

    private static String jsonText(JsonNode node, String field) {
        if (node == null || field == null) {
            return null;
        }
        return jsonNodeText(node.get(field));
    }

    private static String jsonNodeText(JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) {
            return null;
        }
        return node.asText();
    }

    private static Long parseReplayLong(String option, String value) {
        try {
            return Long.parseLong(value.trim());
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid " + option + " value in replay metadata: " + value);
        }
    }

    private static Long parseOptionalReplayLong(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Long.parseLong(value.trim());
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid integer value in replay metadata: " + value);
        }
    }

    private static Integer parseReplayInt(String option, String value) {
        try {
            return Integer.parseInt(value.trim());
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid " + option + " value in replay metadata: " + value);
        }
    }

    private static Double parseReplayDouble(String option, String value) {
        try {
            return Double.parseDouble(value.trim());
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid " + option + " value in replay metadata: " + value);
        }
    }

    private String defaultAudioReplayOutputPath(String originalOutputPath, String replayAudioFormat) {
        String ext = normalizeExtension(replayAudioFormat);
        Path originalPath = originalOutputPath == null || originalOutputPath.isBlank()
                ? Path.of("output." + (ext.isBlank() ? "wav" : ext))
                : Path.of(originalOutputPath);
        Path replayPath = appendReplaySuffix(originalPath, ext);
        return uniquifyPath(replayPath).toString();
    }

    private Path appendReplaySuffix(Path path, String fallbackExt) {
        String filename = path.getFileName() == null ? "output" : path.getFileName().toString();
        String ext = pathExtension(path);
        if (ext.isBlank()) {
            ext = normalizeExtension(fallbackExt);
        }
        String base = filename;
        if (!ext.isBlank() && filename.toLowerCase(Locale.ROOT).endsWith("." + ext.toLowerCase(Locale.ROOT))) {
            base = filename.substring(0, filename.length() - ext.length() - 1);
        }
        String replayFilename = base + "-replay" + (ext.isBlank() ? "" : "." + ext);
        return path.resolveSibling(replayFilename);
    }

    private Path uniquifyPath(Path path) {
        if (!Files.exists(path)) {
            return path;
        }
        String filename = path.getFileName() == null ? "output" : path.getFileName().toString();
        String ext = pathExtension(path);
        String base = filename;
        if (!ext.isBlank() && filename.toLowerCase(Locale.ROOT).endsWith("." + ext.toLowerCase(Locale.ROOT))) {
            base = filename.substring(0, filename.length() - ext.length() - 1);
        }
        for (int i = 2; i < 1000; i++) {
            Path candidate = path.resolveSibling(base + "-" + i + (ext.isBlank() ? "" : "." + ext));
            if (!Files.exists(candidate)) {
                return candidate;
            }
        }
        return path;
    }

    private boolean validateTtsAudioCliOptions() {
        boolean ok = true;
        if (audioFormat != null && !audioFormat.isBlank()) {
            ok &= validateChoice(
                    "--audio-format",
                    audioFormat,
                    Set.of("wav", "flac", "mp3"),
                    "wav, flac, mp3");
        }
        if (audioQuality != null && !audioQuality.isBlank()) {
            ok &= validateChoice(
                    "--audio-quality",
                    audioQuality,
                    Set.of("auto", "default", "compact", "small", "speech", "balanced", "standard", "high",
                            "quality", "lossless", "archive", "flac"),
                    "compact, speech, balanced, high, lossless");
        }
        if (audioChannels != null && !audioChannels.isBlank()) {
            ok &= validateChoice(
                    "--audio-channels",
                    audioChannels,
                    Set.of("auto", "1", "mono", "single", "2", "stereo", "dual-mono", "dual_mono",
                            "native", "source", "raw"),
                    "auto, mono, stereo, dual-mono, native");
        }
        if (ttsCodecDecode != null && !ttsCodecDecode.isBlank()) {
            ok &= validateChoice(
                    "--tts-codec-decode",
                    ttsCodecDecode,
                    Set.of("auto", "full", "full_decode", "decode_full", "stream", "streaming", "step",
                            "decode_step"),
                    "auto, streaming, full");
        }
        if (audioBitrateKbps != null) {
            ok &= validateIntRange("--audio-bitrate", audioBitrateKbps, 8, 320, "kbps");
        }
        if (flacCompression != null) {
            ok &= validateIntRange("--flac-compression", flacCompression, 0, 8, "");
        }
        if (ttsMaxFrames != null) {
            ok &= validateIntRange("--tts-max-frames", ttsMaxFrames, 1, 4096, "frames");
        }
        if (ttsMaxSeconds != null) {
            ok &= validateDoubleRange("--tts-max-seconds", ttsMaxSeconds, 0.01, 300.0, "seconds");
        }
        if (audioStreamChunkKb != null) {
            ok &= validateIntRange("--audio-stream-chunk-kb", audioStreamChunkKb, 4, 1024, "KiB");
        }
        if (ttsStreamProgressFrames != null) {
            ok &= validateIntRange("--tts-stream-progress-frames", ttsStreamProgressFrames, 1, 4096, "frames");
        }
        if (audioFadeMs != null) {
            ok &= validateDoubleRange("--audio-fade-ms", audioFadeMs, 0.0, 10_000.0, "ms");
        }
        if (audioTrimDbfs != null) {
            ok &= validateDoubleRange("--audio-trim-db", audioTrimDbfs, -120.0, -0.1, "dBFS");
        }
        if (audioTrimPaddingMs != null) {
            ok &= validateDoubleRange("--audio-trim-padding-ms", audioTrimPaddingMs, 0.0, 5_000.0, "ms");
        }
        if (audioPeakDb != null) {
            ok &= validateDoubleRange("--audio-peak-db", audioPeakDb, -60.0, 0.0, "dBFS");
        }
        if (audioGainDb != null) {
            ok &= validateDoubleRange("--audio-gain-db", audioGainDb, -60.0, 60.0, "dB");
        }
        return ok;
    }

    private AudioQualityPreset audioQualityPreset() {
        if (audioQuality == null || audioQuality.isBlank()) {
            return null;
        }
        return switch (audioQuality.trim().toLowerCase(Locale.ROOT)) {
            case "auto", "default" -> null;
            case "compact", "small" -> new AudioQualityPreset("mp3", 64, null);
            case "speech" -> new AudioQualityPreset("mp3", 96, null);
            case "balanced", "standard" -> new AudioQualityPreset("mp3", 128, null);
            case "high", "quality" -> new AudioQualityPreset("mp3", 192, null);
            case "lossless", "archive", "flac" -> new AudioQualityPreset("flac", null, 8);
            default -> null;
        };
    }

    private String effectiveAudioFormat(AudioQualityPreset preset) {
        if (audioFormat != null && !audioFormat.isBlank()) {
            return audioFormat.trim().toLowerCase(Locale.ROOT);
        }
        return preset == null ? null : preset.format();
    }

    private Integer effectiveAudioBitrateKbps(AudioQualityPreset preset, String effectiveAudioFormat) {
        if (audioBitrateKbps != null) {
            return audioBitrateKbps;
        }
        if (preset == null || preset.bitrateKbps() == null) {
            return null;
        }
        return "mp3".equalsIgnoreCase(effectiveAudioFormat) ? preset.bitrateKbps() : null;
    }

    private Integer effectiveFlacCompression(AudioQualityPreset preset, String effectiveAudioFormat) {
        if (flacCompression != null) {
            return flacCompression;
        }
        if (preset == null || preset.flacCompression() == null) {
            return null;
        }
        return "flac".equalsIgnoreCase(effectiveAudioFormat) ? preset.flacCompression() : null;
    }

    private boolean validateChoice(String option, String value, Set<String> allowed, String displayAllowed) {
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if (allowed.contains(normalized)) {
            return true;
        }
        System.err.println("Error: invalid " + option + " value: " + value);
        System.err.println("Supported values: " + displayAllowed);
        return false;
    }

    private boolean validateIntRange(String option, int value, int min, int max, String unit) {
        if (value >= min && value <= max) {
            return true;
        }
        String suffix = unit == null || unit.isBlank() ? "" : " " + unit;
        System.err.println("Error: " + option + " must be between " + min + " and " + max + suffix
                + " (got " + value + ").");
        return false;
    }

    private boolean validateDoubleRange(String option, double value, double min, double max, String unit) {
        if (Double.isFinite(value) && value >= min && value <= max) {
            return true;
        }
        String suffix = unit == null || unit.isBlank() ? "" : " " + unit;
        System.err.println("Error: " + option + " must be between "
                + formatRangeNumber(min) + " and " + formatRangeNumber(max) + suffix
                + " (got " + value + ").");
        return false;
    }

    private static String formatRangeNumber(double value) {
        if (Math.rint(value) == value) {
            return String.format(Locale.ROOT, "%.0f", value);
        }
        return String.format(Locale.ROOT, "%.2f", value);
    }

    private String effectiveTtsVoiceSelector() {
        if (ttsVoice != null && !ttsVoice.isBlank()) {
            return ttsVoice.trim();
        }

        List<String> parts = new ArrayList<>();
        if (ttsVoiceLanguage != null && !ttsVoiceLanguage.isBlank()) {
            parts.add(ttsVoiceLanguage.trim());
        }
        if (ttsVoiceGender != null && !ttsVoiceGender.isBlank()) {
            parts.add(ttsVoiceGender.trim());
        }
        return parts.isEmpty() ? null : String.join(" ", parts);
    }

    private int printTtsVoices(String modelId, String finalLocalPath) {
        Optional<Path> ttsDir = resolveMossTtsDirectory(finalLocalPath, modelId);
        if (ttsDir.isEmpty()) {
            System.err.println("Error: --list-voices is only available for local MOSS TTS ONNX model directories.");
            System.err.println("Resolved model path: " + (finalLocalPath == null || finalLocalPath.isBlank()
                    ? "(none)"
                    : finalLocalPath));
            return 1;
        }

        Path manifestPath = ttsDir.get().resolve("browser_poc_manifest.json");
        try {
            JsonNode manifest = new ObjectMapper().readTree(manifestPath.toFile());
            JsonNode voices = manifest.path("builtin_voices");
            if (!voices.isArray() || voices.isEmpty()) {
                System.err.println("Error: MOSS TTS manifest does not include builtin voices: " + manifestPath);
                return 1;
            }

            TtsVoiceListFilter filter = ttsVoiceListFilter();
            List<JsonNode> matchedVoices = new ArrayList<>();
            for (JsonNode voice : voices) {
                if (ttsVoiceMatchesListFilter(voice, filter)) {
                    matchedVoices.add(voice);
                }
            }

            if (ttsVoicesJsonOutput()) {
                printTtsVoicesJson(ttsDir.get(), filter, voices, matchedVoices);
                return 0;
            }

            System.out.println("TTS voices for: " + ttsDir.get().toAbsolutePath());
            if (filter.hasFilters()) {
                System.out.println("Filter: " + filter.describe());
                System.out.println("Matched: " + matchedVoices.size() + "/" + voices.size());
            }
            System.out.println();
            if (matchedVoices.isEmpty()) {
                System.out.println("No voices matched this filter.");
            } else {
                System.out.printf("%-7s %-12s %-8s %-11s %-18s %s%n",
                        "DEFAULT", "VOICE", "LANG", "GENDER", "GROUP", "DISPLAY NAME");
                System.out.println("------------------------------------------------------------------------------------------");
                for (JsonNode voice : matchedVoices) {
                    System.out.printf("%-7s %-12s %-8s %-11s %-18s %s%n",
                            voice == voices.get(0) ? "*" : "",
                            ttsVoiceId(voice),
                            ttsVoiceLanguageLabel(voice),
                            ttsVoiceGenderLabel(voice),
                            truncateDisplay(ttsVoiceGroup(voice), 18),
                            ttsVoiceDisplayName(voice));
                }
            }
            System.out.println();
            System.out.println("Use one of the VOICE values with --voice, for example: --voice Junhao");
            System.out.println("Omit --voice, or use --voice auto, to pick a voice from the prompt language.");
            System.out.println("Language aliases are supported too: --voice en, --voice jp, --voice zh.");
            System.out.println("You can combine language and gender, for example: --voice en-female or --voice jp female.");
            System.out.println("Use --voice default for the first preset, or a unique display-name part such as --voice bitter.");
            System.out.println("Filter this table with --voice-search, --voice-lang, --voice-gender, or --voice.");
            return 0;
        } catch (Exception e) {
            System.err.println("Error: Failed to read MOSS TTS voices: " + e.getMessage());
            return 1;
        }
    }

    private boolean ttsVoicesJsonOutput() {
        return jsonMode || listTtsVoicesJson;
    }

    private boolean quietRouteResolutionOutput() {
        return ttsVoicesJsonOutput() || routeReportJson || noBanner;
    }

    boolean shouldAllowRepositoryResolutionDuringRouteReport() {
        return !routeReportJson || routeReportAllowPull;
    }

    private String routeExecutionProviderId(String provider, String localPath) {
        if ("libtorch".equalsIgnoreCase(provider)
                && localPath != null
                && !localPath.isBlank()
                && (isSafetensorCheckpointDir(localPath) || isSafetensorWeightFile(localPath))) {
            return "safetensor";
        }
        return provider;
    }

    private void printTtsVoicesJson(
            Path ttsDir,
            TtsVoiceListFilter filter,
            JsonNode voices,
            List<JsonNode> matchedVoices) throws Exception {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "moss_tts_voices");
        payload.put("model_path", ttsDir.toAbsolutePath().normalize().toString());
        payload.put("total", voices.size());
        payload.put("matched", matchedVoices.size());
        payload.put("default_voice", voices.isEmpty() ? null : ttsVoiceId(voices.get(0)));
        payload.put("filter", ttsVoiceFilterJson(filter));

        List<Map<String, Object>> voiceItems = new ArrayList<>();
        for (JsonNode voice : matchedVoices) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("default", voice.equals(voices.get(0)));
            item.put("voice", ttsVoiceId(voice));
            item.put("language", ttsVoiceLanguageLabel(voice));
            item.put("gender", ttsVoiceGenderLabel(voice));
            item.put("group", ttsVoiceGroup(voice));
            item.put("display_name", ttsVoiceDisplayName(voice));
            item.put("selector", ttsVoiceSelectorExample(voice));
            voiceItems.add(item);
        }
        payload.put("voices", voiceItems);
        payload.put("examples", List.of(
                "--voice " + (voices.isEmpty() ? "Junhao" : ttsVoiceId(voices.get(0))),
                "--voice en",
                "--voice jp female",
                "--voice-search bitter"));
        System.out.println(new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(payload));
    }

    private Map<String, Object> ttsVoiceFilterJson(TtsVoiceListFilter filter) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("active", filter != null && filter.hasFilters());
        value.put("language", filter == null ? null : filter.language());
        value.put("gender", filter == null ? null : filter.gender());
        value.put("terms", filter == null ? List.of() : filter.terms());
        value.put("description", filter == null ? "(none)" : filter.describe());
        return value;
    }

    private String ttsVoiceSelectorExample(JsonNode voice) {
        String language = ttsVoiceLanguageLabel(voice);
        String gender = ttsVoiceGenderLabel(voice);
        if (!"other".equals(language) && !"unspecified".equals(gender)) {
            return language + " " + gender;
        }
        if (!"other".equals(language)) {
            return language;
        }
        return ttsVoiceId(voice);
    }

    private TtsVoiceListFilter ttsVoiceListFilter() {
        String language = normalizeTtsVoiceLanguageFilter(ttsVoiceLanguage);
        String gender = normalizeTtsVoiceGenderFilter(ttsVoiceGender);
        List<String> terms = new ArrayList<>();

        if (ttsVoiceGender != null && !ttsVoiceGender.isBlank() && gender == null) {
            terms.addAll(ttsVoiceFilterTerms(ttsVoiceGender));
        }

        String selectorText = ttsVoiceSearch != null && !ttsVoiceSearch.isBlank()
                ? ttsVoiceSearch
                : ttsVoice;
        for (String term : ttsVoiceFilterTerms(selectorText)) {
            String termLanguage = normalizeTtsVoiceLanguageFilter(term);
            if (termLanguage != null) {
                language = termLanguage;
                continue;
            }
            String termGender = normalizeTtsVoiceGenderFilter(term);
            if (termGender != null) {
                gender = termGender;
                continue;
            }
            if (!Set.of("auto", "any", "all").contains(term)) {
                terms.add(term);
            }
        }

        return new TtsVoiceListFilter(language, gender, List.copyOf(terms));
    }

    private List<String> ttsVoiceFilterTerms(String value) {
        String normalized = normalizeTtsVoiceFilterText(value);
        if (normalized.isBlank()) {
            return List.of();
        }
        return List.of(normalized.split(" "));
    }

    private boolean ttsVoiceMatchesListFilter(JsonNode voice, TtsVoiceListFilter filter) {
        if (filter == null || !filter.hasFilters()) {
            return true;
        }
        if (filter.language() != null && !filter.language().equals(ttsVoiceLanguageLabel(voice))) {
            return false;
        }
        if (filter.gender() != null && !filter.gender().equals(ttsVoiceGenderLabel(voice))) {
            return false;
        }
        if (!filter.terms().isEmpty()) {
            String searchText = ttsVoiceSearchText(voice);
            for (String term : filter.terms()) {
                if (!searchText.contains(term)) {
                    return false;
                }
            }
        }
        return true;
    }

    private String normalizeTtsVoiceLanguageFilter(String value) {
        String normalized = normalizeTtsVoiceFilterText(value);
        return switch (normalized) {
            case "en", "eng", "english" -> "en";
            case "jp", "ja", "jpn", "japanese" -> "jp";
            case "cn", "zh", "zho", "chi", "chinese", "mandarin" -> "zh";
            default -> null;
        };
    }

    private String normalizeTtsVoiceGenderFilter(String value) {
        String normalized = normalizeTtsVoiceFilterText(value);
        return switch (normalized) {
            case "f", "female", "woman", "women", "girl" -> "female";
            case "m", "male", "man", "men", "boy" -> "male";
            default -> null;
        };
    }

    private String normalizeTtsVoiceFilterText(String value) {
        if (value == null) {
            return "";
        }
        return value.trim()
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^\\p{L}\\p{N}]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private int printOnnxDiagnostics(String modelId, String finalLocalPath) {
        System.out.println("ONNX diagnostics");
        System.out.println("================");
        System.out.println("Requested model: " + nullToDash(modelId));
        System.out.println("Resolved path: " + nullToDash(finalLocalPath));

        Path diagnosticPath = resolveOnnxDiagnosticPath(modelId, finalLocalPath);
        if (diagnosticPath == null) {
            System.err.println();
            System.err.println("Error: no local ONNX path was resolved for diagnostics.");
            System.err.println("Pull the model first or pass --modelDir /path/to/repository.");
            return 1;
        }

        OnnxModelDiagnostics.Report report = OnnxModelDiagnostics.inspect(diagnosticPath);
        System.out.println("Diagnostic path: " + displayPath(diagnosticPath));
        System.out.println("Model directory: " + (report.modelDir() == null ? "-" : displayPath(report.modelDir())));
        System.out.println("Pipeline: " + nullToDash(report.pipelineType()));
        System.out.println("Model type: " + nullToDash(report.modelType()));
        if (!report.architectures().isEmpty()) {
            System.out.println("Architectures: " + String.join(", ", report.architectures()));
        }
        if (!report.capabilities().isEmpty()) {
            System.out.println("Capabilities: " + String.join(", ", report.capabilities()));
        }
        if (report.genAiInfo() != null) {
            printGenAiDiagnostics(report.genAiInfo());
        }
        int exitCode = report.graphs().isEmpty() ? 1 : 0;

        System.out.println();
        System.out.println("Graphs");
        System.out.println("------");
        if (report.graphs().isEmpty()) {
            System.out.println("(none)");
        } else {
            for (OnnxModelDiagnostics.GraphInfo graph : report.graphs()) {
                List<String> tags = new ArrayList<>();
                tags.add(graph.role());
                tags.add(graph.quantization());
                if (graph.externalData()) {
                    tags.add("external-data");
                }
                System.out.println("- " + graph.relativePath()
                        + " [" + String.join(", ", tags) + ", " + formatBytes(graph.sizeBytes()) + "]");
                printOnnxGraphIo(graph);
            }
        }

        System.out.println();
        System.out.println("Sidecars");
        System.out.println("--------");
        System.out.println(report.sidecars().isEmpty() ? "(none)" : formatDiagnosticList(report.sidecars(), 12));

        if (report.isPaddleOcrVl()) {
            exitCode = Math.max(exitCode, printPaddleOcrVlPreflight(report));
        }

        if (!report.warnings().isEmpty()) {
            System.out.println();
            System.out.println("Warnings");
            System.out.println("--------");
            for (String warning : report.warnings()) {
                System.out.println("- " + warning);
            }
        }

        if (!report.recommendations().isEmpty()) {
            System.out.println();
            System.out.println("Next steps");
            System.out.println("----------");
            for (String recommendation : report.recommendations()) {
                System.out.println("- " + recommendation);
            }
        }

        System.out.println();
        System.out.println(exitCode == 0 ? "Status: inspected" : "Status: attention needed");
        return exitCode;
    }

    private void printGenAiDiagnostics(OnnxModelDiagnostics.GenAiInfo info) {
        List<String> details = new ArrayList<>();
        if (info.type() != null && !info.type().isBlank()) {
            details.add("type=" + info.type());
        }
        if (info.layers() > 0) {
            details.add("layers=" + info.layers());
        }
        if (info.attentionHeads() > 0) {
            details.add("heads=" + info.attentionHeads());
        }
        if (info.keyValueHeads() > 0) {
            details.add("kv_heads=" + info.keyValueHeads());
        }
        if (info.headSize() > 0) {
            details.add("head_size=" + info.headSize());
        }
        if (info.contextLength() > 0) {
            details.add("ctx=" + info.contextLength());
        }
        if (info.vocabSize() > 0) {
            details.add("vocab=" + info.vocabSize());
        }
        if (!details.isEmpty()) {
            System.out.println("ORT GenAI: " + String.join(", ", details));
        }
        System.out.println("GenAI IO: inputs="
                + info.inputIdsName() + ", " + info.attentionMaskName() + ", " + info.positionIdsName()
                + "; kv=" + info.pastKeyNameTemplate() + " / " + info.pastValueNameTemplate()
                + "; outputs=" + info.logitsName()
                + ", " + info.presentKeyNameTemplate() + " / " + info.presentValueNameTemplate());
    }

    private void printOnnxGraphIo(OnnxModelDiagnostics.GraphInfo graph) {
        String ioStatus = graph.ioStatus();
        if (ioStatus != null && !ioStatus.isBlank() && !"parsed".equalsIgnoreCase(ioStatus)) {
            System.out.println("  io: " + ioStatus);
        }
        if (!graph.inputs().isEmpty()) {
            System.out.println("  inputs: " + formatTensorIoSummary(graph.inputs()));
        }
        if (!graph.outputs().isEmpty()) {
            System.out.println("  outputs: " + formatTensorIoSummary(graph.outputs()));
        }
    }

    private String formatTensorIoSummary(List<OnnxModelDiagnostics.TensorIoInfo> tensors) {
        Map<String, List<OnnxModelDiagnostics.TensorIoInfo>> grouped = new LinkedHashMap<>();
        List<OnnxModelDiagnostics.TensorIoInfo> direct = new ArrayList<>();
        for (OnnxModelDiagnostics.TensorIoInfo tensor : tensors) {
            String group = tensorIoGroup(tensor.name());
            if (group == null) {
                direct.add(tensor);
            } else {
                grouped.computeIfAbsent(group, ignored -> new ArrayList<>()).add(tensor);
            }
        }

        List<String> parts = new ArrayList<>();
        int maxDirect = grouped.isEmpty() ? 10 : 6;
        int directLimit = Math.min(direct.size(), maxDirect);
        for (int i = 0; i < directLimit; i++) {
            parts.add(formatTensorIo(direct.get(i)));
        }
        if (direct.size() > directLimit) {
            parts.add("+" + (direct.size() - directLimit) + " more tensors");
        }
        for (Map.Entry<String, List<OnnxModelDiagnostics.TensorIoInfo>> entry : grouped.entrySet()) {
            List<OnnxModelDiagnostics.TensorIoInfo> values = entry.getValue();
            if (values.isEmpty()) {
                continue;
            }
            parts.add(entry.getKey()
                    + " (" + values.size()
                    + " tensors, e.g. " + formatTensorIo(values.get(0)) + ")");
        }
        return parts.isEmpty() ? "(none)" : String.join("; ", parts);
    }

    private String tensorIoGroup(String name) {
        if (name == null) {
            return null;
        }
        if (name.startsWith("past_key_values.")) {
            return "past_key_values.*";
        }
        if (name.startsWith("present.")) {
            return "present.*";
        }
        if (name.startsWith("past.")) {
            return "past.*";
        }
        return null;
    }

    private String formatTensorIo(OnnxModelDiagnostics.TensorIoInfo tensor) {
        String shape = tensor.shape().isEmpty() ? "[]" : "[" + String.join(",", tensor.shape()) + "]";
        return tensor.name() + ":" + tensor.elementType() + shape;
    }

    private int printOnnxVisionProbe(String modelId, String finalLocalPath, boolean includePipelineGraphs) {
        System.out.println();
        System.out.println(includePipelineGraphs ? "PaddleOCR-VL pipeline probe" : "PaddleOCR-VL vision probe");
        System.out.println(includePipelineGraphs ? "--------------------------" : "-------------------------");

        Path diagnosticPath = resolveOnnxDiagnosticPath(modelId, finalLocalPath);
        if (diagnosticPath == null) {
            System.err.println("Error: no local ONNX path was resolved for vision probe.");
            return 1;
        }
        OnnxModelDiagnostics.Report report = OnnxModelDiagnostics.inspect(diagnosticPath);
        if (!report.isPaddleOcrVl() || report.modelDir() == null) {
            System.err.println("Error: --onnx-vision-probe is only available for local PaddleOCR-VL ONNX repositories.");
            System.err.println("Resolved path: " + displayPath(diagnosticPath));
            System.err.println("Detected pipeline: " + nullToDash(report.pipelineType()));
            return 1;
        }

        List<String> normalizedImages = normalizedInputImagePaths();
        if (!validateInputImages(normalizedImages)) {
            return 1;
        }
        if (normalizedImages.isEmpty()) {
            System.err.println("Error: --onnx-vision-probe requires --image /path/to/page.png.");
            return 1;
        }

        int threads = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
        try {
            if (includePipelineGraphs) {
                printOnnxPipelineProbeDetails(report, Path.of(normalizedImages.get(0)), threads);
                return 0;
            }
            PaddleOcrVlOnnxProbe.VisionEncoderProbeResult result =
                    PaddleOcrVlOnnxProbe.runVisionEncoder(
                            report.modelDir(),
                            Path.of(normalizedImages.get(0)),
                            onnxVariant,
                            threads);
            PaddleOcrVlOnnxPlanner.ImageTensor tensor = result.imageTensor();
            PaddleOcrVlOnnxPlanner.ImagePlan image = tensor.image();
            System.out.println("Graph variant: " + result.plan().graphs().variant()
                    + (onnxVariant == null || onnxVariant.isBlank() ? " (auto)" : " (requested " + onnxVariant + ")"));
            System.out.println("Vision graph: " + relativeOrDisplay(result.plan().modelDir(), result.graph()));
            System.out.println("Threads: " + threads);
            System.out.println("Image: " + displayPath(image.path())
                    + " " + image.originalWidth() + "x" + image.originalHeight()
                    + " -> " + image.resizedWidth() + "x" + image.resizedHeight()
                    + ", grid=[" + image.gridT() + "," + image.gridH() + "," + image.gridW() + "]"
                    + ", patches=" + image.patchCount());
            System.out.println("Input tensor: pixel_values" + formatShape(tensor.pixelValuesShape())
                    + ", image_grid_thw" + formatShape(tensor.imageGridThwShape())
                    + "=" + java.util.Arrays.toString(tensor.imageGridThw())
                    + ", checksum=" + tensor.checksum()
                    + ", range=[" + formatDiagnosticNumber(tensor.min())
                    + "," + formatDiagnosticNumber(tensor.max())
                    + "], mean=" + formatDiagnosticNumber(tensor.mean()));
            System.out.println("Session load: " + formatDuration(result.loadDuration())
                    + ", run: " + formatDuration(result.runDuration()));

            printProbeIo("Inputs", result.inputs());
            printProbeIo("Declared outputs", result.declaredOutputs());
            printProbeIo("Actual outputs", result.outputs());

            System.out.println("Status: vision_encoder executed");
            return 0;
        } catch (Exception e) {
            System.err.println("Error: PaddleOCR-VL "
                    + (includePipelineGraphs ? "pipeline" : "vision")
                    + " probe failed: " + e.getMessage());
            return 1;
        }
    }

    private void printOnnxPipelineProbeDetails(OnnxModelDiagnostics.Report report, Path imagePath, int threads) {
        int imageTokenLimit = onnxProbeImageTokens == null ? 64 : Math.max(0, onnxProbeImageTokens);
        int decodeTokens = onnxProbeDecodeTokens == null ? 8 : Math.max(0, onnxProbeDecodeTokens);
        String probePrompt = prompt == null || prompt.isBlank()
                ? "Extract all text from the image."
                : prompt;
        PaddleOcrVlOnnxProbe.PromptDecodeProbeResult result =
                PaddleOcrVlOnnxProbe.runPromptDecode(
                        report.modelDir(),
                        imagePath,
                        probePrompt,
                        onnxVariant,
                        imageTokenLimit,
                        decodeTokens,
                        threads);
        PaddleOcrVlOnnxPlanner.ImageTensor tensor = result.imageTensor();
        PaddleOcrVlOnnxPlanner.ImagePlan image = tensor.image();
        System.out.println("Graph variant: " + result.plan().graphs().variant()
                + (onnxVariant == null || onnxVariant.isBlank() ? " (auto)" : " (requested " + onnxVariant + ")"));
        System.out.println("Vision graph: " + relativeOrDisplay(result.plan().modelDir(), result.visionGraph()));
        System.out.println("Embedding graph: " + relativeOrDisplay(result.plan().modelDir(), result.embeddingGraph()));
        System.out.println("Decoder graph: " + relativeOrDisplay(result.plan().modelDir(), result.decoderGraph()));
        System.out.println("Threads: " + threads);
        System.out.println("Image: " + displayPath(image.path())
                + " " + image.originalWidth() + "x" + image.originalHeight()
                + " -> " + image.resizedWidth() + "x" + image.resizedHeight()
                + ", grid=[" + image.gridT() + "," + image.gridH() + "," + image.gridW() + "]"
                + ", patches=" + image.patchCount());
        System.out.println("Input tensor: pixel_values" + formatShape(tensor.pixelValuesShape())
                + ", image_grid_thw" + formatShape(tensor.imageGridThwShape())
                + "=" + java.util.Arrays.toString(tensor.imageGridThw())
                + ", checksum=" + tensor.checksum()
                + ", range=[" + formatDiagnosticNumber(tensor.min())
                + "," + formatDiagnosticNumber(tensor.max())
                + "], mean=" + formatDiagnosticNumber(tensor.mean()));
        System.out.println("Vision output: image_embeds[" + result.imageTokens()
                + "," + result.hiddenSize() + "]");
        System.out.println("Decoder prefill: sequence=" + result.sequenceLength()
                + ", image embeddings used=" + result.usedImageTokens() + "/" + result.imageTokens()
                + ", prompt_embeds" + formatShape(result.promptEmbedsShape()));
        System.out.println("Prompt tokens: input_ids=" + result.inputIds().length
                + ", image_token_id=" + result.imageTokenId()
                + ", image token positions=" + result.imageTokenPositions());
        System.out.println("Prompt preview: "
                + truncateDisplay(result.promptText().replaceAll("\\s+", " "), 120));
        if (imageTokenLimit > 0 && result.usedImageTokens() < result.imageTokens()) {
            System.out.println("Decoder prefill limit: " + imageTokenLimit
                    + " image embeddings (pass --onnx-probe-image-tokens 0 to use all)");
        }
        printDecoderLogitsSummary(result.decoderLogits());
        System.out.println("Durations: vision load=" + formatDuration(result.visionLoadDuration())
                + ", vision run=" + formatDuration(result.visionRunDuration())
                + ", embedding load=" + formatDuration(result.embeddingLoadDuration())
                + ", embedding run=" + formatDuration(result.embeddingRunDuration())
                + ", decoder load=" + formatDuration(result.decoderLoadDuration())
                + ", decoder prefill=" + formatDuration(result.decoderPrefillDuration())
                + ", decode loop=" + formatDuration(result.decoderDecodeDuration()));
        if (decodeTokens > 0) {
            System.out.println("Generated token ids: " + java.util.Arrays.toString(result.generatedTokenIds()));
            System.out.println("Decoded preview: "
                    + (result.decodedText() == null || result.decodedText().isBlank()
                            ? "(empty)"
                            : truncateDisplay(result.decodedText().replaceAll("\\s+", " "), 160)));
            PaddleOcrVlOnnxProbe.OcrPostProcessResult postProcess =
                    PaddleOcrVlOnnxProbe.postProcessOcrText(result.decodedText(), image);
            if (!postProcess.locations().isEmpty()) {
                System.out.println("Postprocessed OCR preview: "
                        + truncateDisplay(postProcess.displayText().replaceAll("\\s+", " "), 180));
            }
            System.out.println("Decode finish: " + result.finishReason()
                    + ", generated=" + result.generatedTokenIds().length + "/" + decodeTokens);
        }
        printProbeIo("Decoder inputs", result.decoderInputs(), 10);
        printProbeIo("Decoder declared outputs", result.decoderDeclaredOutputs(), 10);
        printProbeIo("Decoder actual outputs", result.decoderOutputs(), 12, true);
        System.out.println("Status: vision_encoder, tokenizer/chat prompt, embedding, decoder prefill, and greedy decode executed");
    }

    private void printDecoderLogitsSummary(PaddleOcrVlOnnxProbe.DecoderLogitsSummary summary) {
        if (summary == null) {
            return;
        }
        StringBuilder top = new StringBuilder();
        int[] tokenIds = summary.topTokenIds();
        float[] logits = summary.topLogits();
        if (tokenIds != null && logits != null) {
            for (int i = 0; i < Math.min(tokenIds.length, logits.length); i++) {
                if (tokenIds[i] < 0 || !Float.isFinite(logits[i])) {
                    continue;
                }
                if (top.length() > 0) {
                    top.append(", ");
                }
                top.append(tokenIds[i]).append('=').append(formatDiagnosticNumber(logits[i]));
            }
        }
        System.out.println("Decoder logits: " + summary.name()
                + " " + summary.type() + formatShape(summary.shape())
                + ", last token index=" + summary.sequenceIndex()
                + ", vocab=" + summary.vocabSize()
                + (top.length() > 0 ? ", top ids: " + top : ""));
    }

    private void printSelectedGraphIo(String title, Path graph, Path modelDir, int threads) {
        System.out.println(title + ":");
        if (graph == null || !Files.isRegularFile(graph)) {
            System.out.println("- missing: " + (graph == null ? "-" : relativeOrDisplay(modelDir, graph)));
            return;
        }
        PaddleOcrVlOnnxProbe.GraphIoProbeResult result =
                PaddleOcrVlOnnxProbe.inspectGraph(graph, threads);
        System.out.println("- graph: " + relativeOrDisplay(modelDir, result.graph())
                + ", load=" + formatDuration(result.loadDuration()));
        printProbeIo("  inputs", result.inputs(), 10);
        printProbeIo("  outputs", result.outputs(), 10);
    }

    private void printProbeIo(String title, List<PaddleOcrVlOnnxProbe.IoInfo> values) {
        printProbeIo(title, values, Integer.MAX_VALUE);
    }

    private void printProbeIo(String title, List<PaddleOcrVlOnnxProbe.IoInfo> values, int maxEntries) {
        printProbeIo(title, values, maxEntries, false);
    }

    private void printProbeIo(
            String title,
            List<PaddleOcrVlOnnxProbe.IoInfo> values,
            int maxEntries,
            boolean includeLogits) {
        System.out.println(title + ":");
        if (values == null || values.isEmpty()) {
            System.out.println("- (none)");
            return;
        }
        int limit = Math.min(values.size(), Math.max(0, maxEntries));
        for (int i = 0; i < limit; i++) {
            printProbeIoValue(values.get(i));
        }
        int highlighted = 0;
        if (includeLogits && values.size() > limit) {
            for (int i = limit; i < values.size(); i++) {
                if (isLogitsIo(values.get(i))) {
                    printProbeIoValue(values.get(i));
                    highlighted++;
                }
            }
        }
        int omitted = values.size() - limit - highlighted;
        if (omitted > 0) {
            System.out.println("- ... " + omitted + " more");
        }
    }

    private void printProbeIoValue(PaddleOcrVlOnnxProbe.IoInfo value) {
        System.out.println("- " + value.name()
                + ": " + value.kind()
                + ", " + value.type()
                + formatShape(value.shape())
                + formatElementCount(value.elements()));
    }

    private boolean isLogitsIo(PaddleOcrVlOnnxProbe.IoInfo value) {
        return value != null
                && value.name() != null
                && value.name().toLowerCase(Locale.ROOT).contains("logit");
    }

    private int printPaddleOcrVlPreflight(OnnxModelDiagnostics.Report report) {
        System.out.println();
        System.out.println("PaddleOCR-VL preflight");
        System.out.println("----------------------");
        if (report.modelDir() == null) {
            System.out.println("Model directory: missing");
            return 1;
        }

        List<String> normalizedImages = normalizedInputImagePaths();
        if (!validateInputImages(normalizedImages)) {
            return 1;
        }

        try {
            List<Path> imagePaths = normalizedImages.stream().map(Path::of).toList();
            PaddleOcrVlOnnxPlanner.Plan plan =
                    PaddleOcrVlOnnxPlanner.plan(report.modelDir(), imagePaths, onnxVariant);
            PaddleOcrVlOnnxPlanner.ProcessorConfig processor = plan.processor();
            System.out.println("Graph variant: " + plan.graphs().variant()
                    + (onnxVariant == null || onnxVariant.isBlank() ? " (auto)" : " (requested " + onnxVariant + ")"));
            System.out.println("Processor: patch=" + processor.patchSize()
                    + ", merge=" + processor.mergeSize()
                    + ", temporal=" + processor.temporalPatchSize()
                    + ", pixels=" + processor.minPixels() + ".." + processor.maxPixels());
            System.out.println("Selected graphs:");
            printSelectedPaddleGraph("vision_encoder", plan.graphs().visionEncoder(), plan.modelDir());
            printSelectedPaddleGraph("embedding", plan.graphs().embedding(), plan.modelDir());
            printSelectedPaddleGraph("decoder", plan.graphs().decoder(), plan.modelDir());
            if (!plan.graphs().warnings().isEmpty()) {
                for (String warning : plan.graphs().warnings()) {
                    System.out.println("- Warning: " + warning);
                }
            }

            if (plan.images().isEmpty()) {
                System.out.println("Images: none provided (pass --image /path/to/page.png for OCR image-grid preflight)");
            } else {
                System.out.println("Images:");
                for (PaddleOcrVlOnnxPlanner.ImagePlan image : plan.images()) {
                    System.out.println("- " + displayPath(image.path())
                            + ": " + image.originalWidth() + "x" + image.originalHeight()
                            + " -> " + image.resizedWidth() + "x" + image.resizedHeight()
                            + ", grid=[" + image.gridT() + "," + image.gridH() + "," + image.gridW() + "]"
                            + ", patches=" + image.patchCount()
                            + ", prompt image tokens=" + image.promptImageTokens());
                    PaddleOcrVlOnnxPlanner.ImageTensor tensor =
                            PaddleOcrVlOnnxPlanner.preprocessImage(image.path(), processor);
                    System.out.println("  tensor: pixel_values" + formatShape(tensor.pixelValuesShape())
                            + ", image_grid_thw" + formatShape(tensor.imageGridThwShape())
                            + "=" + java.util.Arrays.toString(tensor.imageGridThw())
                            + ", " + formatBytes(tensor.byteSize())
                            + ", checksum=" + tensor.checksum()
                            + ", range=[" + formatDiagnosticNumber(tensor.min())
                            + "," + formatDiagnosticNumber(tensor.max())
                            + "], mean=" + formatDiagnosticNumber(tensor.mean()));
                }
                System.out.println("Total prompt image tokens: " + plan.totalPromptImageTokens());
            }
            return 0;
        } catch (Exception e) {
            System.err.println("Error: PaddleOCR-VL preflight failed: " + e.getMessage());
            return 1;
        }
    }

    private void printSelectedPaddleGraph(String label, Path graph, Path modelDir) {
        String path = graph == null ? "-" : relativeOrDisplay(modelDir, graph);
        String status = graph != null && Files.isRegularFile(graph) ? "ok" : "missing";
        long bytes = graph == null ? 0L : onnxGraphBundleSize(graph);
        System.out.println("- " + label + ": " + path
                + " [" + status + (bytes > 0 ? ", " + formatBytes(bytes) : "") + "]");
    }

    private String relativeOrDisplay(Path base, Path path) {
        if (path == null) {
            return "-";
        }
        try {
            return base.toAbsolutePath().normalize().relativize(path.toAbsolutePath().normalize()).toString();
        } catch (Exception ignored) {
            return displayPath(path);
        }
    }

    private long onnxGraphBundleSize(Path graph) {
        if (graph == null) {
            return 0L;
        }
        long bytes = 0L;
        try {
            bytes += Files.size(graph);
        } catch (Exception ignored) {
            // Best-effort diagnostic.
        }
        Path parent = graph.getParent();
        String name = graph.getFileName() == null ? "" : graph.getFileName().toString();
        if (parent == null || name.isBlank()) {
            return bytes;
        }
        for (Path candidate : List.of(
                parent.resolve(name + ".data"),
                parent.resolve(name.replaceFirst("\\.onnx$", ".onnx.data")),
                parent.resolve(name.replaceFirst("\\.onnx$", ".data")))) {
            if (!Files.isRegularFile(candidate)) {
                continue;
            }
            try {
                bytes += Files.size(candidate);
            } catch (Exception ignored) {
                // Best-effort diagnostic.
            }
        }
        return bytes;
    }

    private String formatShape(long[] shape) {
        return shape == null ? "[]" : java.util.Arrays.toString(shape).replace(" ", "");
    }

    private String formatDiagnosticNumber(double value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }

    private String formatDuration(Duration duration) {
        if (duration == null) {
            return "-";
        }
        double millis = duration.toNanos() / 1_000_000.0;
        if (millis < 1000.0) {
            return String.format(Locale.ROOT, "%.1f ms", millis);
        }
        return String.format(Locale.ROOT, "%.2f s", millis / 1000.0);
    }

    private String formatElementCount(long elements) {
        return elements < 0 ? "" : ", elements=" + elements;
    }

    private Path resolveOnnxDiagnosticPath(String modelId, String finalLocalPath) {
        for (String candidate : List.of(finalLocalPath, modelId)) {
            Path direct = existingCliPath(candidate);
            if (direct != null) {
                return direct;
            }
            if (candidate == null || candidate.isBlank()) {
                continue;
            }
            try {
                Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(candidate, "onnx");
                if (indexed.isEmpty()) {
                    indexed = LocalModelIndex.find(candidate);
                }
                if (indexed.isPresent() && indexed.get().path != null) {
                    Path indexedPath = existingCliPath(indexed.get().path);
                    if (indexedPath != null) {
                        return indexedPath;
                    }
                }
            } catch (Exception ignored) {
                // Continue with the next candidate.
            }
        }
        return null;
    }

    private List<String> normalizedInputImagePaths() {
        List<String> images = new ArrayList<>();
        addInputImage(images, inputImagePath);
        if (inputImagePaths != null) {
            for (String image : inputImagePaths) {
                addInputImage(images, image);
            }
        }
        return List.copyOf(images);
    }

    private void addInputImage(List<String> images, String value) {
        if (value == null || value.isBlank()) {
            return;
        }
        Path path = expandCliPath(value.trim()).toAbsolutePath().normalize();
        String normalized = path.toString();
        if (!images.contains(normalized)) {
            images.add(normalized);
        }
    }

    private boolean validateInputImages(List<String> images) {
        boolean ok = true;
        for (String image : images) {
            Path path = Path.of(image);
            if (!Files.isRegularFile(path)) {
                System.err.println("Error: input image not found: " + image);
                ok = false;
            }
        }
        return ok;
    }

    private Path existingCliPath(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            Path path = expandCliPath(value.trim()).toAbsolutePath().normalize();
            return Files.exists(path) ? path : null;
        } catch (Exception ignored) {
            return null;
        }
    }

    private Path expandCliPath(String value) {
        if (value == null || value.isBlank()) {
            return Path.of("");
        }
        if (value.equals("~")) {
            return Path.of(System.getProperty("user.home", "~"));
        }
        if (value.startsWith("~/") || value.startsWith("~\\")) {
            return Path.of(System.getProperty("user.home", "~")).resolve(value.substring(2));
        }
        return Path.of(value);
    }

    private String formatDiagnosticList(List<String> values, int limit) {
        if (values == null || values.isEmpty()) {
            return "(none)";
        }
        int safeLimit = Math.max(1, limit);
        if (values.size() <= safeLimit) {
            return String.join(", ", values);
        }
        return String.join(", ", values.subList(0, safeLimit))
                + ", ... +" + (values.size() - safeLimit) + " more";
    }

    private String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        double value = bytes;
        String[] units = { "B", "KiB", "MiB", "GiB", "TiB" };
        int unit = 0;
        while (value >= 1024.0 && unit < units.length - 1) {
            value /= 1024.0;
            unit++;
        }
        return String.format(Locale.ROOT, "%.1f %s", value, units[unit]);
    }

    private int printTtsDiagnostics(String modelId, String finalLocalPath) {
        if (jsonMode) {
            return printTtsDiagnosticsJson(modelId, finalLocalPath);
        }
        int exitCode = 0;
        System.out.println("MOSS TTS diagnostics");
        System.out.println("===================");
        System.out.println("Requested model: " + nullToDash(modelId));
        System.out.println("Resolved path: " + nullToDash(finalLocalPath));

        Optional<Path> ttsDir = resolveMossTtsDirectory(finalLocalPath, modelId);
        if (ttsDir.isEmpty()) {
            System.err.println();
            System.err.println("Error: resolved model is not a local MOSS TTS ONNX directory.");
            System.err.println("Expected files: browser_poc_manifest.json and tts_browser_onnx_meta.json");
            return 1;
        }

        Path modelDir = ttsDir.get().toAbsolutePath().normalize();
        System.out.println("TTS directory: " + displayPath(modelDir));
        Path manifestPath = modelDir.resolve("browser_poc_manifest.json");
        Path ttsMetaPath = modelDir.resolve("tts_browser_onnx_meta.json");
        System.out.println("Manifest: " + diagnosticFileStatus(manifestPath));
        System.out.println("TTS metadata: " + diagnosticFileStatus(ttsMetaPath));

        JsonNode manifest = null;
        try {
            manifest = new ObjectMapper().readTree(manifestPath.toFile());
            JsonNode voices = manifest.path("builtin_voices");
            int voiceCount = voices.isArray() ? voices.size() : 0;
            System.out.println("Voices: " + voiceCount + (voiceCount == 0 ? " (missing)" : ""));
            if (voiceCount > 0) {
                System.out.println("Voice groups: " + ttsVoiceGroupSummary(voices));
                System.out.println("Default voice: " + ttsVoiceId(voices.get(0))
                        + " (" + ttsVoiceDisplayName(voices.get(0)) + ")");
            } else {
                exitCode = 1;
            }
        } catch (Exception e) {
            System.err.println("Manifest read error: " + e.getMessage());
            exitCode = 1;
        }

        String selector = effectiveTtsVoiceSelector();
        if (selector != null && !selector.isBlank()) {
            System.out.println("Requested voice selector: " + selector);
        }
        if (prompt != null && !prompt.isBlank()) {
            System.out.println("Prompt preview: " + truncateDisplay(prompt.replaceAll("\\s+", " "), 80));
        }

        System.out.println();
        System.out.println("Companion codec");
        System.out.println("---------------");
        Optional<Path> codecDir = diagnosticCodecDirectory();
        if (codecDir.isPresent()) {
            Path dir = codecDir.get().toAbsolutePath().normalize();
            System.out.println("Codec directory: " + displayPath(dir));
            System.out.println("Codec metadata: " + diagnosticFileStatus(dir.resolve(MOSS_CODEC_META_FILE)));
        } else {
            exitCode = 1;
            System.out.println("Codec directory: missing");
            if (ttsCodec != null && !ttsCodec.isBlank()) {
                System.out.println("Explicit codec ref: " + ttsCodec);
            } else if (offline) {
                System.out.println("Auto pull: disabled by --offline/--local");
            } else if (disableTtsAutoCodec) {
                System.out.println("Auto pull: disabled by --no-tts-auto-codec");
            } else {
                System.out.println("Auto pull: enabled for generation");
            }
            System.out.println("Install with: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
        }

        System.out.println();
        System.out.println("ONNX runtime caches");
        System.out.println("-------------------");
        printTtsOnnxCacheDiagnostics(modelDir, codecDir);

        System.out.println();
        System.out.println("Audio output");
        System.out.println("------------");
        AudioQualityPreset preset = audioQualityPreset();
        String effectiveFormat = effectiveAudioFormat(preset);
        String diagnosticFormat = diagnosticAudioFormat(effectiveFormat);
        Integer bitrate = effectiveAudioBitrateKbps(preset, diagnosticFormat);
        Integer compression = effectiveFlacCompression(preset, diagnosticFormat);
        System.out.println("Format: " + diagnosticFormat);
        System.out.println("Quality preset: " + (audioQuality == null || audioQuality.isBlank()
                ? "auto/default"
                : audioQuality.trim().toLowerCase(Locale.ROOT)));
        String encoderStatus = diagnosticAudioEncoderStatus(diagnosticFormat);
        System.out.println("Encoder: " + encoderStatus);
        if (encoderStatus.startsWith("missing")) {
            exitCode = 1;
        }
        if (bitrate != null) {
            System.out.println("Bitrate: " + bitrate + " kbps");
        }
        if (compression != null) {
            System.out.println("FLAC compression: " + compression);
        }
        System.out.println("Channels: " + (audioChannels == null || audioChannels.isBlank()
                ? "auto"
                : audioChannels.trim().toLowerCase(Locale.ROOT)));
        System.out.println("Polish: " + (disableAudioPolish ? "disabled" : "enabled")
                + (disableAudioNormalize ? " (peak normalization disabled)" : ""));
        System.out.println("Trim silence: " + (audioTrimSilence ? "enabled" : "disabled")
                + (audioTrimSilence
                        ? " (threshold " + (audioTrimDbfs == null ? "-48" : formatRangeNumber(audioTrimDbfs))
                                + " dBFS, padding "
                                + (audioTrimPaddingMs == null ? "25" : formatRangeNumber(audioTrimPaddingMs))
                                + " ms)"
                        : ""));
        System.out.println("Low-latency PCM: " + (disableLowLatencyAudio ? "disabled" : "enabled when streaming PCM/live audio"));
        System.out.println("Live audio: " + (liveAudio ? "requested" : "off"));

        System.out.println();
        System.out.println("Suling audio backends");
        System.out.println("---------------------");
        try {
            System.out.println("Formats: " + String.join(", ", Suling.supportedAudioFormats()));
            System.out.println(indentBlock(Suling.diagnostics(), "  "));
        } catch (Throwable e) {
            exitCode = 1;
            System.out.println("Diagnostics unavailable: " + e.getMessage());
        }

        System.out.println();
        System.out.println(exitCode == 0 ? "Status: ready" : "Status: attention needed");
        return exitCode;
    }

    private Optional<Path> diagnosticCodecDirectory() {
        if (ttsCodec != null && !ttsCodec.isBlank()) {
            try {
                return mossCodecDirectoryFrom(Path.of(resolveTtsCodecReference(ttsCodec)));
            } catch (Exception ignored) {
                return Optional.empty();
            }
        }
        return findInstalledMossAudioCodecDir();
    }

    private int printTtsDiagnosticsJson(String modelId, String finalLocalPath) {
        int exitCode = 0;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "moss_tts_diagnostics");
        payload.put("requested_model", modelId);
        payload.put("resolved_path", finalLocalPath);

        Optional<Path> ttsDir = resolveMossTtsDirectory(finalLocalPath, modelId);
        if (ttsDir.isEmpty()) {
            payload.put("status", "attention_needed");
            payload.put("error", "resolved model is not a local MOSS TTS ONNX directory");
            printJsonPayload(payload);
            return 1;
        }

        Path modelDir = ttsDir.get().toAbsolutePath().normalize();
        Path manifestPath = modelDir.resolve("browser_poc_manifest.json");
        Path ttsMetaPath = modelDir.resolve("tts_browser_onnx_meta.json");
        Map<String, Object> tts = new LinkedHashMap<>();
        tts.put("directory", modelDir.toString());
        tts.put("manifest", diagnosticFileJson(manifestPath));
        tts.put("metadata", diagnosticFileJson(ttsMetaPath));
        payload.put("tts", tts);

        try {
            JsonNode manifest = new ObjectMapper().readTree(manifestPath.toFile());
            JsonNode voices = manifest.path("builtin_voices");
            Map<String, Object> voiceInfo = new LinkedHashMap<>();
            int voiceCount = voices.isArray() ? voices.size() : 0;
            voiceInfo.put("count", voiceCount);
            voiceInfo.put("groups", voices.isArray() ? ttsVoiceGroupCounts(voices) : Map.of());
            if (voiceCount > 0) {
                Map<String, Object> defaultVoice = new LinkedHashMap<>();
                defaultVoice.put("voice", ttsVoiceId(voices.get(0)));
                defaultVoice.put("display_name", ttsVoiceDisplayName(voices.get(0)));
                defaultVoice.put("language", ttsVoiceLanguageLabel(voices.get(0)));
                defaultVoice.put("gender", ttsVoiceGenderLabel(voices.get(0)));
                voiceInfo.put("default", defaultVoice);
            } else {
                exitCode = 1;
            }
            String selector = effectiveTtsVoiceSelector();
            if (selector != null && !selector.isBlank()) {
                voiceInfo.put("requested_selector", selector);
            }
            tts.put("voices", voiceInfo);
        } catch (Exception e) {
            tts.put("manifest_error", e.getMessage());
            exitCode = 1;
        }

        Optional<Path> codecDir = diagnosticCodecDirectory();
        Map<String, Object> codec = new LinkedHashMap<>();
        if (codecDir.isPresent()) {
            Path dir = codecDir.get().toAbsolutePath().normalize();
            codec.put("present", true);
            codec.put("directory", dir.toString());
            codec.put("metadata", diagnosticFileJson(dir.resolve(MOSS_CODEC_META_FILE)));
        } else {
            codec.put("present", false);
            codec.put("install_command", "gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
            exitCode = 1;
        }
        payload.put("codec", codec);
        payload.put("onnx_caches", ttsOnnxCacheDiagnosticsJson(modelDir, codecDir));

        AudioQualityPreset preset = audioQualityPreset();
        String diagnosticFormat = diagnosticAudioFormat(effectiveAudioFormat(preset));
        String encoderStatus = diagnosticAudioEncoderStatus(diagnosticFormat);
        Map<String, Object> audio = new LinkedHashMap<>();
        audio.put("format", diagnosticFormat);
        audio.put("quality_preset", audioQuality == null || audioQuality.isBlank()
                ? "auto/default"
                : audioQuality.trim().toLowerCase(Locale.ROOT));
        audio.put("encoder", encoderStatus);
        audio.put("channels", audioChannels == null || audioChannels.isBlank()
                ? "auto"
                : audioChannels.trim().toLowerCase(Locale.ROOT));
        audio.put("polish", !disableAudioPolish);
        audio.put("trim_silence", audioTrimSilence);
        audio.put("live_audio", liveAudio);
        if (encoderStatus.startsWith("missing")) {
            exitCode = 1;
        }
        payload.put("audio", audio);

        try {
            Map<String, Object> suling = new LinkedHashMap<>();
            suling.put("formats", Suling.supportedAudioFormats());
            suling.put("diagnostics", Suling.diagnostics());
            payload.put("suling", suling);
        } catch (Throwable e) {
            payload.put("suling", Map.of("error", e.getMessage()));
            exitCode = 1;
        }

        payload.put("status", exitCode == 0 ? "ready" : "attention_needed");
        printJsonPayload(payload);
        return exitCode;
    }

    private int warmupMossTts(String modelId, String finalLocalPath) {
        if (!jsonMode) {
            System.out.println("MOSS TTS warmup");
            System.out.println("==============");
        }
        Optional<Path> ttsDir = resolveMossTtsDirectory(finalLocalPath, modelId);
        if (ttsDir.isEmpty()) {
            if (jsonMode) {
                printTtsJsonError("moss_tts_warmup", "resolved model is not a local MOSS TTS ONNX directory");
                return 1;
            }
            System.err.println("Error: resolved model is not a local MOSS TTS ONNX directory.");
            return 1;
        }
        Optional<Path> codecDir = diagnosticCodecDirectory();
        if (codecDir.isEmpty()) {
            if (jsonMode) {
                printTtsJsonError("moss_tts_warmup", "MOSS TTS companion codec is not installed");
                return 1;
            }
            System.err.println("Error: MOSS TTS companion codec is not installed.");
            System.err.println("Install with: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
            return 1;
        }

        Path modelDir = ttsDir.get().toAbsolutePath().normalize();
        Path codecPath = codecDir.get().toAbsolutePath().normalize();
        int threads = onnxProviderThreads();
        if (!jsonMode) {
            System.out.println("TTS directory: " + displayPath(modelDir));
            System.out.println("Codec directory: " + displayPath(codecPath));
            System.out.println("Threads: " + threads);
        }
        try {
            MossTtsOnnxRunner runner = mossTtsWarmupRunner != null && !mossTtsWarmupRunner.isUnsatisfied()
                    ? mossTtsWarmupRunner.get()
                    : new MossTtsOnnxRunner();
            MossTtsOnnxRunner.MossTtsWarmup warmup = runner.warmup(modelDir, codecPath, threads);
            Map<String, Object> metadata = warmup.metadata();
            if (jsonMode) {
                printMossTtsWarmupJson(modelId, modelDir, codecPath, threads, warmup, metadata);
                return 0;
            }
            System.out.println("Warmup duration: " + String.format(Locale.ROOT, "%.2fs", warmup.durationMs() / 1000.0));
            String graphCache = onnxOptimizedCacheDetail(metadata);
            if (graphCache != null) {
                System.out.println("Optimized graph cache: " + graphCache);
            }
            String sessionCache = onnxSessionCacheDetail(metadata);
            if (sessionCache != null) {
                System.out.println("Session cache: " + sessionCache);
            }
            String assetCache = onnxAssetCacheDetail(metadata);
            if (assetCache != null) {
                System.out.println("Asset cache: " + assetCache);
            }
            System.out.println("Opened sessions: " + Math.round(defaultedDouble(
                    metadata,
                    "onnx_optimized_model_cache_sessions")));
            System.out.println("Status: warmed");
            return 0;
        } catch (Exception e) {
            if (jsonMode) {
                printTtsJsonError("moss_tts_warmup", "MOSS TTS warmup failed: " + e.getMessage());
                return 1;
            }
            System.err.println("Error: MOSS TTS warmup failed: " + e.getMessage());
            return 1;
        }
    }

    private int onnxProviderThreads() {
        String raw = System.getProperty("gollek.providers.onnx.threads");
        if (raw == null || raw.isBlank()) {
            raw = System.getenv("GOLLEK_ONNX_THREADS");
        }
        try {
            return Math.max(1, raw == null || raw.isBlank()
                    ? 4
                    : Integer.parseInt(raw.trim()));
        } catch (NumberFormatException ignored) {
            return 4;
        }
    }

    private int pruneMossTtsOptimizedCache(String modelId, String finalLocalPath) {
        boolean dryRun = !ttsCachePruneApply;
        Optional<Path> ttsDir = resolveMossTtsDirectory(finalLocalPath, modelId);
        if (ttsDir.isEmpty()) {
            if (jsonMode) {
                printTtsJsonError("moss_tts_cache_prune", "resolved model is not a local MOSS TTS ONNX directory");
                return 1;
            }
            System.err.println("Error: resolved model is not a local MOSS TTS ONNX directory.");
            return 1;
        }
        Optional<Path> codecDir = diagnosticCodecDirectory();
        if (codecDir.isEmpty()) {
            if (jsonMode) {
                printTtsJsonError("moss_tts_cache_prune", "MOSS TTS companion codec is not installed");
                return 1;
            }
            System.err.println("Error: MOSS TTS companion codec is not installed.");
            System.err.println("Install with: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
            return 1;
        }

        Path modelDir = ttsDir.get().toAbsolutePath().normalize();
        Path codecPath = codecDir.get().toAbsolutePath().normalize();
        Path cacheRoot = onnxOptimizedCacheRoot();
        try {
            OptimizedCacheScope cacheScope = currentMossTtsOptimizedCacheScope(modelDir, codecPath, cacheRoot);
            OptimizedCacheDirectoryStats before = optimizedCacheDirectoryStats(cacheRoot, cacheScope);
            List<CachePruneOperation> operations = pruneOperations(before, dryRun);
            int errors = 0;
            long affectedBytes = 0L;
            if (!dryRun) {
                for (CachePruneOperation operation : operations) {
                    if (operation.error() != null && !operation.error().isBlank()) {
                        errors++;
                    } else if (operation.deleted()) {
                        affectedBytes += operation.sizeBytes();
                    }
                }
            } else {
                for (CachePruneOperation operation : operations) {
                    affectedBytes += operation.sizeBytes();
                }
            }
            OptimizedCacheDirectoryStats after = dryRun
                    ? before
                    : optimizedCacheDirectoryStats(cacheRoot, cacheScope);
            if (jsonMode) {
                printMossTtsCachePruneJson(
                        modelId,
                        modelDir,
                        codecPath,
                        cacheRoot,
                        dryRun,
                        before,
                        after,
                        operations,
                        affectedBytes,
                        errors);
            } else {
                printMossTtsCachePrune(
                        modelDir,
                        codecPath,
                        cacheRoot,
                        dryRun,
                        before,
                        after,
                        operations,
                        affectedBytes,
                        errors);
            }
            return errors == 0 ? 0 : 1;
        } catch (Exception e) {
            if (jsonMode) {
                printTtsJsonError("moss_tts_cache_prune", "MOSS TTS cache prune failed: " + e.getMessage());
                return 1;
            }
            System.err.println("Error: MOSS TTS cache prune failed: " + e.getMessage());
            return 1;
        }
    }

    private OptimizedCacheScope currentMossTtsOptimizedCacheScope(
            Path modelDir,
            Path codecPath,
            Path cacheRoot) throws Exception {
        Set<Path> currentPaths = new java.util.HashSet<>();
        Set<String> currentPrefixes = new java.util.HashSet<>();
        for (DiagnosticOnnxGraph graph : mossTtsDiagnosticGraphs(modelDir, codecPath)) {
            currentPaths.add(optimizedModelCachePath(graph.path(), cacheRoot).toAbsolutePath().normalize());
            currentPrefixes.add(optimizedCacheFilenamePrefix(graph.path()));
        }
        return new OptimizedCacheScope(currentPaths, currentPrefixes);
    }

    private List<CachePruneOperation> pruneOperations(OptimizedCacheDirectoryStats before, boolean dryRun) {
        List<CachePruneOperation> operations = new ArrayList<>();
        for (Path path : before.staleOptimizedGraphPaths()) {
            long sizeBytes = safeSize(path);
            if (dryRun) {
                operations.add(new CachePruneOperation("optimized_graph", path, sizeBytes, false, true, null));
                continue;
            }
            boolean deleted = false;
            String error = null;
            try {
                deleted = Files.deleteIfExists(path);
            } catch (Exception e) {
                error = e.getMessage();
            }
            operations.add(new CachePruneOperation("optimized_graph", path, sizeBytes, deleted, false, error));
        }
        for (Path path : orphanSidecarPruneCandidates(before)) {
            long sizeBytes = safeSize(path);
            if (dryRun) {
                operations.add(new CachePruneOperation("external_data_sidecar", path, sizeBytes, false, true, null));
                continue;
            }
            boolean deleted = false;
            String error = null;
            try {
                deleted = Files.deleteIfExists(path);
            } catch (Exception e) {
                error = e.getMessage();
            }
            operations.add(new CachePruneOperation("external_data_sidecar", path, sizeBytes, deleted, false, error));
        }
        return operations;
    }

    private List<Path> orphanSidecarPruneCandidates(OptimizedCacheDirectoryStats before) {
        return before.pruneableExternalDataSidecarPaths();
    }

    private void printMossTtsCachePruneJson(
            String modelId,
            Path modelDir,
            Path codecPath,
            Path cacheRoot,
            boolean dryRun,
            OptimizedCacheDirectoryStats before,
            OptimizedCacheDirectoryStats after,
            List<CachePruneOperation> operations,
            long affectedBytes,
            int errors) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "moss_tts_cache_prune");
        payload.put("status", errors == 0 ? (dryRun ? "dry_run" : "pruned") : "error");
        payload.put("dry_run", dryRun);
        payload.put("requested_model", modelId);
        payload.put("tts_directory", modelDir.toString());
        payload.put("codec_directory", codecPath.toString());
        payload.put("cache_root", cacheRoot.toAbsolutePath().normalize().toString());
        payload.put("affected_files", operations.size());
        payload.put("affected_optimized_graph_files", countOperations(operations, "optimized_graph"));
        payload.put("affected_external_data_sidecar_files", countOperations(operations, "external_data_sidecar"));
        payload.put("affected_size_bytes", affectedBytes);
        payload.put("errors", errors);
        payload.put("before", before.toJson());
        payload.put("after", after.toJson());
        payload.put("operations", operations.stream().map(CachePruneOperation::toJson).toList());
        printJsonPayload(payload);
    }

    private void printMossTtsCachePrune(
            Path modelDir,
            Path codecPath,
            Path cacheRoot,
            boolean dryRun,
            OptimizedCacheDirectoryStats before,
            OptimizedCacheDirectoryStats after,
            List<CachePruneOperation> operations,
            long affectedBytes,
            int errors) {
        System.out.println("MOSS TTS optimized cache prune");
        System.out.println("=============================");
        System.out.println("TTS directory: " + displayPath(modelDir));
        System.out.println("Codec directory: " + displayPath(codecPath));
        System.out.println("Cache root: " + displayPath(cacheRoot));
        System.out.println("Mode: " + (dryRun ? "dry-run (use --tts-cache-prune-apply to delete)" : "apply"));
        System.out.println("Before: " + before.currentOptimizedGraphFiles() + " current, "
                + before.staleOptimizedGraphFiles() + " stale, "
                + before.unrelatedOptimizedGraphFiles() + " unrelated, "
                + before.optimizedGraphFiles() + " optimized total, "
                + before.externalDataSidecarFiles() + " sidecar(s), "
                + formatBytes(before.sizeBytes()));
        if (before.externalDataSidecarFiles() > 0) {
            System.out.println("Sidecar prune policy: " + before.externalDataSidecarPrunePolicy());
            if (before.retainedExternalDataSidecarFiles() > 0) {
                System.out.println("Retained external data sidecars: "
                        + before.retainedExternalDataSidecarFiles() + " file(s), "
                        + formatBytes(before.retainedExternalDataSidecarSizeBytes()) + " ("
                        + formatDiagnosticList(before.retainedExternalDataSidecarFileNames(), 10) + ")");
            }
        }
        if (before.unrelatedOptimizedGraphFiles() > 0) {
            System.out.println("Unrelated optimized graph files kept: "
                    + formatDiagnosticList(before.unrelatedOptimizedGraphFileNames(), 10));
        }
        List<CachePruneOperation> graphOperations = operationsOfKind(operations, "optimized_graph");
        List<CachePruneOperation> sidecarOperations = operationsOfKind(operations, "external_data_sidecar");
        if (graphOperations.isEmpty()) {
            System.out.println("Stale optimized graph files: none");
        } else {
            System.out.println((dryRun ? "Would delete" : "Deleted") + " stale optimized graph files: "
                    + graphOperations.size() + " file(s), " + formatBytes(totalOperationBytes(graphOperations)));
            printCachePruneOperations(graphOperations, dryRun, 10);
        }
        if (sidecarOperations.isEmpty()) {
            System.out.println("Orphan external data sidecars: none");
        } else {
            System.out.println((dryRun ? "Would delete" : "Deleted") + " orphan external data sidecars: "
                    + sidecarOperations.size() + " file(s), " + formatBytes(totalOperationBytes(sidecarOperations)));
            printCachePruneOperations(sidecarOperations, dryRun, 10);
        }
        if (!dryRun) {
            System.out.println("After: " + after.currentOptimizedGraphFiles() + " current, "
                    + after.staleOptimizedGraphFiles() + " stale, "
                    + after.unrelatedOptimizedGraphFiles() + " unrelated, "
                    + after.optimizedGraphFiles() + " optimized total, "
                    + after.externalDataSidecarFiles() + " sidecar(s), "
                    + formatBytes(after.sizeBytes()));
        }
        System.out.println(errors == 0
                ? "Status: " + (dryRun ? "dry run complete" : "pruned")
                : "Status: completed with " + errors + " error(s)");
    }

    private void printCachePruneOperations(
            List<CachePruneOperation> operations,
            boolean dryRun,
            int limit) {
        for (CachePruneOperation operation : operations.stream().limit(limit).toList()) {
            String status = dryRun
                    ? "would delete"
                    : (operation.deleted() ? "deleted" : (operation.error() == null ? "missing" : "failed"));
            System.out.println("  " + operation.path().getFileName()
                    + " (" + formatBytes(operation.sizeBytes()) + ", " + status
                    + (operation.error() == null ? "" : ": " + operation.error())
                    + ")");
        }
        if (operations.size() > limit) {
            System.out.println("  ... +" + (operations.size() - limit) + " more");
        }
    }

    private int countOperations(List<CachePruneOperation> operations, String kind) {
        return (int) operations.stream()
                .filter(operation -> kind.equals(operation.kind()))
                .count();
    }

    private long totalOperationBytes(List<CachePruneOperation> operations) {
        long total = 0L;
        for (CachePruneOperation operation : operations) {
            total += operation.sizeBytes();
        }
        return total;
    }

    private List<CachePruneOperation> operationsOfKind(List<CachePruneOperation> operations, String kind) {
        return operations.stream()
                .filter(operation -> kind.equals(operation.kind()))
                .toList();
    }

    private void printMossTtsWarmupJson(
            String modelId,
            Path modelDir,
            Path codecPath,
            int threads,
            MossTtsOnnxRunner.MossTtsWarmup warmup,
            Map<String, Object> metadata) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "moss_tts_warmup");
        payload.put("status", "warmed");
        payload.put("requested_model", modelId);
        payload.put("tts_directory", modelDir.toString());
        payload.put("codec_directory", codecPath.toString());
        payload.put("threads", threads);
        payload.put("duration_ms", warmup.durationMs());
        payload.put("duration_seconds", warmup.durationMs() / 1000.0);
        payload.put("optimized_graph_cache", onnxOptimizedCacheDetail(metadata));
        payload.put("session_cache", onnxSessionCacheDetail(metadata));
        payload.put("asset_cache", onnxAssetCacheDetail(metadata));
        payload.put("opened_sessions", Math.round(defaultedDouble(
                metadata,
                "onnx_optimized_model_cache_sessions")));
        payload.put("metadata", metadata);
        printJsonPayload(payload);
    }

    private void printTtsJsonError(String type, String message) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", type);
        payload.put("status", "error");
        payload.put("error", message);
        printJsonPayload(payload);
    }

    private Map<String, Object> diagnosticFileJson(Path path) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("path", path == null ? null : path.toAbsolutePath().normalize().toString());
        boolean present = path != null && Files.isRegularFile(path);
        value.put("present", present);
        if (present) {
            try {
                value.put("size_bytes", Files.size(path));
            } catch (Exception ignored) {
            }
        }
        return value;
    }

    private Map<String, Integer> ttsVoiceGroupCounts(JsonNode voices) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (JsonNode voice : voices) {
            String key = ttsVoiceLanguageLabel(voice) + " " + ttsVoiceGenderLabel(voice);
            counts.put(key, counts.getOrDefault(key, 0) + 1);
        }
        return counts;
    }

    private Map<String, Object> ttsOnnxCacheDiagnosticsJson(Path modelDir, Optional<Path> codecDir) {
        Map<String, Object> payload = new LinkedHashMap<>();
        boolean graphCacheEnabled = cacheBoolean(
                ONNX_OPTIMIZED_CACHE_ENABLED_PROPERTY,
                ONNX_OPTIMIZED_CACHE_ENABLED_ENV,
                true);
        Path cacheRoot = onnxOptimizedCacheRoot();
        payload.put("optimized_graph_cache_enabled", graphCacheEnabled);
        payload.put("graph_cache_root", cacheRoot.toString());
        payload.put("onnx_runtime_log_level", onnxLogLevelName());
        OptimizedCacheScope cacheScope = new OptimizedCacheScope(new java.util.HashSet<>(), new java.util.HashSet<>());

        if (codecDir.isEmpty()) {
            payload.put("graphs_available", false);
        } else {
            try {
                List<Map<String, Object>> graphItems = new ArrayList<>();
                List<DiagnosticOnnxGraph> graphs = mossTtsDiagnosticGraphs(modelDir, codecDir.get());
                int present = 0;
                for (DiagnosticOnnxGraph graph : graphs) {
                    Path optimizedPath = optimizedModelCachePath(graph.path(), cacheRoot);
                    cacheScope.currentPaths().add(optimizedPath.toAbsolutePath().normalize());
                    cacheScope.currentPrefixes().add(optimizedCacheFilenamePrefix(graph.path()));
                    boolean warm = Files.isRegularFile(optimizedPath);
                    if (warm) {
                        present++;
                    }
                    Map<String, Object> item = new LinkedHashMap<>();
                    item.put("label", graph.label());
                    item.put("state", warm ? "warm" : "cold");
                    item.put("source_path", graph.path().toAbsolutePath().normalize().toString());
                    item.put("optimized_path", optimizedPath.toAbsolutePath().normalize().toString());
                    item.put("optimized_file", optimizedPath.getFileName().toString());
                    item.put("external_data_sidecars", externalDataSidecars(graph.path()).size());
                    graphItems.add(item);
                }
                payload.put("graphs_available", true);
                payload.put("warmable_graphs", graphs.size());
                payload.put("normal_generation_graphs", 4);
                payload.put("optimized_graph_files_present", present);
                payload.put("graphs", graphItems);
            } catch (Exception e) {
                payload.put("graphs_available", false);
                payload.put("graph_error", e.getMessage());
            }
        }
        payload.put("cache_directory", optimizedCacheDirectoryStats(cacheRoot, cacheScope).toJson());

        payload.put("session_cache", Map.of(
                "enabled", cacheBoolean(
                        MOSS_SESSION_CACHE_ENABLED_PROPERTY,
                        MOSS_SESSION_CACHE_ENABLED_ENV,
                        true),
                "max_entries", cacheInt(
                        MOSS_SESSION_CACHE_MAX_ENTRIES_PROPERTY,
                        MOSS_SESSION_CACHE_MAX_ENTRIES_ENV,
                        2),
                "scope", "in-process"));
        payload.put("asset_cache", Map.of(
                "enabled", cacheBoolean(
                        MOSS_ASSET_CACHE_ENABLED_PROPERTY,
                        MOSS_ASSET_CACHE_ENABLED_ENV,
                        true),
                "max_entries", cacheInt(
                        MOSS_ASSET_CACHE_MAX_ENTRIES_PROPERTY,
                        MOSS_ASSET_CACHE_MAX_ENTRIES_ENV,
                        4),
                "scope", "in-process"));
        return payload;
    }

    private void printTtsOnnxCacheDiagnostics(Path modelDir, Optional<Path> codecDir) {
        boolean graphCacheEnabled = cacheBoolean(
                ONNX_OPTIMIZED_CACHE_ENABLED_PROPERTY,
                ONNX_OPTIMIZED_CACHE_ENABLED_ENV,
                true);
        Path cacheRoot = onnxOptimizedCacheRoot();
        System.out.println("Optimized graph cache: " + (graphCacheEnabled ? "enabled" : "disabled"));
        System.out.println("Graph cache root: " + displayPath(cacheRoot));
        System.out.println("ONNX Runtime log level: " + onnxLogLevelName());
        OptimizedCacheScope cacheScope = new OptimizedCacheScope(new java.util.HashSet<>(), new java.util.HashSet<>());

        if (codecDir.isEmpty()) {
            System.out.println("Optimized graph files: unavailable until companion codec is installed");
        } else {
            try {
                List<DiagnosticOnnxGraph> graphs = mossTtsDiagnosticGraphs(modelDir, codecDir.get());
                long present = 0L;
                for (DiagnosticOnnxGraph graph : graphs) {
                    Path optimizedPath = optimizedModelCachePath(graph.path(), cacheRoot);
                    cacheScope.currentPaths().add(optimizedPath.toAbsolutePath().normalize());
                    cacheScope.currentPrefixes().add(optimizedCacheFilenamePrefix(graph.path()));
                    if (Files.isRegularFile(optimizedPath)) {
                        present++;
                    }
                }
                System.out.println("Warmable graphs: " + graphs.size() + " (normal generation opens 4)");
                System.out.println("Optimized graph files: " + present + "/" + graphs.size()
                        + (graphCacheEnabled ? " present" : " present but cache disabled"));
                for (DiagnosticOnnxGraph graph : graphs) {
                    Path optimizedPath = optimizedModelCachePath(graph.path(), cacheRoot);
                    String status = Files.isRegularFile(optimizedPath) ? "warm" : "cold";
                    int sidecars = externalDataSidecars(graph.path()).size();
                    System.out.println("  " + graph.label() + ": " + status
                            + " -> " + optimizedPath.getFileName()
                            + (sidecars > 0 ? " (" + sidecars + " external data sidecars)" : ""));
                }
            } catch (Exception e) {
                System.out.println("Optimized graph files: unavailable (" + e.getMessage() + ")");
            }
        }
        printOptimizedCacheDirectoryStats(optimizedCacheDirectoryStats(cacheRoot, cacheScope));

        boolean sessionCacheEnabled = cacheBoolean(
                MOSS_SESSION_CACHE_ENABLED_PROPERTY,
                MOSS_SESSION_CACHE_ENABLED_ENV,
                true);
        int sessionCacheMax = cacheInt(
                MOSS_SESSION_CACHE_MAX_ENTRIES_PROPERTY,
                MOSS_SESSION_CACHE_MAX_ENTRIES_ENV,
                2);
        boolean assetCacheEnabled = cacheBoolean(
                MOSS_ASSET_CACHE_ENABLED_PROPERTY,
                MOSS_ASSET_CACHE_ENABLED_ENV,
                true);
        int assetCacheMax = cacheInt(
                MOSS_ASSET_CACHE_MAX_ENTRIES_PROPERTY,
                MOSS_ASSET_CACHE_MAX_ENTRIES_ENV,
                4);
        System.out.println("Session cache: "
                + (sessionCacheEnabled && sessionCacheMax > 0 ? "enabled" : "disabled")
                + " (max entries " + Math.max(0, sessionCacheMax) + ", in-process)");
        System.out.println("Asset cache: "
                + (assetCacheEnabled && assetCacheMax > 0 ? "enabled" : "disabled")
                + " (max entries " + Math.max(0, assetCacheMax) + ", in-process)");
    }

    private List<DiagnosticOnnxGraph> mossTtsDiagnosticGraphs(Path modelDir, Path codecDir) throws Exception {
        JsonNode ttsMeta = new ObjectMapper().readTree(modelDir.resolve("tts_browser_onnx_meta.json").toFile());
        JsonNode codecMeta = new ObjectMapper().readTree(codecDir.resolve(MOSS_CODEC_META_FILE).toFile());
        JsonNode ttsFiles = ttsMeta.path("files");
        JsonNode codecFiles = codecMeta.path("files");
        return List.of(
                new DiagnosticOnnxGraph(
                        "tts prefill",
                        modelDir.resolve(ttsFiles.path("prefill").asText("moss_tts_prefill.onnx"))),
                new DiagnosticOnnxGraph(
                        "tts decode",
                        modelDir.resolve(ttsFiles.path("decode_step").asText("moss_tts_decode_step.onnx"))),
                new DiagnosticOnnxGraph(
                        "local sampler",
                        modelDir.resolve(ttsFiles.path("local_fixed_sampled_frame")
                                .asText("moss_tts_local_fixed_sampled_frame.onnx"))),
                new DiagnosticOnnxGraph(
                        "codec full decode",
                        codecDir.resolve(codecFiles.path("decode_full")
                                .asText("moss_audio_tokenizer_decode_full.onnx"))),
                new DiagnosticOnnxGraph(
                        "codec streaming decode",
                        codecDir.resolve(codecFiles.path("decode_step")
                                .asText("moss_audio_tokenizer_decode_step.onnx"))));
    }

    private Path optimizedModelCachePath(Path modelPath, Path cacheRoot) throws Exception {
        String fingerprint = optimizedModelFingerprint(modelPath.toAbsolutePath().normalize());
        String filename = optimizedCacheFilenamePrefix(modelPath)
                + fingerprint.substring(0, 16) + ".optimized.onnx";
        return cacheRoot.resolve(filename);
    }

    private String optimizedCacheFilenamePrefix(Path modelPath) {
        return sanitizeCacheFilename(modelPath.getFileName().toString()) + "-";
    }

    private String optimizedModelFingerprint(Path modelPath) throws Exception {
        java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
        updateDigestLine(digest, "os=" + System.getProperty("os.name", ""));
        updateDigestLine(digest, "arch=" + System.getProperty("os.arch", ""));
        updateDigestLine(digest, "runtime=ort-java");
        updateDigestLine(digest, "optimization=all-opt");
        updateDigestLine(digest, "backend=cpu");
        updateGraphIdentityDigest(digest, modelPath);
        return java.util.HexFormat.of().formatHex(digest.digest());
    }

    private void updateGraphIdentityDigest(java.security.MessageDigest digest, Path modelPath) throws Exception {
        updateFileIdentityDigest(digest, "onnx", modelPath);
        List<Path> sidecars = externalDataSidecars(modelPath);
        updateDigestLine(digest, "external_data_sidecars=" + sidecars.size());
        for (Path sidecar : sidecars) {
            updateFileIdentityDigest(digest, "data", sidecar);
        }
    }

    private void updateFileIdentityDigest(java.security.MessageDigest digest, String kind, Path path) throws Exception {
        Path modelPath = path.toAbsolutePath().normalize();
        updateDigestLine(digest, String.join("\t",
                kind,
                modelPath.toRealPath().toString(),
                String.valueOf(Files.size(modelPath)),
                String.valueOf(Files.getLastModifiedTime(modelPath).toMillis())));
    }

    private void updateDigestLine(java.security.MessageDigest digest, String text) {
        digest.update(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        digest.update((byte) '\n');
    }

    private List<Path> externalDataSidecars(Path modelPath) throws Exception {
        Path sourceDir = modelPath.toAbsolutePath().normalize().getParent();
        if (sourceDir == null || !Files.isDirectory(sourceDir)) {
            return List.of();
        }
        try (var stream = Files.list(sourceDir)) {
            return stream
                    .filter(path -> Files.isRegularFile(path) && path.getFileName().toString().endsWith(".data"))
                    .sorted((left, right) -> left.getFileName().toString()
                            .compareTo(right.getFileName().toString()))
                    .toList();
        }
    }

    private OptimizedCacheDirectoryStats optimizedCacheDirectoryStats(
            Path cacheRoot,
            OptimizedCacheScope scope) {
        Path root = cacheRoot.toAbsolutePath().normalize();
        if (!Files.isDirectory(root)) {
            return new OptimizedCacheDirectoryStats(
                    root,
                    false,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0L,
                    0L,
                    0L,
                    0L,
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of());
        }
        Set<Path> currentOptimizedPaths = scope.currentPaths();
        Set<String> currentPrefixes = scope.currentPrefixes();
        int fileCount = 0;
        int optimizedGraphFiles = 0;
        int currentOptimizedGraphFiles = 0;
        int staleOptimizedGraphFiles = 0;
        int unrelatedOptimizedGraphFiles = 0;
        int externalDataSidecarFiles = 0;
        long sizeBytes = 0L;
        long staleBytes = 0L;
        long unrelatedBytes = 0L;
        long externalDataSidecarBytes = 0L;
        List<Path> externalDataSidecarPaths = new ArrayList<>();
        List<Path> stalePaths = new ArrayList<>();
        List<Path> unrelatedPaths = new ArrayList<>();
        try (var stream = Files.list(root)) {
            for (Path file : stream.toList()) {
                if (!Files.isRegularFile(file)) {
                    continue;
                }
                fileCount++;
                try {
                    sizeBytes += Files.size(file);
                } catch (Exception ignored) {
                }
                String filename = file.getFileName().toString();
                if (filename.endsWith(".data")) {
                    externalDataSidecarFiles++;
                    externalDataSidecarBytes += safeSize(file);
                    externalDataSidecarPaths.add(file.toAbsolutePath().normalize());
                }
                if (!filename.endsWith(".optimized.onnx")) {
                    continue;
                }
                optimizedGraphFiles++;
                Path normalized = file.toAbsolutePath().normalize();
                if (currentOptimizedPaths.contains(normalized)) {
                    currentOptimizedGraphFiles++;
                } else if (hasOptimizedCachePrefix(filename, currentPrefixes)) {
                    staleOptimizedGraphFiles++;
                    staleBytes += safeSize(file);
                    stalePaths.add(file.toAbsolutePath().normalize());
                } else {
                    unrelatedOptimizedGraphFiles++;
                    unrelatedBytes += safeSize(file);
                    unrelatedPaths.add(file.toAbsolutePath().normalize());
                }
            }
        } catch (Exception ignored) {
        }
        stalePaths.sort((left, right) -> left.getFileName().toString().compareTo(right.getFileName().toString()));
        unrelatedPaths.sort((left, right) -> left.getFileName().toString().compareTo(right.getFileName().toString()));
        externalDataSidecarPaths.sort((left, right) -> left.getFileName().toString()
                .compareTo(right.getFileName().toString()));
        List<Path> orphanSidecars = optimizedGraphFiles == 0 ? externalDataSidecarPaths : List.of();
        return new OptimizedCacheDirectoryStats(
                root,
                true,
                fileCount,
                optimizedGraphFiles,
                currentOptimizedGraphFiles,
                staleOptimizedGraphFiles,
                unrelatedOptimizedGraphFiles,
                externalDataSidecarFiles,
                sizeBytes,
                staleBytes,
                unrelatedBytes,
                externalDataSidecarBytes,
                externalDataSidecarPaths,
                orphanSidecars,
                unrelatedPaths,
                stalePaths);
    }

    private boolean hasOptimizedCachePrefix(String filename, Set<String> prefixes) {
        if (prefixes == null || prefixes.isEmpty()) {
            return false;
        }
        for (String prefix : prefixes) {
            if (filename.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    private void printOptimizedCacheDirectoryStats(OptimizedCacheDirectoryStats stats) {
        if (!stats.exists()) {
            System.out.println("Cache directory files: not created yet");
            return;
        }
        System.out.println("Cache directory files: " + stats.fileCount()
                + " file(s), " + formatBytes(stats.sizeBytes()));
        System.out.println("Optimized graph cache inventory: "
                + stats.currentOptimizedGraphFiles() + " current, "
                + stats.staleOptimizedGraphFiles() + " stale, "
                + stats.unrelatedOptimizedGraphFiles() + " unrelated, "
                + stats.optimizedGraphFiles() + " optimized total, "
                + stats.externalDataSidecarFiles() + " external data sidecar(s)");
        if (stats.orphanExternalDataSidecarFiles() > 0) {
            System.out.println("Orphan external data sidecars: "
                    + formatDiagnosticList(stats.orphanExternalDataSidecarFileNames(), 5));
        }
        if (stats.externalDataSidecarFiles() > 0) {
            System.out.println("External data sidecar prune policy: " + stats.externalDataSidecarPrunePolicy());
            if (stats.retainedExternalDataSidecarFiles() > 0) {
                System.out.println("Retained external data sidecars: "
                        + stats.retainedExternalDataSidecarFiles() + " file(s), "
                        + formatBytes(stats.retainedExternalDataSidecarSizeBytes()) + " ("
                        + formatDiagnosticList(stats.retainedExternalDataSidecarFileNames(), 5) + ")");
            }
        }
        if (stats.staleOptimizedGraphFiles() > 0) {
            System.out.println("Stale optimized graph files: "
                    + formatDiagnosticList(stats.staleOptimizedGraphFileNames(), 5));
        }
        if (stats.unrelatedOptimizedGraphFiles() > 0) {
            System.out.println("Unrelated optimized graph files kept: "
                    + formatDiagnosticList(stats.unrelatedOptimizedGraphFileNames(), 5));
        }
    }

    private static long safeSize(Path path) {
        try {
            return Files.size(path);
        } catch (Exception ignored) {
            return 0L;
        }
    }

    private Path onnxOptimizedCacheRoot() {
        String root = firstNonBlank(
                System.getProperty(ONNX_OPTIMIZED_CACHE_DIR_PROPERTY),
                System.getenv(ONNX_OPTIMIZED_CACHE_DIR_ENV));
        return root.isBlank()
                ? Path.of(System.getProperty("user.home"), ".gollek", "cache", "onnx", "optimized", "moss-tts")
                : Path.of(root).resolve("moss-tts");
    }

    private boolean cacheBoolean(String property, String env, boolean fallback) {
        String raw = firstNonBlank(System.getProperty(property), System.getenv(env), String.valueOf(fallback));
        return !Set.of("0", "false", "no", "off", "disable", "disabled")
                .contains(raw.trim().toLowerCase(Locale.ROOT));
    }

    private int cacheInt(String property, String env, int fallback) {
        String raw = firstNonBlank(System.getProperty(property), System.getenv(env), String.valueOf(fallback));
        try {
            return Math.max(0, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private String onnxLogLevelName() {
        String raw = firstNonBlank(
                System.getProperty(ONNX_LOG_LEVEL_PROPERTY),
                System.getenv(ONNX_LOG_LEVEL_ENV),
                "error");
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "verbose", "info", "warning", "warn", "error", "fatal" -> normalized;
            default -> "error";
        };
    }

    private static String sanitizeCacheFilename(String value) {
        String normalized = value == null ? "model" : value.toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9._-]+", "-")
                .replaceAll("[-_.]{2,}", "-")
                .replaceAll("^[-_.]+|[-_.]+$", "");
        return normalized.isBlank() ? "model" : normalized;
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return "";
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return "";
    }

    private String diagnosticAudioEncoderStatus(String format) {
        return switch (format == null ? "" : format.toLowerCase(Locale.ROOT)) {
            case "wav" -> "ready (pure Java)";
            case "flac" -> FlacLibraryCheck.isAvailable()
                    ? "ready (libFLAC " + FlacLibraryCheck.getVersion() + ")"
                    : "missing libFLAC; install libFLAC or use --audio-format wav";
            case "mp3" -> FfmpegAudioEncoder.isMp3EncodingAvailable()
                    ? "ready (FFmpeg/libmp3lame)"
                    : "missing FFmpeg/libmp3lame; install FFmpeg libraries or use --audio-format wav";
            default -> "missing unsupported format";
        };
    }

    private String diagnosticAudioFormat(String effectiveFormat) {
        if (effectiveFormat != null && !effectiveFormat.isBlank()) {
            return effectiveFormat;
        }
        String explicitExt = normalizeExtension(extension);
        if (isKnownAudioExtension(explicitExt)) {
            return explicitExt;
        }
        try {
            String outputExt = pathExtension(outputPath == null || outputPath.isBlank() ? null : Path.of(outputPath));
            if (isKnownAudioExtension(outputExt)) {
                return outputExt.toLowerCase(Locale.ROOT);
            }
        } catch (Exception ignored) {
            // Fall through to runner default.
        }
        return "wav";
    }

    private String diagnosticFileStatus(Path path) {
        if (path == null) {
            return "missing";
        }
        if (!Files.isRegularFile(path)) {
            return "missing (" + displayPath(path) + ")";
        }
        try {
            return "ok (" + displayPath(path) + ", " + Files.size(path) + " bytes)";
        } catch (Exception ignored) {
            return "ok (" + displayPath(path) + ")";
        }
    }

    private record DiagnosticOnnxGraph(String label, Path path) {
    }

    private record OptimizedCacheScope(Set<Path> currentPaths, Set<String> currentPrefixes) {
    }

    private record OptimizedCacheDirectoryStats(
            Path path,
            boolean exists,
            int fileCount,
            int optimizedGraphFiles,
            int currentOptimizedGraphFiles,
            int staleOptimizedGraphFiles,
            int unrelatedOptimizedGraphFiles,
            int externalDataSidecarFiles,
            long sizeBytes,
            long staleOptimizedGraphSizeBytes,
            long unrelatedOptimizedGraphSizeBytes,
            long externalDataSidecarSizeBytes,
            List<Path> externalDataSidecarPaths,
            List<Path> orphanExternalDataSidecarPaths,
            List<Path> unrelatedOptimizedGraphPaths,
            List<Path> staleOptimizedGraphPaths) {
        private Map<String, Object> toJson() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("path", path.toString());
            value.put("exists", exists);
            value.put("file_count", fileCount);
            value.put("size_bytes", sizeBytes);
            value.put("optimized_graph_files", optimizedGraphFiles);
            value.put("current_optimized_graph_files", currentOptimizedGraphFiles);
            value.put("stale_optimized_graph_files", staleOptimizedGraphFiles);
            value.put("stale_optimized_graph_size_bytes", staleOptimizedGraphSizeBytes);
            value.put("unrelated_optimized_graph_files", unrelatedOptimizedGraphFiles);
            value.put("unrelated_optimized_graph_size_bytes", unrelatedOptimizedGraphSizeBytes);
            value.put("external_data_sidecar_files", externalDataSidecarFiles);
            value.put("external_data_sidecar_size_bytes", externalDataSidecarSizeBytes);
            value.put("external_data_sidecar_prune_policy", externalDataSidecarPrunePolicy());
            value.put("optimized_graph_files_after_stale_prune", optimizedGraphFilesAfterStalePrune());
            value.put("pruneable_external_data_sidecar_files", pruneableExternalDataSidecarFiles());
            value.put("pruneable_external_data_sidecar_size_bytes", pruneableExternalDataSidecarSizeBytes());
            value.put("pruneable_external_data_sidecar_file_names", pruneableExternalDataSidecarFileNames());
            value.put("pruneable_external_data_sidecar_paths", pruneableExternalDataSidecarPaths().stream()
                    .map(Path::toString)
                    .toList());
            value.put("retained_external_data_sidecar_files", retainedExternalDataSidecarFiles());
            value.put("retained_external_data_sidecar_size_bytes", retainedExternalDataSidecarSizeBytes());
            value.put("retained_external_data_sidecar_file_names", retainedExternalDataSidecarFileNames());
            value.put("retained_external_data_sidecar_paths", retainedExternalDataSidecarPaths().stream()
                    .map(Path::toString)
                    .toList());
            value.put("external_data_sidecar_file_names", externalDataSidecarFileNames());
            value.put("external_data_sidecar_paths", externalDataSidecarPaths.stream()
                    .map(Path::toString)
                    .toList());
            value.put("orphan_external_data_sidecar_files", orphanExternalDataSidecarFiles());
            value.put("orphan_external_data_sidecar_size_bytes", orphanExternalDataSidecarSizeBytes());
            value.put("orphan_external_data_sidecar_file_names", orphanExternalDataSidecarFileNames());
            value.put("orphan_external_data_sidecar_paths", orphanExternalDataSidecarPaths.stream()
                    .map(Path::toString)
                    .toList());
            value.put("stale_optimized_graph_file_names", staleOptimizedGraphFileNames());
            value.put("stale_optimized_graph_paths", staleOptimizedGraphPaths.stream()
                    .map(Path::toString)
                    .toList());
            value.put("unrelated_optimized_graph_file_names", unrelatedOptimizedGraphFileNames());
            value.put("unrelated_optimized_graph_paths", unrelatedOptimizedGraphPaths.stream()
                    .map(Path::toString)
                    .toList());
            return value;
        }

        private List<String> staleOptimizedGraphFileNames() {
            return staleOptimizedGraphPaths.stream()
                    .map(path -> path.getFileName().toString())
                    .toList();
        }

        private List<String> unrelatedOptimizedGraphFileNames() {
            return unrelatedOptimizedGraphPaths.stream()
                    .map(path -> path.getFileName().toString())
                    .toList();
        }

        private int orphanExternalDataSidecarFiles() {
            return orphanExternalDataSidecarPaths.size();
        }

        private long orphanExternalDataSidecarSizeBytes() {
            long total = 0L;
            for (Path path : orphanExternalDataSidecarPaths) {
                total += safeSize(path);
            }
            return total;
        }

        private List<String> externalDataSidecarFileNames() {
            return externalDataSidecarPaths.stream()
                    .map(path -> path.getFileName().toString())
                    .toList();
        }

        private List<String> orphanExternalDataSidecarFileNames() {
            return orphanExternalDataSidecarPaths.stream()
                    .map(path -> path.getFileName().toString())
                    .toList();
        }

        private int optimizedGraphFilesAfterStalePrune() {
            return Math.max(0, optimizedGraphFiles - staleOptimizedGraphFiles);
        }

        private List<Path> pruneableExternalDataSidecarPaths() {
            if (externalDataSidecarPaths.isEmpty()) {
                return List.of();
            }
            return optimizedGraphFilesAfterStalePrune() == 0 ? externalDataSidecarPaths : List.of();
        }

        private int pruneableExternalDataSidecarFiles() {
            return pruneableExternalDataSidecarPaths().size();
        }

        private long pruneableExternalDataSidecarSizeBytes() {
            long total = 0L;
            for (Path path : pruneableExternalDataSidecarPaths()) {
                total += safeSize(path);
            }
            return total;
        }

        private List<String> pruneableExternalDataSidecarFileNames() {
            return pruneableExternalDataSidecarPaths().stream()
                    .map(path -> path.getFileName().toString())
                    .toList();
        }

        private List<Path> retainedExternalDataSidecarPaths() {
            if (externalDataSidecarPaths.isEmpty()) {
                return List.of();
            }
            List<Path> pruneable = pruneableExternalDataSidecarPaths();
            if (pruneable.isEmpty()) {
                return externalDataSidecarPaths;
            }
            if (pruneable.size() == externalDataSidecarPaths.size()) {
                return List.of();
            }
            Set<Path> pruneableSet = new java.util.HashSet<>(pruneable);
            return externalDataSidecarPaths.stream()
                    .filter(path -> !pruneableSet.contains(path))
                    .toList();
        }

        private int retainedExternalDataSidecarFiles() {
            return retainedExternalDataSidecarPaths().size();
        }

        private long retainedExternalDataSidecarSizeBytes() {
            long total = 0L;
            for (Path path : retainedExternalDataSidecarPaths()) {
                total += safeSize(path);
            }
            return total;
        }

        private List<String> retainedExternalDataSidecarFileNames() {
            return retainedExternalDataSidecarPaths().stream()
                    .map(path -> path.getFileName().toString())
                    .toList();
        }

        private String externalDataSidecarPrunePolicy() {
            if (externalDataSidecarPaths.isEmpty()) {
                return "none";
            }
            if (optimizedGraphFiles == 0) {
                return "prune orphan sidecars; no optimized graphs are present";
            }
            if (optimizedGraphFilesAfterStalePrune() == 0) {
                return "prune sidecars with stale graphs; no optimized graphs remain after stale cleanup";
            }
            return "keep sidecars; optimized graphs remain after stale cleanup";
        }
    }

    private record CachePruneOperation(
            String kind,
            Path path,
            long sizeBytes,
            boolean deleted,
            boolean dryRun,
            String error) {
        private Map<String, Object> toJson() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("kind", kind);
            value.put("path", path.toString());
            value.put("file", path.getFileName().toString());
            value.put("size_bytes", sizeBytes);
            value.put("dry_run", dryRun);
            value.put("deleted", deleted);
            value.put("action", dryRun ? "would_delete" : (deleted ? "deleted" : (error == null ? "missing" : "failed")));
            if (error != null && !error.isBlank()) {
                value.put("error", error);
            }
            return value;
        }
    }

    private String ttsVoiceGroupSummary(JsonNode voices) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (JsonNode voice : voices) {
            String key = ttsVoiceLanguageLabel(voice) + " " + ttsVoiceGenderLabel(voice);
            counts.put(key, counts.getOrDefault(key, 0) + 1);
        }
        List<String> parts = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            parts.add(entry.getKey() + "=" + entry.getValue());
        }
        return parts.isEmpty() ? "(none)" : String.join(", ", parts);
    }

    private String ttsVoiceLanguageLabel(JsonNode voice) {
        String text = ttsVoiceSearchText(voice);
        if (text.contains("english") || text.contains(" en ") || text.startsWith("en ")) {
            return "en";
        }
        if (text.contains("japanese") || text.contains(" jp ") || text.startsWith("jp ")) {
            return "jp";
        }
        if (text.contains("chinese") || text.contains(" cn ") || text.contains(" zh ")
                || text.startsWith("cn ") || text.startsWith("zh ")) {
            return "zh";
        }
        return "other";
    }

    private String ttsVoiceGenderLabel(JsonNode voice) {
        String text = ttsVoiceSearchText(voice);
        if (text.contains("female")) {
            return "female";
        }
        if (text.contains("male")) {
            return "male";
        }
        return "unspecified";
    }

    private String ttsVoiceSearchText(JsonNode voice) {
        String raw = ttsVoiceId(voice) + " " + ttsVoiceGroup(voice) + " " + ttsVoiceDisplayName(voice);
        return " " + normalizeTtsVoiceFilterText(raw) + " ";
    }

    private String indentBlock(String text, String indent) {
        if (text == null || text.isBlank()) {
            return indent + "(empty)";
        }
        return indent + text.replace("\n", "\n" + indent);
    }

    private String nullToDash(String value) {
        return value == null || value.isBlank() ? "-" : value;
    }

    private Optional<Path> resolveMossTtsDirectory(String finalLocalPath, String modelId) {
        for (String candidate : new String[] { finalLocalPath, modelId }) {
            if (candidate == null || candidate.isBlank()) {
                continue;
            }
            Path path;
            try {
                path = Paths.get(candidate);
            } catch (Exception ignored) {
                continue;
            }
            Optional<Path> resolved = mossTtsDirectoryFrom(path);
            if (resolved.isPresent()) {
                return resolved;
            }
        }
        return Optional.empty();
    }

    private Optional<Path> mossTtsDirectoryFrom(Path path) {
        if (path == null) {
            return Optional.empty();
        }
        if (isMossTtsDirectory(path)) {
            return Optional.of(path);
        }
        Path parent = path.getParent();
        if (parent != null && isMossTtsDirectory(parent)) {
            return Optional.of(parent);
        }
        return Optional.empty();
    }

    private boolean isMossTtsDirectory(Path path) {
        return Files.isRegularFile(path.resolve("browser_poc_manifest.json"))
                && Files.isRegularFile(path.resolve("tts_browser_onnx_meta.json"));
    }

    private boolean allowsPromptlessExtensionRun(String modelId, String finalLocalPath) {
        return looksLikeWebWorldReference(modelId) || looksLikeWebWorldReference(finalLocalPath);
    }

    private boolean looksLikeWebWorldReference(String value) {
        if (value == null || value.isBlank()) {
            return false;
        }
        String normalized = value.toLowerCase(Locale.ROOT);
        if (normalized.contains("webworld")) {
            return true;
        }
        try {
            Path path = Paths.get(value);
            if (Files.isRegularFile(path)) {
                path = path.getParent();
            }
            if (path != null && Files.isDirectory(path)) {
                if (path.toAbsolutePath().normalize().toString().toLowerCase(Locale.ROOT).contains("webworld")) {
                    return true;
                }
                Path readme = path.resolve("README.md");
                if (Files.isRegularFile(readme)) {
                    String text = Files.readString(readme).toLowerCase(Locale.ROOT);
                    return text.contains("webworld") && text.contains("world model");
                }
            }
        } catch (Exception ignored) {
            // Treat non-path model references with the text checks above.
        }
        return false;
    }

    private String providerForModelDirectory(Path dirPath) {
        String detectedFormat = formatForModelDirectory(dirPath);
        String provider = providerForFormat(detectedFormat);
        return provider != null ? provider : "safetensor";
    }

    private String formatForModelDirectory(Path dirPath) {
        String detectedFormat = ManifestStore.detectFormat(dirPath);
        if (detectedFormat == null || detectedFormat.isBlank() || "unknown".equalsIgnoreCase(detectedFormat)) {
            return null;
        }
        return canonicalModelFormat(detectedFormat);
    }

    private String formatForLocalModelPath(Path modelPath) {
        if (modelPath == null || modelPath.getFileName() == null) {
            return null;
        }
        String name = modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf")) {
            return "gguf";
        }
        if (name.endsWith(".litertlm") || name.endsWith(".tflite") || name.endsWith(".task")) {
            return "litert";
        }
        if (name.endsWith(".onnx")) {
            return "onnx";
        }
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return "safetensors";
        }
        if (name.endsWith(".pt") || name.endsWith(".pth") || name.endsWith(".pts")) {
            return "torchscript";
        }
        return null;
    }

    private boolean shouldPreferOnnxResolution(String requestedModel, String requestedFormat, String requestedProvider) {
        if (isOnnxFormat(requestedFormat) || isOnnxProvider(requestedProvider)) {
            return true;
        }
        if ((requestedFormat != null && !requestedFormat.isBlank())
                || (requestedProvider != null && !requestedProvider.isBlank())) {
            return false;
        }
        return looksLikeOnnxModelRef(requestedModel);
    }

    private String preferredLocalIndexFormat(String requestedModel, String requestedFormat, String requestedProvider) {
        String canonicalFormat = canonicalModelFormat(requestedFormat);
        if (canonicalFormat != null) {
            return canonicalFormat;
        }
        if (isOnnxProvider(requestedProvider) || looksLikeOnnxModelRef(requestedModel)) {
            return "onnx";
        }
        return null;
    }

    private boolean isOnnxProvider(String requestedProvider) {
        return requestedProvider != null && "onnx".equalsIgnoreCase(requestedProvider.trim());
    }

    private boolean isOnnxFormat(String requestedFormat) {
        return "onnx".equals(canonicalModelFormat(requestedFormat));
    }

    private String canonicalModelFormat(String requestedFormat) {
        if (requestedFormat == null || requestedFormat.isBlank()) {
            return null;
        }
        return switch (requestedFormat.trim().toLowerCase(Locale.ROOT)) {
            case "onnx" -> "onnx";
            case "safetensor", "safetensors" -> "safetensors";
            case "litert", "tflite", "task", "litertlm" -> "litert";
            case "gguf" -> "gguf";
            case "torchscript", "pytorch", "pt", "pth", "pts" -> "torchscript";
            default -> requestedFormat.trim().toLowerCase(Locale.ROOT);
        };
    }

    private boolean looksLikeOnnxModelRef(String requestedModel) {
        String normalized = stripHfPrefix(requestedModel).toLowerCase(Locale.ROOT);
        return normalized.startsWith("onnx-community/")
                || normalized.endsWith("-onnx")
                || normalized.endsWith("_onnx")
                || normalized.contains("/onnx")
                || normalized.contains("-onnx-")
                || normalized.contains("_onnx_");
    }

    private String stripHfPrefix(String value) {
        if (value == null) {
            return "";
        }
        String trimmed = value.trim();
        if (trimmed.startsWith("hf:")) {
            return trimmed.substring("hf:".length()).trim();
        }
        if (trimmed.startsWith("huggingface:")) {
            return trimmed.substring("huggingface:".length()).trim();
        }
        return trimmed;
    }

    private LocalModelIndex.Entry autoPullHfRepositoryForRunIfNeeded(String requestedModel, String preferredFormat)
            throws Exception {
        String repoId = hfRepoIdFromModelRef(requestedModel);
        if (repoId == null || isCloudProviderName(providerId) || isMcpProvider()) {
            return null;
        }

        Optional<LocalModelIndex.Entry> existing = findLocalIndexEntry(repoId, preferredFormat);
        if (existing.isPresent()) {
            return existing.get();
        }

        if (offline) {
            System.err.println("Error: Model " + repoId
                    + " is not installed locally, and --offline/--local disables automatic downloads.");
            System.err.println("Run once online: gollek pull " + repoId);
            return null;
        }
        if (huggingFaceClient == null || huggingFaceClient.isUnsatisfied()) {
            throw new IllegalStateException("Hugging Face repository downloader is not available in this runtime");
        }

        String revision = branch == null || branch.isBlank() ? "main" : branch.trim();
        String manifestName = GollekManifest.computeName(repoId, revision);
        Path targetDir = ManifestStore.resolveBlobDir(repoId, manifestName);
        Files.createDirectories(targetDir);

        if (!ttsVoicesJsonOutput()) {
            System.out.println("Model not found locally; pulling hf:" + repoId
                    + ("main".equalsIgnoreCase(revision) ? "" : "@" + revision) + "...");
        }
        HuggingFaceClient client = huggingFaceClient.get();
        List<String> files = client.listFiles(repoId, revision);
        try (CodecPullProgressRenderer progress = new CodecPullProgressRenderer(files == null ? 0 : files.size())) {
            client.downloadRepository(repoId, revision, targetDir, progress);
        }
        saveRunHfDownloadManifest(repoId, targetDir, files, manifestName, revision, preferredFormat);
        LocalModelIndex.refreshFromDisk();

        Optional<LocalModelIndex.Entry> downloaded = findLocalIndexEntry(repoId, preferredFormat);
        if (downloaded.isPresent()) {
            return downloaded.get();
        }
        throw new IllegalStateException("download finished, but no local"
                + (preferredFormat == null ? "" : " " + preferredFormat)
                + " artifact was indexed for " + repoId);
    }

    private Optional<LocalModelIndex.Entry> findLocalIndexEntry(String repoId, String preferredFormat) {
        for (String ref : List.of(repoId, "hf:" + repoId)) {
            try {
                Optional<LocalModelIndex.Entry> entry = preferredFormat == null
                        ? LocalModelIndex.find(ref)
                        : LocalModelIndex.find(ref, preferredFormat);
                if (entry.isPresent() && entry.get().path != null && Files.exists(Path.of(entry.get().path))) {
                    return entry;
                }
            } catch (Exception ignored) {
                // Best effort lookup; caller will produce the final diagnostic.
            }
        }
        return Optional.empty();
    }

    private String hfRepoIdFromModelRef(String requestedModel) {
        String value = stripHfPrefix(requestedModel);
        if (value.isBlank()
                || value.contains("://")
                || value.startsWith("/")
                || value.startsWith(".")
                || value.startsWith("~")
                || value.contains("\\")
                || !value.contains("/")) {
            return null;
        }
        String[] parts = value.split("/");
        if (parts.length != 2 || parts[0].isBlank() || parts[1].isBlank()) {
            return null;
        }
        return value;
    }

    private boolean isCloudProviderName(String provider) {
        if (provider == null || provider.isBlank()) {
            return false;
        }
        return switch (provider.trim().toLowerCase(Locale.ROOT)) {
            case "openai", "mistral", "anthropic", "gemini", "cerebras", "deepseek", "ollama" -> true;
            default -> false;
        };
    }

    private boolean requiresLocalHfRepositoryResolution(String requestedModel, String requestedProvider) {
        return hfRepoIdFromModelRef(requestedModel) != null
                && !isCloudProviderName(requestedProvider)
                && !isMcpProvider();
    }

    private void saveRunHfDownloadManifest(
            String repoId,
            Path targetDir,
            List<String> files,
            String manifestName,
            String revision,
            String preferredFormat) throws Exception {
        ManifestStore store = manifestStore != null && !manifestStore.isUnsatisfied()
                ? manifestStore.get()
                : new ManifestStore();

        String detectedFormat = ManifestStore.detectFormat(targetDir);
        String manifestFormat = "unknown".equalsIgnoreCase(detectedFormat) && preferredFormat != null
                ? preferredFormat
                : detectedFormat;
        GollekManifest manifest = new GollekManifest();
        manifest.setId(manifestName);
        manifest.setModelId(repoId);
        manifest.setName(manifestName);
        manifest.setFormat(manifestFormat);
        manifest.setPipeline(files != null && files.stream().anyMatch(name -> name.endsWith("model_index.json")));
        manifest.setSource("huggingface");
        manifest.setRepository(repoId);
        manifest.setBranch(revision == null || revision.isBlank() ? "main" : revision);
        manifest.setBlobPath(targetDir.toAbsolutePath().toString());
        manifest.setFiles(ManifestStore.listBlobFiles(targetDir));
        manifest.setCreatedAt(Instant.now());
        manifest.setSizeBytes(computePathSize(targetDir));
        manifest.setArchitecture(ManifestStore.detectArchitecture(manifest));

        Map<String, String> metadata = new LinkedHashMap<>();
        metadata.put("fallback", "automatic-run-huggingface-repository");
        metadata.put("format", manifestFormat != null ? manifestFormat : "unknown");
        if (preferredFormat != null) {
            metadata.put("preferred_format", preferredFormat);
        }
        manifest.setMetadata(metadata);

        store.save(manifest);
    }

    private String ttsVoiceId(JsonNode voice) {
        for (String field : List.of("voice", "id", "name", "display_name")) {
            String value = voice.path(field).asText("");
            if (!value.isBlank()) {
                return value;
            }
        }
        return "default";
    }

    private String ttsVoiceDisplayName(JsonNode voice) {
        String displayName = voice.path("display_name").asText("");
        return displayName.isBlank() ? ttsVoiceId(voice) : displayName;
    }

    private String ttsVoiceGroup(JsonNode voice) {
        String group = voice.path("group").asText("");
        return group.isBlank() ? "-" : group;
    }

    private String truncateDisplay(String value, int width) {
        if (value == null) {
            return "";
        }
        if (value.length() <= width) {
            return value;
        }
        return value.substring(0, Math.max(0, width - 3)) + "...";
    }

    private String resolveTtsCodecReference(String ref) {
        String value = ref == null ? "" : ref.trim();
        if (value.isEmpty()) {
            return value;
        }

        try {
            Path direct = Paths.get(value);
            if (Files.exists(direct)) {
                return direct.toAbsolutePath().normalize().toString();
            }
        } catch (Exception ignored) {
            // Not a local path; try the model index below.
        }

        try {
            Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(value);
            if (indexed.isPresent()) {
                String path = indexed.get().path;
                if (path != null && !path.isBlank() && Files.exists(Path.of(path))) {
                    return Path.of(path).toAbsolutePath().normalize().toString();
                }
            }
        } catch (Exception ignored) {
            // Provider will emit the final codec-not-found diagnostic.
        }

        return value;
    }

    private boolean ensureMossTtsCodecAvailableIfNeeded(String modelId, String finalLocalPath) {
        if (providerId != null && !providerId.isBlank() && !"onnx".equalsIgnoreCase(providerId)) {
            return true;
        }
        if (ttsCodec != null && !ttsCodec.isBlank()) {
            return true;
        }
        if (resolveMossTtsDirectory(finalLocalPath, modelId).isEmpty()) {
            return true;
        }
        Optional<Path> installedCodec = findInstalledMossAudioCodecDir();
        if (installedCodec.isPresent()) {
            return true;
        }
        if (offline) {
            System.err.println("Error: MOSS TTS companion codec is not installed, and --offline/--local disables automatic downloads.");
            System.err.println("Run once online: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
            return false;
        }
        if (disableTtsAutoCodec) {
            System.err.println("Error: MOSS TTS companion codec is not installed, and automatic codec pull is disabled.");
            System.err.println("Pull it with: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
            return false;
        }

        System.out.println("MOSS TTS companion codec not found; pulling " + MOSS_AUDIO_TOKENIZER_REPOSITORY + "...");
        try {
            pullMossAudioCodecRepository();
            LocalModelIndex.refreshFromDisk();
        } catch (Exception e) {
            System.err.println("Error: Failed to pull MOSS TTS companion codec: " + e.getMessage());
            if (e.getCause() != null && e.getCause().getMessage() != null) {
                System.err.println("Detail: " + e.getCause().getMessage());
            }
            System.err.println("You can also run manually: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
            return false;
        }

        installedCodec = findInstalledMossAudioCodecDir();
        if (installedCodec.isPresent()) {
            System.out.println("MOSS TTS companion codec ready at: " + displayPath(installedCodec.get()));
            return true;
        }

        System.err.println("Error: Codec pull finished, but " + MOSS_CODEC_META_FILE + " was not found.");
        System.err.println("Try manually: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY);
        return false;
    }

    private Optional<Path> findInstalledMossAudioCodecDir() {
        for (String ref : List.of(MOSS_AUDIO_TOKENIZER_REPOSITORY, MOSS_AUDIO_TOKENIZER_MODEL_SPEC)) {
            try {
                Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(ref);
                if (indexed.isPresent() && indexed.get().path != null) {
                    Optional<Path> dir = mossCodecDirectoryFrom(Path.of(indexed.get().path));
                    if (dir.isPresent()) {
                        return dir;
                    }
                }
            } catch (Exception ignored) {
                // Fall back to scanning the model directory.
            }
        }

        Path modelsRoot = GollekHome.path("models");
        if (!Files.isDirectory(modelsRoot)) {
            return Optional.empty();
        }
        try (Stream<Path> walk = Files.walk(modelsRoot, 8)) {
            List<Path> codecDirs = walk.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().equals(MOSS_CODEC_META_FILE))
                    .map(Path::getParent)
                    .filter(Objects::nonNull)
                    .distinct()
                    .sorted((left, right) -> left.toString().compareToIgnoreCase(right.toString()))
                    .toList();
            Optional<Path> official = codecDirs.stream()
                    .filter(path -> pathLooksLikeMossAudioTokenizer(path))
                    .findFirst();
            return official.isPresent() ? official : codecDirs.stream().findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private Optional<Path> mossCodecDirectoryFrom(Path path) {
        if (path == null) {
            return Optional.empty();
        }
        if (Files.isRegularFile(path)) {
            if (path.getFileName().toString().equals(MOSS_CODEC_META_FILE)) {
                return Optional.ofNullable(path.getParent());
            }
            Path parent = path.getParent();
            return isMossAudioCodecDirectory(parent) ? Optional.of(parent) : Optional.empty();
        }
        if (isMossAudioCodecDirectory(path)) {
            return Optional.of(path);
        }
        if (Files.isDirectory(path)) {
            try (Stream<Path> walk = Files.walk(path, 6)) {
                return walk.filter(Files::isRegularFile)
                        .filter(file -> file.getFileName().toString().equals(MOSS_CODEC_META_FILE))
                        .map(Path::getParent)
                        .findFirst();
            } catch (Exception ignored) {
                return Optional.empty();
            }
        }
        return Optional.empty();
    }

    private boolean isMossAudioCodecDirectory(Path path) {
        return path != null && Files.isRegularFile(path.resolve(MOSS_CODEC_META_FILE));
    }

    private boolean pathLooksLikeMossAudioTokenizer(Path path) {
        if (path == null) {
            return false;
        }
        String normalizedPath = normalizeModelRef(path.toAbsolutePath().normalize().toString());
        String normalizedRepo = normalizeModelRef(MOSS_AUDIO_TOKENIZER_REPOSITORY);
        return normalizedPath.contains(normalizedRepo);
    }

    private String normalizeModelRef(String value) {
        return value == null
                ? ""
                : value.toLowerCase(Locale.ROOT)
                        .replace("\\", "/")
                        .replace("__", "/")
                        .replace("--", "/")
                        .replaceAll("[^a-z0-9]+", "/")
                        .replaceAll("/+", "/")
                        .replaceAll("^/|/$", "");
    }

    private void pullMossAudioCodecRepository() throws Exception {
        if (huggingFaceClient == null || huggingFaceClient.isUnsatisfied()) {
            throw new IllegalStateException("Hugging Face repository downloader is not available in this runtime");
        }

        String repoId = MOSS_AUDIO_TOKENIZER_REPOSITORY;
        String manifestName = GollekManifest.computeName(repoId, "main");
        Path targetDir = ManifestStore.resolveBlobDir(repoId, manifestName);
        Files.createDirectories(targetDir);

        HuggingFaceClient client = huggingFaceClient.get();
        List<String> files = client.listFiles(repoId);
        try (CodecPullProgressRenderer progress = new CodecPullProgressRenderer(files.size())) {
            client.downloadRepository(repoId, targetDir, progress);
        }
        saveMossAudioCodecManifest(repoId, targetDir, files, manifestName);
    }

    private void saveMossAudioCodecManifest(String repoId, Path targetDir, List<String> files, String manifestName)
            throws Exception {
        ManifestStore store = manifestStore != null && !manifestStore.isUnsatisfied()
                ? manifestStore.get()
                : new ManifestStore();

        String format = ManifestStore.detectFormat(targetDir);
        GollekManifest manifest = new GollekManifest();
        manifest.setId(manifestName);
        manifest.setModelId(repoId);
        manifest.setName(manifestName);
        manifest.setFormat(format);
        manifest.setPipeline(files != null && files.stream().anyMatch(name -> name.endsWith("model_index.json")));
        manifest.setSource("huggingface");
        manifest.setRepository(repoId);
        manifest.setBranch("main");
        manifest.setBlobPath(targetDir.toAbsolutePath().toString());
        manifest.setFiles(ManifestStore.listBlobFiles(targetDir));
        manifest.setCreatedAt(Instant.now());
        manifest.setSizeBytes(computePathSize(targetDir));
        manifest.setArchitecture("moss-audio-tokenizer");
        manifest.setMetadata(Map.of(
                "fallback", "automatic-moss-tts-codec",
                "format", format != null ? format : "unknown"));

        store.save(manifest);
    }

    private long computePathSize(Path path) {
        if (path == null || !Files.exists(path)) {
            return 0L;
        }
        try {
            if (Files.isRegularFile(path)) {
                return Files.size(path);
            }
            try (Stream<Path> walk = Files.walk(path)) {
                return walk.filter(Files::isRegularFile)
                        .mapToLong(file -> {
                            try {
                                return Files.size(file);
                            } catch (Exception ignored) {
                                return 0L;
                            }
                        })
                        .sum();
            }
        } catch (Exception ignored) {
            return 0L;
        }
    }

    private String displayPath(Path path) {
        if (path == null) {
            return "";
        }
        String value = path.toAbsolutePath().normalize().toString();
        String userHome = System.getProperty("user.home");
        if (userHome != null && value.startsWith(userHome)) {
            return "~" + value.substring(userHome.length());
        }
        return value;
    }

    private boolean allowCpuFallbackWhenMetalRequested() {
        return CliMetalRuntime.allowCpuFallbackWhenMetalRequested();
    }

    private static final class CodecPullProgressRenderer implements DownloadProgressListener, AutoCloseable {
        private static final String[] SPINNER = { "|", "/", "-", "\\" };
        private static final int BAR_WIDTH = 30;
        private static final long MIN_REDRAW_NS = 80_000_000L;

        private final int totalFiles;
        private int completedFiles;
        private int spinnerTick;
        private long fileStartNanos = System.nanoTime();
        private long lastRedrawNanos;

        private CodecPullProgressRenderer(int totalFiles) {
            this.totalFiles = Math.max(1, totalFiles);
        }

        @Override
        public synchronized void onProgress(long downloadedBytes, long totalBytes, double progress) {
            long now = System.nanoTime();
            if (now - lastRedrawNanos < MIN_REDRAW_NS && downloadedBytes < totalBytes) {
                return;
            }
            lastRedrawNanos = now;

            double pct = Math.max(0.0, Math.min(1.0, progress));
            int filled = Math.min(BAR_WIDTH, (int) Math.round(BAR_WIDTH * pct));
            StringBuilder bar = new StringBuilder(BAR_WIDTH);
            for (int i = 0; i < BAR_WIDTH; i++) {
                bar.append(i < filled ? "=" : ".");
            }

            double elapsedSec = Math.max(0.001, (now - fileStartNanos) / 1_000_000_000.0);
            double mbDone = downloadedBytes / 1024.0 / 1024.0;
            double mbTotal = totalBytes > 0 ? totalBytes / 1024.0 / 1024.0 : 0.0;
            double speed = mbDone / elapsedSec;
            String spin = SPINNER[spinnerTick++ % SPINNER.length];

            System.out.printf("\r%s [%-30s] %3d%% %.1f/%.1f MB %.1f MB/s file %d/%d",
                    spin,
                    bar,
                    (int) Math.round(pct * 100.0),
                    mbDone,
                    mbTotal,
                    speed,
                    Math.min(completedFiles + 1, totalFiles),
                    totalFiles);
            System.out.flush();
        }

        @Override
        public synchronized void onComplete(long totalBytes) {
            completedFiles++;
            fileStartNanos = System.nanoTime();
            System.out.printf("\r* [%-30s] 100%% file %d/%d done%n",
                    "==============================",
                    Math.min(completedFiles, totalFiles),
                    totalFiles);
            System.out.flush();
        }

        @Override
        public void close() {
            System.out.flush();
        }
    }

    private void printQuantizationInfo() {
        if (quantizeStrategy == null || quantizeStrategy.isBlank() || noBanner) {
            return;
        }

        String effective = effectiveQuantizationStrategy(quantizeStrategy);
        String effectiveKv = effectiveKvQuantization(quantizeKv);
        System.out.println("Quantization: " + quantizeStrategy.toLowerCase()
                + (effective.equalsIgnoreCase(quantizeStrategy) ? "" : " -> " + effective)
                + " (" + quantizeBits + "-bit)");
        if (quantizeKv != null && !quantizeKv.equalsIgnoreCase("none")) {
            System.out.println("KV cache quantization: " + quantizeKv.toLowerCase()
                    + (effectiveKv.equalsIgnoreCase(quantizeKv) ? "" : " -> " + effectiveKv));
        } else {
            System.out.println("KV cache quantization: off");
        }
        System.out.println("--------------------------------------------------");
    }

    private boolean shouldUseLocalModelPath(String providerId, String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        if (providerId == null || providerId.isBlank()) {
            return false;
        }
        String normalized = providerId.trim().toLowerCase();
        return normalized.equals("safetensor")
                || normalized.equals("gguf")
                || normalized.equals("native")
                || normalized.equals("litert")
                || normalized.equals("onnx")
                || normalized.equals("libtorch");
    }

    private boolean shouldUseDirectSafetensorRunPath(String currentProvider, String localPath) {
        Path checkpointPath = resolveSafetensorCheckpointPath(localPath);
        return DirectSafetensorRoutePolicy.shouldUseDirectRun(
                currentProvider,
                checkpointPath,
                checkpointPath != null && isSafetensorCheckpointDir(checkpointPath.toString()),
                sessionId,
                grammar,
                prompt,
                directInferenceEngine() != null);
    }

    private String effectiveRunSystemPrompt(String providerId, String localPath) {
        return DirectSafetensorRoutePolicy.effectiveSystemPrompt(
                systemPrompt,
                directSafetensorRunProfile(providerId, localPath));
    }

    private DirectSafetensorRunProfile directSafetensorRunProfile(String providerId, String localPath) {
        return DirectSafetensorRoutePolicy.profileForProvider(
                providerId,
                resolveSafetensorCheckpointPath(localPath));
    }

    private boolean shouldUseDirectLiteRtStreamPath(String currentProvider, String localPath) {
        return DirectSafetensorRoutePolicy.shouldUseDirectLiteRtStreamPath(currentProvider, localPath);
    }

    private InferenceResponse runDirectSafetensorCompletion(String localPath, String userPrompt,
            String effectiveSystemPrompt) throws Exception {
        Path modelPath = resolveSafetensorCheckpointPath(localPath);
        if (modelPath == null) {
            throw new IllegalArgumentException("Invalid safetensor model path: " + localPath);
        }
        DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(modelPath);
        float effectiveRepeatPenalty = DirectSafetensorTextPolicy.normalizeRepeatPenalty(profile, repeatPenalty);
        GenerationConfig.KvCacheQuantization kvQuant = GenerationConfig.KvCacheQuantization.NONE;
        if ("int8".equalsIgnoreCase(quantizeKv)) {
            kvQuant = GenerationConfig.KvCacheQuantization.INT8;
        } else if ("int4".equalsIgnoreCase(quantizeKv) || "turbo".equalsIgnoreCase(quantizeKv)) {
            kvQuant = GenerationConfig.KvCacheQuantization.INT4;
        }

        GenerationConfig config = GenerationConfig.builder()
                .maxNewTokens(maxTokens)
                .strategy(resolveDirectSamplingStrategy())
                .temperature((float) temperature)
                .topK(topK)
                .topP((float) topP)
                .repetitionPenalty(effectiveRepeatPenalty)
                .kvCacheQuant(kvQuant)
                .seed(seed != null ? seed.longValue() : -1L)
                .build();
        String preparedPrompt = DirectSafetensorTextPolicy.preparePrompt(profile, effectiveSystemPrompt, userPrompt);
        
        directInferenceEngine().loadModel(modelPath, null, quantizeStrategy);
        System.err.println("Directly generating...");
        InferenceResponse response = directInferenceEngine().generate(preparedPrompt, modelPath, config)
                .await()
                .atMost(Duration.ofMinutes(5));
        System.err.println("Generated response: '" + response.getContent() + "'");
        return DirectSafetensorTextPolicy.sanitizeResponse(response, profile);
    }

    private Path resolveSafetensorCheckpointPath(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return null;
        }
        try {
            Path path = Path.of(localPath).toAbsolutePath().normalize();
            if (Files.isDirectory(path)) {
                return path;
            }
            if (!Files.isRegularFile(path)) {
                return null;
            }
            String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
            if (!name.endsWith(".safetensors")) {
                return null;
            }
            Path parent = path.getParent();
            return parent != null && Files.isDirectory(parent) ? parent : null;
        } catch (Exception ignored) {
            return null;
        }
    }

    private GenerationConfig.SamplingStrategy resolveDirectSamplingStrategy() {
        if (temperature < 1.0e-4f || topK == 1) {
            return GenerationConfig.SamplingStrategy.GREEDY;
        }
        boolean hasTopK = topK > 0;
        boolean hasTopP = topP > 0.0 && topP < 1.0;
        if (hasTopK && hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_K_TOP_P;
        }
        if (hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_P;
        }
        if (hasTopK) {
            return GenerationConfig.SamplingStrategy.TOP_K;
        }
        return GenerationConfig.SamplingStrategy.GREEDY;
    }

    private DirectInferenceEngine directInferenceEngine() {
        if (directInferenceEngine != null) {
            return directInferenceEngine;
        }
        try {
            var instance = Arc.container().instance(DirectInferenceEngine.class);
            if (instance != null && instance.isAvailable()) {
                return instance.get();
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private boolean isSafetensorCheckpointDir(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        try {
            Path path = Path.of(localPath);
            if (!Files.isDirectory(path)) {
                return false;
            }
            if (!Files.exists(path.resolve("config.json"))) {
                return false;
            }
            return Files.exists(path.resolve("model.safetensors"))
                    || Files.exists(path.resolve("model.safetensor"))
                    || Files.exists(path.resolve("model.safetensors.index.json"));
        } catch (Exception ignored) {
            return false;
        }
    }

    private String detectUnsupportedLiteRtPreflight(String provider, String localPath) {
        if (!"litert".equalsIgnoreCase(provider) || localPath == null || localPath.isBlank()) {
            return null;
        }
        if (Boolean.getBoolean("gollek.litert.enable_experimental_gemma4_task_runner")
                || Boolean.getBoolean("gollek.litert.enable_experimental_raw_litertlm")) {
            return null;
        }
        try {
            Path path = Path.of(localPath);
            if (!Files.exists(path) || !looksLikeGemma4LiteRtArtifact(path)) {
                return null;
            }
            if (hasNativeGemmaLiteRtLm(path)) {
                return null;
            }
            return "Gemma 4 LiteRT runner is disabled by default because the current Java/Metal paths are slow "
                    + "and produce incorrect repeated-token output for this export. Set -D"
                    + "gollek.litert.enable_experimental_gemma4_task_runner=true only for diagnostics.";
        } catch (Exception ignored) {
            return null;
        }
    }

    private boolean looksLikeGemma4LiteRtArtifact(Path path) {
        String lower = path.toString().toLowerCase(Locale.ROOT);
        if (Files.isRegularFile(path)) {
            if (!lower.contains("gemma-4") && !lower.contains("gemma4")) {
                return false;
            }
            String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
            return isLiteRtModelFileName(name);
        }
        if (!Files.isDirectory(path)) {
            return false;
        }
        try (var stream = Files.list(path)) {
            return stream.anyMatch(candidate -> {
                String name = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
                return (name.contains("gemma-4") || name.contains("gemma4"))
                        && isLiteRtModelFileName(name);
            });
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean hasNativeGemmaLiteRtLm(Path path) {
        try {
            if (Files.isRegularFile(path)) {
                String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
                return name.endsWith(".litertlm")
                        && (name.contains("gemma-4") || name.contains("gemma4"));
            }
            if (!Files.isDirectory(path)) {
                return false;
            }
            try (var stream = Files.list(path)) {
                return stream.anyMatch(candidate -> {
                    String name = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
                    return Files.isRegularFile(candidate)
                            && name.endsWith(".litertlm")
                            && (name.contains("gemma-4") || name.contains("gemma4"))
                            && !name.contains("qualcomm");
                });
            }
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean isLiteRtModelFileName(String name) {
        return name.endsWith(".litertlm")
                || name.endsWith(".task")
                || name.endsWith(".tflite")
                || name.endsWith(".tfl");
    }

    private String detectSafetensorCompatibilityIssue(String providerId, String localPath) {
        if (!"safetensor".equalsIgnoreCase(providerId) || localPath == null || localPath.isBlank()) {
            return null;
        }
        if (!isSafetensorCheckpointDir(localPath)) {
            return null;
        }

        try {
            Path configPath = Path.of(localPath).resolve("config.json");
            ModelConfig parsed = new ModelConfigLoader(new ObjectMapper()).load(configPath);
            String modelType = parsed.getModelType() == null ? "" : parsed.getModelType().toLowerCase(Locale.ROOT);
            String architecture = parsed.getPrimaryArchitecture() == null
                    ? ""
                    : parsed.getPrimaryArchitecture().toLowerCase(Locale.ROOT);
            boolean gemma4UnifiedWrapper = isGemma4UnifiedWrapper(modelType, architecture);
            boolean resolvedGemma4Text = modelType.startsWith("gemma4")
                    && parsed.getHiddenSize() > 0
                    && parsed.getNumHiddenLayers() > 0
                    && parsed.getNumAttentionHeads() > 0;
            if (resolvedGemma4Text) {
                return null;
            }
            if (gemma4UnifiedWrapper) {
                return "this local safetensor runtime build recognizes Gemma 4 unified checkpoints like "
                        + Path.of(localPath).getFileName()
                        + ", but does not run their unified multimodal embedder path yet. "
                        + "Use a runtime plugin with Gemma 4 unified support, or convert to a supported artifact.";
            }
        } catch (Exception ignored) {
            // Fall back to raw config heuristics only when structured parsing cannot
            // reconcile the checkpoint into a supported text model shape.
        }

        try {
            Path configPath = Path.of(localPath).resolve("config.json");
            String config = Files.readString(configPath);
            boolean gemma4Conditional = config.contains("\"Gemma4ForConditionalGeneration\"")
                    || config.contains("\"Gemma4ForMultimodalLM\"")
                    || config.contains("\"Gemma4ForImageTextToText\"")
                    || config.contains("\"model_type\": \"gemma4\"");
            boolean gemma4Unified = config.contains("\"model_type\": \"gemma4_unified\"")
                    || config.contains("\"Gemma4ForMultimodalLM\"")
                    || config.contains("\"Gemma4ForImageTextToText\"");
            boolean multimodalWrapper = config.contains("\"vision_config\"")
                    || config.contains("\"audio_config\"");
            boolean hasTextConfig = config.contains("\"text_config\"");

            if (gemma4Unified && multimodalWrapper && hasTextConfig) {
                return null;
            }
            if (gemma4Unified && multimodalWrapper) {
                return "this local safetensor runtime build recognizes Gemma 4 unified checkpoints like "
                        + Path.of(localPath).getFileName()
                        + ", but does not run their unified multimodal embedder path yet. "
                        + "Use a runtime plugin with Gemma 4 unified support, or convert to a supported artifact.";
            }
            if (gemma4Conditional && multimodalWrapper && !hasTextConfig) {
                return "this local safetensor runtime build does not reliably support Gemma4 multimodal text checkpoints like "
                        + Path.of(localPath).getFileName()
                        + ". It may run but produce incorrect output. Use a newer safetensor engine build or convert the model to a supported runtime first.";
            }
        } catch (Exception ignored) {
            return null;
        }

        return null;
    }

    private boolean isGemma4UnifiedWrapper(String modelType, String architecture) {
        return modelType.equals("gemma4_unified")
                || architecture.equals("gemma4formultimodallm")
                || architecture.equals("gemma4forimagetexttotext");
    }

    private String detectNativeGgufCompatibilityIssue(String providerId, String modelId, String localPath) {
        if (!"native".equalsIgnoreCase(providerId) || !isGgufModelPath(localPath)) {
            return null;
        }

        String fingerprint = ((modelId == null ? "" : modelId) + " "
                + (localPath == null ? "" : Path.of(localPath).getFileName())).toLowerCase(java.util.Locale.ROOT);
        if (!fingerprint.contains("gemma4") && !fingerprint.contains("gemma-4")) {
            return null;
        }

        return "the local native GGUF runtime in this build does not reliably support Gemma 4 text checkpoints like "
                + Path.of(localPath).getFileName()
                + ". It can tokenize the prompt but still ignore Gemma 4 attention/KV/RoPE semantics and produce incorrect output. "
                + "Use a newer Gemma 4-capable runtime instead of the local GGUF/native fallback.";
    }

    private boolean isGgufModelPath(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        try {
            Path path = Path.of(localPath);
            return Files.isRegularFile(path)
                    && path.getFileName().toString().toLowerCase(java.util.Locale.ROOT).endsWith(".gguf");
        } catch (Exception ignored) {
            return false;
        }
    }

    private String effectiveQuantizationStrategy(String raw) {
        String normalized = raw == null ? "" : raw.trim().toLowerCase();
        return switch (normalized) {
            case "turbo" -> "bnb";
            case "gptq", "autoround" -> "int4";
            default -> normalized;
        };
    }

    private String effectiveKvQuantization(String raw) {
        String normalized = raw == null ? "" : raw.trim().toLowerCase();
        return switch (normalized) {
            case "", "none" -> "off";
            case "int8" -> "int8";
            case "int4" -> "int4";
            case "turbo" -> "int4";
            default -> normalized;
        };
    }

    private boolean isMcpProvider() {
        return providerId != null && "mcp".equalsIgnoreCase(providerId.trim());
    }

    boolean shouldTryStandaloneLiteRtFastPath(String localPath) {
        if (routeReportJson) {
            return false;
        }
        if (!Boolean.parseBoolean(System.getProperty("gollek.cli.enable_standalone_litert_fast_path", "true"))) {
            return false;
        }
        if (prompt == null || prompt.isBlank()) {
            return false;
        }
        if (providerId != null && !providerId.isBlank() && !isLiteRtProviderAlias(providerId)) {
            return false;
        }
        return looksLikeLiteRtLmPath(localPath) && isSimpleStandaloneTextRun();
    }

    String[] buildStandaloneLiteRtFastRunArgs(String localPath) {
        List<String> args = new java.util.ArrayList<>();
        args.add("run");
        args.add("--modelFile");
        args.add(localPath);
        args.add("--prompt");
        args.add(prompt);
        args.add("--max-tokens");
        args.add(Integer.toString(maxTokens));
        args.add("--temperature");
        args.add(Double.toString(temperature));
        args.add("--top-k");
        args.add(Integer.toString(topK));
        args.add("--top-p");
        args.add(Double.toString(topP));
        args.add("--provider");
        args.add("litert");
        return args.toArray(String[]::new);
    }

    private boolean tryStandaloneLiteRtFastPath(String localPath) {
        if (!shouldTryStandaloneLiteRtFastPath(localPath)) {
            return false;
        }
        int status = LiteRtLmFastRun.run(buildStandaloneLiteRtFastRunArgs(localPath));
        if (status == 42) {
            return false;
        }
        if (status == 0) {
            requestProcessExit();
            return true;
        }
        System.exit(status);
        return true;
    }

    private boolean tryCachedGemma4MobileQatLiteRtFastPath(boolean providerExplicit) {
        return tryCachedGemma4MobileQatLiteRtFastPath(providerExplicit, providerId);
    }

    private boolean tryCachedGemma4MobileQatLiteRtFastPath(boolean providerExplicit, String requestedProviderId) {
        if (!shouldTryCachedGemma4MobileQatLiteRtFastPath(providerExplicit, requestedProviderId)) {
            return false;
        }
        Optional<LocalModelIndex.Entry> indexed = findSafetensorIndexEntry(modelId);
        if (indexed.isEmpty()) {
            return false;
        }
        LocalModelIndex.Entry entry = indexed.get();
        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCachedGemma4MobileQatLiteRtAlternateRuntime(
                        "safetensor",
                        entry.path,
                        entry.path,
                        providerExplicit,
                        preferAlternateRuntime);
        if (!selection.selected()) {
            return false;
        }
        providerId = selection.provider();
        int status = LiteRtLmFastRun.run(buildStandaloneLiteRtFastRunArgs(selection.localPath()));
        if (status == 42) {
            return false;
        }
        if (status == 0) {
            requestProcessExit();
            return true;
        }
        System.exit(status);
        return true;
    }

    boolean shouldTryCachedGemma4MobileQatLiteRtFastPath(boolean providerExplicit) {
        return shouldTryCachedGemma4MobileQatLiteRtFastPath(providerExplicit, providerId);
    }

    boolean shouldTryCachedGemma4MobileQatLiteRtFastPath(boolean providerExplicit, String requestedProviderId) {
        if (!Boolean.parseBoolean(System.getProperty("gollek.cli.enable_cached_gemma4_litert_fast_path", "true"))) {
            return false;
        }
        if (modelId == null || modelId.isBlank()) {
            return false;
        }
        if (modelFile != null && !modelFile.isBlank()) {
            return false;
        }
        if (modelDir != null && !modelDir.isBlank()) {
            return false;
        }
        if (modelPath != null && !modelPath.isEmpty()) {
            return false;
        }
        if (requestedProviderId != null && !requestedProviderId.isBlank() && !isLiteRtProviderAlias(requestedProviderId)) {
            return false;
        }
        if (providerExplicit && !isLiteRtProviderAlias(requestedProviderId)) {
            return false;
        }
        return isSimpleStandaloneTextRun();
    }

    private Optional<LocalModelIndex.Entry> findSafetensorIndexEntry(String modelRef) {
        if (modelRef == null || modelRef.isBlank()) {
            return Optional.empty();
        }
        try {
            Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(modelRef, "safetensor");
            if (indexed.isPresent()
                    && indexed.get().path != null
                    && !indexed.get().path.isBlank()
                    && Files.exists(Path.of(indexed.get().path))) {
                return indexed;
            }
        } catch (Exception ignored) {
            // Fall back below to the default index lookup.
        }
        try {
            Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(modelRef);
            if (indexed.isPresent()
                    && "safetensor".equalsIgnoreCase(indexed.get().format)
                    && indexed.get().path != null
                    && !indexed.get().path.isBlank()
                    && Files.exists(Path.of(indexed.get().path))) {
                return indexed;
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
        return Optional.empty();
    }

    boolean shouldTryStandaloneGgufFastPath() {
        if (routeReportJson) {
            return false;
        }
        if (prompt == null || prompt.isBlank()) {
            return false;
        }
        if (providerId != null && !providerId.isBlank() && !isGgufProviderAlias(providerId)) {
            return false;
        }
        return isGgufProviderAlias(providerId)
                || isGgufPluginAlias(pluginId)
                || hasExplicitGgufEngine()
                || forceGguf
                || isGgufFormat(format)
                || looksLikeGgufPath(modelFile)
                || looksLikeGgufPath(modelPath)
                || looksLikeGgufPath(modelId);
    }

    String[] buildStandaloneGgufFastRunArgs() {
        return buildStandaloneGgufFastRunArgs(false);
    }

    String[] buildStandaloneGgufFastRunArgs(boolean suppressBanner) {
        List<String> args = new java.util.ArrayList<>();
        args.add("run");
        if (suppressBanner) {
            args.add("--no-banner");
        }
        if (modelFile != null && !modelFile.isBlank()) {
            args.add("--modelFile");
            args.add(modelFile);
        } else if (modelPath != null && !modelPath.isBlank()) {
            args.add("--model-path");
            args.add(modelPath);
        } else if (modelId != null && !modelId.isBlank()) {
            args.add("--model");
            args.add(modelId);
        }
        if (prompt != null) {
            args.add("--prompt");
            args.add(prompt);
        }
        args.add("--max-tokens");
        args.add(Integer.toString(maxTokens));
        args.add("--temperature");
        args.add(Double.toString(temperature));
        args.add("--top-k");
        args.add(Integer.toString(topK));
        args.add("--top-p");
        args.add(Double.toString(topP));

        String engine = requestedGgufEngine();
        if (engine != null && !engine.isBlank()) {
            args.add("--engine");
            args.add(engine);
        }
        String backend = requestedGgufBackend();
        if (backend != null && !backend.isBlank()) {
            args.add("--backend");
            args.add(backend);
        }
        if (providerId != null && !providerId.isBlank()) {
            args.add("--provider");
            args.add(providerId);
        } else {
            args.add("--provider");
            args.add("gguf");
        }
        return args.toArray(String[]::new);
    }

    private boolean tryStandaloneGgufFastPath() {
        return tryStandaloneGgufFastPath(false);
    }

    private boolean tryStandaloneGgufFastPath(boolean suppressBanner) {
        if (!shouldTryStandaloneGgufFastPath()) {
            return false;
        }
        int status = GgufFastRun.run(buildStandaloneGgufFastRunArgs(suppressBanner));
        if (GgufFastRun.isFallbackToFullCliStatus(status)) {
            return false;
        }
        if (status == 0) {
            if (GgufFastRun.hardExitAfterRun()) {
                GgufFastRun.hardExitProcess(0);
            }
            requestProcessExit();
            return true;
        }
        System.exit(status);
        return true;
    }

    private boolean hasExplicitGgufEngine() {
        return javaNativeGguf
                || llamaCppGguf
                || benchmarkGguf
                || (ggufEngine != null && !ggufEngine.isBlank());
    }

    private boolean isSimpleStandaloneTextRun() {
        return isBlank(systemPrompt)
                && isBlank(grammar)
                && isBlank(sessionId)
                && isBlank(toolChoice)
                && isBlank(embeddingModel)
                && isBlank(inputImagePath)
                && isEmpty(toolFiles)
                && isEmpty(mcpTools)
                && isEmpty(ragContexts)
                && isEmpty(ragFiles)
                && isEmpty(inputImagePaths)
                && !jsonMode
                && !enableJsonSse
                && !ocrMode
                && !ocrJsonOutput
                && isBlank(ocrJsonOutputPath)
                && isBlank(ocrOverlayOutputPath);
    }

    private static boolean isLiteRtProviderAlias(String value) {
        if (value == null) {
            return false;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("litert") || normalized.equals("litertlm");
    }

    private static boolean looksLikeLiteRtLmPath(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        try {
            Path path = Path.of(localPath);
            return Files.isRegularFile(path)
                    && path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".litertlm");
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static boolean isEmpty(List<?> values) {
        return values == null || values.isEmpty();
    }

    private String requestedGgufEngine() {
        if (benchmarkGguf) {
            return "benchmark";
        }
        if (javaNativeGguf) {
            return "java";
        }
        if (llamaCppGguf) {
            return "llamacpp";
        }
        return ggufEngine == null ? null : ggufEngine.trim();
    }

    private String requestedGgufBackend() {
        if (ggufBackend != null && !ggufBackend.isBlank()) {
            return ggufBackend.trim();
        }
        if (parentCommand != null && parentCommand.isUseCpu()) {
            return "cpu";
        }
        if (parentCommand != null && parentCommand.platform() != null && !parentCommand.platform().isBlank()) {
            return parentCommand.platform().trim();
        }
        return null;
    }

    private boolean isGgufProviderAlias(String provider) {
        if (provider == null || provider.isBlank()) {
            return false;
        }
        String normalized = provider.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("gguf")
                || normalized.equals("llamacpp")
                || normalized.equals("llama.cpp")
                || normalized.equals("llama-cpp")
                || normalized.equals("java")
                || normalized.equals("java-native")
                || normalized.equals("jvm");
    }

    private boolean isGgufPluginAlias(String plugin) {
        if (plugin == null || plugin.isBlank()) {
            return false;
        }
        String normalized = plugin.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("gguf")
                || normalized.equals("gguf-runner")
                || normalized.equals("llamacpp")
                || normalized.equals("llama.cpp")
                || normalized.equals("llama-cpp");
    }

    private boolean isGgufFormat(String value) {
        return value != null && value.trim().equalsIgnoreCase("gguf");
    }

    private boolean looksLikeGgufPath(String value) {
        return value != null && value.trim().toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private String normalizeRequestedProvider(String rawProviderId) {
        if (rawProviderId == null) {
            return null;
        }
        String trimmed = rawProviderId.trim();
        if (trimmed.isEmpty()) {
            return trimmed;
        }
        return switch (trimmed.toLowerCase(Locale.ROOT)) {
            case "torch", "torchscript", "pytorch" -> "libtorch";
            default -> trimmed.toLowerCase(Locale.ROOT);
        };
    }

    private String inferProviderFromIndex(LocalModelIndex.Entry entry) {
        if (entry == null || entry.format == null) {
            return null;
        }
        return providerForFormat(entry.format);
    }

    private String inferFormatFromIndex(LocalModelIndex.Entry entry) {
        if (entry == null || entry.format == null || entry.format.isBlank()) {
            return null;
        }
        return canonicalModelFormat(entry.format);
    }

    private String providerForFormat(String format) {
        if (format == null) {
            return null;
        }
        String normalized = format.trim().toLowerCase(java.util.Locale.ROOT);
        return switch (normalized) {
            case "gguf" -> "gguf";
            case "safetensors", "safetensor" -> "safetensor";
            case "litert", "task", "tflite" -> "litert";
            case "onnx" -> "onnx";
            case "torchscript", "pytorch" -> "libtorch";
            default -> null;
        };
    }

    private record AudioQualityPreset(String format, Integer bitrateKbps, Integer flacCompression) {
    }

    private record TtsVoiceListFilter(String language, String gender, List<String> terms) {
        private boolean hasFilters() {
            return language != null || gender != null || (terms != null && !terms.isEmpty());
        }

        private String describe() {
            List<String> parts = new ArrayList<>();
            if (language != null) {
                parts.add("lang=" + language);
            }
            if (gender != null) {
                parts.add("gender=" + gender);
            }
            if (terms != null && !terms.isEmpty()) {
                parts.add("search=" + String.join(" ", terms));
            }
            return parts.isEmpty() ? "(none)" : String.join(", ", parts);
        }
    }

    private boolean isProviderActive(String provider) {
        if (provider == null || provider.isBlank()) {
            return false;
        }
        try {
            java.util.List<Object> providers = java.util.List.of();
            if (providers == null || providers.isEmpty()) {
                return false;
            }
            String normalized = provider.trim().toLowerCase(Locale.ROOT);
            return providers.stream()
                    .map(Object::toString)
                    .filter(Objects::nonNull)
                    .map(id -> id.trim().toLowerCase(Locale.ROOT))
                    .anyMatch(normalized::equals);
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean isRuntimeRouteActive(String runtimeId) {
        if (isProviderActive(runtimeId)) {
            return true;
        }
        String normalized = runtimeId == null ? "" : runtimeId.trim().toLowerCase(Locale.ROOT);
        if (!"gguf".equals(normalized)) {
            return false;
        }
        try {
            return pluginChecker != null
                    && pluginChecker.getRunnerPluginIds().stream()
                    .map(id -> id == null ? "" : id.toLowerCase(Locale.ROOT))
                    .anyMatch(id -> id.equals("gguf")
                            || id.equals("gguf-runner")
                            || id.contains("gguf"));
        } catch (Exception ignored) {
            return false;
        }
    }

    private void ensureBuiltinProviderRegistration() {
        ensureProviderRegistered(
                "litert",
                "tech.kayys.gollek.provider.litert.LiteRTProvider");
        ensureProviderRegistered(
                "onnx",
                "tech.kayys.gollek.provider.onnx.OnnxProvider");
    }

    private void ensureProviderRegistered(String providerId, String className) {
        try {
            
            Class<?> clazz = Class.forName(className);
            Object instance = clazz.getDeclaredConstructor().newInstance();
            
        } catch (Throwable t) {
            System.err.printf("Provider bootstrap failed for %s (%s): %s%n", providerId, className, t.getMessage());
        }
    }

    private String extractModelPath(ModelInfo modelInfo) {
        if (modelInfo == null || modelInfo.getMetadata() == null) {
            return null;
        }
        Object value = modelInfo.getMetadata().get("path");
        return value != null ? value.toString() : null;
    }

    private ProviderLocalPathResolution resolveProviderSpecificLocalPath(
            String currentProvider,
            String requestedModelId,
            String localPath) {
        if (localPath == null || localPath.isBlank() || currentProvider == null || currentProvider.isBlank()) {
            return new ProviderLocalPathResolution(true, localPath);
        }

        String normalizedProvider = normalizeRequestedProvider(currentProvider);
        if (!"libtorch".equals(normalizedProvider)) {
            return new ProviderLocalPathResolution(true, localPath);
        }

        try {
            Path originalPath = Path.of(localPath).toAbsolutePath().normalize();
            Optional<Path> directArtifact = resolveDirectLibtorchArtifact(originalPath);
            if (directArtifact.isPresent()) {
                Path resolved = directArtifact.get().toAbsolutePath().normalize();
                if (!resolved.equals(originalPath)) {
                    System.out.println("Provider 'libtorch' selected; using local TorchScript artifact: " + resolved);
                }
                return new ProviderLocalPathResolution(true, resolved.toString());
            }
        } catch (Exception ignored) {
            // Fall through to registry-based lookup and explicit compatibility hint.
        }

        Optional<ModelInfo> alternative = findLocalArtifactForProvider(requestedModelId, localPath, normalizedProvider);
        if (alternative.isPresent()) {
            String candidatePath = extractModelPath(alternative.get());
            if (candidatePath != null && !candidatePath.isBlank()) {
                try {
                    Optional<Path> resolved = resolveDirectLibtorchArtifact(Path.of(candidatePath));
                    if (resolved.isPresent()) {
                        Path artifactPath = resolved.get().toAbsolutePath().normalize();
                        System.out.println("Provider 'libtorch' selected; rerouting to registered TorchScript artifact: "
                                + artifactPath);
                        return new ProviderLocalPathResolution(true, artifactPath.toString());
                    }
                } catch (Exception ignored) {
                    // Keep the explicit error below if the candidate is not actually runnable.
                }
            }
        }

        if (isSafetensorCheckpointDir(localPath) || isSafetensorWeightFile(localPath)) {
            System.out.println("Provider 'libtorch' selected; using safetensor bridge mode for this checkpoint.");
            return new ProviderLocalPathResolution(true, localPath);
        }

        System.err.println("Error: Provider 'libtorch' currently needs a runnable TorchScript-style artifact "
                + "(.pt, .pts, .pth, .bin).");
        System.err.println("Resolved local path: " + localPath);
        return new ProviderLocalPathResolution(false, localPath);
    }

    private Optional<ModelInfo> findLocalArtifactForProvider(
            String requestedModelId,
            String localPath,
            String currentProvider) {
        if (!"libtorch".equalsIgnoreCase(currentProvider)) {
            return Optional.empty();
        }
        try {
            List<ModelInfo> models = sdk.listModels();
            if (models.isEmpty()) {
                return Optional.empty();
            }

            String canonicalModelId = models.stream()
                    .filter(model -> requestedModelId != null && (requestedModelId.equalsIgnoreCase(model.getModelId())
                            || requestedModelId.equalsIgnoreCase(model.getShortId())))
                    .map(ModelInfo::getModelId)
                    .findFirst()
                    .orElseGet(() -> models.stream()
                            .filter(model -> {
                                String candidatePath = extractModelPath(model);
                                return candidatePath != null && candidatePath.equals(localPath);
                            })
                            .map(ModelInfo::getModelId)
                            .findFirst()
                            .orElse(requestedModelId));

            return models.stream()
                    .filter(model -> canonicalModelId != null && canonicalModelId.equalsIgnoreCase(model.getModelId()))
                    .filter(model -> {
                        String candidatePath = extractModelPath(model);
                        if (candidatePath == null || candidatePath.equals(localPath)) {
                            return false;
                        }
                        try {
                            return resolveDirectLibtorchArtifact(Path.of(candidatePath)).isPresent();
                        } catch (Exception ignored) {
                            return false;
                        }
                    })
                    .sorted((left, right) -> Integer.compare(
                            libtorchArtifactPriority(extractModelPath(right), right.getFormat()),
                            libtorchArtifactPriority(extractModelPath(left), left.getFormat())))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private Optional<Path> resolveDirectLibtorchArtifact(Path root) {
        if (root == null || !Files.exists(root)) {
            return Optional.empty();
        }
        if (Files.isRegularFile(root) && isLibtorchModelFile(root)) {
            return Optional.of(root);
        }
        if (!Files.isDirectory(root)) {
            return Optional.empty();
        }
        try (var paths = Files.walk(root, 3)) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(this::isLibtorchModelFile)
                    .sorted((left, right) -> Integer.compare(libtorchArtifactPriority(right, null),
                            libtorchArtifactPriority(left, null)))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private boolean isLibtorchModelFile(Path path) {
        if (path == null) {
            return false;
        }
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".pt")
                || name.endsWith(".pts")
                || name.endsWith(".pth")
                || name.endsWith(".bin");
    }

    private int libtorchArtifactPriority(Path path, String format) {
        if (path == null) {
            return 0;
        }
        return libtorchArtifactPriority(path.toString(), format);
    }

    private int libtorchArtifactPriority(String path, String format) {
        int score = 0;
        String normalizedPath = path == null ? "" : path.toLowerCase(Locale.ROOT);
        String normalizedFormat = format == null ? "" : format.toLowerCase(Locale.ROOT);
        if (normalizedFormat.equals("torchscript")) {
            score += 50;
        } else if (normalizedFormat.equals("pytorch")) {
            score += 20;
        }
        if (normalizedPath.contains("/libtorchscript/")) {
            score += 40;
        }
        if (normalizedPath.endsWith(".pt")) {
            score += 30;
        } else if (normalizedPath.endsWith(".pts")) {
            score += 25;
        } else if (normalizedPath.endsWith(".pth")) {
            score += 15;
        } else if (normalizedPath.endsWith(".bin")) {
            score += 10;
        }
        return score;
    }

    private boolean isSafetensorWeightFile(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        try {
            Path path = Path.of(localPath);
            if (!Files.isRegularFile(path)) {
                return false;
            }
            String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
            return name.endsWith(".safetensors") || name.endsWith(".safetensor");
        } catch (Exception ignored) {
            return false;
        }
    }

    private void printResponse(InferenceResponse response, long startTime) {
        Map<String, Object> metadata = effectiveExecutionRouteMetadata(response.getMetadata());
        if (isPaddleOcrMetadata(metadata)) {
            Map<String, Object> ocrPayload = ocrJsonPayload(response, metadata);
            Path overlayPath = maybeWriteOcrOverlayOutput(ocrPayload, ocrJsonOutput || jsonMode);
            if (overlayPath != null) {
                ocrPayload.put("overlay_path", overlayPath.toString());
            }
            maybeWriteOcrJsonOutput(ocrPayload, ocrJsonOutput || jsonMode);
            if (ocrJsonOutput || jsonMode) {
                printJsonPayload(ocrPayload);
                return;
            }
        }
        printExecutionRouteInfo(metadata);
        printQuantCacheInfo(metadata);
        printKvCacheQuantizationInfo(metadata);
        System.out.println();
        System.out.println(ChatUIRenderer.GREEN + response.getContent() + ChatUIRenderer.RESET);
        if (response.getSessionId() != null && !response.getSessionId().isBlank()) {
            System.out.println(ChatUIRenderer.DIM + "Session: " + response.getSessionId() + ChatUIRenderer.RESET);
        }

        boolean audioResponse = metadata.containsKey("audio") || isAudioMetadata(metadata);
        if (metadata.containsKey("audio")) {
            saveAudio(metadata.get("audio").toString(), audioDefaultExtension(metadata), metadata);
        } else if (metadata.containsKey("image")) {
            saveImage(metadata.get("image").toString(), metadata);
        }

        double seconds = Math.max(response.getDurationMs() / 1000.0, 0.001);
        double tps = response.getTokensUsed() / seconds;
        printCompletionStatsForOutput(
                response.getTokensUsed(),
                response.getDurationMs(),
                tps,
                ttftMillis(metadata),
                metadata,
                audioResponse);
        int outputTokens = response.getOutputTokens() > 0 ? response.getOutputTokens() : response.getTokensUsed();
        recordRouteBenchmarkProfile(
                response.getModel(),
                response.getModel(),
                metadata,
                outputTokens,
                response.getDurationMs(),
                audioResponse);
                
        try {
            ObservabilityManager obs = ObservabilityManager.global();
            if (obs != null) {
                MetricsRegistry reg = obs.getMetricsRegistry();
                String resolvedEngine = metadata.containsKey("engine") ? String.valueOf(metadata.get("engine")) : "standard";
                if (metadata.containsKey("runner")) {
                    resolvedEngine = String.valueOf(metadata.get("runner"));
                }
                AttributesBuilder attrs = Attributes.builder()
                        .put("engine", resolvedEngine)
                        .put("fast_path", false);

                Attributes finalAttrs = attrs.build();
                long durationMs = response.getDurationMs();
                reg.record("gollek.llm.generation.duration", durationMs, finalAttrs);
                reg.record("gollek.llm.generation.speed", tps, finalAttrs);
                
                double tokenLatency = outputTokens > 0 ? (double) durationMs / outputTokens : 0.0d;
                reg.record("gollek.llm.generation.token_latency", tokenLatency, finalAttrs);
            }
        } catch (Exception e) {
            // Ignore telemetry exceptions
        }
    }

    private boolean isPaddleOcrMetadata(Map<String, Object> metadata) {
        return metadata != null && "paddleocr-vl".equals(String.valueOf(metadata.get("pipeline")));
    }

    private Map<String, Object> ocrJsonPayload(InferenceResponse response, Map<String, Object> metadata) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "gollek_ocr_result");
        payload.put("request_id", response.getRequestId());
        payload.put("model", response.getModel());
        payload.put("content", response.getContent());
        payload.put("timestamp", response.getTimestamp().toString());
        payload.put("duration_ms", response.getDurationMs());
        payload.put("input_tokens", response.getInputTokens());
        payload.put("output_tokens", response.getOutputTokens());
        payload.put("image_path", metadata.get("image_path"));
        payload.put("text", metadata.get("ocr_text_without_locations"));
        payload.put("raw_text", metadata.get("ocr_raw_text"));
        payload.put("regions", metadata.getOrDefault("ocr_regions", List.of()));
        payload.put("boxes", metadata.getOrDefault("ocr_location_boxes", List.of()));
        payload.put("location_tokens", metadata.getOrDefault("ocr_location_tokens", List.of()));
        payload.put("generated_token_ids", metadata.getOrDefault("generated_token_ids", List.of()));
        payload.put("finish_reason", metadata.get("finish_reason"));

        Map<String, Object> timings = new LinkedHashMap<>();
        putIfPresent(timings, "vision_ms", metadata.get("vision_ms"));
        putIfPresent(timings, "decoder_prefill_ms", metadata.get("decoder_prefill_ms"));
        putIfPresent(timings, "decoder_decode_ms", metadata.get("decoder_decode_ms"));
        putIfPresent(timings, "latency_ms", metadata.get("latency_ms"));
        payload.put("timings", timings);
        payload.put("metadata", new LinkedHashMap<>(metadata));
        return payload;
    }

    private static void putIfPresent(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private void maybeWriteOcrJsonOutput(Map<String, Object> payload, boolean quiet) {
        if (ocrJsonOutputPath == null || ocrJsonOutputPath.isBlank()) {
            return;
        }
        try {
            Path path = Path.of(ocrJsonOutputPath).toAbsolutePath().normalize();
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            String json = new ObjectMapper()
                    .writerWithDefaultPrettyPrinter()
                    .writeValueAsString(payload);
            Files.writeString(path, json);
            if (!quiet) {
                System.out.println(ChatUIRenderer.DIM + "OCR JSON saved to: "
                        + path + ChatUIRenderer.RESET);
            }
        } catch (Exception e) {
            System.err.println("\nFailed to save OCR JSON: " + e.getMessage());
        }
    }

    private Path maybeWriteOcrOverlayOutput(Map<String, Object> payload, boolean quiet) {
        if (ocrOverlayOutputPath == null || ocrOverlayOutputPath.isBlank()) {
            return null;
        }
        try {
            String imagePath = stringValue(payload.get("image_path"));
            if (imagePath == null || imagePath.isBlank()) {
                throw new IllegalArgumentException("OCR payload does not include image_path");
            }
            Path inputPath = Path.of(imagePath).toAbsolutePath().normalize();
            BufferedImage source = ImageIO.read(inputPath.toFile());
            if (source == null) {
                throw new IllegalArgumentException("unsupported image file: " + inputPath);
            }

            BufferedImage overlay = new BufferedImage(
                    source.getWidth(),
                    source.getHeight(),
                    BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = overlay.createGraphics();
            try {
                g.drawImage(source, 0, 0, null);
                drawOcrOverlay(g, overlay.getWidth(), overlay.getHeight(), payload);
            } finally {
                g.dispose();
            }

            Path outputPath = Path.of(ocrOverlayOutputPath).toAbsolutePath().normalize();
            if (pathExtension(outputPath) == null || pathExtension(outputPath).isBlank()) {
                outputPath = outputPath.resolveSibling(outputPath.getFileName() + ".png");
            }
            Path parent = outputPath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            ImageIO.write(overlay, "png", outputPath.toFile());
            if (!quiet) {
                System.out.println(ChatUIRenderer.DIM + "OCR overlay saved to: "
                        + outputPath + ChatUIRenderer.RESET);
            }
            return outputPath;
        } catch (Exception e) {
            System.err.println("\nFailed to save OCR overlay: " + e.getMessage());
            return null;
        }
    }

    private void drawOcrOverlay(Graphics2D g, int imageWidth, int imageHeight, Map<String, Object> payload) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(Math.max(2.0f, Math.min(imageWidth, imageHeight) / 400.0f)));
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, Math.max(13, Math.min(imageWidth, imageHeight) / 42)));

        List<Map<String, Object>> regions = objectMapList(payload.get("regions"));
        if (!regions.isEmpty()) {
            for (Map<String, Object> region : regions) {
                Map<String, Object> box = objectMap(region.get("box"));
                String label = stringValue(region.get("text"));
                drawOcrBox(g, box, label, imageWidth, imageHeight);
            }
            return;
        }

        for (Map<String, Object> box : objectMapList(payload.get("boxes"))) {
            drawOcrBox(g, box, null, imageWidth, imageHeight);
        }
    }

    private void drawOcrBox(Graphics2D g, Map<String, Object> box, String label, int imageWidth, int imageHeight) {
        if (box == null || box.isEmpty()) {
            return;
        }
        int x1 = clampInt(numberValue(box.get("box_pixel_x1"), numberValue(box.get("pixel_x1"), -1)), 0, imageWidth);
        int y1 = clampInt(numberValue(box.get("box_pixel_y1"), numberValue(box.get("pixel_y1"), -1)), 0, imageHeight);
        int x2 = clampInt(numberValue(box.get("box_pixel_x2"), numberValue(box.get("pixel_x2"), -1)), 0, imageWidth);
        int y2 = clampInt(numberValue(box.get("box_pixel_y2"), numberValue(box.get("pixel_y2"), -1)), 0, imageHeight);
        if (x1 < 0 || y1 < 0 || x2 <= x1 || y2 <= y1) {
            return;
        }

        Color stroke = new Color(0, 142, 255, 235);
        Color fill = new Color(0, 142, 255, 40);
        g.setColor(fill);
        g.fillRect(x1, y1, x2 - x1, y2 - y1);
        g.setColor(stroke);
        g.drawRect(x1, y1, x2 - x1, y2 - y1);

        if (label == null || label.isBlank()) {
            label = "#" + stringValue(box.getOrDefault("index", ""));
        }
        if (label == null || label.isBlank()) {
            return;
        }
        String compact = truncateDisplay(label.replaceAll("\\s+", " ").trim(), 48);
        FontMetrics metrics = g.getFontMetrics();
        int pad = Math.max(4, metrics.getHeight() / 4);
        int labelWidth = metrics.stringWidth(compact) + pad * 2;
        int labelHeight = metrics.getHeight() + pad;
        int labelX = Math.max(0, Math.min(x1, imageWidth - labelWidth));
        int labelY = y1 - labelHeight >= 0 ? y1 - labelHeight : Math.min(imageHeight - labelHeight, y2);
        g.setColor(new Color(0, 0, 0, 190));
        g.fillRoundRect(labelX, labelY, labelWidth, labelHeight, 6, 6);
        g.setColor(Color.WHITE);
        g.drawString(compact, labelX + pad, labelY + metrics.getAscent() + (pad / 2));
    }

    private static int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static int numberValue(Object value, int fallback) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text && !text.isBlank()) {
            try {
                return Integer.parseInt(text.trim());
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static String stringValue(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    private static Map<String, Object> objectMap(Object value) {
        if (!(value instanceof Map<?, ?> raw)) {
            return Map.of();
        }
        Map<String, Object> result = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            if (entry.getKey() != null) {
                result.put(String.valueOf(entry.getKey()), entry.getValue());
            }
        }
        return result;
    }

    private static List<Map<String, Object>> objectMapList(Object value) {
        if (!(value instanceof List<?> raw)) {
            return List.of();
        }
        List<Map<String, Object>> result = new ArrayList<>();
        for (Object item : raw) {
            Map<String, Object> map = objectMap(item);
            if (!map.isEmpty()) {
                result.add(map);
            }
        }
        return result;
    }

    private void printJsonPayload(Map<String, Object> payload) {
        try {
            System.out.println(new ObjectMapper()
                    .writerWithDefaultPrettyPrinter()
                    .writeValueAsString(payload));
        } catch (Exception e) {
            System.err.println("\nFailed to render JSON: " + e.getMessage());
        }
    }

    private static Double ttftMillis(Map<String, Object> metadata) {
        return metadataDouble(metadata, "bench.ttft_ms");
    }

    private static Double ttftMillis(
            Map<String, Object> metadata,
            long streamStartTimeMs,
            java.util.concurrent.atomic.AtomicLong firstTokenTimeMs) {
        Double metadataTtft = ttftMillis(metadata);
        if (metadataTtft != null) {
            return metadataTtft;
        }
        long first = firstTokenTimeMs != null ? firstTokenTimeMs.get() : 0L;
        if (first <= 0L || first < streamStartTimeMs) {
            return null;
        }
        return (double) (first - streamStartTimeMs);
    }

    private static Map<String, Object> observedStreamMetrics(
            Map<String, Object> metadata,
            int outputTokens,
            long durationMs,
            Double ttftMs) {
        Map<String, Object> metrics = metadata == null || metadata.isEmpty()
                ? new LinkedHashMap<>()
                : new LinkedHashMap<>(metadata);
        if (outputTokens > 0) {
            metrics.putIfAbsent("tokens.output", outputTokens);
            double durationSeconds = Math.max(durationMs / 1000.0, 0.001);
            metrics.putIfAbsent("bench.generation_tps", outputTokens / durationSeconds);
            if (ttftMs != null) {
                metrics.putIfAbsent("bench.ttft_ms", ttftMs);
                if (outputTokens > 1 && durationMs > ttftMs) {
                    metrics.putIfAbsent("bench.tpot_ms", (durationMs - ttftMs) / (outputTokens - 1));
                }
            }
        } else if (ttftMs != null) {
            metrics.putIfAbsent("bench.ttft_ms", ttftMs);
        }
        return metrics;
    }

    private void printCompletionStatsForOutput(
            int tokenCount,
            long durationMs,
            double tps,
            Double ttftMs,
            Map<String, Object> metadata,
            boolean audioOutput) {
        if (audioOutput || isAudioMetadata(metadata)) {
            printAudioRunStats(metadata, durationMs);
            return;
        }
        uiRenderer.printStats(tokenCount, durationMs / 1000.0, tps, ttftMs, noBanner);
        uiRenderer.printBenchmarks(metadata, noBanner);
    }

    private void printAudioRunStats(Map<String, Object> metadata, long fallbackDurationMs) {
        if (enableJsonSse || noBanner) {
            return;
        }
        List<String> parts = new ArrayList<>();
        Double audioSeconds = audioDurationSeconds(metadata);
        Double generationSeconds = audioGenerationSeconds(metadata, fallbackDurationMs);
        if (audioSeconds != null && audioSeconds > 0.0) {
            parts.add(String.format(Locale.ROOT, "Audio: %.2fs", audioSeconds));
        }
        if (generationSeconds != null && generationSeconds > 0.0) {
            parts.add(String.format(Locale.ROOT, "Generation: %.2fs", generationSeconds));
        }

        Double speed = metadataDouble(metadata, "audio_realtime_speed");
        if (speed == null && audioSeconds != null && generationSeconds != null && generationSeconds > 0.0) {
            speed = audioSeconds / generationSeconds;
        }
        if (speed != null && speed > 0.0) {
            parts.add(String.format(Locale.ROOT, "Speed: %.2fx realtime", speed));
        }

        Double realtimeFactor = metadataDouble(metadata, "audio_realtime_factor");
        if (realtimeFactor == null && audioSeconds != null && audioSeconds > 0.0
                && generationSeconds != null && generationSeconds > 0.0) {
            realtimeFactor = generationSeconds / audioSeconds;
        }
        if (realtimeFactor != null && realtimeFactor > 0.0) {
            parts.add(String.format(Locale.ROOT, "RTF: %.2f", realtimeFactor));
        }

        Double framesPerSecond = metadataDouble(metadata, "tts_frames_per_second");
        Double frames = metadataDouble(metadata, "tts_frames");
        if (framesPerSecond == null && frames != null && generationSeconds != null && generationSeconds > 0.0) {
            framesPerSecond = frames / generationSeconds;
        }
        if (framesPerSecond != null && framesPerSecond > 0.0) {
            parts.add(String.format(Locale.ROOT, "Synthesis: %.1f frames/s", framesPerSecond));
        }

        if (!parts.isEmpty()) {
            System.out.println(ChatUIRenderer.DIM + "\n[" + String.join(", ", parts) + "]" + ChatUIRenderer.RESET);
        }
    }

    private static boolean isAudioMetadata(Map<String, Object> metadata) {
        return metadata != null
                && (metadata.containsKey("audio_format")
                || metadata.containsKey("audio_duration_seconds")
                || metadata.containsKey("tts_frames"));
    }

    private static Double audioGenerationSeconds(Map<String, Object> metadata, long fallbackDurationMs) {
        Double generationSeconds = metadataDouble(metadata, "audio_generation_duration_seconds");
        if (generationSeconds != null && generationSeconds > 0.0) {
            return generationSeconds;
        }
        Double latencyMs = metadataDouble(metadata, "bench.latency_ms");
        if (latencyMs != null && latencyMs > 0.0) {
            return latencyMs / 1000.0;
        }
        if (fallbackDurationMs > 0L) {
            return fallbackDurationMs / 1000.0;
        }
        return null;
    }

    private static long observedStreamDurationMillis(
            long streamStartTimeMs,
            long completionTimeMs,
            java.util.concurrent.atomic.AtomicLong lastTokenTimeMs) {
        long last = lastTokenTimeMs != null ? lastTokenTimeMs.get() : 0L;
        if (last >= streamStartTimeMs) {
            return Math.max(1L, last - streamStartTimeMs);
        }
        return Math.max(1L, completionTimeMs - streamStartTimeMs);
    }

    private static Double metadataDouble(Map<String, Object> metadata, String key) {
        if (metadata == null || metadata.isEmpty()) {
            return null;
        }
        Object value = metadata.get(key);
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text && !text.isBlank()) {
            try {
                return Double.parseDouble(text.trim());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private static String metadataText(Map<String, Object> metadata, String key) {
        if (metadata == null || metadata.isEmpty()) {
            return null;
        }
        Object value = metadata.get(key);
        return value == null ? null : String.valueOf(value);
    }

    private void printExecutionRouteInfo(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty() || noBanner) {
            return;
        }
        Object requested = metadata.get("requested_provider");
        Object effective = metadata.get("effective_provider");
        Object backend = metadata.get("execution_backend");
        Object bridgeMode = metadata.get("provider_bridge_mode");
        Object bridgeReason = metadata.get("provider_bridge_reason");
        Object runnerRoute = metadata.get(
                RunnerRouteReportFields.metadataKey(RunnerRouteReportFields.Report.NORMALIZED_RUNNER));
        Object runnerMode = metadata.get(
                RunnerRouteReportFields.metadataKey(RunnerRouteReportFields.Report.MODE));

        if (requested == null && effective == null && backend == null && bridgeMode == null && runnerRoute == null) {
            return;
        }

        StringBuilder line = new StringBuilder("Execution route: ");
        if (requested != null && effective != null
                && !String.valueOf(requested).equalsIgnoreCase(String.valueOf(effective))) {
            line.append(requested).append(" -> ").append(effective);
        } else if (effective != null) {
            line.append(effective);
        } else if (requested != null) {
            line.append(requested);
        } else {
            line.append(backend);
        }
        if (backend != null) {
            line.append(" (backend=").append(backend).append(")");
        }
        if (runnerRoute != null) {
            line.append(" (runner=").append(runnerRoute);
            if (runnerMode != null && !RunnerRoutePolicy.AUTO.equals(String.valueOf(runnerMode))) {
                line.append(", mode=").append(runnerMode);
            }
            line.append(")");
        }
        if (bridgeMode != null) {
            line.append(" [").append(bridgeMode);
            if (bridgeReason != null) {
                line.append(": ").append(bridgeReason);
            }
            line.append("]");
        }
        System.out.println(line);
        System.out.println("--------------------------------------------------");
    }

    private Map<String, Object> effectiveExecutionRouteMetadata(Map<String, Object> responseMetadata) {
        if (responseMetadata == null || responseMetadata.isEmpty()) {
            return forcedExecutionRouteMetadata;
        }
        if (forcedExecutionRouteMetadata == null || forcedExecutionRouteMetadata.isEmpty()) {
            return responseMetadata;
        }
        return mergeExecutionRouteMetadata(forcedExecutionRouteMetadata, responseMetadata);
    }

    private RunnerRouteReport applyCachedBenchmarkRouteProfile(
            RunnerRouteReport report,
            String requestedModel,
            String effectiveModel,
            String localPath) {
        return RunnerRouteBenchmarkCache.profileFor(report, requestedModel, effectiveModel, localPath)
                .map(report::withRouteProfile)
                .orElse(report);
    }

    private void recordRouteBenchmarkProfile(
            String model,
            String localPath,
            Map<String, Object> metadata,
            int outputTokens,
            long durationMs,
            boolean audioOutput) {
        if (audioOutput || isAudioMetadata(metadata)) {
            return;
        }
        RunnerRouteBenchmarkCache.record(
                model,
                model,
                localPath,
                metadata,
                outputTokens,
                durationMs);
    }

    private Map<String, Object> bridgeExecutionRouteMetadata() {
        Map<String, Object> route = new LinkedHashMap<>();
        route.put("requested_provider", "libtorch");
        route.put("effective_provider", "safetensor");
        route.put("provider_bridge_mode", "cli_libtorch_to_safetensor");
        route.put("provider_bridge_reason", "raw_safetensor_checkpoint");
        return route;
    }

    private Map<String, Object> mergeExecutionRouteMetadata(
            Map<String, Object> baseMetadata,
            Map<String, Object> overlayMetadata) {
        Map<String, Object> merged = new LinkedHashMap<>();
        if (baseMetadata != null && !baseMetadata.isEmpty()) {
            merged.putAll(baseMetadata);
        }
        if (overlayMetadata != null && !overlayMetadata.isEmpty()) {
            merged.putAll(overlayMetadata);
        }
        return merged;
    }

    private void printQuantCacheInfo(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return;
        }
        Object state = metadata.get("quant_cache_state");
        if (state == null) {
            return;
        }
        StringBuilder line = new StringBuilder("Quant cache: ").append(state);
        Object path = metadata.get("quant_cache_path");
        if (path != null) {
            line.append(" (").append(path).append(")");
        }
        System.out.println(line);
        System.out.println("--------------------------------------------------");
    }

    private void printKvCacheQuantizationInfo(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return;
        }
        Object state = metadata.get("kv_cache_quantization");
        if (state == null) {
            return;
        }
        System.out.println("Effective KV cache quantization: " + state);
        System.out.println("--------------------------------------------------");
    }

    private void handleOutputs(
            java.io.ByteArrayOutputStream imageBuffer,
            StreamingAudioOutput audioOutput,
            Map<String, Object> metadata) {
        if (imageBuffer.size() > 0) saveBuffer(imageBuffer, "png", "Image", metadata);
        if (audioOutput != null && audioOutput.hasOutput()) {
            audioOutput.finish(metadata);
        }
    }

    private void handleAudioChunk(
            StreamingInferenceChunk chunk,
            StreamingAudioOutput encodedAudioOutput,
            LivePcmAudioSink liveAudioSink) {
        if (chunk == null || chunk.getDelta() == null) {
            return;
        }
        try {
            byte[] decoded = java.util.Base64.getDecoder().decode(chunk.getDelta());
            if (isPcmStreamChunk(chunk.metadata())) {
                liveAudioSink.write(decoded, chunk.metadata());
                return;
            }
            encodedAudioOutput.write(decoded, chunk.metadata());
        } catch (Exception ignored) {
        }
    }

    private static boolean isPcmStreamChunk(Map<String, Object> metadata) {
        String payload = metadataText(metadata, "audio_stream_payload");
        return "pcm_s16le".equalsIgnoreCase(payload)
                || "true".equalsIgnoreCase(metadataText(metadata, "audio_stream_preview"));
    }

    private void saveBuffer(java.io.ByteArrayOutputStream buffer, String defaultExt, String label) {
        saveBuffer(buffer, defaultExt, label, null);
    }

    private void saveBuffer(
            java.io.ByteArrayOutputStream buffer,
            String defaultExt,
            String label,
            Map<String, Object> metadata) {
        try {
            String actualExt = normalizeExtension(defaultExt);
            String requestedExt = normalizeExtension(extension);
            String ext = requestedExt.isEmpty() ? actualExt : requestedExt;
            if ("Audio".equals(label) && !requestedExt.isEmpty() && !requestedExt.equalsIgnoreCase(actualExt)) {
                System.err.println("\nWarning: requested --ext " + requestedExt
                        + ", but this provider returned " + actualExt
                        + " audio. Saving as ." + actualExt + " to avoid a mislabeled file.");
                ext = actualExt;
            }
            Path outPath = (outputPath != null && !outputPath.isEmpty()) ? Path.of(outputPath) : Path.of("output." + ext);
            if (extension != null && !extension.isBlank()) {
                outPath = replaceExtension(outPath, ext);
            } else if ("Audio".equals(label) && outputPath != null && !outputPath.isEmpty()
                    && isKnownAudioExtension(pathExtension(outPath))
                    && !pathExtension(outPath).equalsIgnoreCase(ext)) {
                outPath = replaceExtension(outPath, ext);
            }
            Files.write(outPath, buffer.toByteArray());
            System.out.println("\n" + ChatUIRenderer.GREEN + ChatUIRenderer.BOLD + "✓ " + label + " saved to: " + ChatUIRenderer.RESET + outPath.toAbsolutePath());
            if ("Audio".equals(label)) {
                printAudioDetails(metadata);
                maybeWriteAudioMetadataSidecar(outPath, metadata);
            } else if ("Image".equals(label)) {
                maybeWriteImageMetadataSidecar(outPath, metadata);
            }
            if ("png".equals(ext)) autoOpenImage(outPath); else autoOpenAudio(outPath);
        } catch (Exception e) {
            System.err.println("\nFailed to save " + label.toLowerCase() + ": " + e.getMessage());
        }
    }

    private final class StreamingAudioOutput implements AutoCloseable {
        private final java.io.ByteArrayOutputStream memory = new java.io.ByteArrayOutputStream();
        private java.io.OutputStream stream;
        private Path tempPath;
        private Path finalPath;
        private long bytes;
        private boolean finished;

        void write(byte[] chunk, Map<String, Object> metadata) throws java.io.IOException {
            if (chunk == null || chunk.length == 0) {
                return;
            }
            if (shouldStreamEncodedAudioToFile(metadata)) {
                ensureFile(metadata);
                stream.write(chunk);
            } else {
                memory.write(chunk);
            }
            bytes += chunk.length;
        }

        boolean hasOutput() {
            return bytes > 0 || memory.size() > 0 || tempPath != null;
        }

        void finish(Map<String, Object> metadata) {
            if (finished) {
                return;
            }
            finished = true;
            try {
                if (stream != null) {
                    stream.close();
                    stream = null;
                    moveTempToFinal();
                    System.out.println("\n" + ChatUIRenderer.GREEN + ChatUIRenderer.BOLD
                            + "✓ Audio saved to: " + ChatUIRenderer.RESET + finalPath.toAbsolutePath());
                    printAudioDetails(metadata);
                    maybeWriteAudioMetadataSidecar(finalPath, metadata);
                    autoOpenAudio(finalPath);
                    return;
                }
                if (memory.size() > 0) {
                    saveBuffer(memory, audioDefaultExtension(metadata), "Audio", metadata);
                }
            } catch (Exception e) {
                System.err.println("\nFailed to save audio: " + e.getMessage());
                cleanupTemp();
            }
        }

        private void ensureFile(Map<String, Object> metadata) throws java.io.IOException {
            if (stream != null) {
                return;
            }
            finalPath = resolveAudioOutputPath(metadata);
            Path parent = finalPath.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path tempDir = parent == null ? Path.of(".") : parent;
            String name = finalPath.getFileName() == null ? "gollek-audio" : finalPath.getFileName().toString();
            tempPath = Files.createTempFile(tempDir, "." + name + ".", ".part");
            stream = Files.newOutputStream(
                    tempPath,
                    java.nio.file.StandardOpenOption.WRITE,
                    java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
            if (memory.size() > 0) {
                memory.writeTo(stream);
                memory.reset();
            }
        }

        private Path resolveAudioOutputPath(Map<String, Object> metadata) {
            String actualExt = normalizeExtension(audioDefaultExtension(metadata));
            String requestedExt = normalizeExtension(extension);
            String ext = requestedExt.isEmpty() ? actualExt : requestedExt;
            if (!requestedExt.isEmpty() && !requestedExt.equalsIgnoreCase(actualExt)) {
                System.err.println("\nWarning: requested --ext " + requestedExt
                        + ", but this provider returned " + actualExt
                        + " audio. Saving as ." + actualExt + " to avoid a mislabeled file.");
                ext = actualExt;
            }
            Path outPath = (outputPath != null && !outputPath.isEmpty())
                    ? Path.of(outputPath)
                    : Path.of("output." + ext);
            if (extension != null && !extension.isBlank()) {
                outPath = replaceExtension(outPath, ext);
            } else if (outputPath != null && !outputPath.isEmpty()
                    && isKnownAudioExtension(pathExtension(outPath))
                    && !pathExtension(outPath).equalsIgnoreCase(ext)) {
                outPath = replaceExtension(outPath, ext);
            }
            return outPath;
        }

        private void moveTempToFinal() throws java.io.IOException {
            if (tempPath == null || finalPath == null) {
                return;
            }
            try {
                Files.move(
                        tempPath,
                        finalPath,
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException ignored) {
                Files.move(tempPath, finalPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            tempPath = null;
        }

        private void cleanupTemp() {
            if (tempPath == null) {
                return;
            }
            try {
                Files.deleteIfExists(tempPath);
            } catch (Exception ignored) {
            } finally {
                tempPath = null;
            }
        }

        void closeQuietly() {
            try {
                close();
            } catch (Exception ignored) {
            }
            if (!finished) {
                cleanupTemp();
            }
        }

        @Override
        public void close() throws java.io.IOException {
            if (stream != null) {
                stream.close();
                stream = null;
            }
        }
    }

    private static boolean shouldStreamEncodedAudioToFile(Map<String, Object> metadata) {
        String payload = metadataText(metadata, "audio_stream_payload");
        return "encoded".equalsIgnoreCase(payload)
                || metadataText(metadata, "audio_format") != null
                || metadataText(metadata, "audio_mime") != null;
    }

    private void printAudioDetails(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return;
        }
        List<String> parts = new ArrayList<>();

        Double durationSeconds = audioDurationSeconds(metadata);
        if (durationSeconds != null) {
            parts.add(String.format(Locale.ROOT, "%.2fs", durationSeconds));
        }
        Double sampleRate = metadataDouble(metadata, "audio_sample_rate");
        if (sampleRate != null && sampleRate > 0.0) {
            parts.add(String.format(Locale.ROOT, "%d Hz", Math.round(sampleRate)));
        }
        Double channels = metadataDouble(metadata, "audio_channels");
        if (channels != null && channels > 0.0) {
            parts.add(Math.round(channels) + "ch");
        }
        Double bitrateKbps = metadataDouble(metadata, "audio_bitrate_kbps");
        if (bitrateKbps != null && bitrateKbps > 0.0) {
            parts.add(Math.round(bitrateKbps) + " kbps");
        }
        String channelMode = metadataText(metadata, "audio_channel_mode");
        if ("stereo_dual_mono".equalsIgnoreCase(channelMode)) {
            parts.add("dual-mono");
        } else if ("mono_collapsed_duplicate".equalsIgnoreCase(channelMode)) {
            parts.add("mono collapsed");
        }
        if ("true".equalsIgnoreCase(metadataText(metadata, "audio_processing"))) {
            String peak = metadataText(metadata, "audio_processing_peak_after_dbfs");
            parts.add(peak == null || peak.isBlank() ? "polished" : "polished peak " + peak + " dBFS");
        }
        if ("true".equalsIgnoreCase(metadataText(metadata, "audio_processing_trim_silence"))) {
            String removed = metadataText(metadata, "audio_processing_trim_removed_ms");
            parts.add(removed == null || removed.isBlank() ? "trimmed silence" : "trimmed " + removed + " ms");
        }
        Double frames = metadataDouble(metadata, "tts_frames");
        if (frames != null && frames > 0.0) {
            long count = Math.round(frames);
            String frameDetail = count + " " + (count == 1 ? "frame" : "frames");
            String stopReason = metadataText(metadata, "tts_stop_reason");
            if ("max_frames".equalsIgnoreCase(stopReason)) {
                frameDetail += " (frame cap)";
            } else if ("max_seconds".equalsIgnoreCase(stopReason)) {
                frameDetail += " (duration cap)";
            } else if ("max_tokens".equalsIgnoreCase(stopReason)) {
                frameDetail += " (token cap)";
            } else if ("manifest_limit".equalsIgnoreCase(stopReason)) {
                frameDetail += " (model cap)";
            } else if ("stop_token".equalsIgnoreCase(stopReason)) {
                frameDetail += " (natural stop)";
            }
            parts.add(frameDetail);
        }
        String onnxCacheDetail = onnxOptimizedCacheDetail(metadata);
        if (onnxCacheDetail != null) {
            parts.add(onnxCacheDetail);
        }
        String onnxSessionCacheDetail = onnxSessionCacheDetail(metadata);
        if (onnxSessionCacheDetail != null) {
            parts.add(onnxSessionCacheDetail);
        }
        String onnxAssetCacheDetail = onnxAssetCacheDetail(metadata);
        if (onnxAssetCacheDetail != null) {
            parts.add(onnxAssetCacheDetail);
        }
        String voice = metadataText(metadata, "tts_voice");
        if (voice != null && !voice.isBlank()) {
            String voiceDetail = "voice " + voice;
            String voiceMode = metadataText(metadata, "tts_voice_mode");
            String voiceLanguage = metadataText(metadata, "tts_voice_language");
            if ("auto".equalsIgnoreCase(voiceMode)) {
                voiceDetail += voiceLanguage == null || voiceLanguage.isBlank()
                        ? " (auto)"
                        : " (auto " + voiceLanguage + ")";
            } else if ("language".equalsIgnoreCase(voiceMode)) {
                voiceDetail += voiceLanguage == null || voiceLanguage.isBlank()
                        ? " (language)"
                        : " (" + voiceLanguage + " alias)";
            } else if ("selector".equalsIgnoreCase(voiceMode)) {
                voiceDetail += voiceLanguage == null || voiceLanguage.isBlank()
                        ? " (selector)"
                        : " (" + voiceLanguage + " selector)";
            }
            parts.add(voiceDetail);
        }
        String ttsSeed = metadataText(metadata, "tts_seed");
        if (ttsSeed != null && !ttsSeed.isBlank()) {
            String seedSource = metadataText(metadata, "tts_seed_source");
            parts.add("seed " + ttsSeed
                    + ("generated".equalsIgnoreCase(seedSource) ? " (generated)" : ""));
        }
        String embeddedMetadata = metadataText(metadata, "audio_metadata_embedded");
        if ("true".equalsIgnoreCase(embeddedMetadata)) {
            parts.add("metadata embedded");
        }

        if (!parts.isEmpty()) {
            System.out.println(ChatUIRenderer.DIM + "Audio details: "
                    + String.join(", ", parts)
                    + ChatUIRenderer.RESET);
        }
    }

    private static String onnxOptimizedCacheDetail(Map<String, Object> metadata) {
        Double sessionValue = metadataDouble(metadata, "onnx_optimized_model_cache_sessions");
        if (sessionValue == null || sessionValue <= 0.0) {
            return null;
        }
        long sessions = Math.round(sessionValue);
        long hits = Math.round(defaultedDouble(metadata, "onnx_optimized_model_cache_hits"));
        long created = Math.round(defaultedDouble(metadata, "onnx_optimized_model_cache_created"));
        long rebuilt = Math.round(defaultedDouble(metadata, "onnx_optimized_model_cache_rebuilt"));
        long disabled = Math.round(defaultedDouble(metadata, "onnx_optimized_model_cache_disabled"));
        if (disabled >= sessions && "false".equalsIgnoreCase(metadataText(metadata, "onnx_optimized_model_cache_enabled"))) {
            return "onnx cache disabled";
        }

        List<String> notes = new ArrayList<>();
        if (created > 0L) {
            notes.add(created + " built");
        }
        if (rebuilt > 0L) {
            notes.add(rebuilt + " rebuilt");
        }
        if (disabled > 0L) {
            notes.add(disabled + " disabled");
        }
        String detail = "onnx cache " + hits + "/" + sessions + " hit";
        return notes.isEmpty() ? detail : detail + " (" + String.join(", ", notes) + ")";
    }

    private static String onnxSessionCacheDetail(Map<String, Object> metadata) {
        String state = metadataText(metadata, "onnx_session_cache_state");
        if (state == null || state.isBlank() || "disabled".equalsIgnoreCase(state)) {
            return null;
        }
        if ("hit".equalsIgnoreCase(state)) {
            return "onnx sessions hot";
        }
        if ("miss".equalsIgnoreCase(state)) {
            return "onnx sessions warmed";
        }
        return null;
    }

    private static String onnxAssetCacheDetail(Map<String, Object> metadata) {
        String state = metadataText(metadata, "onnx_asset_cache_state");
        if (state == null || state.isBlank() || "disabled".equalsIgnoreCase(state)) {
            return null;
        }
        if ("hit".equalsIgnoreCase(state)) {
            return "onnx assets hot";
        }
        if ("miss".equalsIgnoreCase(state)) {
            return "onnx assets warmed";
        }
        return null;
    }

    private static double defaultedDouble(Map<String, Object> metadata, String key) {
        Double value = metadataDouble(metadata, key);
        return value == null ? 0.0 : value;
    }

    private void maybeWriteAudioMetadataSidecar(Path audioPath, Map<String, Object> metadata) {
        boolean explicitPath = audioMetadataOutputPath != null && !audioMetadataOutputPath.isBlank();
        if (!explicitPath && !audioMetadataSidecar) {
            return;
        }
        try {
            Path metadataPath = explicitPath
                    ? Path.of(audioMetadataOutputPath)
                    : replaceExtension(audioPath, "json");
            Path absoluteAudioPath = audioPath.toAbsolutePath().normalize();
            Path absoluteMetadataPath = metadataPath.toAbsolutePath().normalize();
            if (absoluteMetadataPath.equals(absoluteAudioPath)) {
                System.err.println("\nWarning: audio metadata path is the same as the audio output; metadata sidecar skipped.");
                return;
            }
            Path parent = absoluteMetadataPath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }

            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "gollek_audio_metadata");
            payload.put("schema_version", 2);
            payload.put("created_at", Instant.now().toString());
            payload.put("output_path", absoluteAudioPath.toString());
            payload.put("output_size_bytes", Files.size(absoluteAudioPath));
            payload.put("output_sha256", sha256Hex(absoluteAudioPath));
            payload.put("model", modelId);
            payload.put("provider", providerId);
            if (runner != null && !runner.isBlank()) {
                payload.put("runner", runner);
            }
            payload.put("prompt", prompt);
            String voiceSelector = effectiveTtsVoiceSelector();
            if (voiceSelector != null && !voiceSelector.isBlank()) {
                payload.put("voice_selector", voiceSelector);
            }
            payload.put("audio_format", audioDefaultExtension(metadata));
            List<String> replayArgs = audioMetadataReplayArgs(absoluteAudioPath, metadata);
            payload.put("replay_args", replayArgs);
            payload.put("replay_command", shellCommand(replayArgs));
            payload.put("metadata", metadata == null ? Map.of() : new LinkedHashMap<>(metadata));

            String json = new ObjectMapper()
                    .writerWithDefaultPrettyPrinter()
                    .writeValueAsString(payload);
            Files.writeString(absoluteMetadataPath, json);
            System.out.println(ChatUIRenderer.DIM + "Audio metadata saved to: "
                    + absoluteMetadataPath
                    + ChatUIRenderer.RESET);
        } catch (Exception e) {
            System.err.println("\nFailed to save audio metadata: " + e.getMessage());
        }
    }

    private void maybeWriteImageMetadataSidecar(Path imagePath, Map<String, Object> metadata) {
        boolean explicitPath = imageMetadataOutputPath != null && !imageMetadataOutputPath.isBlank();
        if (!explicitPath && !imageMetadataSidecar) {
            return;
        }
        try {
            Path metadataPath = explicitPath
                    ? Path.of(imageMetadataOutputPath)
                    : replaceExtension(imagePath, "json");
            Path absoluteImagePath = imagePath.toAbsolutePath().normalize();
            Path absoluteMetadataPath = metadataPath.toAbsolutePath().normalize();
            if (absoluteMetadataPath.equals(absoluteImagePath)) {
                System.err.println("\nWarning: image metadata path is the same as the image output; metadata sidecar skipped.");
                return;
            }
            Path parent = absoluteMetadataPath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }

            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "gollek_image_metadata");
            payload.put("schema_version", 1);
            payload.put("created_at", Instant.now().toString());
            payload.put("output_path", absoluteImagePath.toString());
            payload.put("output_size_bytes", Files.size(absoluteImagePath));
            payload.put("output_sha256", sha256Hex(absoluteImagePath));
            payload.put("model", modelId);
            payload.put("provider", providerId);
            if (runner != null && !runner.isBlank()) {
                payload.put("runner", runner);
            }
            payload.put("prompt", prompt);
            payload.put("image_format", imageOutputExtension(absoluteImagePath, metadata));
            addImageParameter(payload, "seed", imageReplaySeed(metadata));
            addImageParameter(payload, "steps", steps);
            addImageParameter(payload, "guidance_scale", guidanceScale);
            addImageParameter(payload, "width", width);
            addImageParameter(payload, "height", height);
            List<String> replayArgs = imageMetadataReplayArgs(absoluteImagePath, metadata);
            payload.put("replay_args", replayArgs);
            payload.put("replay_command", shellCommand(replayArgs));
            payload.put("metadata", metadata == null ? Map.of() : new LinkedHashMap<>(metadata));

            String json = new ObjectMapper()
                    .writerWithDefaultPrettyPrinter()
                    .writeValueAsString(payload);
            Files.writeString(absoluteMetadataPath, json);
            System.out.println(ChatUIRenderer.DIM + "Image metadata saved to: "
                    + absoluteMetadataPath
                    + ChatUIRenderer.RESET);
        } catch (Exception e) {
            System.err.println("\nFailed to save image metadata: " + e.getMessage());
        }
    }

    private List<String> imageMetadataReplayArgs(Path imagePath, Map<String, Object> metadata) {
        List<String> args = new ArrayList<>();
        args.add("gollek");
        args.add("run");
        addReplayModelArgs(args, modelId);
        addStringArg(args, "--provider", providerId);
        addStringArg(args, "--runner", runner);
        addStringArg(args, "--prompt", prompt);
        addStringArg(args, "--seed", imageReplaySeed(metadata));
        addStringArg(args, "--steps", steps);
        addStringArg(args, "--guidance-scale", guidanceScale);
        addStringArg(args, "--width", width);
        addStringArg(args, "--height", height);
        addStringArg(args, "--output", imagePath == null ? null : imagePath.toString());
        return args;
    }

    private static void addImageParameter(Map<String, Object> payload, String key, Object value) {
        if (value == null) {
            return;
        }
        String text = String.valueOf(value);
        if (!text.isBlank()) {
            payload.put(key, value);
        }
    }

    private String imageReplaySeed(Map<String, Object> metadata) {
        for (String key : List.of("image_seed", "seed", "generation_seed")) {
            String metadataSeed = metadataText(metadata, key);
            if (metadataSeed != null && !metadataSeed.isBlank()) {
                return metadataSeed;
            }
        }
        return seed == null ? null : seed.toString();
    }

    private static String imageOutputExtension(Path imagePath, Map<String, Object> metadata) {
        String ext = normalizeExtension(pathExtension(imagePath));
        if (!ext.isBlank()) {
            return ext;
        }
        String metadataFormat = metadataText(metadata, "image_format");
        if (metadataFormat == null || metadataFormat.isBlank()) {
            metadataFormat = metadataText(metadata, "image_ext");
        }
        String normalized = normalizeExtension(metadataFormat);
        return normalized.isBlank() ? "png" : normalized;
    }

    private List<String> audioMetadataReplayArgs(Path audioPath, Map<String, Object> metadata) {
        List<String> args = new ArrayList<>();
        args.add("gollek");
        args.add("run");
        addReplayModelArgs(args, modelId);
        addStringArg(args, "--provider", providerId);
        addStringArg(args, "--runner", runner);
        addStringArg(args, "--prompt", prompt);
        addStringArg(args, "--voice", effectiveTtsVoiceSelector());
        addStringArg(args, "--audio-format", audioDefaultExtension(metadata));
        addStringArg(args, "--seed", replaySeed(metadata));
        addStringArg(args, "--output", audioPath == null ? null : audioPath.toString());

        addStringArg(args, "--audio-quality", audioQuality);
        addStringArg(args, "--audio-bitrate", audioBitrateKbps);
        addStringArg(args, "--audio-channels", audioChannels);
        addStringArg(args, "--tts-codec-decode", ttsCodecDecode);
        addStringArg(args, "--tts-max-frames", ttsMaxFrames);
        addStringArg(args, "--tts-max-seconds", ttsMaxSeconds);
        addStringArg(args, "--audio-gain-db", audioGainDb);
        addStringArg(args, "--audio-peak-db", audioPeakDb);
        addStringArg(args, "--audio-fade-ms", audioFadeMs);
        addStringArg(args, "--audio-trim-db", audioTrimDbfs);
        addStringArg(args, "--audio-trim-padding-ms", audioTrimPaddingMs);
        addStringArg(args, "--flac-compression", flacCompression);
        if (disableAudioPolish) {
            args.add("--no-audio-polish");
        }
        if (disableAudioNormalize) {
            args.add("--no-audio-normalize");
        }
        if (audioTrimSilence) {
            args.add("--audio-trim-silence");
        }
        return args;
    }

    private static void addReplayModelArgs(List<String> args, String modelValue) {
        if (modelValue == null || modelValue.isBlank()) {
            return;
        }
        try {
            Path path = Path.of(modelValue);
            if (Files.isDirectory(path)) {
                addStringArg(args, "--modelDir", modelValue);
                return;
            }
            if (Files.isRegularFile(path)) {
                addStringArg(args, "--modelFile", modelValue);
                return;
            }
        } catch (Exception ignored) {
        }
        addStringArg(args, "--model", modelValue);
    }

    private String replaySeed(Map<String, Object> metadata) {
        String metadataSeed = metadataText(metadata, "tts_seed");
        if (metadataSeed != null && !metadataSeed.isBlank()) {
            return metadataSeed;
        }
        return seed == null ? null : seed.toString();
    }

    private static void addStringArg(List<String> args, String flag, Object value) {
        if (value == null) {
            return;
        }
        String text = String.valueOf(value);
        if (text.isBlank()) {
            return;
        }
        args.add(flag);
        args.add(text);
    }

    private static String shellCommand(List<String> args) {
        List<String> quoted = new ArrayList<>();
        for (String arg : args) {
            quoted.add(shellQuote(arg));
        }
        return String.join(" ", quoted);
    }

    private static String shellQuote(String value) {
        if (value == null || value.isEmpty()) {
            return "''";
        }
        if (value.matches("[A-Za-z0-9_@%+=:,./-]+")) {
            return value;
        }
        return "'" + value.replace("'", "'\"'\"'") + "'";
    }

    private static String sha256Hex(Path path) throws Exception {
        java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[64 * 1024];
        try (java.io.InputStream input = Files.newInputStream(path)) {
            int read;
            while ((read = input.read(buffer)) >= 0) {
                if (read > 0) {
                    digest.update(buffer, 0, read);
                }
            }
        }
        return hexLower(digest.digest());
    }

    private static String hexLower(byte[] bytes) {
        char[] table = "0123456789abcdef".toCharArray();
        char[] out = new char[bytes.length * 2];
        for (int i = 0; i < bytes.length; i++) {
            int value = bytes[i] & 0xff;
            out[i * 2] = table[value >>> 4];
            out[i * 2 + 1] = table[value & 0x0f];
        }
        return new String(out);
    }

    private static Double audioDurationSeconds(Map<String, Object> metadata) {
        Double duration = metadataDouble(metadata, "audio_duration_seconds");
        if (duration != null && duration > 0.0) {
            return duration;
        }
        Double samples = metadataDouble(metadata, "audio_samples");
        Double sampleRate = metadataDouble(metadata, "audio_sample_rate");
        if (samples != null && sampleRate != null && samples > 0.0 && sampleRate > 0.0) {
            return samples / sampleRate;
        }
        return null;
    }

    private void saveAudio(String base64) {
        saveAudio(base64, "flac");
    }

    private void saveAudio(String base64, String defaultExt) {
        saveAudio(base64, defaultExt, null);
    }

    private void saveAudio(String base64, String defaultExt, Map<String, Object> metadata) {
        saveBufferFromBase64(base64, defaultExt, "Audio", metadata);
    }

    private void saveImage(String base64) {
        saveImage(base64, null);
    }

    private void saveImage(String base64, Map<String, Object> metadata) {
        saveBufferFromBase64(base64, "png", "Image", metadata);
    }

    private void saveBufferFromBase64(String base64, String defaultExt, String label) {
        saveBufferFromBase64(base64, defaultExt, label, null);
    }

    private void saveBufferFromBase64(
            String base64,
            String defaultExt,
            String label,
            Map<String, Object> metadata) {
        try {
            byte[] decoded = java.util.Base64.getDecoder().decode(base64);
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            bos.write(decoded);
            saveBuffer(bos, defaultExt, label, metadata);
        } catch (Exception ignored) {}
    }

    private Path replaceExtension(Path path, String newExt) {
        String filename = path.getFileName().toString();
        int lastDot = filename.lastIndexOf('.');
        if (lastDot == -1) return path.resolveSibling(filename + "." + newExt);
        return path.resolveSibling(filename.substring(0, lastDot) + "." + newExt);
    }

    private static String normalizeExtension(String ext) {
        if (ext == null) {
            return "";
        }
        String value = ext.trim().toLowerCase(Locale.ROOT);
        while (value.startsWith(".")) {
            value = value.substring(1);
        }
        return value;
    }

    private static String audioDefaultExtension(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return "flac";
        }
        Object format = metadata.get("audio_format");
        if (format == null) {
            format = metadata.get("audio_ext");
        }
        if (format == null) {
            Object mime = metadata.get("audio_mime");
            if (mime != null && String.valueOf(mime).toLowerCase(Locale.ROOT).contains("wav")) {
                return "wav";
            }
        }
        String ext = format == null ? "" : String.valueOf(format).trim().toLowerCase(Locale.ROOT);
        if (ext.startsWith(".")) {
            ext = ext.substring(1);
        }
        return ext.isBlank() ? "flac" : ext;
    }

    private static String pathExtension(Path path) {
        if (path == null || path.getFileName() == null) {
            return "";
        }
        String filename = path.getFileName().toString();
        int dot = filename.lastIndexOf('.');
        return dot >= 0 && dot + 1 < filename.length() ? filename.substring(dot + 1) : "";
    }

    private static boolean isKnownAudioExtension(String ext) {
        if (ext == null || ext.isBlank()) {
            return false;
        }
        return Set.of("wav", "mp3", "flac", "ogg", "m4a", "aac").contains(ext.toLowerCase(Locale.ROOT));
    }

    private void printOpenAiSseDelta(String requestId, String model, String delta) {
        long created = System.currentTimeMillis() / 1000L;
        String id = "chatcmpl-" + (requestId != null ? requestId : UUID.randomUUID().toString());
        String payload = String.format("{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
                escapeJson(id), created, escapeJson(model != null ? model : ""), escapeJson(delta));
        System.out.println("data: " + payload);
    }

    private void printOpenAiSseFinal(String requestId, String model) {
        long created = System.currentTimeMillis() / 1000L;
        String id = "chatcmpl-" + (requestId != null ? requestId : UUID.randomUUID().toString());
        String payload = String.format("{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}",
                escapeJson(id), created, escapeJson(model != null ? model : ""));
        System.out.println("data: " + payload);
        System.out.println("data: [DONE]");
    }

    private String escapeJson(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    private tech.kayys.gollek.spi.inference.InferenceResponse inferDirectWithProvider(String id, InferenceRequest request) { return null; }

    private void streamDirectWithProvider(String id, InferenceRequest request, long startTime) { }

    private Object buildDirectProviderRequest(InferenceRequest request, boolean streaming) { return null; }

    private boolean shouldIgnoreDirectProviderError(Throwable error, int emittedTokens) {
        if (error == null || emittedTokens <= 0) {
            return false;
        }
        String message = error.getMessage();
        return message != null && message.contains("Attempted to close a non-closeable session");
    }

    private boolean ensureProviderHealthy(String provider) { return true; }

    private void printCompatibilityHintBeforeInference() {
        if (providerId == null || modelId == null) return;
        String provider = providerId.toLowerCase();
        String model = modelId.toLowerCase();
        if ("safetensor".equals(provider)) {
            if (model.contains("vlm") || model.contains("vision")) {
                System.err.println("Hint: 'safetensor' provider is text-oriented; VLM models may fail.");
            }
        } else if ("llamacpp".equals(provider) && !model.endsWith(".gguf")) {
            System.err.println("Hint: 'llamacpp' works best with GGUF models.");
        }
    }

    private void printProviderHintFromError(Throwable throwable) {
        String detail = throwable.getMessage() != null ? throwable.getMessage().toLowerCase() : "";
        if (detail.contains("429") || detail.contains("quota")) {
            System.err.println("Hint: Provider quota reached. Wait or switch provider.");
        }
    }

    private void autoOpenImage(Path imagePath) {
        try {
            String os = System.getProperty("os.name").toLowerCase();
            if (os.contains("mac")) new ProcessBuilder("open", imagePath.toString()).start();
            else if (os.contains("linux")) new ProcessBuilder("xdg-open", imagePath.toString()).start();
        } catch (Exception ignored) {}
    }

    private void autoOpenAudio(Path audioPath) {
        try {
            String os = System.getProperty("os.name").toLowerCase();
            if (os.contains("mac")) new ProcessBuilder("afplay", audioPath.toString()).start();
        } catch (Exception ignored) {}
    }

    private static final class LivePcmAudioSink implements AutoCloseable {
        private final boolean enabled;
        private SourceDataLine line;
        private AudioFormat format;
        private boolean warned;

        private LivePcmAudioSink(boolean enabled) {
            this.enabled = enabled;
        }

        void write(byte[] pcm, Map<String, Object> metadata) {
            if (!enabled || pcm == null || pcm.length == 0) {
                return;
            }
            try {
                ensureOpen(metadata);
                if (line != null) {
                    line.write(pcm, 0, pcm.length);
                }
            } catch (Exception e) {
                warnOnce(e);
            }
        }

        private void ensureOpen(Map<String, Object> metadata) throws Exception {
            int sampleRate = (int) Math.round(optionalMetadataDouble(metadata, "audio_sample_rate", 48_000.0));
            int channels = Math.max(1, (int) Math.round(optionalMetadataDouble(metadata, "audio_channels", 1.0)));
            AudioFormat requested = new AudioFormat(
                    AudioFormat.Encoding.PCM_SIGNED,
                    sampleRate,
                    16,
                    channels,
                    channels * Short.BYTES,
                    sampleRate,
                    false);
            if (line != null && requested.matches(format)) {
                return;
            }
            closeLine();
            DataLine.Info info = new DataLine.Info(SourceDataLine.class, requested);
            line = (SourceDataLine) AudioSystem.getLine(info);
            int bufferSize = Math.max(requested.getFrameSize(), sampleRate * requested.getFrameSize() / 4);
            line.open(requested, bufferSize);
            line.start();
            format = requested;
        }

        private static double optionalMetadataDouble(
                Map<String, Object> metadata,
                String key,
                double fallback) {
            Double value = metadataDouble(metadata, key);
            return value == null || value <= 0.0 ? fallback : value;
        }

        private void warnOnce(Exception e) {
            if (warned) {
                return;
            }
            warned = true;
            System.err.println("\nWarning: live audio preview unavailable: " + e.getMessage());
        }

        @Override
        public void close() {
            closeLine();
        }

        private void closeLine() {
            if (line == null) {
                return;
            }
            try {
                line.drain();
                line.stop();
                line.close();
            } catch (Exception ignored) {
            } finally {
                line = null;
                format = null;
            }
        }
    }

    private boolean tryStandaloneLiteRtExecution(String localPath, long startTime) { return false; }

    private void requestProcessExit() {
        requestProcessExit(0);
    }

    private void requestProcessExit(int status) {
        try {
            Quarkus.asyncExit(status);
        } catch (Throwable ignored) {
        }
        System.exit(status);
    }

    private record ProviderLocalPathResolution(boolean ok, String localPath) {
    }
}
