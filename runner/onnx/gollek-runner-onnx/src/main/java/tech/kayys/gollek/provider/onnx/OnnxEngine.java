package tech.kayys.gollek.provider.onnx;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.onnx.runner.MossTtsOnnxRunner;
import tech.kayys.gollek.onnx.runner.OnnxModelDiagnostics;
import tech.kayys.gollek.onnx.runner.OnnxRuntimeRunner;
import tech.kayys.gollek.onnx.runner.PaddleOcrVlOnnxPlanner;
import tech.kayys.gollek.onnx.runner.PaddleOcrVlOnnxProbe;
import tech.kayys.gollek.onnx.runner.StableDiffusionOnnxRunner;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ArtifactLocation;
import tech.kayys.gollek.spi.model.ModelFormatDetector;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Set;

/**
 * ONNX Runtime Provider for Gollek.
 * Provides a bridge between the Inference Engine and OnnxRuntimeRunner.
 */
@ApplicationScoped
@io.quarkus.arc.Unremovable
public class OnnxEngine implements tech.kayys.gollek.spi.inference.LocalInferenceEngine {

    private static final Logger LOG = Logger.getLogger(OnnxEngine.class);
    private static final String PROVIDER_ID = "onnx";
    private static final String PROVIDER_VERSION = "1.0.0";
    private static final String MOSS_AUDIO_TOKENIZER_REPOSITORY = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX";
    private static final List<String> MOSS_CODEC_PARAMETER_KEYS = List.of("tts_codec", "tts_codec_path", "codec",
            "codec_path", "moss_codec", "moss_codec_path");

    @Inject
    OnnxProviderConfig config;

    @Inject
    OnnxRuntimeRunner runner;

    @Inject
    MossTtsOnnxRunner mossTtsRunner;

    private volatile boolean initialized = false;

    public OnnxEngine() {
    }

    public OnnxEngine(OnnxProviderConfig config, OnnxRuntimeRunner runner, StableDiffusionOnnxRunner sdRunner) {
        this.config = config;
        this.runner = runner;
        this.sdRunner = sdRunner;
    }

    public boolean supports(String modelId, tech.kayys.gollek.spi.inference.InferenceRequest request) {
        Path path = resolveModelPath(modelId, request);
        return path != null && Files.exists(path);
    }

    @Override
    public Uni<InferenceResponse> infer(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        return Uni.createFrom().item(() -> {
            try {
                ensureComponents();
                Path modelPath = resolveModelPath(request.getModel(), request);
                if (modelPath == null || !Files.exists(modelPath)) {
                    throw new RuntimeException("Model path not found: " + request.getModel());
                }
                rejectStandaloneMossAudioTokenizer(modelPath);
                if (isMossTtsPipeline(modelPath)) {
                    return inferMossTts(request, modelPath);
                }
                OnnxModelDiagnostics.Report diagnostics = OnnxModelDiagnostics.inspect(modelPath);
                if (diagnostics.isPaddleOcrVl()) {
                    return inferPaddleOcrVl(request, diagnostics);
                }
                rejectUnsupportedCustomOnnxPipeline(diagnostics, request);

                ensureInitialized(request, modelPath);

                InferenceRequest inferenceRequest = InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(convertMessages(request.getMessages()))
                        .parameters(request.getParameters())
                        .streaming(request.isStreaming())
                        .build();

                tech.kayys.gollek.spi.inference.InferenceResponse runnerResponse = runner.infer(inferenceRequest);
                return convertResponse(runnerResponse);
            } catch (Exception e) {
                LOG.error("ONNX Inference failed: " + e.getMessage(), e);
                throw new RuntimeException(cleanErrorMessage(e), e);
            }
        });
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                ensureComponents();
                Path modelPath = resolveModelPath(request.getModel(), request);

                if (modelPath == null || !Files.exists(modelPath)) {
                    emitter.fail(new RuntimeException("Model path not found: " + request.getModel()));
                    return;
                }
                rejectStandaloneMossAudioTokenizer(modelPath);
                if (isMossTtsPipeline(modelPath)) {
                    streamMossTts(request, modelPath, emitter);
                    return;
                }
                OnnxModelDiagnostics.Report diagnostics = OnnxModelDiagnostics.inspect(modelPath);
                if (diagnostics.isPaddleOcrVl()) {
                    streamPaddleOcrVl(request, diagnostics, emitter);
                    return;
                }
                rejectUnsupportedCustomOnnxPipeline(diagnostics, request);

                ensureInitialized(request, modelPath);

                var activeRunner = isStableDiffusion(modelPath) ? sdRunner : runner;

                activeRunner.stream(InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(convertMessages(request.getMessages()))
                        .parameters(request.getParameters())
                        .streaming(true)
                        .build()).subscribe().with(
                                chunk -> emitter.emit(convertChunk(chunk)),
                                emitter::fail,
                                emitter::complete);

            } catch (Exception e) {
                emitter.fail(new RuntimeException(cleanErrorMessage(e), e));
            }
        });
    }

    @Inject
    StableDiffusionOnnxRunner sdRunner;

    private void ensureComponents() {
        if (config == null) {
            config = defaultConfig();
        }
        if (runner == null) {
            runner = new OnnxRuntimeRunner();
        }
        if (sdRunner == null) {
            sdRunner = new StableDiffusionOnnxRunner();
        }
        if (mossTtsRunner == null) {
            mossTtsRunner = new MossTtsOnnxRunner();
        }
    }

    private OnnxProviderConfig defaultConfig() {
        return null;
    }

    private void ensureInitialized(tech.kayys.gollek.spi.inference.InferenceRequest request, Path modelPath)
            throws Exception {
        ensureComponents();
        boolean isSd = isStableDiffusion(modelPath);
        var activeRunner = isSd ? sdRunner : runner;

        if (!activeRunner.health()) {
            // For SD pipelines, resolve the base directory with ONNX models
            Path effectivePath = modelPath;
            if (isSd) {
                effectivePath = resolveOnnxBaseDir(modelPath);
            }
            LOG.infof("Initializing ONNX runner %s with path: %s", activeRunner.name(), effectivePath);
            tech.kayys.gollek.spi.model.ModelManifest manifest = tech.kayys.gollek.spi.model.ModelManifest.builder()
                    .modelId(request.getModel())
                    .name(request.getModel())
                    .version("latest")
                    .path(effectivePath.toAbsolutePath().toString())
                    .apiKey("none")
                    .requestId(request.getRequestId())
                    .artifacts(Map.of(ModelFormat.ONNX,
                            new tech.kayys.gollek.spi.model.ArtifactLocation(effectivePath.toUri().toString(), null,
                                    null, null)))
                    .build();

            RunnerConfiguration runnerCfg = RunnerConfiguration.builder()
                    .putParameter("intra_op_threads", config.threads())
                    .build();
            activeRunner.initialize(manifest, runnerCfg);
        }
    }

    private InferenceResponse inferMossTts(tech.kayys.gollek.spi.inference.InferenceRequest request, Path modelPath)
            throws Exception {
        Path codecDir = resolveMossCodecDir(request);
        if (codecDir == null) {
            throw missingMossCodecException(null);
        }
        InferenceRequest inferenceRequest = toInferenceRequest(request, false);
        MossTtsOnnxRunner.MossTtsAudio audio = mossTtsRunner.synthesize(modelPath, codecDir, inferenceRequest,
                config.threads());
        String base64 = mossTtsRunner.encodeAudioBase64(audio);
        Map<String, Object> metadata = new LinkedHashMap<>(audio.metadata());
        metadata.put("audio", base64);
        metadata.put("tts_codec_path", codecDir.toAbsolutePath().normalize().toString());
        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .content("Audio synthesized (" + audio.frames() + " frame(s), "
                        + audio.sampleRate() + " Hz " + audio.channels() + "ch "
                        + audio.format().toUpperCase(Locale.ROOT) + ").")
                .model(request.getModel())
                .durationMs(((Number) metadata.getOrDefault("bench.latency_ms", 0L)).longValue())
                .inputTokens(((Number) metadata.getOrDefault("tokens.input", 0)).intValue())
                .outputTokens(audio.frames())
                .metadata(metadata)
                .build();
    }

    private InferenceResponse inferPaddleOcrVl(
            tech.kayys.gollek.spi.inference.InferenceRequest request,
            OnnxModelDiagnostics.Report diagnostics) {
        if (diagnostics.modelDir() == null) {
            throw new UnsupportedOperationException("PaddleOCR-VL model directory could not be resolved.");
        }
        String imagePath = firstImagePathParameter(request);
        if (imagePath == null) {
            throw new UnsupportedOperationException(
                    "PaddleOCR-VL needs an input image. Pass --image /path/to/page.png or --input-images /path/to/page.png.");
        }
        String prompt = paddleOcrPrompt(request);
        String variant = firstStringParameter(request, "onnx_variant", "onnx_precision");
        Integer requestedImageTokens = intParameter(
                request,
                "ocr_image_tokens",
                "onnx_ocr_image_tokens",
                "onnx_image_tokens");
        int imageTokenLimit = requestedImageTokens == null ? 0 : Math.max(0, requestedImageTokens);
        int maxNewTokens = Math.max(1, request.getMaxTokens());
        long started = System.nanoTime();
        PaddleOcrVlOnnxProbe.PromptDecodeProbeResult result = PaddleOcrVlOnnxProbe.runPromptDecode(
                diagnostics.modelDir(),
                Path.of(imagePath),
                prompt,
                variant,
                imageTokenLimit,
                maxNewTokens,
                config.threads());
        PaddleOcrVlOnnxProbe.OcrPostProcessResult postProcess = PaddleOcrVlOnnxProbe.postProcessOcrText(
                result.decodedText(),
                result.imageTensor() == null ? null : result.imageTensor().image());
        String content = booleanParameter(request, "ocr_raw_output", "onnx_ocr_raw_output")
                ? postProcess.rawText().trim()
                : postProcess.displayText().trim();
        if (content.isBlank() && result.generatedTokenIds().length > 0) {
            content = "Generated token ids: " + java.util.Arrays.toString(result.generatedTokenIds());
        }
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("pipeline", "paddleocr-vl");
        metadata.put("provider", PROVIDER_ID);
        metadata.put("onnx_variant", result.plan().graphs().variant());
        metadata.put("image_path", imagePath);
        metadata.put("image_tokens", result.imageTokens());
        metadata.put("image_tokens_used", result.usedImageTokens());
        metadata.put("prompt_tokens", result.sequenceLength());
        metadata.put("image_token_id", result.imageTokenId());
        metadata.put("generated_token_ids", longArrayMetadata(result.generatedTokenIds()));
        metadata.put("ocr_raw_text", postProcess.rawText());
        metadata.put("ocr_text_without_locations", postProcess.textWithoutLocations());
        metadata.put("ocr_location_tokens", postProcess.locations());
        metadata.put("ocr_location_boxes", locationBoxesMetadata(postProcess.boxes()));
        metadata.put("ocr_location_box_count", postProcess.boxes().size());
        metadata.put("ocr_regions", ocrRegionsMetadata(postProcess.regions()));
        metadata.put("ocr_region_count", postProcess.regions().size());
        metadata.put("finish_reason", result.finishReason());
        metadata.put("vision_ms", result.visionRunDuration().toMillis());
        metadata.put("decoder_prefill_ms", result.decoderPrefillDuration().toMillis());
        metadata.put("decoder_decode_ms", result.decoderDecodeDuration().toMillis());
        metadata.put("latency_ms", java.time.Duration.ofNanos(System.nanoTime() - started).toMillis());
        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .content(content)
                .model(request.getModel())
                .inputTokens(result.sequenceLength())
                .outputTokens(result.generatedTokenIds().length)
                .durationMs(((Number) metadata.get("latency_ms")).longValue())
                .metadata(metadata)
                .finishReason("stop".equals(result.finishReason())
                        ? InferenceResponse.FinishReason.STOP
                        : InferenceResponse.FinishReason.LENGTH)
                .build();
    }

    private void streamPaddleOcrVl(
            tech.kayys.gollek.spi.inference.InferenceRequest request,
            OnnxModelDiagnostics.Report diagnostics,
            io.smallrye.mutiny.subscription.MultiEmitter<? super StreamingInferenceChunk> emitter) {
        try {
            InferenceResponse response = inferPaddleOcrVl(request, diagnostics);
            emitter.emit(new StreamingInferenceChunk(
                    request.getRequestId(),
                    0,
                    ModalityType.TEXT,
                    response.getContent(),
                    null,
                    true,
                    response.getFinishReason() == InferenceResponse.FinishReason.STOP ? "stop" : "length",
                    null,
                    Instant.now(),
                    response.getMetadata(),
                    null,
                    null,
                    null));
            emitter.complete();
        } catch (Exception e) {
            emitter.fail(new RuntimeException(cleanErrorMessage(e), e));
        }
    }

    private void streamMossTts(
            tech.kayys.gollek.spi.inference.InferenceRequest request,
            Path modelPath,
            io.smallrye.mutiny.subscription.MultiEmitter<? super StreamingInferenceChunk> emitter) {
        try {
            Path codecDir = resolveMossCodecDir(request);
            if (codecDir == null) {
                emitter.fail(missingMossCodecException(null));
                return;
            }
            InferenceRequest inferenceRequest = toInferenceRequest(request, true);
            java.util.concurrent.atomic.AtomicInteger index = new java.util.concurrent.atomic.AtomicInteger(0);
            MossTtsOnnxRunner.MossTtsProgressListener progressListener = mossTtsStreamProgressEnabled(request)
                    ? progress -> emitter.emit(mossTtsProgressChunk(
                            request.getRequestId(),
                            index.getAndIncrement(),
                            progress))
                    : null;
            MossTtsOnnxRunner.MossTtsPcmChunkListener pcmChunkListener = mossTtsPcmStreamingEnabled(request)
                    ? chunk -> emitter.emit(mossTtsPcmChunk(
                            request.getRequestId(),
                            index.getAndIncrement(),
                            chunk))
                    : null;
            MossTtsOnnxRunner.MossTtsAudio audio = mossTtsRunner.synthesize(
                    modelPath,
                    codecDir,
                    inferenceRequest,
                    config.threads(),
                    progressListener,
                    pcmChunkListener);
            Map<String, Object> metadata = new LinkedHashMap<>(audio.metadata());
            metadata.put("audio_bytes", audio.bytes().length);
            metadata.put("audio_stream_payload", "encoded");
            metadata.put("tts_codec_path", codecDir.toAbsolutePath().normalize().toString());
            emitAudioChunks(request, audio, metadata, index, emitter);
            emitter.complete();
        } catch (Exception e) {
            emitter.fail(new RuntimeException(cleanErrorMessage(e), e));
        }
    }

    private StreamingInferenceChunk mossTtsProgressChunk(
            String requestId,
            int index,
            MossTtsOnnxRunner.MossTtsProgress progress) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("tts_stream_event", progress.stage());
        metadata.put("tts_stream_frames", progress.frames());
        metadata.put("tts_stream_max_frames", progress.maxFrames());
        metadata.put("tts_stream_message", progress.message());
        String delta = "[" + progress.message() + "]";
        return new StreamingInferenceChunk(
                requestId,
                index,
                ModalityType.TEXT,
                delta,
                null,
                false,
                null,
                null,
                Instant.now(),
                metadata,
                null,
                null,
                null);
    }

    private StreamingInferenceChunk mossTtsPcmChunk(
            String requestId,
            int index,
            MossTtsOnnxRunner.MossTtsPcmChunk chunk) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("tts_stream_event", "pcm_chunk");
        metadata.put("audio_stream_payload", "pcm_s16le");
        metadata.put("audio_stream_preview", true);
        metadata.put("audio_format", "pcm_s16le");
        metadata.put("audio_mime", "audio/L16");
        metadata.put("audio_sample_rate", chunk.sampleRate());
        metadata.put("audio_channels", chunk.channels());
        metadata.put("audio_source_channels", chunk.sourceChannels());
        metadata.put("audio_samples", chunk.samples());
        metadata.put("audio_chunk_index", index);
        metadata.put("audio_chunk_bytes", chunk.pcmS16le().length);
        metadata.put("tts_frame_index", chunk.frameIndex());
        metadata.put("tts_frames", chunk.totalFrames());
        metadata.put("tts_max_frames", chunk.totalFrames());
        metadata.put("tts_stream_stage", chunk.codecDecodeMode().startsWith("generation_") ? "generation" : "decode");
        metadata.put("audio_codec_decode_mode", chunk.codecDecodeMode());
        metadata.put("audio_channel_mode", chunk.channelMode());
        metadata.put("audio_processing", false);
        return new StreamingInferenceChunk(
                requestId,
                index,
                ModalityType.AUDIO,
                java.util.Base64.getEncoder().encodeToString(chunk.pcmS16le()),
                null,
                false,
                null,
                null,
                Instant.now(),
                metadata,
                null,
                null,
                null);
    }

    private void emitAudioChunks(
            tech.kayys.gollek.spi.inference.InferenceRequest request,
            MossTtsOnnxRunner.MossTtsAudio audio,
            Map<String, Object> metadata,
            java.util.concurrent.atomic.AtomicInteger index,
            io.smallrye.mutiny.subscription.MultiEmitter<? super StreamingInferenceChunk> emitter) {
        byte[] bytes = audio.bytes();
        int chunkBytes = requestedAudioStreamChunkBytes(request);
        int chunkCount = Math.max(1, (bytes.length + chunkBytes - 1) / chunkBytes);
        metadata.put("audio_stream_chunks", chunkCount);
        metadata.put("audio_stream_chunk_bytes", chunkBytes);
        for (int offset = 0, chunkIndex = 0; offset < bytes.length; offset += chunkBytes, chunkIndex++) {
            int end = Math.min(bytes.length, offset + chunkBytes);
            boolean finalChunk = end >= bytes.length;
            byte[] slice = java.util.Arrays.copyOfRange(bytes, offset, end);
            Map<String, Object> chunkMetadata = finalChunk
                    ? new LinkedHashMap<>(metadata)
                    : audioChunkMetadata(audio, chunkIndex, chunkCount, offset, slice.length, chunkBytes);
            emitter.emit(new StreamingInferenceChunk(
                    request.getRequestId(),
                    index.getAndIncrement(),
                    ModalityType.AUDIO,
                    java.util.Base64.getEncoder().encodeToString(slice),
                    null,
                    finalChunk,
                    finalChunk ? "stop" : null,
                    null,
                    Instant.now(),
                    chunkMetadata,
                    null,
                    null,
                    null));
        }
    }

    private Map<String, Object> audioChunkMetadata(
            MossTtsOnnxRunner.MossTtsAudio audio,
            int chunkIndex,
            int chunkCount,
            int offset,
            int length,
            int chunkBytes) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("tts_stream_event", "audio_chunk");
        metadata.put("audio_format", audio.format());
        metadata.put("audio_mime", audio.mimeType());
        metadata.put("audio_stream_payload", "encoded");
        metadata.put("audio_chunk_index", chunkIndex);
        metadata.put("audio_chunk_count", chunkCount);
        metadata.put("audio_chunk_offset", offset);
        metadata.put("audio_chunk_bytes", length);
        metadata.put("audio_stream_chunk_bytes", chunkBytes);
        return metadata;
    }

    private boolean mossTtsStreamProgressEnabled(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        if (request == null || request.getParameters() == null) {
            return true;
        }
        Object value = request.getParameters().get("tts_stream_progress");
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value instanceof String text && !text.isBlank()) {
            return Boolean.parseBoolean(text.trim());
        }
        return true;
    }

    private boolean mossTtsPcmStreamingEnabled(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        if (request == null || request.getParameters() == null) {
            return false;
        }
        for (String key : List.of("tts_stream_pcm", "audio_stream_pcm", "tts_live_audio", "live_audio")) {
            Object value = request.getParameters().get(key);
            if (value instanceof Boolean bool) {
                return bool;
            }
            if (value instanceof String text && !text.isBlank()) {
                return Boolean.parseBoolean(text.trim());
            }
        }
        return false;
    }

    private int requestedAudioStreamChunkBytes(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        int defaultBytes = 64 * 1024;
        if (request == null || request.getParameters() == null) {
            return defaultBytes;
        }
        Integer explicitBytes = intParameter(request, "audio_stream_chunk_bytes", "tts_stream_chunk_bytes");
        if (explicitBytes != null) {
            return clampAudioStreamChunkBytes(explicitBytes);
        }
        Integer explicitKb = intParameter(
                request,
                "audio_stream_chunk_kb",
                "tts_stream_chunk_kb",
                "audio_stream_chunk_size_kb");
        if (explicitKb != null) {
            return clampAudioStreamChunkBytes(explicitKb * 1024);
        }
        return defaultBytes;
    }

    private static int clampAudioStreamChunkBytes(int value) {
        return Math.max(4 * 1024, Math.min(1024 * 1024, value));
    }

    private Integer intParameter(tech.kayys.gollek.spi.inference.InferenceRequest request, String... keys) {
        if (request == null || request.getParameters() == null) {
            return null;
        }
        for (String key : keys) {
            Object value = request.getParameters().get(key);
            if (value instanceof Number number) {
                return number.intValue();
            }
            if (value instanceof String text && !text.isBlank()) {
                try {
                    return Integer.parseInt(text.trim());
                } catch (NumberFormatException ignored) {
                    // Continue to the next alias.
                }
            }
        }
        return null;
    }

    private boolean booleanParameter(tech.kayys.gollek.spi.inference.InferenceRequest request, String... keys) {
        if (request == null || request.getParameters() == null) {
            return false;
        }
        for (String key : keys) {
            Object value = request.getParameters().get(key);
            if (value instanceof Boolean bool) {
                return bool;
            }
            if (value instanceof String text && !text.isBlank()) {
                return Boolean.parseBoolean(text.trim());
            }
        }
        return false;
    }

    private List<Long> longArrayMetadata(long[] values) {
        if (values == null || values.length == 0) {
            return List.of();
        }
        List<Long> result = new ArrayList<>(values.length);
        for (long value : values) {
            result.add(value);
        }
        return List.copyOf(result);
    }

    private List<Map<String, Integer>> locationBoxesMetadata(List<PaddleOcrVlOnnxProbe.LocationBox> boxes) {
        if (boxes == null || boxes.isEmpty()) {
            return List.of();
        }
        List<Map<String, Integer>> values = new ArrayList<>();
        for (PaddleOcrVlOnnxProbe.LocationBox box : boxes) {
            Map<String, Integer> item = new LinkedHashMap<>();
            item.put("index", box.index());
            item.put("x1", box.x1());
            item.put("y1", box.y1());
            item.put("x2", box.x2());
            item.put("y2", box.y2());
            item.put("box_x1", box.normalizedX1());
            item.put("box_y1", box.normalizedY1());
            item.put("box_x2", box.normalizedX2());
            item.put("box_y2", box.normalizedY2());
            item.put("pixel_x1", box.pixelX1());
            item.put("pixel_y1", box.pixelY1());
            item.put("pixel_x2", box.pixelX2());
            item.put("pixel_y2", box.pixelY2());
            item.put("box_pixel_x1", box.normalizedPixelX1());
            item.put("box_pixel_y1", box.normalizedPixelY1());
            item.put("box_pixel_x2", box.normalizedPixelX2());
            item.put("box_pixel_y2", box.normalizedPixelY2());
            values.add(item);
        }
        return List.copyOf(values);
    }

    private List<Map<String, Object>> ocrRegionsMetadata(List<PaddleOcrVlOnnxProbe.OcrTextRegion> regions) {
        if (regions == null || regions.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> values = new ArrayList<>();
        for (PaddleOcrVlOnnxProbe.OcrTextRegion region : regions) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("index", region.index());
            item.put("text", region.text());
            item.put("box", locationBoxMetadata(region.box()));
            values.add(item);
        }
        return List.copyOf(values);
    }

    private Map<String, Integer> locationBoxMetadata(PaddleOcrVlOnnxProbe.LocationBox box) {
        Map<String, Integer> item = new LinkedHashMap<>();
        if (box == null) {
            return item;
        }
        item.put("index", box.index());
        item.put("x1", box.x1());
        item.put("y1", box.y1());
        item.put("x2", box.x2());
        item.put("y2", box.y2());
        item.put("box_x1", box.normalizedX1());
        item.put("box_y1", box.normalizedY1());
        item.put("box_x2", box.normalizedX2());
        item.put("box_y2", box.normalizedY2());
        item.put("pixel_x1", box.pixelX1());
        item.put("pixel_y1", box.pixelY1());
        item.put("pixel_x2", box.pixelX2());
        item.put("pixel_y2", box.pixelY2());
        item.put("box_pixel_x1", box.normalizedPixelX1());
        item.put("box_pixel_y1", box.normalizedPixelY1());
        item.put("box_pixel_x2", box.normalizedPixelX2());
        item.put("box_pixel_y2", box.normalizedPixelY2());
        return item;
    }

    private InferenceRequest toInferenceRequest(tech.kayys.gollek.spi.inference.InferenceRequest request,
            boolean streaming) {
        return InferenceRequest.builder()
                .requestId(request.getRequestId())
                .model(request.getModel())
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .streaming(streaming)
                .build();
    }

    private Path resolveMossCodecDir(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        String explicitRef = requestedMossCodecRef(request);
        if (explicitRef != null) {
            Path explicit = resolveExplicitMossCodecDir(explicitRef);
            if (explicit == null) {
                throw missingMossCodecException(explicitRef);
            }
            return explicit;
        }
        return findInstalledMossAudioTokenizerDir(null);
    }

    private String requestedMossCodecRef(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        if (request == null || request.getParameters() == null) {
            return null;
        }
        for (String key : MOSS_CODEC_PARAMETER_KEYS) {
            Object value = request.getParameters().get(key);
            if (value instanceof String text && !text.isBlank()) {
                return text.trim();
            }
        }
        return null;
    }

    private Path resolveExplicitMossCodecDir(String ref) {
        if (ref == null || ref.isBlank()) {
            return null;
        }

        try {
            Path direct = Paths.get(ref);
            if (Files.exists(direct)) {
                Path resolved = codecDirFrom(direct);
                if (resolved != null) {
                    return resolved;
                }
            }
        } catch (Exception ignored) {
            // Treat non-path values as model ids/names and scan the model base path.
        }

        return findInstalledMossAudioTokenizerDir(ref);
    }

    private RuntimeException missingMossCodecException(String requestedRef) {
        String modelBase = safeModelBasePath().toString();
        if (requestedRef != null && !requestedRef.isBlank()) {
            return new UnsupportedOperationException(
                    "MOSS TTS companion codec not found for --tts-codec " + requestedRef
                            + ". Expected a directory containing codec_browser_onnx_meta.json. "
                            + "Pull it with: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY
                            + ", or pass --tts-codec /path/to/" + MOSS_AUDIO_TOKENIZER_REPOSITORY);
        }
        return new UnsupportedOperationException(
                "MOSS-TTS-Nano ONNX requires the companion codec repository, but no "
                        + "codec_browser_onnx_meta.json was found under " + modelBase
                        + ". Pull it with: gollek pull " + MOSS_AUDIO_TOKENIZER_REPOSITORY
                        + ", or pass --tts-codec /path/to/" + MOSS_AUDIO_TOKENIZER_REPOSITORY);
    }

    private void rejectStandaloneMossAudioTokenizer(Path modelPath) {
        Path modelDir = Files.isDirectory(modelPath) ? modelPath : modelPath.getParent();
        if (modelDir == null || !Files.isRegularFile(modelDir.resolve("codec_browser_onnx_meta.json"))) {
            return;
        }
        throw new UnsupportedOperationException(
                "This ONNX repository is the MOSS audio tokenizer/codec, not a text-to-speech model. "
                        + "Run the TTS model instead, for example: gollek run --model "
                        + "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --prompt \"hello\" --output output.wav");
    }

    private void rejectUnsupportedCustomOnnxPipeline(OnnxModelDiagnostics.Report report,
            tech.kayys.gollek.spi.inference.InferenceRequest request) {
        if (report == null || report.modelPath() == null || isStableDiffusion(report.modelPath())) {
            return;
        }
        if (!report.requiresCustomPipelineOrchestration()) {
            return;
        }
        throw new UnsupportedOperationException(unsupportedCustomPipelineMessage(report, request));
    }

    private String unsupportedCustomPipelineMessage(OnnxModelDiagnostics.Report report,
            tech.kayys.gollek.spi.inference.InferenceRequest request) {
        StringBuilder message = new StringBuilder();
        message.append(report.pipelineType())
                .append(" was detected at ")
                .append(report.modelDir() == null ? report.modelPath() : report.modelDir())
                .append(". This repository is a multi-session OCR/VL pipeline, not a single text-generation ONNX graph. ");
        message.append("Gollek inspected the ONNX files, but OCR execution still needs image preprocessing, ")
                .append("vision_encoder, embedding, decoder orchestration, and tokenizer postprocessing. ");
        message.append(
                "Current ONNX runtime support covers MOSS TTS, Stable Diffusion, and single text-generation-style graphs.");

        List<String> graphSummary = report.graphs().stream()
                .map(graph -> graph.role() + "=" + graph.relativePath())
                .limit(6)
                .toList();
        if (!graphSummary.isEmpty()) {
            message.append(" Detected graphs: ").append(String.join(", ", graphSummary)).append(".");
        }

        String imagePath = firstImagePathParameter(request);
        appendPaddleOcrPlanSummary(message, report, request, imagePath);
        if (imagePath == null) {
            message.append(" When the OCR path is wired, pass the input page with --image /path/to/page.png.");
        } else {
            message.append(" Received image: ").append(imagePath).append(".");
        }
        String modelRef = firstStringParameter(request, "requested_model_ref");
        if (modelRef == null && request != null) {
            modelRef = request.getModel();
        }
        message.append(" Inspect this model with: gollek run --model ")
                .append(modelRef == null ? "<model>" : modelRef)
                .append(" --onnx-diagnostics");
        if (imagePath != null) {
            message.append(". Current executable OCR bridge probe: gollek run --model ")
                    .append(modelRef == null ? "<model>" : quoteCliArg(modelRef))
                    .append(" --onnx-pipeline-probe --image ")
                    .append(quoteCliArg(imagePath));
            String variant = firstStringParameter(request, "onnx_variant", "onnx_precision");
            if (variant != null) {
                message.append(" --onnx-variant ").append(quoteCliArg(variant));
            }
        }
        return message.toString();
    }

    private void appendPaddleOcrPlanSummary(
            StringBuilder message,
            OnnxModelDiagnostics.Report report,
            tech.kayys.gollek.spi.inference.InferenceRequest request,
            String imagePath) {
        if (!report.isPaddleOcrVl() || report.modelDir() == null) {
            return;
        }
        try {
            String variant = firstStringParameter(request, "onnx_variant", "onnx_precision");
            List<Path> images = imagePath == null ? List.of() : List.of(Path.of(imagePath));
            PaddleOcrVlOnnxPlanner.Plan plan = PaddleOcrVlOnnxPlanner.plan(report.modelDir(), images, variant);
            message.append(" Planned graph variant: ")
                    .append(plan.graphs().variant())
                    .append(" (vision=")
                    .append(fileName(plan.graphs().visionEncoder()))
                    .append(", embedding=")
                    .append(fileName(plan.graphs().embedding()))
                    .append(", decoder=")
                    .append(fileName(plan.graphs().decoder()))
                    .append(").");
            if (!plan.images().isEmpty()) {
                PaddleOcrVlOnnxPlanner.ImagePlan image = plan.images().get(0);
                message.append(" Image preflight: ")
                        .append(image.originalWidth()).append("x").append(image.originalHeight())
                        .append(" -> ")
                        .append(image.resizedWidth()).append("x").append(image.resizedHeight())
                        .append(", grid=[")
                        .append(image.gridT()).append(",").append(image.gridH()).append(",").append(image.gridW())
                        .append("], prompt image tokens=")
                        .append(image.promptImageTokens())
                        .append(", pixel_values=[1,")
                        .append(image.patchCount()).append(",3,")
                        .append(image.patchSize()).append(",")
                        .append(image.patchSize()).append("]")
                        .append(".");
            }
        } catch (Exception e) {
            message.append(" PaddleOCR-VL preflight failed: ").append(e.getMessage()).append(".");
        }
    }

    private String firstStringParameter(tech.kayys.gollek.spi.inference.InferenceRequest request, String... keys) {
        if (request == null || request.getParameters() == null) {
            return null;
        }
        for (String key : keys) {
            Object value = request.getParameters().get(key);
            if (value instanceof String text && !text.isBlank()) {
                return text.trim();
            }
        }
        return null;
    }

    private String paddleOcrPrompt(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        String prompt = firstStringParameter(request, "prompt", "ocr_prompt", "vision_prompt");
        if (prompt != null) {
            return prompt;
        }
        if (request != null && request.getMessages() != null && !request.getMessages().isEmpty()) {
            for (int i = request.getMessages().size() - 1; i >= 0; i--) {
                Message message = request.getMessages().get(i);
                if (message != null
                        && message.getRole() == Message.Role.USER
                        && message.getContent() != null
                        && !message.getContent().isBlank()) {
                    return message.getContent().trim();
                }
            }
        }
        return "Extract all text from the image.";
    }

    private String firstImagePathParameter(tech.kayys.gollek.spi.inference.InferenceRequest request) {
        String direct = firstStringParameter(request, "image_path", "vision_input_path", "input_image");
        if (direct != null) {
            return direct;
        }
        if (request == null || request.getParameters() == null) {
            return null;
        }
        for (String key : List.of("image_paths", "vision_input_paths", "input_images")) {
            Object value = request.getParameters().get(key);
            if (value instanceof Iterable<?> items) {
                for (Object item : items) {
                    if (item instanceof String text && !text.isBlank()) {
                        return text.trim();
                    }
                }
            } else if (value instanceof String[] items) {
                for (String item : items) {
                    if (item != null && !item.isBlank()) {
                        return item.trim();
                    }
                }
            } else if (value instanceof String text && !text.isBlank()) {
                return text.trim();
            }
        }
        return null;
    }

    private static String quoteCliArg(String value) {
        if (value == null || value.isBlank()) {
            return "''";
        }
        if (value.matches("[A-Za-z0-9_./:@%+=,-]+")) {
            return value;
        }
        return "'" + value.replace("'", "'\"'\"'") + "'";
    }

    private static String fileName(Path path) {
        return path == null || path.getFileName() == null ? "-" : path.getFileName().toString();
    }

    private Path findInstalledMossAudioTokenizerDir(String preferredRef) {
        Path basePath = safeModelBasePath();
        if (!Files.isDirectory(basePath)) {
            return null;
        }
        List<Path> codecDirs = findCodecDirs(basePath);
        if (codecDirs.isEmpty()) {
            return null;
        }
        if (preferredRef != null && !preferredRef.isBlank()) {
            return codecDirs.stream()
                    .filter(path -> pathMatchesRef(path, preferredRef))
                    .sorted(Comparator.comparing(path -> path.toAbsolutePath().normalize().toString()))
                    .findFirst()
                    .orElse(null);
        }
        return codecDirs.stream()
                .sorted(Comparator
                        .comparingInt((Path path) -> codecMatchScore(path, preferredRef)).reversed()
                        .thenComparing(path -> path.toAbsolutePath().normalize().toString()))
                .findFirst()
                .orElse(null);
    }

    private Path safeModelBasePath() {
        try {
            if (config != null && config.modelBasePath() != null && !config.modelBasePath().isBlank()) {
                return Paths.get(config.modelBasePath());
            }
        } catch (Exception ignored) {
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "models");
    }

    private List<Path> findCodecDirs(Path basePath) {
        List<Path> codecDirs = new ArrayList<>();
        try (var walk = Files.walk(basePath, 8)) {
            walk.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().equals("codec_browser_onnx_meta.json"))
                    .map(Path::getParent)
                    .filter(path -> path != null)
                    .forEach(codecDirs::add);
        } catch (Exception ignored) {
            // Best effort discovery.
        }
        return codecDirs;
    }

    private Path codecDirFrom(Path path) {
        if (path == null) {
            return null;
        }
        if (Files.isRegularFile(path)) {
            if (path.getFileName().toString().equals("codec_browser_onnx_meta.json")) {
                return path.getParent();
            }
            Path parent = path.getParent();
            return parent != null && isMossAudioTokenizerPipeline(parent) ? parent : null;
        }
        if (isMossAudioTokenizerPipeline(path)) {
            return path;
        }
        if (Files.isDirectory(path)) {
            try (var walk = Files.walk(path, 6)) {
                return walk.filter(Files::isRegularFile)
                        .filter(file -> file.getFileName().toString().equals("codec_browser_onnx_meta.json"))
                        .map(Path::getParent)
                        .findFirst()
                        .orElse(null);
            } catch (Exception ignored) {
                return null;
            }
        }
        return null;
    }

    private int codecMatchScore(Path codecDir, String preferredRef) {
        int score = 0;
        if (preferredRef != null && !preferredRef.isBlank() && pathMatchesRef(codecDir, preferredRef)) {
            score += 1_000;
        }
        if (pathMatchesRef(codecDir, MOSS_AUDIO_TOKENIZER_REPOSITORY)) {
            score += 100;
        }
        return score;
    }

    private boolean pathMatchesRef(Path path, String ref) {
        if (path == null || ref == null || ref.isBlank()) {
            return false;
        }
        String normalizedRef = normalizeRef(ref);
        String normalizedPath = normalizeRef(path.toAbsolutePath().normalize().toString());
        if (normalizedPath.contains(normalizedRef)) {
            return true;
        }
        Path fileName = path.getFileName();
        if (fileName != null && normalizeRef(fileName.toString()).contains(normalizedRef)) {
            return true;
        }
        Path parent = path.getParent();
        return parent != null
                && parent.getFileName() != null
                && normalizeRef(parent.getFileName().toString()).contains(normalizedRef);
    }

    private String normalizeRef(String value) {
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

    private String cleanErrorMessage(Throwable error) {
        if (error == null) {
            return "ONNX inference failed";
        }
        String message = error.getMessage();
        if (message == null || message.isBlank()) {
            message = error.getClass().getSimpleName();
        }
        if (message.startsWith("ONNX inference failed: ")) {
            return message;
        }
        return message;
    }

    /**
     * Detects Stable Diffusion pipeline by checking for UNet + VAE subdirectories.
     * Delegates to shared utility in {@link ModelFormatDetector}.
     */
    private boolean isStableDiffusion(Path modelPath) {
        return tech.kayys.gollek.spi.model.ModelFormatDetector.isStableDiffusion(modelPath);
    }

    /**
     * For SD pipelines, locates the directory that contains the actual ONNX model
     * files.
     * The safetensors path may not contain .onnx files — the blobs path typically
     * does.
     */
    private Path resolveOnnxBaseDir(Path modelPath) {
        // If the path already has unet/model.onnx, use it directly
        if (Files.exists(modelPath.resolve("unet/model.onnx"))) {
            return modelPath;
        }

        // Try the local model registry's resolve for the blob path
        if (false) {
            try {
                var entries = java.util.List.of();
                for (var entry : entries) {

                }
            } catch (Exception ignored) {
            }
        }

        // Fallback: search for unet/model.onnx in common model storage paths
        Path gollekModels = Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs");
        if (Files.isDirectory(gollekModels)) {
            try (var walk = Files.walk(gollekModels, 2)) {
                return walk.filter(p -> Files.isDirectory(p) && Files.exists(p.resolve("unet/model.onnx")))
                        .findFirst()
                        .orElse(modelPath);
            } catch (Exception ignored) {
            }
        }

        return modelPath;
    }

    private Path resolveModelPath(String modelId, tech.kayys.gollek.spi.inference.InferenceRequest request) {
        ensureComponents();
        if (modelId == null)
            return null;

        Path targetPath = null;

        // Try metadata first (provided by ModelRouterService)
        if (request != null && request.getMetadata() != null) {
            Object pathObj = request.getMetadata().get("model_path");
            if (pathObj instanceof String pt && !pt.isBlank()) {
                Path p = Paths.get(pt);
                if (Files.exists(p)) {
                    targetPath = p;
                }
            }
        }

        // Try registry next
        if (targetPath == null && false) {
            java.util.Optional<tech.kayys.gollek.spi.registry.ModelEntry> entry = java.util.Optional.empty();
            if (entry.isPresent() && entry.get().physicalPath() != null) {
                Path blobDir = entry.get().physicalPath();
                if (Files.exists(blobDir)) {
                    targetPath = blobDir;
                }
            }
        }

        // Same logic as LiteRTProvider for now
        if (targetPath == null) {
            try {
                if (modelId.startsWith("file://")) {
                    targetPath = Paths.get(modelId.substring("file://".length()));
                } else {
                    Path asPath = Paths.get(modelId);
                    if (asPath.isAbsolute() && Files.exists(asPath)) {
                        targetPath = asPath;
                    } else if (config != null) {
                        Path basePath = Paths.get(config.modelBasePath());
                        Path modelDir = basePath.resolve(modelId);
                        if (Files.exists(modelDir)) {
                            targetPath = modelDir;
                        }
                    }
                }
            } catch (Exception e) {
                targetPath = Paths.get(modelId);
            }
        }

        if (targetPath != null && Files.isDirectory(targetPath)) {
            if (isMossTtsPipeline(targetPath)) {
                return targetPath;
            }
            if (isMossAudioTokenizerPipeline(targetPath)) {
                return targetPath;
            }

            // For ONNX pipelines (like SD), the "model" might be a directory.
            // Check if it's a Stable Diffusion pipeline first.
            if (isStableDiffusion(targetPath)) {
                return targetPath;
            }

            // Otherwise, look for a single .onnx file (legacy behavior for simple models in
            // dirs)
            try (var stream = Files.walk(targetPath, 3)) {
                return stream.filter(p -> p.toString().endsWith(".onnx"))
                        .findFirst()
                        .orElse(targetPath);
            } catch (Exception e) {
                return targetPath;
            }
        }
        if (targetPath != null && Files.isRegularFile(targetPath)) {
            Path parent = targetPath.getParent();
            if (parent != null && isMossTtsPipeline(parent)) {
                return parent;
            }
            if (parent != null && isMossAudioTokenizerPipeline(parent)) {
                return parent;
            }
        }
        return targetPath;
    }

    private boolean isMossTtsPipeline(Path modelPath) {
        return modelPath != null && Files.isRegularFile(modelPath.resolve("tts_browser_onnx_meta.json"));
    }

    private boolean isMossAudioTokenizerPipeline(Path modelPath) {
        return modelPath != null && Files.isRegularFile(modelPath.resolve("codec_browser_onnx_meta.json"));
    }

    @Override
    public void shutdown() {
        if (runner != null) {
            runner.close();
        }
        if (sdRunner != null) {
            sdRunner.close();
        }
        if (mossTtsRunner != null) {
            mossTtsRunner.close();
        }
        initialized = false;
        LOG.info("ONNX Provider shutdown complete");
    }

    private List<tech.kayys.gollek.spi.Message> convertMessages(List<tech.kayys.gollek.spi.Message> messages) {
        if (messages == null)
            return List.of();
        return messages.stream()
                .map(m -> new tech.kayys.gollek.spi.Message(
                        tech.kayys.gollek.spi.Message.Role.valueOf(m.getRole().name()),
                        m.getContent()))
                .toList();
    }

    private InferenceResponse convertResponse(tech.kayys.gollek.spi.inference.InferenceResponse response) {
        if (response == null)
            return null;
        return InferenceResponse.builder()
                .requestId(response.getRequestId())
                .content(response.getContent())
                .model(response.getModel())
                .durationMs(response.getDurationMs())
                .inputTokens(response.getInputTokens())
                .outputTokens(response.getOutputTokens())
                .finishReason(convertFinishReason(response.getFinishReason()))
                .metadata(response.getMetadata())
                .build();
    }

    private InferenceResponse.FinishReason convertFinishReason(
            tech.kayys.gollek.spi.inference.InferenceResponse.FinishReason reason) {
        if (reason == null)
            return null;
        try {
            return InferenceResponse.FinishReason.valueOf(reason.name());
        } catch (Exception e) {
            return null;
        }
    }

    private StreamingInferenceChunk convertChunk(tech.kayys.gollek.spi.inference.StreamingInferenceChunk chunk) {
        if (chunk == null)
            return null;
        return new StreamingInferenceChunk(
                chunk.requestId(),
                chunk.index(),
                convertModality(chunk.modality()),
                chunk.delta(),
                null,
                chunk.finished(),
                chunk.finishReason(),
                chunk.usage() != null
                        ? new StreamingInferenceChunk.ChunkUsage(chunk.usage().inputTokens(),
                                chunk.usage().outputTokens(), 0)
                        : null,
                chunk.emittedAt(),
                chunk.metadata(),
                null,
                null,
                null);
    }

    private ModalityType convertModality(tech.kayys.gollek.spi.model.ModalityType modality) {
        if (modality == null)
            return null;
        try {
            return ModalityType.valueOf(modality.name());
        } catch (Exception e) {
            return null;
        }
    }

}
