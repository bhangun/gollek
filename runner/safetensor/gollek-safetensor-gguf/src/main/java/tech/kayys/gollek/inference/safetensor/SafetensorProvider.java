package tech.kayys.gollek.inference.safetensor;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.inference.gguf.ModelConverterService;
import tech.kayys.gollek.inference.libtorch.LibTorchProvider;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.safetensor.engine.warmup.DirectSafetensorBackend;
import tech.kayys.gollek.converter.GGUFConverter;
import tech.kayys.gollek.converter.model.GGUFConversionParams;
import tech.kayys.gollek.converter.model.QuantizationType;
import tech.kayys.gollek.spi.observability.AdapterMetricTagResolver;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.provider.AdapterCapabilityProfile;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import org.jboss.logging.Logger;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Arrays;
import java.util.Locale;
import java.util.LinkedHashSet;
import java.util.Set;

@ApplicationScoped
public class SafetensorProvider implements StreamingProvider {

    private static final String PROVIDER_ID = "safetensor";
    private static final Logger log = Logger.getLogger(SafetensorProvider.class);

    @Inject
    SafetensorProviderConfig config;

    @Inject
    LibTorchProvider libTorchProvider;

    @Inject
    SafetensorGgufBackend ggufBackend;

    @Inject
    DirectSafetensorBackend directBackend;

    @Inject
    GGUFConverter ggufConverter;

    @Inject
    ModelConverterService modelConverterService;

    @Inject
    AdapterMetricsRecorder adapterMetricsRecorder = new NoopAdapterMetricsRecorder();

    @Override
    public boolean isEnabled() {
        return config.enabled();
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "SafeTensor Provider";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(name())
                .version(version())
                .description("SafeTensor provider using GGUF conversion (default) or LibTorch backend")
                .vendor("Golek / Kayys")
                .homepage("https://github.com/huggingface/safetensors")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        var features = new LinkedHashSet<>(Set.of("safetensors", "libtorch-backend", "streaming-native"));
        if (modelConverterService != null && modelConverterService.isAvailable()) {
            features.add("gguf-conversion");
        }
        features.addAll(AdapterCapabilityProfile.builder()
                .adapterSupported(true)
                .adapterTypes(Set.of("lora"))
                .runtimeApply(false)
                .precompiledModelPath(false)
                .rolloutGuard(false)
                .metricsSchema(true)
                .build()
                .toFeatureFlags());
        return ProviderCapabilities.builder()
                .streaming(true)
                .embeddings(true)
                .multimodal(false)
                .functionCalling(false)
                .toolCalling(false)
                .structuredOutputs(false)
                .supportedFormats(Set.of(ModelFormat.SAFETENSORS))
                .supportedDevices(mergedSupportedDevices())
                .features(Set.copyOf(features))
                .build();
    }

    @Override
    public void initialize(ProviderConfig providerConfig) throws ProviderException.ProviderInitializationException {
        libTorchProvider.initialize(providerConfig);
        if (directBackend != null) {
            directBackend.initialize(providerConfig);
        }
        Backend backend = resolveBackend();
        if (backend != Backend.LIBTORCH && backend != Backend.DIRECT && ggufBackend != null) {
            try {
                ggufBackend.initialize(providerConfig);
            } catch (ProviderException.ProviderInitializationException e) {
                if (backend == Backend.GGUF) {
                    throw e;
                }
                log.warnf(e, "SafetensorProvider: GGUF backend init failed; auto mode will fall back");
            }
        }
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (!config.enabled() || modelId == null || modelId.isBlank()) {
            return false;
        }
        Path modelPath = resolveModelPath(modelId);
        if (modelPath == null || !Files.exists(modelPath) || Files.isDirectory(modelPath)) {
            return false;
        }
        String name = modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        return configuredExtensions().stream().anyMatch(name::endsWith);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {

        Path resolved = resolveModelPath(request.getModel());
        if (resolved == null || !Files.exists(resolved)) {
            String modelName = friendlyModelName(request.getModel());
            return Uni.createFrom().failure(new ProviderException(PROVIDER_ID,
                    "Model not found: " + modelName, null, false));
        }
        ProviderException validation = validateModelCompatibility(resolved);
        if (validation != null) {
            return Uni.createFrom().failure(validation);
        }
        Backend backend = resolveBackend();
        if (backend == Backend.DIRECT) {
            return directBackend.infer(requestWithModelPath(request, resolved));
        }
        if (backend == Backend.LIBTORCH) {
            return Uni.createFrom().failure(new ProviderException(PROVIDER_ID,
                    "Safetensors cannot be executed directly in LibTorch. Set safetensor.provider.backend=gguf, direct or auto.",
                    null, false));
        }
        if (backend == Backend.GGUF || backend == Backend.AUTO) {
            // Try DIRECT first if AUTO
            if (backend == Backend.AUTO && directBackend != null) {
                return directBackend.infer(requestWithModelPath(request, resolved));
            }
            return inferViaGguf(request, resolved);
        }
        ProviderRequest delegated = requestWithModelPath(request, resolved);
        return libTorchProvider.infer(delegated);
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {

        Path resolved = resolveModelPath(request.getModel());
        if (resolved == null || !Files.exists(resolved)) {
            String modelName = friendlyModelName(request.getModel());
            return Multi.createFrom().failure(new ProviderException(PROVIDER_ID,
                    "Model not found: " + modelName, null, false));
        }
        ProviderException validation = validateModelCompatibility(resolved);
        if (validation != null) {
            return Multi.createFrom().failure(validation);
        }
        Backend backend = resolveBackend();
        if (backend == Backend.DIRECT) {
            return directBackend.inferStream(requestWithModelPath(request, resolved));
        }
        if (backend == Backend.LIBTORCH) {
            return Multi.createFrom().failure(new ProviderException(PROVIDER_ID,
                    "Safetensors cannot be executed directly in LibTorch. Set safetensor.provider.backend=gguf, direct or auto.",
                    null, false));
        }
        if (backend == Backend.GGUF || backend == Backend.AUTO) {
            // Try DIRECT first if AUTO
            if (backend == Backend.AUTO && directBackend != null) {
                return directBackend.inferStream(requestWithModelPath(request, resolved));
            }
            return inferStreamViaGguf(request, resolved);
        }
        ProviderRequest delegated = requestWithModelPath(request, resolved);
        return libTorchProvider.inferStream(delegated);
    }

    @Override
    public Uni<ProviderHealth> health() {
        if (!config.enabled()) {
            return Uni.createFrom().item(ProviderHealth.builder()
                    .status(ProviderHealth.Status.UNHEALTHY)
                    .message("SafeTensor provider disabled by config")
                    .timestamp(Instant.now())
                    .build());
        }
        Backend backend = resolveBackend();
        if (backend == Backend.GGUF || backend == Backend.AUTO) {
            if (ggufBackend == null) {
                return Uni.createFrom().item(ProviderHealth.builder()
                        .status(ProviderHealth.Status.DEGRADED)
                        .message("GGUF backend not available")
                        .timestamp(Instant.now())
                        .detail("delegate", "gguf")
                        .build());
            }
            return ggufBackend.health().map(h -> ProviderHealth.builder()
                    .status(h.status())
                    .message(h.message())
                    .timestamp(Instant.now())
                    .detail("delegate", "gguf")
                    .build());
        }
        return libTorchProvider.health().map(h -> ProviderHealth.builder()
                .status(h.status())
                .message(h.message())
                .timestamp(Instant.now())
                .detail("delegate", "libtorch")
                .build());
    }

    @Override
    public void shutdown() {
        // Delegate lifecycle is managed by the underlying provider beans.
    }

    private ProviderRequest requestWithModelPath(ProviderRequest request, Path modelPath) {
        return new ProviderRequest(
                request.getRequestId(),
                modelPath.toString(),
                request.getMessages(),
                request.getParameters(),
                request.getTools(),
                request.getToolChoice(),
                request.isStreaming(),
                request.getTimeout(),
                request.getUserId().orElse(null),
                request.getSessionId().orElse(null),
                request.getTraceId().orElse(null),
                request.getApiKey().orElse(null),
                request.getMetadata());
    }

    private Path resolveModelPath(String modelId) {
        Path asGiven = Path.of(modelId);
        if (asGiven.isAbsolute()) {
            if (Files.exists(asGiven)) {
                return resolveToSafetensorFile(asGiven);
            }
            for (String ext : configuredExtensions()) {
                Path withExt = Path.of(modelId + ext);
                if (Files.exists(withExt)) {
                    return resolveToSafetensorFile(withExt);
                }
            }
            return resolveToSafetensorFile(asGiven);
        }

        Path fromBase = Path.of(config.basePath(), modelId);
        if (Files.exists(fromBase)) {
            return resolveToSafetensorFile(fromBase);
        }
        for (String ext : configuredExtensions()) {
            Path withExt = Path.of(config.basePath(), modelId + ext);
            if (Files.exists(withExt)) {
                return resolveToSafetensorFile(withExt);
            }
        }

        return resolveToSafetensorFile(Path.of(config.basePath(), modelId));
    }

    private java.util.List<String> configuredExtensions() {
        String raw = config.extensions();
        if (raw == null || raw.isBlank()) {
            return java.util.List.of(".safetensors", ".safetensor");
        }
        java.util.List<String> exts = Arrays.stream(raw.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> s.startsWith(".") ? s.toLowerCase(Locale.ROOT) : "." + s.toLowerCase(Locale.ROOT))
                .toList();
        return exts.isEmpty() ? java.util.List.of(".safetensors", ".safetensor") : exts;
    }

    private Set<tech.kayys.gollek.spi.model.DeviceType> mergedSupportedDevices() {
        var devices = new LinkedHashSet<tech.kayys.gollek.spi.model.DeviceType>();
        if (ggufBackend != null && ggufBackend.capabilities() != null) {
            devices.addAll(ggufBackend.capabilities().getSupportedDevices());
        }
        if (libTorchProvider != null && libTorchProvider.capabilities() != null) {
            devices.addAll(libTorchProvider.capabilities().getSupportedDevices());
        }
        if (devices.isEmpty()) {
            devices.add(tech.kayys.gollek.spi.model.DeviceType.CPU);
        }
        return Set.copyOf(devices);
    }

    private ProviderException validateModelCompatibility(Path resolvedModelPath) {
        String lowerPath = resolvedModelPath.toString().toLowerCase(Locale.ROOT);
        String modelName = friendlyModelName(resolvedModelPath.toString());
        if (lowerPath.contains("vlm")
                || lowerPath.contains("vision")
                || lowerPath.contains("llava")
                || lowerPath.contains("idefics")) {
            return new ProviderException(
                    PROVIDER_ID,
                    "Model appears to be multimodal/VLM and is not supported by local safetensor text runtime: "
                            + modelName,
                    null,
                    false);
        }

        Path checkpointDir = Files.isDirectory(resolvedModelPath) ? resolvedModelPath : resolvedModelPath.getParent();
        if (checkpointDir != null && !Files.exists(checkpointDir.resolve("config.json"))) {
            return new ProviderException(
                    PROVIDER_ID,
                    "Incomplete safetensor checkpoint (missing config.json). Re-pull model to download metadata sidecars.",
                    null,
                    true);
        }
        return null;
    }

    private Uni<InferenceResponse> inferViaGguf(ProviderRequest request, Path resolvedSafetensor) {
        try {
            Path ggufPath = ensureGgufModel(request.getModel(), resolvedSafetensor);
            ProviderRequest delegated = requestWithModelPath(request, ggufPath);
            return ggufBackend.infer(delegated);
        } catch (ProviderException e) {
            return Uni.createFrom().failure(e);
        } catch (Exception e) {
            return Uni.createFrom().failure(new ProviderException(PROVIDER_ID,
                    "Safetensor conversion failed: " + e.getMessage(), e, true));
        }
    }

    private Multi<StreamingInferenceChunk> inferStreamViaGguf(ProviderRequest request, Path resolvedSafetensor) {
        try {
            Path ggufPath = ensureGgufModel(request.getModel(), resolvedSafetensor);
            ProviderRequest delegated = requestWithModelPath(request, ggufPath);
            return ggufBackend.inferStream(delegated);
        } catch (ProviderException e) {
            return Multi.createFrom().failure(e);
        } catch (Exception e) {
            return Multi.createFrom().failure(new ProviderException(PROVIDER_ID,
                    "Safetensor conversion failed: " + e.getMessage(), e, true));
        }
    }

    private Path ensureGgufModel(String modelId, Path resolvedSafetensor) throws Exception {
        if (ggufBackend == null || (ggufConverter == null && modelConverterService == null)) {
            throw new ProviderException(PROVIDER_ID,
                    "GGUF backend/converter unavailable. Ensure the GGUF runner is on the classpath.", null, true);
        }

        Path modelDir = Files.isDirectory(resolvedSafetensor) ? resolvedSafetensor : resolvedSafetensor.getParent();
        if (modelDir == null) {
            throw new ProviderException(PROVIDER_ID, "Invalid safetensor path: " + resolvedSafetensor, null, false);
        }

        Path outputDir = Path.of(config.ggufOutputDir());
        Files.createDirectories(outputDir);
        String normalized = normalizeModelId(modelId);
        Path outputFile = outputDir.resolve(normalized + ".gguf");

        if (Files.exists(outputFile) && Files.size(outputFile) > 0) {
            return outputFile;
        }

        // Use native GGUFConverter if available
        if (ggufConverter != null) {
            log.infof("SafetensorProvider: converting %s → %s (Native Java)", modelDir, outputFile);
            GGUFConversionParams params = GGUFConversionParams.builder()
                    .inputPath(modelDir)
                    .outputPath(outputFile)
                    .quantization(QuantizationType.Q4_K_M) // Default to standard 4-bit
                    .build();
            ggufConverter.convert(params, null);
            return outputFile;
        }

        // Fallback to Python converter if available
        if (modelConverterService != null && modelConverterService.isAvailable()) {
            log.infof("SafetensorProvider: converting %s → %s (Python Fallback)", modelDir, outputFile);
            modelConverterService.convert(modelDir, outputFile);
            return outputFile;
        }

        throw new ProviderException(PROVIDER_ID,
                "No GGUF converter available. Install llama.cpp or enable native converter.", null, true);
    }

    private Backend resolveBackend() {
        String raw = config.backend();
        if (raw == null || raw.isBlank()) {
            return Backend.AUTO;
        }
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "gguf" -> Backend.GGUF;
            case "libtorch" -> Backend.LIBTORCH;
            case "direct" -> Backend.DIRECT;
            case "auto" -> Backend.AUTO;
            default -> Backend.AUTO;
        };
    }

    private String normalizeModelId(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return "unknown";
        }
        String normalized = modelId.replace('/', '_').replace('\\', '_');
        return normalized.isBlank() ? "model" : normalized;
    }

    private Path resolveToSafetensorFile(Path candidate) {
        if (candidate == null) {
            return null;
        }
        if (!Files.exists(candidate)) {
            return candidate;
        }
        if (!Files.isDirectory(candidate)) {
            return candidate;
        }

        Path preferred = candidate.resolve("model.safetensors");
        if (Files.isRegularFile(preferred)) {
            return preferred;
        }
        preferred = candidate.resolve("model.safetensor");
        if (Files.isRegularFile(preferred)) {
            return preferred;
        }

        try (var files = Files.walk(candidate, 3)) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(this::isSafetensorFilePath)
                    .findFirst()
                    .orElse(candidate);
        } catch (Exception ignored) {
            return candidate;
        }
    }

    private boolean isSafetensorFilePath(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".safetensors") || name.endsWith(".safetensor");
    }

    private String friendlyModelName(String modelRef) {
        if (modelRef == null || modelRef.isBlank()) {
            return "unknown";
        }
        try {
            Path path = Path.of(modelRef);
            Path fileName = path.getFileName();
            String file = fileName != null ? fileName.toString() : modelRef;
            String lower = file.toLowerCase(Locale.ROOT);
            if (lower.endsWith(".safetensors") || lower.endsWith(".safetensor")) {
                Path parent = path.getParent();
                if (parent != null && parent.getFileName() != null) {
                    return parent.getFileName().toString();
                }
            }
            return file;
        } catch (Exception ignored) {
            return modelRef;
        }
    }

    private enum Backend {
        AUTO,
        GGUF,
        LIBTORCH,
        DIRECT
    }
}
