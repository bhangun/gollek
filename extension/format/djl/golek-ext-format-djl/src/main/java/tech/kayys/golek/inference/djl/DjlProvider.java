package tech.kayys.golek.inference.djl;

import ai.djl.Application;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.translator.SimpleText2TextTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.exception.ProviderException;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.model.DeviceType;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderConfig;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderMetadata;
import tech.kayys.golek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicBoolean;

@ApplicationScoped
public class DjlProvider implements LLMProvider {

    private static final Logger LOG = Logger.getLogger(DjlProvider.class);
    private static final String PROVIDER_ID = "djl";

    @Inject
    DjlProviderConfig config;

    private final AtomicReference<ProviderHealth.Status> status = new AtomicReference<>(ProviderHealth.Status.UNKNOWN);
    private volatile String startupFailure;
    private final ConcurrentHashMap<String, ZooModel<String, String>> models = new ConcurrentHashMap<>();
    private final AtomicBoolean engineInitialized = new AtomicBoolean(false);

    void onStart(@Observes StartupEvent ignored) {
        if (!config.enabled()) {
            status.set(ProviderHealth.Status.UNHEALTHY);
            startupFailure = "DJL provider disabled by config";
            return;
        }
        // Lazy init to avoid loading heavyweight native runtimes unless DJL is
        // selected.
        status.set(ProviderHealth.Status.DEGRADED);
        startupFailure = null;
        LOG.infof("DJL provider registered in lazy mode (engine=%s)", config.engine());
    }

    void onShutdown(@Observes ShutdownEvent ignored) {
        shutdown();
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "DJL Runtime";
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
                .description("Deep Java Library runtime (PyTorch engine)")
                .vendor("Golek / DJL")
                .homepage("https://docs.djl.ai/")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(false)
                .embeddings(false)
                .multimodal(false)
                .functionCalling(false)
                .toolCalling(false)
                .structuredOutputs(false)
                .supportedFormats(Set.of(ModelFormat.TORCHSCRIPT, ModelFormat.SAFETENSORS, ModelFormat.PYTORCH))
                .supportedDevices(Set.of(DeviceType.CPU))
                .features(Set.of("djl", "pytorch-engine"))
                .build();
    }

    @Override
    public void initialize(ProviderConfig providerConfig) throws ProviderException.ProviderInitializationException {
        if (!config.enabled()) {
            throw new ProviderException.ProviderInitializationException(
                    PROVIDER_ID,
                    startupFailure != null ? startupFailure : "DJL provider not enabled");
        }
        // Keep lazy semantics: defer heavyweight native engine initialization until
        // first infer().
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (!config.enabled()) {
            return false;
        }
        if (status.get() == ProviderHealth.Status.UNHEALTHY) {
            return false;
        }
        Path modelPath = resolveModelPath(modelId);
        if (modelPath == null || !Files.exists(modelPath)) {
            return false;
        }
        if (Files.isDirectory(modelPath)) {
            if (!Files.exists(modelPath.resolve("config.json"))) {
                return false;
            }
            try (var files = Files.walk(modelPath, 3)) {
                return files
                        .filter(Files::isRegularFile)
                        .map(path -> path.getFileName().toString().toLowerCase())
                        .anyMatch(name -> configuredExtensions().stream().anyMatch(name::endsWith));
            } catch (Exception ignored) {
                return false;
            }
        }
        String name = modelPath.getFileName().toString().toLowerCase();
        return configuredExtensions().stream().anyMatch(name::endsWith);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            ensureEngineInitialized();
            Path modelPath = resolveModelPath(request.getModel());
            if (modelPath == null || !Files.exists(modelPath)) {
                throw new IllegalArgumentException("Model not found: " + request.getModel());
            }

            long start = System.currentTimeMillis();
            String prompt = buildPrompt(request);
            String output = runInference(modelPath, prompt);

            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(request.getModel())
                    .content(output)
                    .durationMs(System.currentTimeMillis() - start)
                    .build();
        });
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            ProviderHealth.Builder builder = ProviderHealth.builder()
                    .status(status.get())
                    .timestamp(Instant.now());
            if (status.get() == ProviderHealth.Status.HEALTHY) {
                builder.message("DJL provider is healthy")
                        .detail("engine", config.engine())
                        .detail("loadedModels", models.size());
            } else if (status.get() == ProviderHealth.Status.DEGRADED) {
                builder.message("DJL provider is ready (lazy init)")
                        .detail("engine", config.engine())
                        .detail("initialized", engineInitialized.get());
            } else {
                builder.message("DJL provider is unhealthy");
                if (startupFailure != null && !startupFailure.isBlank()) {
                    builder.detail("reason", startupFailure);
                }
            }
            return builder.build();
        });
    }

    @Override
    public void shutdown() {
        for (ZooModel<String, String> model : models.values()) {
            try {
                model.close();
            } catch (Exception ignored) {
                // no-op
            }
        }
        models.clear();
        engineInitialized.set(false);
        status.set(ProviderHealth.Status.UNKNOWN);
    }

    private synchronized void ensureEngineInitialized() {
        if (engineInitialized.get()) {
            return;
        }
        try {
            Engine.getEngine(config.engine());
            engineInitialized.set(true);
            status.set(ProviderHealth.Status.HEALTHY);
            startupFailure = null;
            LOG.infof("DJL engine initialized on demand (engine=%s)", config.engine());
        } catch (Exception e) {
            status.set(ProviderHealth.Status.UNHEALTHY);
            startupFailure = e.getMessage();
            throw new RuntimeException("Failed to initialize DJL engine: " + startupFailure, e);
        }
    }

    private String runInference(Path modelPath, String prompt) {
        try {
            ZooModel<String, String> model = models.computeIfAbsent(modelPath.toString(), this::loadModel);
            try (Predictor<String, String> predictor = model.newPredictor()) {
                return predictor.predict(prompt);
            }
        } catch (Exception e) {
            throw new RuntimeException("DJL inference failed: " + e.getMessage(), e);
        }
    }

    private ZooModel<String, String> loadModel(String modelLocation) {
        Path modelPath = Path.of(modelLocation);
        try {
            // Local files are mapped to parent directory for DJL repository scanning.
            Path criteriaPath = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (criteriaPath == null) {
                criteriaPath = modelPath;
            }

            Exception last = null;
            for (Application app : List.of(Application.NLP.TEXT_GENERATION, Application.NLP.MACHINE_TRANSLATION)) {
                try {
                    Criteria<String, String> criteria = Criteria.builder()
                            .setTypes(String.class, String.class)
                            .optApplication(app)
                            .optEngine(config.engine())
                            .optModelPath(criteriaPath)
                            .optTranslator(new SimpleText2TextTranslator())
                            .build();
                    return criteria.loadModel();
                } catch (Exception e) {
                    last = e;
                }
            }
            throw new RuntimeException("Unable to load model with DJL", last);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load DJL model: " + modelLocation, e);
        }
    }

    private Path resolveModelPath(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return null;
        }

        Path direct = Path.of(modelId);
        if (direct.isAbsolute() && Files.exists(direct)) {
            return direct;
        }

        Path base = Path.of(config.model().basePath(), modelId);
        if (Files.exists(base)) {
            return base;
        }

        Path normalized = Path.of(config.model().basePath(), modelId.replace("/", "_"));
        if (Files.exists(normalized)) {
            return normalized;
        }

        // Backward compatibility for previous local layout.
        String userHome = System.getProperty("user.home");
        for (String ext : configuredExtensions()) {
            Path legacy = Path.of(userHome, ".golek", "models", "torchscript", modelId + ext);
            if (Files.exists(legacy)) {
                return legacy;
            }
        }
        Path legacyNormalized = Path.of(userHome, ".golek", "models", "torchscript", modelId.replace("/", "_"));
        if (Files.exists(legacyNormalized)) {
            return legacyNormalized;
        }

        return null;
    }

    private List<String> configuredExtensions() {
        String raw = config.model().extensions();
        if (raw == null || raw.isBlank()) {
            return List.of(".pt", ".pts", ".jit", ".ts");
        }
        return Arrays.stream(raw.split(","))
                .map(String::trim)
                .filter(s -> !s.isBlank())
                .map(s -> s.startsWith(".") ? s.toLowerCase() : "." + s.toLowerCase())
                .toList();
    }

    private String buildPrompt(ProviderRequest request) {
        if (request.getMessages() == null || request.getMessages().isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (Message msg : request.getMessages()) {
            if (msg == null || msg.getContent() == null || msg.getContent().isBlank()) {
                continue;
            }
            sb.append(msg.getContent()).append('\n');
        }
        return sb.toString().trim();
    }
}
