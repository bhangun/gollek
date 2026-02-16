package tech.kayys.golek.inference.transformers;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import io.quarkus.runtime.StartupEvent;
import io.quarkus.runtime.ShutdownEvent;
import org.jboss.logging.Logger;

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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Transformers runtime provider.
 * Executes local HuggingFace models via python transformers runtime.
 */
@ApplicationScoped
public class TransformersProvider implements LLMProvider {

    private static final Logger log = Logger.getLogger(TransformersProvider.class);
    private static final String PROVIDER_ID = "transformers";

    @Inject
    TransformersProviderConfig config;

    private final AtomicReference<ProviderHealth.Status> status = new AtomicReference<>(ProviderHealth.Status.UNKNOWN);
    private volatile String startupFailure;
    private volatile Path runnerScript;

    void onStart(@Observes StartupEvent ignored) {
        if (!config.enabled()) {
            status.set(ProviderHealth.Status.UNHEALTHY);
            startupFailure = "Transformers provider disabled by config";
            return;
        }

        try {
            runnerScript = extractRunnerScript();
            verifyPythonRuntime();
            status.set(ProviderHealth.Status.HEALTHY);
            startupFailure = null;
            log.info("Transformers provider started");
        } catch (Exception e) {
            status.set(ProviderHealth.Status.UNHEALTHY);
            startupFailure = e.getMessage();
            log.warnf(e, "Transformers provider unavailable: %s", startupFailure);
        }
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
        return "Transformers Runtime";
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
                .description("Python transformers runtime for local HuggingFace models")
                .vendor("Golek / Kayys")
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
                .supportedFormats(Set.of(ModelFormat.PYTORCH, ModelFormat.SAFETENSORS))
                .supportedDevices(Set.of(DeviceType.CPU))
                .features(Set.of("hf-transformers", "pytorch-checkpoint"))
                .build();
    }

    @Override
    public void initialize(ProviderConfig providerConfig) throws ProviderException.ProviderInitializationException {
        if (status.get() != ProviderHealth.Status.HEALTHY) {
            throw new ProviderException.ProviderInitializationException(
                    PROVIDER_ID,
                    startupFailure != null ? startupFailure : "Transformers provider not healthy");
        }
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (status.get() != ProviderHealth.Status.HEALTHY) {
            return false;
        }

        Path modelPath = resolveModelPath(modelId);
        if (modelPath == null) {
            return false;
        }

        if (Files.isRegularFile(modelPath)) {
            modelPath = modelPath.getParent();
        }
        if (modelPath == null || !Files.isDirectory(modelPath)) {
            return false;
        }

        return Files.exists(modelPath.resolve("config.json"))
                && (Files.exists(modelPath.resolve("pytorch_model.bin"))
                        || Files.exists(modelPath.resolve("model.safetensors"))
                        || hasShardWeights(modelPath));
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            Path modelPath = resolveModelPath(request.getModel());
            if (modelPath == null) {
                throw new RuntimeException("Model not found: " + request.getModel());
            }
            if (Files.isRegularFile(modelPath)) {
                modelPath = modelPath.getParent();
            }
            if (modelPath == null || !Files.isDirectory(modelPath)) {
                throw new RuntimeException("Transformers model directory not found: " + request.getModel());
            }

            String prompt = buildPrompt(request);
            int maxNewTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : 128;
            double temperature = request.getTemperature();
            double topP = request.getTopP();

            long start = System.currentTimeMillis();
            String output = runPythonInference(modelPath, prompt, maxNewTokens, temperature, topP,
                    request.getTimeout().toSeconds() > 0 ? request.getTimeout().toSeconds() : config.timeoutSeconds());
            long duration = System.currentTimeMillis() - start;

            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(request.getModel())
                    .content(output)
                    .durationMs(duration)
                    .build();
        });
    }

    @Override
    public Uni<Boolean> isAvailable() {
        return Uni.createFrom().item(() -> status.get() == ProviderHealth.Status.HEALTHY);
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            ProviderHealth.Builder builder = ProviderHealth.builder()
                    .status(status.get())
                    .timestamp(Instant.now());
            if (status.get() == ProviderHealth.Status.HEALTHY) {
                builder.message("Transformers provider is healthy");
            } else {
                builder.message("Transformers provider is unhealthy");
                if (startupFailure != null && !startupFailure.isBlank()) {
                    builder.detail("reason", startupFailure);
                }
            }
            return builder.build();
        });
    }

    @Override
    public void shutdown() {
        status.set(ProviderHealth.Status.UNKNOWN);
    }

    private void verifyPythonRuntime() throws IOException, InterruptedException {
        Process process = new ProcessBuilder(
                config.pythonCommand(),
                "-c",
                "import transformers, torch; print('ok')")
                .redirectErrorStream(true)
                .start();

        boolean finished = process.waitFor(15, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            throw new IOException("Python runtime check timeout");
        }
        if (process.exitValue() != 0) {
            String out = readAll(process.getInputStream());
            throw new IOException("Python transformers runtime unavailable: " + out.trim());
        }
    }

    private Path extractRunnerScript() throws IOException {
        try (InputStream in = getClass().getResourceAsStream("/scripts/transformers_infer.py")) {
            if (in == null) {
                throw new IOException("transformers_infer.py not found in resources");
            }
            Path dir = Files.createTempDirectory("golek-transformers-");
            Path script = dir.resolve("transformers_infer.py");
            Files.copy(in, script, StandardCopyOption.REPLACE_EXISTING);
            script.toFile().setExecutable(true);
            return script;
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

        Path base = Path.of(config.basePath(), modelId);
        if (Files.exists(base)) {
            return base;
        }

        Path normalized = Path.of(config.basePath(), modelId.replace("/", "_"));
        if (Files.exists(normalized)) {
            return normalized;
        }

        return null;
    }

    private String buildPrompt(ProviderRequest request) {
        if (request.getMessages() == null || request.getMessages().isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        request.getMessages().forEach(m -> {
            if (m == null || m.getContent() == null) {
                return;
            }
            if (!m.getContent().isBlank()) {
                sb.append(m.getContent()).append("\n");
            }
        });
        return sb.toString().trim();
    }

    private String runPythonInference(
            Path modelPath,
            String prompt,
            int maxNewTokens,
            double temperature,
            double topP,
            long timeoutSeconds) {

        try {
            Process process = new ProcessBuilder(
                    config.pythonCommand(),
                    runnerScript.toString(),
                    "--model-path", modelPath.toString(),
                    "--prompt", prompt,
                    "--max-new-tokens", Integer.toString(maxNewTokens),
                    "--temperature", Double.toString(temperature),
                    "--top-p", Double.toString(topP))
                    .start();

            boolean finished = process.waitFor(Math.max(10, timeoutSeconds), TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new RuntimeException("Transformers inference timeout");
            }

            String stdout = readAll(process.getInputStream()).trim();
            String stderr = readAll(process.getErrorStream()).trim();

            if (process.exitValue() != 0) {
                String detail = stderr.isEmpty() ? stdout : stderr;
                throw new RuntimeException("Transformers runtime failed: " + detail);
            }

            if (stdout.isBlank()) {
                return "";
            }
            return stdout;

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Failed to execute transformers runtime", e);
        } catch (IOException e) {
            throw new RuntimeException("Failed to execute transformers runtime", e);
        }
    }

    private boolean hasShardWeights(Path modelDir) {
        try (var files = Files.list(modelDir)) {
            return files.map(Path::getFileName)
                    .map(Path::toString)
                    .anyMatch(name -> name.startsWith("pytorch_model-") && name.endsWith(".bin")
                            || name.startsWith("model-") && name.endsWith(".safetensors"));
        } catch (IOException e) {
            return false;
        }
    }

    private String readAll(InputStream inputStream) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            return sb.toString();
        }
    }
}
