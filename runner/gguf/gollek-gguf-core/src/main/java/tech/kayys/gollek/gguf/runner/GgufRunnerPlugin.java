package tech.kayys.gollek.plugin.runner.gguf;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.*;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Unified GGUF runner plugin that defaults to Java-native execution but can
 * fall back to or explicitly use llama.cpp bindings.
 */
@ApplicationScoped
@Unremovable
public class GgufRunnerPlugin implements RunnerPlugin {
    private static final Logger LOG = Logger.getLogger(GgufRunnerPlugin.class);
    public static final String ID = "gguf-runner";

    private final Map<String, GgufBackend> backends = new ConcurrentHashMap<>();
    private RunnerConfig runnerConfig;
    private boolean initialized = false;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "GGUF Unified Runner";
    }

    @Override
    public String version() {
        return "2.0.0";
    }

    @Override
    public String description() {
        return "Support for GGUF models with Java-native and Llama.cpp backends";
    }

    @Override
    public String format() {
        return "gguf";
    }

    @Override
    public Set<String> supportedFormats() {
        return Set.of(".gguf");
    }

    @Override
    public Set<String> supportedArchitectures() {
        // llama.cpp supports many, but we'll advertise the core ones
        return Set.of("llama", "mistral", "mixtral", "phi", "gemma", "gemma4", "qwen");
    }

    @Override
    public RunnerValidationResult validate() {
        return RunnerValidationResult.valid();
    }

    @Override
    public void initialize(RunnerContext context) throws RunnerException {
        LOG.info("Initializing GGUF runner plugin...");
        this.runnerConfig = context.getConfig();
        this.initialized = true;
    }

    @Override
    public boolean isAvailable() {
        return true; // Java-native is always available
    }

    @Override
    public ModelHandle loadModel(ModelLoadRequest request, RunnerContext context) throws RunnerException {
        if (!initialized) {
            throw new RunnerInitializationException("gguf", "Plugin not initialized");
        }

        LOG.infof("Loading GGUF model: %s", request.getModelPath());

        try {
            // Choice of backend: preference (plugin parameter) -> request metadata ->
            // context config -> default (java)
            String requestedPlugin = (String) request.getMetadata().get("plugin");
            String backendType = requestedPlugin;

            if (backendType == null) {
                backendType = (String) request.getMetadata().getOrDefault("gguf.backend",
                        context.getMetadataValue("gguf.backend").orElse("java"));
            }

            GgufBackend backend;
            if ("llamacpp".equalsIgnoreCase(backendType)) {
                backend = createLlamaCppBackend(request, context);
            } else if ("java".equalsIgnoreCase(backendType) || backendType == null) {
                backend = createJavaNativeBackend(request, context);
            } else {
                LOG.warnf("Requested plugin '%s' is not supported by GGUF runner. Falling back to Java-native.",
                        backendType);
                backend = createJavaNativeBackend(request, context);
            }

            ModelHandle handle = new ModelHandle(request.getModelPath(), "gguf");
            backends.put(handle.getModelId(), backend);

            return handle;

        } catch (Exception e) {
            throw new ModelLoadException(request.getModelPath(), "Failed to load GGUF model: " + e.getMessage(), e);
        }
    }

    @Override
    public void unloadModel(ModelHandle handle, RunnerContext context) throws RunnerException {
        GgufBackend backend = backends.remove(handle.getModelId());
        if (backend != null) {
            backend.close();
        }
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request, RunnerContext context) throws RunnerException {
        if (!initialized) {
            throw new RunnerException("GGUF runner not initialized");
        }

        // This execute method in v2.0 usually expects a ModelHandle in the context
        // but here we might be using the stateless execute or per-session execute.
        // For unified runner, we'll try to find the backend.

        return RunnerResult.failed("Stateless execute not supported. Use session-based execution.");
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down GGUF runner plugin...");
        backends.values().forEach(GgufBackend::close);
        backends.clear();
        initialized = false;
    }

    private GgufBackend createJavaNativeBackend(ModelLoadRequest request, RunnerContext context) throws Exception {
        LOG.info("Creating Java-native GGUF backend");
        return new JavaNativeGgufBackend(java.nio.file.Path.of(request.getModelPath()));
    }

    private GgufBackend createLlamaCppBackend(ModelLoadRequest request, RunnerContext context) throws Exception {
        LOG.info("Creating Llama.cpp GGUF backend");
        try {
            // Use CDI to resolve the provider to avoid direct instantiation
            // and respect the Quarkus lifecycle.
            var provider = jakarta.enterprise.inject.spi.CDI.current()
                    .select(tech.kayys.gollek.inference.llamacpp.LlamaCppProvider.class)
                    .get();
            return new LlamaCppGgufBackend(provider);
        } catch (Exception e) {
            LOG.error("Failed to resolve LlamaCppProvider via CDI. Native backend may not be available.", e);
            throw new RunnerException("Llama.cpp provider not available: " + e.getMessage());
        }
    }
}
