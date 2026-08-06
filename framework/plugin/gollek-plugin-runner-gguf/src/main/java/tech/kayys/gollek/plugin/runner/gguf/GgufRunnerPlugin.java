package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.*;

import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Unified GGUF runner plugin.
 *
 * <p>Auto/default mode uses the llama.cpp backend because it is the production
 * generation path today. The Java backend stays explicit until its decoder can
 * generate correctly and competitively.</p>
 */
public final class GgufRunnerPlugin implements RunnerPlugin {
    public static final String ID = "gguf-runner";

    private final Map<String, GgufBackend> backends = new ConcurrentHashMap<>();
    private volatile boolean initialized;

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
        return "2.1.0";
    }

    @Override
    public String description() {
        return "GGUF model support with safe auto-selection between Java-native inspection and llama.cpp generation";
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
        return Set.of("llama", "mistral", "mixtral", "phi", "gemma", "qwen");
    }

    @Override
    public void initialize(RunnerContext context) {
        initialized = true;
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public ModelHandle loadModel(ModelLoadRequest request, RunnerContext context) throws RunnerException {
        ensureInitialized();
        GgufBackendSelection selection = GgufBackendSelection.resolve(request, context);

        try {
            GgufBackend backend = createBackend(request, context, selection);
            Map<String, Object> metadata = new LinkedHashMap<>(backend.metadata());
            metadata.put("backendSelection", selection.normalizedValue());
            metadata.put("backendSelectionSource", selection.source());
            metadata.put("backendSelectionExplicit", selection.explicit());

            ModelHandle handle = ModelHandle.of(request.getModelPath(), "gguf", Map.copyOf(metadata));
            backends.put(handle.getModelId(), backend);
            return handle;
        } catch (RunnerException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelLoadException(request.getModelPath(), "Failed to load GGUF model: " + e.getMessage(), e);
        }
    }

    @Override
    public void unloadModel(ModelHandle handle, RunnerContext context) {
        Optional.ofNullable(backends.remove(handle.getModelId())).ifPresent(GgufBackend::close);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request, RunnerContext context) throws RunnerException {
        ensureInitialized();
        Optional<GgufBackend> backend = backendForRequest(request);
        if (backend.isEmpty()) {
            return RunnerResult.failed("No GGUF model loaded for request. Pass modelId or load exactly one model.");
        }
        return backend.get().execute(request);
    }

    @Override
    public void shutdown() {
        backends.values().forEach(GgufBackend::close);
        backends.clear();
        initialized = false;
    }

    private void ensureInitialized() throws RunnerException {
        if (!initialized) {
            throw new RunnerInitializationException("gguf", "Plugin not initialized");
        }
    }

    private GgufBackend createBackend(
            ModelLoadRequest request,
            RunnerContext context,
            GgufBackendSelection selection) throws Exception {
        if (selection.requestedJavaNative()) {
            return new JavaNativeGgufBackend(Path.of(request.getModelPath()));
        }

        Optional<Object> provider = resolveProvider(request, context);
        if (provider.isPresent()) {
            return new LlamaCppGgufBackend(provider.get());
        }

        if (selection.explicit()) {
            throw new RunnerException("llama.cpp GGUF backend requested but LlamaCppProvider is unavailable");
        }

        return new JavaNativeGgufBackend(Path.of(request.getModelPath()));
    }

    private Optional<GgufBackend> backendForRequest(RunnerRequest request) {
        Optional<String> requestedModel = request.getParameter("modelId", String.class)
                .or(() -> valueAsString(request.metadata().get("modelId")))
                .or(() -> valueAsString(request.metadata().get("model.id")));
        if (requestedModel.isPresent()) {
            return Optional.ofNullable(backends.get(requestedModel.get()));
        }
        if (backends.size() == 1) {
            return backends.values().stream().findFirst();
        }
        return Optional.empty();
    }

    private static Optional<Object> resolveProvider(ModelLoadRequest request, RunnerContext context) {
        Optional<Object> requestProvider = providerFrom(request.getMetadata().get("llamacpp.provider"));
        if (requestProvider.isPresent()) {
            return requestProvider;
        }

        Optional<Object> contextProvider = context.getMetadataValue("llamacpp.provider")
                .flatMap(GgufRunnerPlugin::providerFrom)
                .or(() -> context.getParameter("llamacpp.provider").flatMap(GgufRunnerPlugin::providerFrom));
        if (contextProvider.isPresent()) {
            return contextProvider;
        }

        return providerFromCdi();
    }

    private static Optional<Object> providerFrom(Object value) {
        if (value != null && hasInferenceMethod(value)) {
            return Optional.of(value);
        }
        return Optional.empty();
    }

    private static Optional<Object> providerFromCdi() {
        try {
            Class<?> providerClass = Class.forName("tech.kayys.gollek.inference.llamacpp.LlamaCppProvider");
            Class<?> cdiClass = Class.forName("jakarta.enterprise.inject.spi.CDI");
            Object cdi = cdiClass.getMethod("current").invoke(null);
            Method select = cdiClass.getMethod("select", Class.class, Annotation[].class);
            Object instance = select.invoke(cdi, providerClass, new Annotation[0]);
            Object provider = instance.getClass().getMethod("get").invoke(instance);
            return providerFrom(provider);
        } catch (ReflectiveOperationException | LinkageError | IllegalStateException ignored) {
            return Optional.empty();
        }
    }

    private static boolean hasInferenceMethod(Object value) {
        try {
            value.getClass().getMethod("infer", tech.kayys.gollek.spi.inference.InferenceRequest.class);
            return true;
        } catch (NoSuchMethodException ignored) {
            return false;
        }
    }

    private static Optional<String> valueAsString(Object value) {
        if (value == null) {
            return Optional.empty();
        }
        return Optional.of(value.toString());
    }
}
