package tech.kayys.gollek.ml.nlp;

import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Factory for creating and caching NLP and multimodal pipelines.
 *
 * <p>Maintains a registry of pipeline constructors keyed by task name. Built-in
 * tasks are registered at class-load time; custom tasks can be added via
 * {@link #register(String, Function)}. Additionally, tasks contributed by
 * plugins are auto-discovered via {@link ServiceLoader} for {@link PipelineProvider}.
 *
 * <h3>Built-in tasks</h3>
 * <ul>
 *   <li>{@code "text-generation"} → {@link TextGenerationPipeline}</li>
 *   <li>{@code "text-classification"} → {@link TextClassificationPipeline}</li>
 *   <li>{@code "embedding"} → {@link EmbeddingPipeline}</li>
 * </ul>
 *
 * <h3>Plugin-contributed tasks</h3>
 * <p>Plugins register additional tasks by implementing {@link PipelineProvider}
 * and listing the implementation in
 * {@code META-INF/services/tech.kayys.gollek.ml.nlp.PipelineProvider}.
 * These are discovered lazily on first use.</p>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * Pipeline<String, String> gen = PipelineFactory.create("text-generation", "Qwen/Qwen2.5-0.5B");
 * Pipeline<String, float[]> embed = PipelineFactory.create("embedding", "all-MiniLM-L6-v2");
 *
 * // Plugin-contributed task (if gollek-plugin-rag is on classpath):
 * Pipeline<String, String> rag = PipelineFactory.create("rag", "my-rag-model");
 * }</pre>
 */
public final class PipelineFactory {

    private static final Map<String, Function<PipelineConfig, Pipeline<?, ?>>> REGISTRY = new ConcurrentHashMap<>();
    private static final AtomicBoolean PLUGINS_LOADED = new AtomicBoolean(false);

    static {
        // Register built-in pipelines
        REGISTRY.put("text-generation", cfg -> new TextGenerationPipeline(cfg));
        REGISTRY.put("text-classification", cfg -> new TextClassificationPipeline(cfg.modelId()));
        REGISTRY.put("embedding", cfg -> new EmbeddingPipeline(cfg.modelId()));
    }

    private PipelineFactory() {}

    /**
     * Creates a pipeline for the given task and model using default configuration.
     *
     * @param <I>     input type inferred from the task
     * @param <O>     output type inferred from the task
     * @param task    pipeline task name (e.g. {@code "text-generation"}, {@code "embedding"})
     * @param modelId model identifier to use
     * @return a configured {@link Pipeline} instance
     * @throws PipelineException if {@code task} is not registered
     */
    @SuppressWarnings("unchecked")
    public static <I, O> Pipeline<I, O> create(String task, String modelId) {
        PipelineConfig config = PipelineConfig.builder(task, modelId).build();
        return create(config);
    }

    /**
     * Creates a pipeline from a fully specified {@link PipelineConfig}.
     *
     * @param <I>    input type inferred from the task
     * @param <O>    output type inferred from the task
     * @param config pipeline configuration including task, model, and sampling parameters
     * @return a configured {@link Pipeline} instance
     * @throws PipelineException if the task in {@code config} is not registered
     */
    @SuppressWarnings("unchecked")
    public static <I, O> Pipeline<I, O> create(PipelineConfig config) {
        ensurePluginsLoaded();

        Function<PipelineConfig, Pipeline<?, ?>> factory = REGISTRY.get(config.task());
        if (factory == null) {
            throw new PipelineException("Unknown pipeline task: " + config.task()
                + ". Available: " + REGISTRY.keySet());
        }
        return (Pipeline<I, O>) factory.apply(config);
    }

    /**
     * Registers a custom pipeline type, overwriting any existing registration for the same task.
     *
     * @param task    task name to register (e.g. {@code "summarization"})
     * @param factory function that creates a pipeline from a {@link PipelineConfig}
     */
    public static void register(String task, Function<PipelineConfig, Pipeline<?, ?>> factory) {
        REGISTRY.put(task, factory);
    }

    /**
     * Registers a {@link PipelineProvider} as a pipeline source.
     *
     * @param provider the pipeline provider to register
     */
    public static void register(PipelineProvider provider) {
        REGISTRY.put(provider.task(), provider::create);
    }

    /**
     * Returns an unmodifiable view of all currently registered task names.
     *
     * @return set of registered task identifiers
     */
    public static java.util.Set<String> availableTasks() {
        ensurePluginsLoaded();
        return java.util.Collections.unmodifiableSet(REGISTRY.keySet());
    }

    /**
     * Checks whether a given task is available (either built-in or plugin-contributed).
     *
     * @param task task name to check
     * @return {@code true} if the task is registered
     */
    public static boolean hasTask(String task) {
        ensurePluginsLoaded();
        return REGISTRY.containsKey(task);
    }

    /**
     * Forces re-discovery of plugin-contributed pipeline tasks.
     * <p>
     * Useful after dynamically adding plugins to the classpath at runtime.
     */
    public static void reloadPlugins() {
        PLUGINS_LOADED.set(false);
        ensurePluginsLoaded();
    }

    // ── Plugin discovery ─────────────────────────────────────────────────

    /**
     * Lazily discovers and registers {@link PipelineProvider} implementations
     * from the classpath via {@link ServiceLoader}.
     * <p>
     * Plugin-contributed tasks do <strong>not</strong> overwrite built-in tasks.
     */
    private static void ensurePluginsLoaded() {
        if (PLUGINS_LOADED.compareAndSet(false, true)) {
            try {
                ServiceLoader<PipelineProvider> loader = ServiceLoader.load(PipelineProvider.class);
                for (PipelineProvider provider : loader) {
                    // Plugin tasks don't overwrite built-in tasks
                    REGISTRY.putIfAbsent(provider.task(), provider::create);
                }
            } catch (Exception e) {
                // ServiceLoader failure is non-fatal — log and continue with built-in tasks
                System.err.println("[Gollek] Warning: Failed to load pipeline plugins: " + e.getMessage());
            }
        }
    }
}
