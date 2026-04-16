package tech.kayys.gollek.inference.gguf;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.model.ModelManifest;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.micrometer.core.instrument.MeterRegistry;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Refactored GGUF runner - now an orchestrator that wires all components
 * together.
 * 
 * Components:
 * - LlamaCppModelInitializer: Model/context loading
 * - LlamaCppTokenSampler: Token sampling (temp, top-k, top-p, penalties)
 * - LlamaCppKVCacheManager: KV cache, sessions, tokenization
 * - LlamaCppMetricsRecorder: Metrics collection
 * - LlamaCppAdapterManager: LoRA adapter lifecycle
 * - LlamaCppCoalescer: Request batching (optional)
 */
public class LlamaCppRunner {

    private static final Logger log = Logger.getLogger(LlamaCppRunner.class);
    private volatile boolean initialized = false;
    private ModelManifest manifest;

    // Components
    private final LlamaCppBinding binding;
    private final GGUFProviderConfig providerConfig;
    private final GGUFChatTemplateService templateService;
    private LlamaCppModelInitializer modelInitializer;
    private LlamaCppMetricsRecorder metricsRecorder;
    private LlamaCppAdapterManager adapterManager;
    private LlamaCppKVCacheManager kvCacheManager;
    private LlamaCppTokenSampler tokenSampler;
    private LlamaCppCoalescer coalescer;

    // State from initialization
    private java.lang.foreign.MemorySegment model;
    private java.lang.foreign.MemorySegment context;
    private int contextSize, vocabSize, eosToken, bosToken, runtimeBatchSize;
    private String chatTemplate;

    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final Semaphore concurrencyLimit;

    public LlamaCppRunner(LlamaCppBinding binding, GGUFProviderConfig config, GGUFChatTemplateService templateService) {
        this.binding = binding;
        this.providerConfig = config;
        this.templateService = templateService;
        this.concurrencyLimit = new Semaphore(config.maxConcurrentRequests(), true);
    }

    public void initialize(ModelManifest manifest, Map<String, Object> runnerConfig) {
        if (initialized) {
            log.warnf("Runner already initialized: %s", manifest.modelId());
            return;
        }
        try {
            this.manifest = manifest;

            // 1. Initialize components
            this.modelInitializer = new LlamaCppModelInitializer(binding, providerConfig);
            this.metricsRecorder = new LlamaCppMetricsRecorder();
            this.adapterManager = new LlamaCppAdapterManager(binding);

            // 2. Load model and context using ModelInitializer component
            LlamaCppModelInitializer.InitializationResult result = modelInitializer.initialize(manifest, runnerConfig);
            this.model = result.model;
            this.context = result.context;
            this.contextSize = result.contextSize;
            this.vocabSize = result.vocabSize;
            this.eosToken = result.eosToken;
            this.bosToken = result.bosToken;
            this.chatTemplate = result.chatTemplate;
            this.runtimeBatchSize = result.runtimeBatchSize;

            // 3. Initialize remaining components
            this.kvCacheManager = new LlamaCppKVCacheManager(binding, providerConfig, manifest);
            this.tokenSampler = new LlamaCppTokenSampler(binding, vocabSize);

            // 4. Configure adapter using AdapterManager component
            adapterManager.configureAdapter(model, context, runnerConfig);

            // 5. Start coalescer if enabled using Coalescer component
            if (providerConfig.coalesceEnabled()) {
                this.coalescer = new LlamaCppCoalescer(
                        binding,
                        providerConfig,
                        metricsRecorder,
                        new ComponentInferenceExecutor());
                coalescer.start();
            }

            // 6. Register metrics using MetricsRecorder component
            metricsRecorder.registerMetrics(
                    null,
                    manifest.requestId(),
                    manifest.modelId(),
                    providerConfig.coalesceMaxQueue());

            initialized = true;
            log.infof("GGUF runner initialized: %s", manifest.modelId());

        } catch (Exception e) {
            cleanup();
            throw new RuntimeException("Failed to initialize GGUF runner: " + e.getMessage(), e);
        }
    }

    public InferenceResponse infer(InferenceRequest request) {
        checkInitialized();
        if (coalescer != null) {
            return coalescer.submit(request, null, () -> {
                executeWithComponents(request, null);
                return null;
            });
        }
        return executeWithComponents(request, null);
    }

    public Multi<StreamingInferenceChunk> inferStream(InferenceRequest request) {
        checkInitialized();
        return Multi.createFrom().emitter(emitter -> executorService.execute(() -> {
            try {
                int[] counter = { 0 };
                Consumer<String> onToken = piece -> {
                    if (!emitter.isCancelled()) {
                        emitter.emit(StreamingInferenceChunk.of(request.getRequestId(), counter[0]++, piece));
                    }
                };
                if (coalescer != null) {
                    coalescer.submit(request, onToken, () -> {
                        executeWithComponents(request, onToken);
                        return null;
                    });
                } else {
                    executeWithComponents(request, onToken);
                }
                emitter.complete();
            } catch (Exception e) {
                log.error("Streaming failed", e);
                emitter.fail(e);
            }
        }));
    }

    public Uni<EmbeddingResponse> embed(EmbeddingRequest request) {
        checkInitialized();
        return Uni.createFrom().item(() -> executeEmbedding(request));
    }

    public void registerMetrics(MeterRegistry registry, String tenantId, String modelId) {
        if (metricsRecorder != null) {
            metricsRecorder.registerMetrics(registry, tenantId, modelId, providerConfig.coalesceMaxQueue());
        }
    }

    public void close() {
        if (!initialized)
            return;
        cleanup();
        executorService.shutdownNow();
        if (coalescer != null)
            coalescer.shutdown();
        initialized = false;
    }

    public List<InferenceRequest> createDefaultWarmupRequests() {
        return List.of(InferenceRequest.builder()
                .model(manifest != null ? manifest.modelId() : "unknown")
                .message(tech.kayys.gollek.spi.Message.user("warmup"))
                .parameter("prompt", "Hello")
                .build());
    }

    public void warmup(List<InferenceRequest> requests) {
        if (requests == null)
            return;
        requests.forEach(r -> {
            try {
                infer(r);
            } catch (Exception e) {
                log.debug("Warmup failed: " + e.getMessage());
            }
        });
    }

    /**
     * Helper for verifying multi-sequence batching in tests.
     */
    public List<java.util.concurrent.CompletableFuture<InferenceResponse>> runMultiSequenceForTests(
            List<InferenceRequest> requests) {
        checkInitialized();
        List<java.util.concurrent.CompletableFuture<InferenceResponse>> futures = new java.util.ArrayList<>();
        for (InferenceRequest req : requests) {
            java.util.concurrent.CompletableFuture<InferenceResponse> f = new java.util.concurrent.CompletableFuture<>();
            executorService.execute(() -> {
                try {
                    f.complete(infer(req));
                } catch (Exception e) {
                    f.completeExceptionally(e);
                }
            });
            futures.add(f);
        }
        return futures;
    }

    private void checkInitialized() {
        if (!initialized)
            throw new IllegalStateException("Not initialized");
    }

    private boolean shouldBypassCoalesce(InferenceRequest request) {
        return coalescer != null && coalescer.shouldBypassCoalesce(request);
    }

    private InferenceResponse executeWithComponents(InferenceRequest request, Consumer<String> onTokenPiece) {
        boolean permit = false;
        try {
            permit = concurrencyLimit.tryAcquire(providerConfig.defaultTimeout().toMillis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted", e);
        }
        if (!permit)
            throw new RuntimeException("Runner busy");
        try {
            return executeInference(request, onTokenPiece);
        } finally {
            if (permit)
                concurrencyLimit.release();
        }
    }

    private InferenceResponse executeInference(InferenceRequest request, Consumer<String> onTokenPiece) {
        // Delegate to inference logic that uses all components
        return new InferenceLogicExecutor(
                binding, providerConfig, templateService,
                model, context, contextSize, vocabSize, eosToken, bosToken, runtimeBatchSize, chatTemplate,
                kvCacheManager, tokenSampler, metricsRecorder, manifest).execute(request, onTokenPiece);
    }

    private EmbeddingResponse executeEmbedding(EmbeddingRequest request) {
        try {
            boolean permit = concurrencyLimit.tryAcquire(providerConfig.defaultTimeout().toMillis(),
                    TimeUnit.MILLISECONDS);
            if (!permit)
                throw new RuntimeException("Runner busy");
            try {
                // Use kvCacheManager for tokenization
                String input = request.inputs().get(0);
                int[] tokens = kvCacheManager.tokenizeWithCache(model, input, true);
                if (tokens.length == 0) {
                    return new EmbeddingResponse(request.requestId(), manifest.modelId(), List.of(new float[0]), 0,
                            Map.of());
                }
                int nEmbd;
                try {
                    nEmbd = binding.nEmbd(model);
                } catch (Throwable t) {
                    throw new RuntimeException("Failed to get embedding dimension", t);
                }
                return new EmbeddingResponse(request.requestId(), manifest.modelId(), List.of(new float[nEmbd]), nEmbd,
                        Map.of());
            } finally {
                concurrencyLimit.release();
            }
        } catch (Exception e) {
            throw new RuntimeException("Embedding failed", e);
        }
    }

    private void cleanup() {
        if (adapterManager != null) {
            adapterManager.removeAdapter(context);
            adapterManager.cleanup();
        }
        if (context != null)
            binding.freeContext(context);
        if (model != null)
            binding.freeModel(model);
    }

    /**
     * Adapter to make component-based inference work with Coalescer
     */
    private class ComponentInferenceExecutor implements LlamaCppCoalescer.InferenceExecutor {
        @Override
        public InferenceResponse execute(InferenceRequest request, Consumer<String> onTokenPiece) {
            return executeWithComponents(request, onTokenPiece);
        }

        @Override
        public void executeMultiSequence(List<LlamaCppCoalescer.InferenceTask> tasks) {
            for (LlamaCppCoalescer.InferenceTask task : tasks) {
                try {
                    executeWithComponents(task.request, task.onTokenPiece);
                } catch (Exception e) {
                    log.warnf("Multi-sequence task failed: %s", e.getMessage());
                }
            }
        }
    }
}
