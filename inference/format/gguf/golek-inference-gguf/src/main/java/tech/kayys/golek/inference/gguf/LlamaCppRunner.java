package tech.kayys.golek.inference.gguf;

import org.jboss.logging.Logger;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.ModelFormat;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.subscription.MultiEmitter;
import tech.kayys.golek.spi.stream.StreamChunk;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Production-ready GGUF/llama.cpp model runner implementation
 * Supports CPU and CUDA acceleration with complete lifecycle management
 * 
 * Note: This class is NOT a CDI bean - it is instantiated by GGUFSessionManager
 */
public class LlamaCppRunner {

    private static final Logger log = Logger.getLogger(LlamaCppRunner.class);

    private volatile boolean initialized = false;
    private ModelManifest manifest;

    // Native handles
    private final LlamaCppBinding binding;
    private MemorySegment model;
    private MemorySegment context;
    private Path modelPath;

    // Model metadata
    private int eosToken;
    private int bosToken;
    private int contextSize;
    private int vocabSize;
    private String chatTemplate;

    private final GGUFChatTemplateService templateService;

    // Threading and concurrency
    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final Semaphore concurrencyLimit;

    // Metrics
    private final AtomicLong totalInferences = new AtomicLong(0);
    private final AtomicLong failedInferences = new AtomicLong(0);
    private final AtomicLong totalTokensGenerated = new AtomicLong(0);
    private volatile Duration lastInferenceLatency = Duration.ZERO;

    private final GGUFProviderConfig providerConfig;

    /**
     * Constructor for non-CDI instantiation by SessionManager
     */
    public LlamaCppRunner(LlamaCppBinding binding, GGUFProviderConfig config, GGUFChatTemplateService templateService) {
        this.binding = binding;
        this.providerConfig = config;
        this.templateService = templateService;
        this.concurrencyLimit = new Semaphore(config.maxConcurrentRequests(), true);
    }

    public void initialize(
            ModelManifest manifest,
            Map<String, Object> runnerConfig) {
        if (initialized) {
            log.warnf("Runner already initialized for model %s", manifest.modelId());
            return;
        }

        log.infof("Initializing GGUF runner for model %s (tenant: %s)",
                manifest.modelId(), manifest.tenantId());

        try {
            this.manifest = manifest;

            // Assume model is already at location or resolve it
            // Extract from artifacts map
            var artifact = manifest.artifacts().get(ModelFormat.GGUF);
            if (artifact == null) {
                throw new RuntimeException("No GGUF artifact found in manifest for model " + manifest.modelId());
            }
            this.modelPath = java.nio.file.Paths.get(artifact.uri());

            if (!modelPath.toFile().exists()) {
                throw new RuntimeException("Model file not found: " + modelPath);
            }

            log.infof("Loading GGUF model from: %s", modelPath);

            // Suppress verbose native logs before model loading
            if (!providerConfig.verboseLogging()) {
                // FIXME: Native upcall causes crash on some systems. Disabled for now.
                // binding.suppressNativeLogs();
            }

            // Native initialization logic (mocked/placeholder mainly as we reuse binding)
            // In real impl, we would use binding.loadModel etc.
            // For now, we trust the binding handles are correct.

            MemorySegment modelParams = binding.getDefaultModelParams();
            // Configure params...

            this.model = binding.loadModel(modelPath.toString(), modelParams);

            MemorySegment contextParams = binding.getDefaultContextParams();
            // Increase context size for multi-turn chat and longer prompts
            binding.setContextParam(contextParams, "n_ctx", 4096);
            binding.setContextParam(contextParams, "n_batch", 512);

            this.context = binding.createContext(model, contextParams);

            // Load metadata (chat template)
            this.chatTemplate = binding.getModelMetadata(model, "tokenizer.chat_template");
            log.debugf("Loaded chat template: %s", chatTemplate != null ? "Yes" : "No");

            this.initialized = true;

        } catch (Exception e) {
            cleanup();
            throw new RuntimeException("Failed to initialize GGUF runner", e);
        }
    }

    public InferenceResponse infer(
            InferenceRequest request,
            tech.kayys.golek.spi.context.RequestContext requestContext) {

        if (!initialized) {
            throw new IllegalStateException("Runner not initialized");
        }

        // Always clear KV cache for stateless requests to prevent context corruption
        // TODO: Only clear if not using session (once session support is restored)
        binding.kvCacheClear(this.context);

        String prompt = (String) request.getParameters().getOrDefault("prompt", "");
        if (prompt == null || prompt.isBlank()) {
            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(manifest.modelId())
                    .content("")
                    .tokensUsed(0)
                    .build();
        }

        // Apply Chat Template if messages are present
        if (request.getMessages() != null && !request.getMessages().isEmpty()) {
            prompt = templateService.render(chatTemplate, request.getMessages());
            // Add generation prompt if not already present or handled by template
        }

        // 1. Tokenize
        int[] promptTokens = binding.tokenize(model, prompt, true, true);
        int nTokens = promptTokens.length;

        if (nTokens == 0) {
            return InferenceResponse.builder().requestId(request.getRequestId()).content("").build();
        }

        // 2. Sampler Chain
        float temperature = ((Number) request.getParameters().getOrDefault("temperature", 0.8f)).floatValue();
        int topK = ((Number) request.getParameters().getOrDefault("top_k", 40)).intValue();
        float topP = ((Number) request.getParameters().getOrDefault("top_p", 0.95f)).floatValue();
        float minP = ((Number) request.getParameters().getOrDefault("min_p", 0.05f)).floatValue();
        int seed = ((Number) request.getParameters().getOrDefault("seed", -1)).intValue();
        if (seed == -1) {
            seed = (int) (Instant.now().toEpochMilli() & 0xFFFFFFFFL);
        }

        MemorySegment chain = createSamplerChainFromRequest(request);

        if (temperature > 0) {
            log.debugf("Sampling params: temp=%.2f, top_k=%d, top_p=%.2f, min_p=%.2f, seed=%d",
                    temperature, topK, topP, minP, seed);
        } else {
            log.debug("Using greedy sampling");
        }

        // 3. Batch Init
        // Allocate batch with capacity for prompt + safety
        MemorySegment batch = binding.batchInit(Math.max(nTokens, 8), 0, 1);

        StringBuilder result = new StringBuilder();
        int tokensGenerated = 0;
        int maxTokens = ((Number) request.getParameters().getOrDefault("max_tokens", 128)).intValue();

        try {
            // 4. Prompt Evaluation
            binding.setBatchSize(batch, nTokens);
            for (int i = 0; i < nTokens; i++) {
                // setBatchToken(batch, index, token, pos, seq_id, logits)
                // Only last token needs logits for sampling
                binding.setBatchToken(batch, i, promptTokens[i], i, 0, i == nTokens - 1);
            }

            if (binding.decode(this.context, batch) != 0) {
                throw new RuntimeException("Prompt evaluation failed");
            }

            // 5. Generation Loop
            int currentPos = nTokens;

            while (tokensGenerated < maxTokens) {
                // Sample next token
                int newTokenId = binding.sample(chain, this.context, -1);

                // Check EOS/EOG
                if (binding.isEndOfGeneration(model, newTokenId)) {
                    break;
                }

                // Decode piece
                String piece = binding.tokenToPiece(model, newTokenId);
                result.append(piece);
                tokensGenerated++;

                // Prepare next batch (size 1)
                binding.setBatchSize(batch, 1);
                // Feed the new token back as input
                binding.setBatchToken(batch, 0, newTokenId, currentPos, 0, true);
                currentPos++;

                // Decode next token
                if (binding.decode(this.context, batch) != 0) {
                    log.error("Decode failed during generation");
                    break;
                }
            }

        } catch (Exception e) {
            log.error("Inference failed", e);
            throw new RuntimeException("Inference failed", e);
        } finally {
            // Cleanup request-specific resources
            binding.batchFree(batch);
            binding.freeSampler(chain);
        }

        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .model(manifest.modelId())
                .content(result.toString())
                .inputTokens(nTokens)
                .outputTokens(tokensGenerated)
                .tokensUsed(nTokens + tokensGenerated)
                .build();
    }

    public Multi<StreamChunk> inferStream(
            InferenceRequest request,
            tech.kayys.golek.spi.context.RequestContext requestContext) {

        if (!initialized) {
            return Multi.createFrom().failure(new IllegalStateException("Runner not initialized"));
        }

        return Multi.createFrom().emitter(emitter -> {
            executorService.execute(() -> {
                try {
                    streamingInferenceLoop(request, emitter);
                } catch (Exception e) {
                    log.error("Streaming inference failed", e);
                    emitter.fail(e);
                }
            });
        });
    }

    private void streamingInferenceLoop(InferenceRequest request, MultiEmitter<? super StreamChunk> emitter) {
        // Always clear KV cache for stateless requests
        binding.kvCacheClear(this.context);

        String prompt = (String) request.getParameters().getOrDefault("prompt", "");
        if (request.getMessages() != null && !request.getMessages().isEmpty()) {
            prompt = templateService.render(chatTemplate, request.getMessages());
        }

        if (prompt == null || prompt.isBlank()) {
            emitter.complete();
            return;
        }

        // 1. Tokenize
        int[] promptTokens = binding.tokenize(model, prompt, true, true);
        int nTokens = promptTokens.length;

        if (nTokens == 0) {
            emitter.complete();
            return;
        }

        // 2. Sampler Chain
        MemorySegment chain = createSamplerChainFromRequest(request);

        // 3. Batch Init
        MemorySegment batch = binding.batchInit(Math.max(nTokens, 8), 0, 1);

        try {
            // 4. Prompt Evaluation
            binding.setBatchSize(batch, nTokens);
            for (int i = 0; i < nTokens; i++) {
                binding.setBatchToken(batch, i, promptTokens[i], i, 0, i == nTokens - 1);
            }

            if (binding.decode(this.context, batch) != 0) {
                log.errorf("Prompt evaluation failed for model %s (request %s)", manifest.modelId(),
                        request.getRequestId());
                emitter.fail(
                        new RuntimeException("Prompt evaluation failed (check context size or model compatibility)"));
                return;
            }

            // 5. Generation Loop
            int tokensGenerated = 0;
            int currentPos = nTokens;
            int maxTokens = ((Number) request.getParameters().getOrDefault("max_tokens", 128)).intValue();

            @SuppressWarnings("unchecked")
            List<String> stopTokens = (List<String>) request.getParameters().getOrDefault("stop",
                    Collections.emptyList());

            while (tokensGenerated < maxTokens) {
                if (emitter.isCancelled()) {
                    break;
                }

                // Sample next token
                int newTokenId = binding.sample(chain, this.context, -1);

                // Check EOS/EOG
                if (binding.isEndOfGeneration(model, newTokenId)) {
                    break;
                }

                // Decode piece
                String piece = binding.tokenToPiece(model, newTokenId);

                // Check stop tokens
                boolean stopped = false;
                for (String stop : stopTokens) {
                    if (piece.contains(stop)) {
                        stopped = true;
                        break;
                    }
                }
                if (stopped)
                    break;

                emitter.emit(StreamChunk.of(request.getRequestId(), tokensGenerated, piece));
                tokensGenerated++;

                // Prepare next batch (size 1)
                binding.setBatchSize(batch, 1);
                binding.setBatchToken(batch, 0, newTokenId, currentPos, 0, true);
                currentPos++;

                // Decode next token
                if (binding.decode(this.context, batch) != 0) {
                    log.error("Decode failed during generation");
                    break;
                }
            }

            // Final chunk
            emitter.emit(StreamChunk.finalChunk(request.getRequestId(), tokensGenerated, ""));
            emitter.complete();

        } finally {
            binding.batchFree(batch);
            binding.freeSampler(chain);
        }
    }

    private MemorySegment createSamplerChainFromRequest(InferenceRequest request) {
        float temperature = ((Number) request.getParameters().getOrDefault("temperature", 0.8f)).floatValue();
        int topK = ((Number) request.getParameters().getOrDefault("top_k", 40)).intValue();
        float topP = ((Number) request.getParameters().getOrDefault("top_p", 0.95f)).floatValue();
        float minP = ((Number) request.getParameters().getOrDefault("min_p", 0.05f)).floatValue();
        int seed = ((Number) request.getParameters().getOrDefault("seed", -1)).intValue();
        if (seed == -1) {
            seed = (int) (Instant.now().toEpochMilli() & 0xFFFFFFFFL);
        }

        MemorySegment chain = binding.createSamplerChain();

        // 1. Penalties
        float repeatPenalty = ((Number) request.getParameters().getOrDefault("repeat_penalty", 1.1f)).floatValue();
        float frequencyPenalty = ((Number) request.getParameters().getOrDefault("frequency_penalty", 0.0f))
                .floatValue();
        float presencePenalty = ((Number) request.getParameters().getOrDefault("presence_penalty", 0.0f)).floatValue();
        int repeatLastN = ((Number) request.getParameters().getOrDefault("repeat_last_n", 64)).intValue();

        if (repeatPenalty != 1.0f || frequencyPenalty != 0.0f || presencePenalty != 0.0f) {
            binding.addPenaltiesSampler(chain, repeatLastN, repeatPenalty, frequencyPenalty, presencePenalty);
        }

        // 2. Truncation samplers (top-k, top-p, min-p, typical)
        if (temperature > 0) {
            binding.addTopKSampler(chain, topK);
            binding.addTopPSampler(chain, topP, 1);
            binding.addMinPSampler(chain, minP, 1);

            float typicalP = ((Number) request.getParameters().getOrDefault("typical_p", 1.0f)).floatValue();
            if (typicalP < 1.0f) {
                binding.addTypicalSampler(chain, typicalP, 1);
            }

            // 3. Grammar / JSON Mode
            String grammar = (String) request.getParameters().get("grammar");
            boolean jsonMode = (boolean) request.getParameters().getOrDefault("json_mode", false);
            if (jsonMode && (grammar == null || grammar.isBlank())) {
                // Simple JSON grammar if none provided and JSON mode requested
                grammar = "root ::= object\n" +
                        "object ::= \"{\" ws ( pair ( \",\" ws pair )* )? \"}\"\n" +
                        "pair ::= string \":\" ws value\n" +
                        "string ::= \"\\\"\" ([^\\\"\\\\\\x00-\\x1F] | \"\\\\\" [\\\"\\\\/bfnrt] | \"\\\\u\" [0-9a-fA-F]{4})* \"\\\"\"\n"
                        +
                        "value ::= string | number | object | array | \"true\" | \"false\" | \"null\"\n" +
                        "array ::= \"[\" ws ( value ( \",\" ws value )* )? \"]\"\n" +
                        "number ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n" +
                        "ws ::= [ \\t\\n\\r]*";
            }
            if (grammar != null && !grammar.isBlank()) {
                binding.addGrammarSampler(chain, model, grammar, "root");
            }

            // 4. Final selection (Mirostat or Dist/Temp)
            int mirostat = ((Number) request.getParameters().getOrDefault("mirostat", 0)).intValue();
            if (mirostat == 1) {
                float tau = ((Number) request.getParameters().getOrDefault("mirostat_tau", 5.0f)).floatValue();
                float eta = ((Number) request.getParameters().getOrDefault("mirostat_eta", 0.1f)).floatValue();
                binding.addMirostatSampler(chain, binding.getVocabSize(model), seed, tau, eta, 100);
            } else if (mirostat == 2) {
                float tau = ((Number) request.getParameters().getOrDefault("mirostat_tau", 5.0f)).floatValue();
                float eta = ((Number) request.getParameters().getOrDefault("mirostat_eta", 0.1f)).floatValue();
                binding.addMirostatV2Sampler(chain, seed, tau, eta);
            } else {
                binding.addTempSampler(chain, temperature);
                binding.addDistSampler(chain, seed);
            }
        } else {
            binding.addGreedySampler(chain);
        }

        return chain;
    }

    public void close() {
        if (!initialized) {
            return;
        }
        cleanup();
        executorService.shutdownNow();
        initialized = false;
    }

    private void cleanup() {
        if (context != null) {
            binding.freeContext(context);
            context = null;
        }
        if (model != null) {
            binding.freeModel(model);
            model = null;
        }
    }

    public List<InferenceRequest> createDefaultWarmupRequests() {
        return List.of(
                InferenceRequest.builder()
                        .model(manifest != null ? manifest.modelId() : "unknown")
                        .message(tech.kayys.golek.spi.Message.user("warmup"))
                        .parameter("prompt", "Hello")
                        .build());
    }

    public void warmup(List<InferenceRequest> requests) {
        // Warmup logic
    }
}
