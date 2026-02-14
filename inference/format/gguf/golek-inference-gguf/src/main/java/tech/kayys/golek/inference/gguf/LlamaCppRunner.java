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
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Comparator;
import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.Random;
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

            // Suppress verbose native logs before model loading.
            if (!providerConfig.verboseLogging()) {
                try {
                    binding.suppressNativeLogs();
                } catch (Throwable t) {
                    log.debugf("Unable to suppress native logs: %s", t.getMessage());
                }
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
            binding.setContextParam(contextParams, "n_batch", providerConfig.batchSize());

            this.context = binding.createContext(model, contextParams);
            this.contextSize = binding.getContextSize(context);
            this.vocabSize = binding.getVocabSize(model);
            this.eosToken = binding.getEosToken(model);
            this.bosToken = binding.getBosToken(model);

            // Load metadata (chat template)
            this.chatTemplate = binding.getModelMetadata(model, "tokenizer.chat_template");
            log.debugf("Loaded chat template: %s", chatTemplate != null ? "Yes" : "No");
            log.debugf("Model initialized: ctx=%d vocab=%d eos=%d bos=%d", contextSize, vocabSize, eosToken, bosToken);

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
        if (isNativeImageRuntime() && request.getMessages() != null && !request.getMessages().isEmpty()) {
            // Native image can degrade when special chat tokens/templates are interpreted
            // inconsistently. Use a plain text instruction format for stability.
            prompt = buildNativeSafePrompt(request.getMessages());
        }

        // 1. Tokenize
        // If prompt already includes chat special tokens, do not prepend BOS again.
        boolean hasChatSpecialTokens = prompt.contains("<|im_start|>")
                || prompt.contains("<|start_header_id|>")
                || prompt.contains("<|assistant|>");
        int[] promptTokens = binding.tokenize(model, prompt, !hasChatSpecialTokens, true);
        int nTokens = promptTokens.length;

        if (nTokens == 0) {
            return InferenceResponse.builder().requestId(request.getRequestId()).content("").build();
        }

        // 2. Sampler params
        float temperature = ((Number) request.getParameters().getOrDefault("temperature", 0.8f)).floatValue();
        int topK = ((Number) request.getParameters().getOrDefault("top_k", 40)).intValue();
        float topP = ((Number) request.getParameters().getOrDefault("top_p", 0.95f)).floatValue();
        float minP = ((Number) request.getParameters().getOrDefault("min_p", 0.05f)).floatValue();
        float repeatPenalty = ((Number) request.getParameters().getOrDefault("repeat_penalty", 1.1f)).floatValue();
        float frequencyPenalty = ((Number) request.getParameters().getOrDefault("frequency_penalty", 0.0f))
                .floatValue();
        float presencePenalty = ((Number) request.getParameters().getOrDefault("presence_penalty", 0.0f)).floatValue();
        int repeatLastN = ((Number) request.getParameters().getOrDefault("repeat_last_n", 64)).intValue();
        int seed = ((Number) request.getParameters().getOrDefault("seed", -1)).intValue();
        if (seed == -1) {
            seed = (int) (Instant.now().toEpochMilli() & 0xFFFFFFFFL);
        }
        Random random = new Random(seed);

        int maxBatch = Math.max(1, providerConfig.batchSize());

        // 3. Batch Init
        // Allocate only up to n_batch capacity; prompt is decoded in chunks.
        MemorySegment batch = binding.batchInit(Math.max(Math.min(nTokens, maxBatch), 8), 0, 1);

        StringBuilder result = new StringBuilder();
        int tokensGenerated = 0;
        int maxTokens = ((Number) request.getParameters().getOrDefault("max_tokens", 128)).intValue();
        int effectiveRepeatLastN = Math.max(0, repeatLastN);
        ArrayDeque<Integer> recentTokens = new ArrayDeque<>();
        Map<Integer, Integer> recentTokenCounts = new HashMap<>();

        try {
            // 4. Prompt Evaluation in chunks to respect n_batch
            int processed = 0;
            while (processed < nTokens) {
                // Decode prompt token-by-token for maximum compatibility across JVM/native
                // runtimes and stable logits indexing.
                int chunk = 1;
                binding.setBatchSize(batch, chunk);
                for (int i = 0; i < chunk; i++) {
                    int absoluteIndex = processed + i;
                    // Single-token decode: logits are always at index 0.
                    binding.setBatchToken(batch, i, promptTokens[absoluteIndex], absoluteIndex, 0,
                            true);
                }
                if (binding.decode(this.context, batch) != 0) {
                    throw new RuntimeException("Prompt evaluation failed");
                }
                processed += chunk;
            }

            // 5. Generation Loop
            int currentPos = nTokens;
            int logitsBatchIndex = 0;
            if (effectiveRepeatLastN > 0) {
                int start = Math.max(0, nTokens - effectiveRepeatLastN);
                for (int i = start; i < nTokens; i++) {
                    pushRecentToken(promptTokens[i], recentTokens, recentTokenCounts, effectiveRepeatLastN);
                }
            }

            while (tokensGenerated < maxTokens) {
                // Sample next token using the correct batch index for logits
                int newTokenId = sampleNextToken(
                        this.context,
                        logitsBatchIndex,
                        temperature,
                        topK,
                        topP,
                        minP,
                        repeatPenalty,
                        frequencyPenalty,
                        presencePenalty,
                        recentTokenCounts,
                        random);

                // Check EOS/EOG
                if (isEndToken(newTokenId)) {
                    break;
                }

                // Decode piece
                String piece = binding.tokenToPiece(model, newTokenId);
                result.append(piece);
                tokensGenerated++;
                if (effectiveRepeatLastN > 0) {
                    pushRecentToken(newTokenId, recentTokens, recentTokenCounts, effectiveRepeatLastN);
                }

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

                // After single-token decode, logits are always at batch index 0
                logitsBatchIndex = 0;
            }

        } catch (Exception e) {
            log.error("Inference failed", e);
            throw new RuntimeException("Inference failed", e);
        } finally {
            // Cleanup request-specific resources
            binding.batchFree(batch);
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
        // Stable fallback: perform normal inference, then emit in small chunks.
        InferenceResponse response = infer(request, null);
        String content = response.getContent() == null ? "" : response.getContent();
        int index = 0;
        int chunkSize = 24;
        int emitted = 0;

        while (index < content.length()) {
            if (emitter.isCancelled()) {
                return;
            }
            int end = Math.min(content.length(), index + chunkSize);
            emitter.emit(StreamChunk.of(request.getRequestId(), emitted++, content.substring(index, end)));
            index = end;
        }

        emitter.emit(StreamChunk.finalChunk(request.getRequestId(), emitted, ""));
        emitter.complete();
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

    private int sampleNextToken(
            MemorySegment context,
            int batchIndex,
            float temperature,
            int topK,
            float topP,
            float minP,
            float repeatPenalty,
            float frequencyPenalty,
            float presencePenalty,
            Map<Integer, Integer> recentTokenCounts,
            Random random) {

        int effectiveVocab = vocabSize > 0 ? vocabSize : 32768;
        MemorySegment logits;
        try {
            logits = binding.getLogitsIth(context, batchIndex);
        } catch (RuntimeException e) {
            logits = binding.getLogits(context);
        }
        if (logits == null || logits.equals(MemorySegment.NULL)) {
            throw new RuntimeException("No logits available for sampling");
        }
        logits = logits.reinterpret((long) effectiveVocab * Float.BYTES);

        if (temperature <= 0.0f) {
            return argMaxToken(logits, effectiveVocab);
        }

        List<TokenProb> candidates = new ArrayList<>(effectiveVocab);
        for (int i = 0; i < effectiveVocab; i++) {
            float value = logits.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            int count = recentTokenCounts.getOrDefault(i, 0);
            if (count > 0 && repeatPenalty > 1.0f) {
                value = value < 0.0f ? value * repeatPenalty : value / repeatPenalty;
            }
            if (count > 0 && presencePenalty != 0.0f) {
                value -= presencePenalty;
            }
            if (count > 0 && frequencyPenalty != 0.0f) {
                value -= frequencyPenalty * count;
            }
            candidates.add(new TokenProb(i, value / Math.max(temperature, 1.0e-6f)));
        }

        candidates.sort(Comparator.comparingDouble((TokenProb t) -> t.logit).reversed());
        if (topK > 0 && topK < candidates.size()) {
            candidates = new ArrayList<>(candidates.subList(0, topK));
        }

        float maxLogit = candidates.get(0).logit;
        double sum = 0.0;
        for (TokenProb c : candidates) {
            c.prob = Math.exp(c.logit - maxLogit);
            sum += c.prob;
        }

        if (sum <= 0.0) {
            return candidates.get(0).tokenId;
        }

        for (TokenProb c : candidates) {
            c.prob /= sum;
        }

        if (topP > 0.0f && topP < 1.0f) {
            double cumulative = 0.0;
            List<TokenProb> nucleus = new ArrayList<>();
            for (TokenProb c : candidates) {
                nucleus.add(c);
                cumulative += c.prob;
                if (cumulative >= topP) {
                    break;
                }
            }
            candidates = nucleus;
            normalizeProbabilities(candidates);
        }

        if (minP > 0.0f) {
            double best = candidates.get(0).prob;
            double threshold = best * minP;
            List<TokenProb> filtered = new ArrayList<>();
            for (TokenProb c : candidates) {
                if (c.prob >= threshold) {
                    filtered.add(c);
                }
            }
            if (!filtered.isEmpty()) {
                candidates = filtered;
                normalizeProbabilities(candidates);
            }
        }

        double r = random.nextDouble();
        double acc = 0.0;
        for (TokenProb c : candidates) {
            acc += c.prob;
            if (r <= acc) {
                return c.tokenId;
            }
        }
        return candidates.get(candidates.size() - 1).tokenId;
    }

    private void normalizeProbabilities(List<TokenProb> candidates) {
        double sum = 0.0;
        for (TokenProb c : candidates) {
            sum += c.prob;
        }
        if (sum <= 0.0) {
            return;
        }
        for (TokenProb c : candidates) {
            c.prob /= sum;
        }
    }

    private void pushRecentToken(
            int tokenId,
            ArrayDeque<Integer> recentTokens,
            Map<Integer, Integer> recentTokenCounts,
            int repeatLastN) {
        if (repeatLastN <= 0) {
            return;
        }
        recentTokens.addLast(tokenId);
        recentTokenCounts.merge(tokenId, 1, Integer::sum);
        while (recentTokens.size() > repeatLastN) {
            Integer evicted = recentTokens.removeFirst();
            Integer count = recentTokenCounts.get(evicted);
            if (count == null || count <= 1) {
                recentTokenCounts.remove(evicted);
            } else {
                recentTokenCounts.put(evicted, count - 1);
            }
        }
    }

    private int argMaxToken(MemorySegment logits, int effectiveVocab) {
        int bestId = 0;
        float best = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < effectiveVocab; i++) {
            float value = logits.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            if (value > best) {
                best = value;
                bestId = i;
            }
        }
        return bestId;
    }

    private static final class TokenProb {
        private final int tokenId;
        private final float logit;
        private double prob;

        private TokenProb(int tokenId, float logit) {
            this.tokenId = tokenId;
            this.logit = logit;
        }
    }

    private boolean isNativeImageRuntime() {
        return System.getProperty("org.graalvm.nativeimage.imagecode") != null;
    }

    private String buildNativeSafePrompt(List<tech.kayys.golek.spi.Message> messages) {
        StringBuilder sb = new StringBuilder();
        // Qwen-style fallback chat format for native mode when Jinjava is unavailable.
        sb.append("<|im_start|>system\n");
        sb.append("You are a helpful assistant. Reply briefly and clearly.");
        sb.append("<|im_end|>\n");
        for (tech.kayys.golek.spi.Message msg : messages) {
            String role = msg.getRole() == null ? "user" : switch (msg.getRole()) {
                case SYSTEM -> "system";
                case USER -> "user";
                case ASSISTANT -> "assistant";
                default -> "user";
            };
            sb.append("<|im_start|>").append(role).append("\n");
            sb.append(msg.getContent() == null ? "" : msg.getContent());
            sb.append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    private boolean isEndToken(int tokenId) {
        if (tokenId < 0) {
            return true;
        }
        try {
            if (model != null && !model.equals(MemorySegment.NULL)) {
                if (binding.isEndOfGeneration(model, tokenId)) {
                    return true;
                }
            }
        } catch (RuntimeException e) {
            log.debugf("EOG check failed, using EOS fallback: %s", e.getMessage());
        }
        return tokenId == eosToken;
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
