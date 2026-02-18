package tech.kayys.gollek.inference.gguf;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ModelFormat;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.subscription.MultiEmitter;
import tech.kayys.gollek.spi.stream.StreamChunk;
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
import java.util.function.Consumer;

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
    private int runtimeBatchSize;

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
                manifest.modelId(), manifest.requestId());

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
            if (!hasGgufHeader(modelPath)) {
                throw new RuntimeException(
                        "Model file is not a valid GGUF binary (missing GGUF header): "
                                + modelPath
                                + ". Re-pull or re-convert the model.");
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

            int configuredGpuLayers = getIntConfig(runnerConfig, "nGpuLayers",
                    providerConfig.gpuEnabled() ? providerConfig.gpuLayers() : 0);
            int configuredThreads = Math.max(1,
                    getIntConfig(runnerConfig, "nThreads", providerConfig.threads()));
            int configuredCtx = Math.max(512,
                    getIntConfig(runnerConfig, "nCtx", providerConfig.maxContextTokens()));
            int configuredBatch = Math.max(1,
                    getIntConfig(runnerConfig, "nBatch", providerConfig.batchSize()));
            boolean useMmap = getBooleanConfig(runnerConfig, "useMmap", providerConfig.mmapEnabled());
            boolean useMlock = getBooleanConfig(runnerConfig, "useMlock", providerConfig.mlockEnabled());

            long modelSizeBytes = safeFileSize(modelPath);
            long largeModelThreshold = 4L * 1024L * 1024L * 1024L; // 4 GiB
            boolean forceGpuForLargeModel = Boolean.parseBoolean(
                    System.getProperty(
                            "gollek.gguf.force.gpu.large-model",
                            System.getenv().getOrDefault("GOLEK_GGUF_FORCE_GPU_FOR_LARGE_MODEL", "false")));

            // Preserve llama.cpp semantics: -1 = all layers.
            if (configuredGpuLayers < -1) {
                configuredGpuLayers = -1;
            }
            if (configuredGpuLayers != 0 && modelSizeBytes >= largeModelThreshold && !forceGpuForLargeModel) {
                int capped = configuredGpuLayers == -1 ? 8 : Math.min(configuredGpuLayers, 8);
                if (capped != configuredGpuLayers) {
                    log.warnf(
                            "Large GGUF model detected (%.2f GiB). Capping initial GPU layers from %d to %d "
                                    + "for faster and safer startup. Set GOLEK_GGUF_FORCE_GPU_FOR_LARGE_MODEL=true "
                                    + "to keep requested layers.",
                            modelSizeBytes / (1024.0 * 1024.0 * 1024.0),
                            configuredGpuLayers,
                            capped);
                }
                configuredGpuLayers = capped;
            }

            int effectiveBatch = Math.min(configuredBatch, 128);

            this.runtimeBatchSize = effectiveBatch;

            MemorySegment modelParams = binding.getDefaultModelParams();
            int activeGpuLayers = configuredGpuLayers;
            binding.setModelParam(modelParams, "n_gpu_layers", activeGpuLayers);
            binding.setModelParam(modelParams, "main_gpu", providerConfig.gpuDeviceId());
            binding.setModelParam(modelParams, "use_mmap", useMmap);
            binding.setModelParam(modelParams, "use_direct_io", false);
            binding.setModelParam(modelParams, "use_mlock", useMlock);
            binding.setModelParam(modelParams, "check_tensors", false);
            binding.setModelParam(modelParams, "use_extra_bufts", false);
            binding.setModelParam(modelParams, "no_host", false);
            binding.setModelParam(modelParams, "no_alloc", false);
            this.model = binding.loadModel(modelPath.toString(), modelParams);

            MemorySegment contextParams = binding.getDefaultContextParams();
            binding.setContextParam(contextParams, "n_ctx", configuredCtx);
            binding.setContextParam(contextParams, "n_batch", this.runtimeBatchSize);
            binding.setContextParam(contextParams, "n_ubatch", this.runtimeBatchSize);
            binding.setContextParam(contextParams, "n_threads", configuredThreads);
            binding.setContextParam(contextParams, "n_threads_batch", configuredThreads);
            binding.setContextParam(contextParams, "offload_kqv", activeGpuLayers != 0);
            binding.setContextParam(contextParams, "flash_attn_type", 0);
            binding.setContextParam(contextParams, "samplers", MemorySegment.NULL);
            binding.setContextParam(contextParams, "n_samplers", 0L);
            try {
                this.context = binding.createContext(model, contextParams);
            } catch (RuntimeException gpuContextError) {
                if (activeGpuLayers == 0) {
                    throw gpuContextError;
                }

                log.warnf("GPU context initialization failed (n_gpu_layers=%d). Retrying on CPU. Cause: %s",
                        activeGpuLayers, gpuContextError.getMessage());
                binding.freeModel(this.model);

                MemorySegment cpuModelParams = binding.getDefaultModelParams();
                activeGpuLayers = 0;
                binding.setModelParam(cpuModelParams, "n_gpu_layers", activeGpuLayers);
                binding.setModelParam(cpuModelParams, "main_gpu", providerConfig.gpuDeviceId());
                binding.setModelParam(cpuModelParams, "use_mmap", useMmap);
                binding.setModelParam(cpuModelParams, "use_direct_io", false);
                binding.setModelParam(cpuModelParams, "use_mlock", useMlock);
                binding.setModelParam(cpuModelParams, "check_tensors", false);
                binding.setModelParam(cpuModelParams, "use_extra_bufts", false);
                binding.setModelParam(cpuModelParams, "no_host", false);
                binding.setModelParam(cpuModelParams, "no_alloc", false);

                this.model = binding.loadModel(modelPath.toString(), cpuModelParams);
                binding.setContextParam(contextParams, "offload_kqv", false);
                this.context = binding.createContext(model, contextParams);
            }
            this.contextSize = binding.getContextSize(context);
            this.vocabSize = binding.getVocabSize(model);
            this.eosToken = binding.getEosToken(model);
            this.bosToken = binding.getBosToken(model);
            log.infof("GGUF runtime config: gpu_layers=%d, n_ctx=%d, n_batch=%d, threads=%d",
                    activeGpuLayers, configuredCtx, this.runtimeBatchSize, configuredThreads);

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

    private int getIntConfig(Map<String, Object> config, String key, int defaultValue) {
        if (config == null || !config.containsKey(key)) {
            return defaultValue;
        }
        Object value = config.get(key);
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String s) {
            try {
                return Integer.parseInt(s.trim());
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    private boolean getBooleanConfig(Map<String, Object> config, String key, boolean defaultValue) {
        if (config == null || !config.containsKey(key)) {
            return defaultValue;
        }
        Object value = config.get(key);
        if (value instanceof Boolean b) {
            return b;
        }
        if (value instanceof String s) {
            return Boolean.parseBoolean(s.trim());
        }
        return defaultValue;
    }

    private long safeFileSize(Path path) {
        try {
            return java.nio.file.Files.size(path);
        } catch (Exception ignored) {
            return -1;
        }
    }

    private boolean hasGgufHeader(Path path) {
        try (var in = java.nio.file.Files.newInputStream(path)) {
            byte[] magic = in.readNBytes(4);
            return magic.length == 4
                    && magic[0] == 'G'
                    && magic[1] == 'G'
                    && magic[2] == 'U'
                    && magic[3] == 'F';
        } catch (Exception ignored) {
            return false;
        }
    }

    public InferenceResponse infer(
            InferenceRequest request) {
        return inferInternal(request, null);
    }

    private InferenceResponse inferInternal(
            InferenceRequest request,
            Consumer<String> onTokenPiece) {

        if (!initialized) {
            throw new IllegalStateException("Runner not initialized");
        }
        boolean permitAcquired = false;
        try {
            permitAcquired = concurrencyLimit.tryAcquire(providerConfig.defaultTimeout().toMillis(),
                    java.util.concurrent.TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting for GGUF runner slot", e);
        }
        if (!permitAcquired) {
            throw new RuntimeException("GGUF runner is busy. Please retry.");
        }

        try {
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

            // Apply chat template if messages are present. In native mode, only force
            // fallback when template is unavailable.
            if (request.getMessages() != null && !request.getMessages().isEmpty()) {
                boolean useNativeFallback = isNativeImageRuntime() && (chatTemplate == null || chatTemplate.isBlank());
                if (useNativeFallback) {
                    prompt = buildNativeSafePrompt(request.getMessages());
                } else {
                    prompt = templateService.render(chatTemplate, request.getMessages());
                    if (prompt == null || prompt.isBlank()) {
                        prompt = buildNativeSafePrompt(request.getMessages());
                    }
                }
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
            float repeatPenalty = ((Number) request.getParameters().getOrDefault("repeat_penalty", 1.1f))
                    .floatValue();
            float frequencyPenalty = ((Number) request.getParameters().getOrDefault("frequency_penalty", 0.0f))
                    .floatValue();
            float presencePenalty = ((Number) request.getParameters().getOrDefault("presence_penalty", 0.0f))
                    .floatValue();
            int repeatLastN = ((Number) request.getParameters().getOrDefault("repeat_last_n", 64)).intValue();
            int seed = ((Number) request.getParameters().getOrDefault("seed", -1)).intValue();
            if (seed == -1) {
                seed = (int) (Instant.now().toEpochMilli() & 0xFFFFFFFFL);
            }
            Random random = new Random(seed);

            int maxBatch = Math.max(1, runtimeBatchSize);
            long timeoutMs = Math.max(
                    1_000L,
                    ((Number) request.getParameters().getOrDefault("inference_timeout_ms", 120_000)).longValue());
            Instant inferenceDeadline = Instant.now().plusMillis(timeoutMs);

            // 3. Batch Init
            // Keep capacity aligned with runtime batch size so prompt evaluation can run in chunks.
            MemorySegment batch = binding.batchInit(maxBatch, 0, 1);

            StringBuilder result = new StringBuilder();
            int tokensGenerated = 0;
            int maxTokens = ((Number) request.getParameters().getOrDefault("max_tokens", 128)).intValue();
            List<String> stopSequences = resolveStopSequences(request);
            int effectiveRepeatLastN = Math.max(0, repeatLastN);
            ArrayDeque<Integer> recentTokens = new ArrayDeque<>();
            Map<Integer, Integer> recentTokenCounts = new HashMap<>();

            try {
                // 4. Prompt Evaluation in chunks to respect n_batch
                int processed = 0;
                int promptLogitsBatchIndex = 0;
                while (processed < nTokens) {
                    if (Instant.now().isAfter(inferenceDeadline)) {
                        throw new RuntimeException(
                                "GGUF prompt evaluation timed out after " + timeoutMs + " ms");
                    }
                    int chunk = Math.min(maxBatch, nTokens - processed);
                    binding.setBatchSize(batch, chunk);
                    for (int i = 0; i < chunk; i++) {
                        int absoluteIndex = processed + i;
                        // Request logits only on the final token in this chunk.
                        binding.setBatchToken(batch, i, promptTokens[absoluteIndex], absoluteIndex, 0,
                                i == chunk - 1);
                    }
                    if (binding.decode(this.context, batch) != 0) {
                        throw new RuntimeException("Prompt evaluation failed");
                    }
                    promptLogitsBatchIndex = chunk - 1;
                    processed += chunk;
                }

                // 5. Generation Loop
                int currentPos = nTokens;
                int logitsBatchIndex = promptLogitsBatchIndex;
                if (effectiveRepeatLastN > 0) {
                    int start = Math.max(0, nTokens - effectiveRepeatLastN);
                    for (int i = start; i < nTokens; i++) {
                        pushRecentToken(promptTokens[i], recentTokens, recentTokenCounts, effectiveRepeatLastN);
                    }
                }

                while (tokensGenerated < maxTokens) {
                    if (Instant.now().isAfter(inferenceDeadline)) {
                        throw new RuntimeException("GGUF generation timed out after " + timeoutMs + " ms");
                    }
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
                    if (onTokenPiece != null && piece != null && !piece.isEmpty()) {
                        onTokenPiece.accept(piece);
                    }
                    tokensGenerated++;

                    // Respect textual stop sequences for chat-style templates.
                    if (!stopSequences.isEmpty()) {
                        String current = result.toString();
                        String matched = firstMatchedStopSequence(current, stopSequences);
                        if (matched != null) {
                            int cut = current.indexOf(matched);
                            result.setLength(Math.max(0, cut));
                            break;
                        }
                    }

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
                String message = e.getMessage();
                if (message == null || message.isBlank()) {
                    message = "Inference failed";
                } else if (message.contains("timed out")) {
                    message = message + " (increase --inference-timeout-ms for larger models)";
                }
                if (message.contains("timed out")) {
                    log.warnf("Inference timeout: %s", message);
                } else {
                    log.errorf(e, "Inference failed: %s", message);
                }
                throw new RuntimeException(message, e);
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
        } finally {
            if (permitAcquired) {
                concurrencyLimit.release();
            }
        }
    }

    public Multi<StreamChunk> inferStream(
            InferenceRequest request) {

        if (!initialized) {
            return Multi.createFrom().failure(new IllegalStateException("Runner not initialized"));
        }

        return Multi.createFrom().emitter(emitter -> {
            executorService.execute(() -> {
                try {
                    streamingInferenceLoop(request, emitter);
                } catch (Exception e) {
                    String message = e.getMessage() == null ? "" : e.getMessage();
                    if (message.contains("timed out")) {
                        log.warnf("Streaming inference timeout: %s", message);
                    } else {
                        log.error("Streaming inference failed", e);
                    }
                    emitter.fail(e);
                }
            });
        });
    }

    private void streamingInferenceLoop(InferenceRequest request, MultiEmitter<? super StreamChunk> emitter) {
        final int[] emitted = { 0 };
        inferInternal(request, piece -> {
            if (!emitter.isCancelled()) {
                emitter.emit(StreamChunk.of(request.getRequestId(), emitted[0]++, piece));
            }
        });
        if (!emitter.isCancelled()) {
            emitter.emit(StreamChunk.finalChunk(request.getRequestId(), emitted[0], ""));
            emitter.complete();
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

    private String buildNativeSafePrompt(List<tech.kayys.gollek.spi.Message> messages) {
        StringBuilder sb = new StringBuilder();
        // Qwen-style fallback chat format for native mode when Jinjava is unavailable.
        sb.append("<|im_start|>system\n");
        sb.append("You are a helpful assistant. Reply briefly and clearly.");
        sb.append("<|im_end|>\n");
        for (tech.kayys.gollek.spi.Message msg : messages) {
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

    private List<String> resolveStopSequences(InferenceRequest request) {
        Object stop = request.getParameters().get("stop");
        if (stop == null) {
            return List.of();
        }
        if (stop instanceof String s) {
            return s.isBlank() ? List.of() : List.of(s);
        }
        if (stop instanceof List<?> list) {
            List<String> values = new ArrayList<>();
            for (Object item : list) {
                if (item != null) {
                    String s = item.toString();
                    if (!s.isBlank()) {
                        values.add(s);
                    }
                }
            }
            return values;
        }
        return List.of();
    }

    private String firstMatchedStopSequence(String text, List<String> stopSequences) {
        for (String stop : stopSequences) {
            if (text.contains(stop)) {
                return stop;
            }
        }
        return null;
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
                        .message(tech.kayys.gollek.spi.Message.user("warmup"))
                        .parameter("prompt", "Hello")
                        .build());
    }

    public void warmup(List<InferenceRequest> requests) {
        // Warmup logic
    }
}
