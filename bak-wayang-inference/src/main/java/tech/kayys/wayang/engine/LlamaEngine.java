package tech.kayys.golek.engine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.golek.engine.LlamaConfig.SamplingConfig;
import tech.kayys.golek.mcp.CircuitBreaker;
import tech.kayys.golek.mcp.DistributedTracing;
import tech.kayys.golek.mcp.Metrics;
import tech.kayys.golek.model.ChatMessage;
import tech.kayys.golek.model.EmbeddingResult;
import tech.kayys.golek.model.GenerationResult;
import tech.kayys.golek.model.ModelInfo;
import tech.kayys.golek.plugin.PluginException;
import tech.kayys.golek.plugin.PluginManager;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

public class LlamaEngine implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(LlamaEngine.class);

    private final LlamaConfig config;
    private final LlamaCppBinding binding;
    private final Arena arena;
    private final MemorySegment model;
    private final MemorySegment context;
    private final ModelInfo modelInfo;
    private final PluginManager pluginManager;
    private final CircuitBreaker circuitBreaker;
    private final Metrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    // Special tokens
    private final int bosToken;
    private final int eosToken;
    private final int nlToken;


    // Performance monitoring
    private final AtomicLong totalTokensProcessed = new AtomicLong(0);
    private final AtomicLong totalGenerations = new AtomicLong(0);

    // Token cache for repeated prompts
    private final Map<String, int[]> promptTokenCache = new ConcurrentHashMap<>();
    private static final int MAX_CACHE_SIZE = 1000;

    public LlamaEngine(LlamaConfig config) {
        this(config, null, null);
    }

    public LlamaEngine(LlamaConfig config, Path pluginsDir, Path dataDir) {
        this.config = config;
        this.metrics = new Metrics();

        try {
            log.info("Initializing LlamaEngine with config: {}", config.modelPath());

            // Validate configuration
            validateConfig(config);

            // Initialize FFM bindings
            arena = Arena.ofShared();
            binding = new LlamaCppBinding(config.libraryPath());

            // Initialize backend with error handling
            int backendResult = binding.backendInit();
            if (backendResult != 0) {
                throw new RuntimeException("Backend initialization failed with code: " + backendResult);
            }

            // Load model with retry logic
            log.info("Loading model: {} (gpu_layers: {}, mmap: {}, mlock: {})",
                    config.modelPath(), config.gpuLayers(), config.useMmap(), config.useMlock());

            model = binding.loadModel(arena, config.modelPath(), config.gpuLayers(),
                    config.useMmap(), config.useMlock());

            if (model.address() == 0) {
                throw new RuntimeException("Failed to load model from: " + config.modelPath());
            }

            // Get model info with comprehensive error checking
            this.modelInfo = loadModelInfo();
            log.info("Model loaded: {} ({}B params)", modelInfo.description(),
                    modelInfo.parameterCount() / 1_000_000_000);

            // Create context with validation
            context = binding.createContext(arena, model, config.contextSize(),
                    config.batchSize(), config.threads(), config.seed(),
                    config.ropeFreqBase(), config.ropeFreqScale(),
                    config.embeddings(), config.flashAttention());

            if (context.address() == 0) {
                throw new RuntimeException("Failed to create context with size: " + config.contextSize());
            }

            // Get special tokens
            this.bosToken = binding.tokenBos(model);
            this.eosToken = binding.tokenEos(model);
            this.nlToken = binding.tokenNl(model);

            log.info("Context created: size={}, batch={}, threads={}",
                    config.contextSize(), config.batchSize(), config.threads());

            // Initialize circuit breaker with appropriate settings for LLM inference
            this.circuitBreaker = new CircuitBreaker(
                    3, // Lower failure threshold for LLM inference
                    java.time.Duration.ofSeconds(10), // Shorter timeout
                    java.time.Duration.ofSeconds(30) // Quicker reset
            );

            // Initialize plugin system
            if (pluginsDir != null) {
                this.pluginManager = new PluginManager(this, pluginsDir, dataDir);
                try {
                    pluginManager.loadPlugins();
                    log.info("Loaded {} plugins", pluginManager.getAllPlugins().size());
                } catch (PluginException e) {
                    log.warn("Some plugins failed to load, continuing without them", e);
                }
            } else {
                this.pluginManager = null;
            }

            // Initialize metrics
            initializeMetrics();



            log.info("LlamaEngine initialized successfully");

        } catch (Throwable e) {
            cleanup();
            throw new RuntimeException("Failed to initialize LlamaEngine", e);
        }
    }

    private void validateConfig(LlamaConfig config) {
        if (config.modelPath() == null || config.modelPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Model path cannot be null or empty");
        }

        if (config.contextSize() <= 0) {
            throw new IllegalArgumentException("Context size must be positive");
        }

        if (config.batchSize() <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }

        if (config.threads() <= 0) {
            throw new IllegalArgumentException("Thread count must be positive");
        }

        // Validate sampling config
        config.sampling().validate();
    }

    private ModelInfo loadModelInfo() {
        try {
            int vocabSize = binding.nVocab(model);
            int ctxTrain = binding.nCtxTrain(model);
            int embdSize = binding.nEmbd(model);
            long modelSize = binding.modelSize(model);
            long nParams = binding.modelNParams(model);
            String desc = binding.modelDesc(arena, model, 256);

            // Detect model type and architecture
            String modelType = detectModelType(desc);
            String quantization = detectQuantization(desc);

            return ModelInfo.builder()
                    .name(Path.of(config.modelPath()).getFileName().toString())
                    .description(desc != null ? desc.trim() : "Unknown")
                    .modelType(modelType)
                    .architecture("Transformer")
                    .parameterCount(nParams)
                    .quantization(quantization)
                    .contextLength(ctxTrain)
                    .vocabSize(vocabSize)
                    .fileSize(modelSize)
                    .metadata(Map.of(
                            "embedding_size", embdSize,
                            "gpu_layers", config.gpuLayers(),
                            "use_mmap", config.useMmap(),
                            "use_mlock", config.useMlock()))
                    .build();

        } catch (Throwable e) {
            log.warn("Failed to get detailed model info, using basic info", e);
            return ModelInfo.builder()
                    .name(Path.of(config.modelPath()).getFileName().toString())
                    .description("Unknown")
                    .vocabSize(-1)
                    .contextLength(config.contextSize())
                    .parameterCount(-1)
                    .fileSize(-1)
                    .metadata(Map.of())
                    .build();
        }
    }

    private String detectModelType(String description) {
        if (description == null)
            return "unknown";
        String descLower = description.toLowerCase();
        if (descLower.contains("llama") && descLower.contains("chat"))
            return "llama-chat";
        if (descLower.contains("llama"))
            return "llama";
        if (descLower.contains("mistral"))
            return "mistral";
        if (descLower.contains("mixtral"))
            return "mixtral";
        if (descLower.contains("codellama"))
            return "code-llama";
        return "unknown";
    }

    private String detectQuantization(String description) {
        if (description == null)
            return "unknown";
        if (description.contains("Q4_0"))
            return "Q4_0";
        if (description.contains("Q4_1"))
            return "Q4_1";
        if (description.contains("Q5_0"))
            return "Q5_0";
        if (description.contains("Q5_1"))
            return "Q5_1";
        if (description.contains("Q8_0"))
            return "Q8_0";
        return "unknown";
    }

    private void initializeMetrics() {
        metrics.setGauge("model.vocab_size", modelInfo.vocabSize());
        metrics.setGauge("model.context_length", modelInfo.contextLength());
        metrics.setGauge("model.parameter_count", modelInfo.parameterCount());
        metrics.setGauge("engine.cache_size", 0);
    }


    public GenerationResult generate(String prompt, SamplingConfig samplingConfig,
            int maxTokens, List<String> stopStrings,
            Consumer<String> streamCallback) {
        return generate(prompt, samplingConfig, maxTokens, stopStrings, streamCallback, true);
    }

    public GenerationResult generate(String prompt, SamplingConfig samplingConfig,
            int maxTokens, List<String> stopStrings,
            Consumer<String> streamCallback, boolean useCache) {
        ensureNotClosed();

        if (prompt == null || prompt.trim().isEmpty()) {
            throw new IllegalArgumentException("Prompt cannot be null or empty");
        }

        if (maxTokens <= 0) {
            throw new IllegalArgumentException("Max tokens must be positive");
        }

        try {
            return circuitBreaker.execute(() -> {
                DistributedTracing.TraceContext trace = DistributedTracing.startTrace("generate");
                long startTime = System.currentTimeMillis();

                try {
                    // Apply preprocessing from plugins
                    String processedPrompt = applyPluginPreprocessing(prompt);

                    // Emit events
                    if (pluginManager != null) {
                        pluginManager.emitGlobalEvent("generation.start",
                                Map.of("prompt", processedPrompt, "maxTokens", maxTokens));
                    }

                    // Tokenize with caching
                    DistributedTracing.startSpan("tokenize");
                    int[] promptTokens;
                    String cacheKey = useCache ? processedPrompt : null;

                    if (useCache && cacheKey != null && promptTokenCache.containsKey(cacheKey)) {
                        promptTokens = promptTokenCache.get(cacheKey);
                        log.debug("Using cached tokens for prompt ({} tokens)", promptTokens.length);
                    } else {
                        boolean addBos;
                        try {
                            addBos = binding.addBosToken(model);
                            promptTokens = binding.tokenize(arena, model, processedPrompt, addBos, false);
                        } catch (Throwable t) {
                            throw new Exception("Tokenization failed", t);
                        }

                        if (useCache && cacheKey != null) {
                            cachePromptTokens(cacheKey, promptTokens);
                        }
                    }
                    DistributedTracing.addTag("prompt_tokens", String.valueOf(promptTokens.length));
                    DistributedTracing.endSpan();

                    log.debug("Prompt tokenized: {} tokens", promptTokens.length);
                    metrics.recordHistogram("prompt.tokens", promptTokens.length);

                    // Validate context size
                    if (promptTokens.length + maxTokens > config.contextSize()) {
                        throw new IllegalArgumentException(
                                String.format("Prompt tokens (%d) + max tokens (%d) exceeds context size (%d)",
                                        promptTokens.length, maxTokens, config.contextSize()));
                    }

                    // Generate
                    DistributedTracing.startSpan("generation");
                    GenerationResult result;
                    try {
                        result = performGeneration(
                                promptTokens, samplingConfig, maxTokens, stopStrings, streamCallback);
                    } catch (Throwable t) {
                        throw new Exception("Generation failed", t);
                    }
                    DistributedTracing.addTag("tokens_generated", String.valueOf(result.tokensGenerated()));
                    DistributedTracing.endSpan();

                    // Apply postprocessing
                    String finalText = applyPluginPostprocessing(result.text());

                    GenerationResult finalResult = new GenerationResult(
                            finalText,
                            result.tokensGenerated(),
                            promptTokens.length,
                            System.currentTimeMillis() - startTime,
                            result.finishReason());

                    // Update statistics
                    totalTokensProcessed.addAndGet(promptTokens.length + result.tokensGenerated());
                    totalGenerations.incrementAndGet();

                    // Emit completion events
                    if (pluginManager != null) {
                        pluginManager.emitGlobalEvent("generation.complete", finalResult);
                    }

                    metrics.incrementCounter("generations.completed");
                    metrics.recordHistogram("generation.time_ms", finalResult.timeMs());
                    metrics.recordHistogram("generation.tokens", finalResult.tokensGenerated());
                    metrics.setGauge("total_tokens_processed", totalTokensProcessed.get());
                    metrics.setGauge("total_generations", totalGenerations.get());

                    return finalResult;

                } catch (Exception e) {
                    metrics.incrementCounter("generations.failed");
                    throw e;
                } finally {
                    DistributedTracing.endSpan();
                }
            });
        } catch (CircuitBreaker.CircuitBreakerOpenException e) {
            throw e;
        } catch (Exception e) {
            if (e.getCause() instanceof RuntimeException) {
                throw (RuntimeException) e.getCause();
            }
            throw new RuntimeException("Generation failed", e);
        }
    }

    private void cachePromptTokens(String prompt, int[] tokens) {
        synchronized (promptTokenCache) {
            if (promptTokenCache.size() >= MAX_CACHE_SIZE) {
                // Remove oldest entry (simple LRU)
                Iterator<String> it = promptTokenCache.keySet().iterator();
                if (it.hasNext()) {
                    promptTokenCache.remove(it.next());
                }
            }
            promptTokenCache.put(prompt, tokens);
            metrics.setGauge("engine.cache_size", promptTokenCache.size());
        }
    }

    private GenerationResult performGeneration(int[] promptTokens, SamplingConfig samplingConfig,
            int maxTokens, List<String> stopStrings,
            Consumer<String> streamCallback) throws Throwable {

        try (Arena sessionArena = Arena.ofConfined()) {
            // Clear KV cache
            binding.kvCacheClear(context);

            // Process prompt in batches with progress tracking
            DistributedTracing.startSpan("process_prompt");
            int nPrompt = promptTokens.length;
            int batchSize = config.batchSize();
            int processed = 0;

            for (int i = 0; i < nPrompt; i += batchSize) {
                int currentBatchSize = Math.min(batchSize, nPrompt - i);
                int[] batchTokens = Arrays.copyOfRange(promptTokens, i, i + currentBatchSize);
                int[] positions = new int[currentBatchSize];
                boolean[] logitsFlags = new boolean[currentBatchSize];

                for (int j = 0; j < currentBatchSize; j++) {
                    positions[j] = i + j;
                    logitsFlags[j] = (i + j == nPrompt - 1);
                }

                MemorySegment batch = LlamaStructs.createBatch(sessionArena, batchTokens, positions, logitsFlags);
                int ret = binding.decode(context, batch);

                if (ret != 0) {
                    throw new RuntimeException("Prompt decode failed at position " + i + " with error: " + ret);
                }

                processed += currentBatchSize;

                // Notify progress for long prompts
                if (streamCallback != null && nPrompt > 1000 && processed % 500 == 0) {
                    log.debug("Processed {}/{} prompt tokens", processed, nPrompt);
                }
            }
            DistributedTracing.endSpan();

            // Generate tokens with enhanced stop condition handling
            DistributedTracing.startSpan("generate_tokens");
            SamplingParams samplingParams = SamplingParams.fromConfig(samplingConfig, config.getEffectiveSeed());
            Sampler sampler = new Sampler(samplingParams);
            StringBuilder result = new StringBuilder();
            int tokensGenerated = 0;
            int pos = nPrompt;
            String finishReason = "length";

            Set<String> stopStringsSet = stopStrings != null ? new HashSet<>(stopStrings) : Set.of();
            StopConditionChecker stopChecker = new StopConditionChecker(stopStringsSet);

            for (int i = 0; i < maxTokens; i++) {
                // Get logits and sample next token
                MemorySegment logits = binding.getLogitsIth(context, 0);
                Set<Integer> stopTokens = Set.of(binding.tokenEos(model)); // Include EOS token
                int nextToken = sampler.sample(arena, logits, modelInfo.vocabSize(), stopTokens);

                // Check for EOS token
                if (nextToken == eosToken) {
                    finishReason = "stop";
                    break;
                }

                // Convert token to text
                String piece = binding.tokenToString(sessionArena, model, nextToken);
                if (piece == null) {
                    log.warn("Received null piece for token: {}", nextToken);
                    piece = "";
                }

                result.append(piece);
                tokensGenerated++;

                // Notify plugins
                if (pluginManager != null) {
                    pluginManager.emitGlobalEvent("token.generated",
                            Map.of("token", piece, "tokenId", nextToken, "position", pos));
                }

                // Stream callback
                if (streamCallback != null && !piece.isEmpty()) {
                    try {
                        streamCallback.accept(piece);
                    } catch (Exception e) {
                        log.warn("Stream callback failed", e);
                    }
                }

                // Check stop conditions
                String currentText = result.toString();
                if (stopChecker.shouldStop(currentText, nextToken)) {
                    if (stopChecker.isStopStringMatch()) {
                        result.setLength(result.length() - stopChecker.getMatchedStopString().length());
                    }
                    finishReason = "stop";
                    break;
                }

                // Decode next token
                int[] nextTokens = { nextToken };
                int[] positionsArr = { pos++ };
                boolean[] logitsFlagsArr = { true };

                MemorySegment batch = LlamaStructs.createBatch(sessionArena, nextTokens, positionsArr, logitsFlagsArr);
                int ret = binding.decode(context, batch);

                if (ret != 0) {
                    log.error("Decode failed at token {} with error: {}", i, ret);
                    finishReason = "error";
                    break;
                }

                // Periodic logging for long generations
                if (tokensGenerated % 50 == 0) {
                    log.debug("Generated {}/{} tokens", tokensGenerated, maxTokens);
                }
            }
            DistributedTracing.endSpan();

            log.debug("Generation completed: {} tokens, reason: {}", tokensGenerated, finishReason);

            return new GenerationResult(
                    result.toString(),
                    tokensGenerated,
                    promptTokens.length,
                    0, // Will be set by caller
                    finishReason);
        }
    }

    // Enhanced stop condition checker
    private static class StopConditionChecker {
        private final Set<String> stopStrings;
        private String matchedStopString;

        StopConditionChecker(Set<String> stopStrings) {
            this.stopStrings = stopStrings;
        }

        boolean shouldStop(String currentText, int token) {
            // Check stop strings
            if (!stopStrings.isEmpty()) {
                for (String stop : stopStrings) {
                    if (currentText.endsWith(stop)) {
                        matchedStopString = stop;
                        return true;
                    }
                }
            }
            return false;
        }

        boolean isStopStringMatch() {
            return matchedStopString != null;
        }

        String getMatchedStopString() {
            return matchedStopString;
        }
    }

    public GenerationResult chat(List<ChatMessage> messages, SamplingConfig samplingConfig,
            int maxTokens, Consumer<String> streamCallback) {
        return chat(messages, samplingConfig, maxTokens, null, streamCallback);
    }

    public GenerationResult chat(List<ChatMessage> messages, SamplingConfig samplingConfig,
            int maxTokens, List<String> stopStrings,
            Consumer<String> streamCallback) {
        ensureNotClosed();

        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("Messages cannot be null or empty");
        }

        // Apply plugin preprocessing to messages
        List<ChatMessage> processedMessages = messages;
        if (pluginManager != null) {
            for (EnginePlugin plugin : pluginManager.getPluginsByType(EnginePlugin.class)) {
                processedMessages = plugin.preprocessChatMessages(processedMessages);
            }
        }

        String prompt = formatChatPrompt(processedMessages);
        return generate(prompt, samplingConfig, maxTokens, stopStrings, streamCallback);
    }

    public EmbeddingResult embeddings(List<String> texts) {
        ensureNotClosed();

        if (!config.embeddings()) {
            throw new IllegalStateException("Embeddings not enabled in configuration");
        }

        if (texts == null || texts.isEmpty()) {
            throw new IllegalArgumentException("Texts cannot be null or empty");
        }

        long startTime = System.currentTimeMillis();
        List<float[]> embeddings = new ArrayList<>();
        int dimensions = (int) modelInfo.metadata().getOrDefault("embedding_size", 0);

        if (dimensions <= 0) {
            throw new IllegalStateException("Invalid embedding dimensions: " + dimensions);
        }

        try (Arena sessionArena = Arena.ofConfined()) {
            for (int textIndex = 0; textIndex < texts.size(); textIndex++) {
                String text = texts.get(textIndex);
                if (text == null || text.trim().isEmpty()) {
                    log.warn("Skipping null or empty text at index {}", textIndex);
                    embeddings.add(new float[dimensions]); // Add zero vector
                    continue;
                }

                binding.kvCacheClear(context);

                boolean addBos = binding.addBosToken(model);
                int[] tokens = binding.tokenize(sessionArena, model, text, addBos, false);

                if (tokens.length == 0) {
                    log.warn("No tokens generated for text at index {}", textIndex);
                    embeddings.add(new float[dimensions]); // Add zero vector
                    continue;
                }

                // Process tokens in batches
                for (int i = 0; i < tokens.length; i += config.batchSize()) {
                    int batchSize = Math.min(config.batchSize(), tokens.length - i);
                    int[] batchTokens = Arrays.copyOfRange(tokens, i, i + batchSize);
                    int[] positions = new int[batchSize];
                    boolean[] logitsFlags = new boolean[batchSize];

                    for (int j = 0; j < batchSize; j++) {
                        positions[j] = i + j;
                        logitsFlags[j] = false;
                    }

                    MemorySegment batch = LlamaStructs.createBatch(sessionArena, batchTokens, positions, logitsFlags);
                    int ret = binding.decode(context, batch);

                    if (ret != 0) {
                        log.warn("Decode failed for batch {}-{}", i, i + batchSize);
                    }
                }

                // Get embeddings
                MemorySegment embdSegment = binding.getEmbeddingsIth(context, 0);
                if (embdSegment.address() == 0) {
                    throw new RuntimeException("Failed to get embeddings for text at index " + textIndex);
                }

                float[] embedding = new float[dimensions];
                for (int i = 0; i < dimensions; i++) {
                    embedding[i] = embdSegment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                }

                embeddings.add(embedding);

                // Log progress for large batches
                if (texts.size() > 10 && (textIndex + 1) % 10 == 0) {
                    log.debug("Processed {}/{} texts for embeddings", textIndex + 1, texts.size());
                }
            }

            long duration = System.currentTimeMillis() - startTime;
            metrics.incrementCounter("embeddings.generated");
            metrics.recordHistogram("embeddings.time_ms", duration);
            metrics.recordHistogram("embeddings.batch_size", texts.size());

            return new EmbeddingResult(embeddings, dimensions, duration);

        } catch (Throwable e) {
            metrics.incrementCounter("embeddings.failed");
            throw new RuntimeException("Embedding generation failed", e);
        }
    }

    private String applyPluginPreprocessing(String prompt) {
        if (pluginManager == null)
            return prompt;

        String processed = prompt;
        for (EnginePlugin plugin : pluginManager.getPluginsByType(EnginePlugin.class)) {
            try {
                processed = plugin.preprocessPrompt(processed);
            } catch (Exception e) {
                log.warn("Plugin {} failed in preprocessing, using previous text", plugin.getName(), e);
            }
        }
        return processed;
    }

    private String applyPluginPostprocessing(String output) {
        if (pluginManager == null)
            return output;

        String processed = output;
        for (EnginePlugin plugin : pluginManager.getPluginsByType(EnginePlugin.class)) {
            try {
                processed = plugin.postprocessOutput(processed);
            } catch (Exception e) {
                log.warn("Plugin {} failed in postprocessing, using previous text", plugin.getName(), e);
            }
        }
        return processed;
    }

    private String formatChatPrompt(List<ChatMessage> messages) {
        // Enhanced chat template handling
        String modelType = modelInfo.modelType();

        if ("llama-chat".equals(modelType)) {
            return formatLlama2ChatPrompt(messages);
        } else if ("mistral".equals(modelType) || "mixtral".equals(modelType)) {
            return formatMistralChatPrompt(messages);
        } else {
            // Default format
            return formatDefaultChatPrompt(messages);
        }
    }

    private String formatLlama2ChatPrompt(List<ChatMessage> messages) {
        StringBuilder prompt = new StringBuilder("<s>");

        for (ChatMessage msg : messages) {
            switch (msg.role()) {
                case "system" -> prompt.append("[INST] <<SYS>>\n")
                        .append(msg.content())
                        .append("\n<</SYS>>\n\n");
                case "user" -> prompt.append(msg.content()).append(" [/INST]");
                case "assistant" -> prompt.append(" ")
                        .append(msg.content())
                        .append(" </s><s>");
                default -> log.warn("Unknown message role: {}", msg.role());
            }
        }

        return prompt.toString();
    }

    private String formatMistralChatPrompt(List<ChatMessage> messages) {
        StringBuilder prompt = new StringBuilder("<s>");

        for (ChatMessage msg : messages) {
            switch (msg.role()) {
                case "user" -> prompt.append("[INST] ").append(msg.content()).append(" [/INST]");
                case "assistant" -> prompt.append(msg.content()).append("</s>");
                case "system" -> // Mistral doesn't have explicit system role, prepend to first user message
                    prompt.append("[INST] ").append(msg.content()).append("\n\n");
                default -> log.warn("Unknown message role: {}", msg.role());
            }
        }

        return prompt.toString();
    }

    private String formatDefaultChatPrompt(List<ChatMessage> messages) {
        StringBuilder prompt = new StringBuilder();

        for (ChatMessage msg : messages) {
            switch (msg.role()) {
                case "system" -> prompt.append("System: ").append(msg.content()).append("\n\n");
                case "user" -> prompt.append("User: ").append(msg.content()).append("\n\n");
                case "assistant" -> prompt.append("Assistant: ").append(msg.content()).append("\n\n");
                default -> prompt.append(msg.role()).append(": ").append(msg.content()).append("\n\n");
            }
        }

        prompt.append("Assistant: ");
        return prompt.toString();
    }

    // State management with validation
    public void saveState(String path) {
        ensureNotClosed();

        if (path == null || path.trim().isEmpty()) {
            throw new IllegalArgumentException("State path cannot be null or empty");
        }

        try {
            boolean success = binding.stateSaveFile(context, path);
            if (!success) {
                throw new RuntimeException("State save failed for path: " + path);
            }

            // If you need the actual bytes written, you might need to call a different
            // method
            // or get the file size separately
            long fileSize = Files.size(Path.of(path));

            log.info("State saved: {} bytes to {}", fileSize, path);
            metrics.incrementCounter("state.saves");
        } catch (Throwable e) {
            metrics.incrementCounter("state.save_errors");
            throw new RuntimeException("Failed to save state to: " + path, e);
        }
    }

    public void loadState(String path) {
        ensureNotClosed();

        if (path == null || path.trim().isEmpty()) {
            throw new IllegalArgumentException("State path cannot be null or empty");
        }

        Path statePath = Path.of(path);
        if (!Files.exists(statePath)) {
            throw new RuntimeException("State file does not exist: " + path);
        }

        try {
            boolean success = binding.stateLoadFile(context, path);
            if (!success) {
                throw new RuntimeException("State load failed for path: " + path);
            }

            long fileSize = Files.size(statePath);
            log.info("State loaded: {} bytes from {}", fileSize, path);
            metrics.incrementCounter("state.loads");
        } catch (Throwable e) {
            metrics.incrementCounter("state.load_errors");
            throw new RuntimeException("Failed to load state from: " + path, e);
        }
    }


    // Clear token cache
    public void clearTokenCache() {
        promptTokenCache.clear();
        metrics.setGauge("engine.cache_size", 0);
        log.info("Token cache cleared");
    }

    // Get cache statistics
    public Map<String, Object> getCacheStats() {
        return Map.of(
                "size", promptTokenCache.size(),
                "max_size", MAX_CACHE_SIZE,
                "hit_ratio", "N/A" // Would need to track hits/misses
        );
    }

    // Getters
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    public LlamaConfig getConfig() {
        return config;
    }

    public PluginManager getPluginManager() {
        return pluginManager;
    }

    public Metrics getMetrics() {
        return metrics;
    }

    public CircuitBreaker getCircuitBreaker() {
        return circuitBreaker;
    }

    public long getTotalTokensProcessed() {
        return totalTokensProcessed.get();
    }

    public long getTotalGenerations() {
        return totalGenerations.get();
    }

    private void ensureNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("LlamaEngine is closed");
        }
    }

    private void cleanup() {
        log.info("Starting LlamaEngine cleanup");

        try {
            // Clear cache
            promptTokenCache.clear();

            // Free resources in correct order
            if (context != null && context.address() != 0) {
                binding.freeContext(context);
            }
            if (model != null && model.address() != 0) {
                binding.freeModel(model);
            }
            binding.backendFree();

        } catch (Throwable e) {
            log.error("Error during native resource cleanup", e);
        } finally {
            // Always close arena
            if (arena != null) {
                try {
                    arena.close();
                } catch (Throwable e) {
                    log.error("Error closing arena", e);
                }
            }
        }

        log.info("LlamaEngine cleanup completed");
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            log.info("Closing LlamaEngine");

            try {
                if (pluginManager != null) {
                    pluginManager.stopAllPlugins();
                }

                cleanup();

                log.info("LlamaEngine closed. Processed {} tokens in {} generations",
                        totalTokensProcessed.get(), totalGenerations.get());

            } catch (Exception e) {
                log.error("Error during LlamaEngine shutdown", e);
            }
        }
    }
}
