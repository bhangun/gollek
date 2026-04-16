package tech.kayys.gollek.inference.gguf;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.Message;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.Arena;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Consumer;

/**
 * Inference logic executor that uses refactored components.
 * Supports both text-only and multimodal inference.
 */
class InferenceLogicExecutor {
    private static final Logger log = Logger.getLogger(InferenceLogicExecutor.class);
    private static final String CHAT_TOKEN = "\u7e26\u7e26";
    private static final String HEADER_START = "<|start_header_id|>";
    private static final String ASSISTANT = "<|assistant|>";

    // Simple multimodal data holder
    private static class MultimodalData {
        private final float[][] embeddings;
        private final int[] embeddingPositions;
        
        MultimodalData(float[][] embeddings, int[] embeddingPositions) {
            this.embeddings = embeddings;
            this.embeddingPositions = embeddingPositions;
        }
        
        float[][] getEmbeddings() { return embeddings; }
        int[] getEmbeddingPositions() { return embeddingPositions; }
        boolean hasEmbeddings() { return embeddings != null && embeddings.length > 0; }
    }

    private final LlamaCppBinding binding;
    private final GGUFProviderConfig providerConfig;
    private final GGUFChatTemplateService templateService;
    private final MemorySegment model, context;
    private final int contextSize, vocabSize, eosToken, bosToken, runtimeBatchSize;
    private final String chatTemplate;
    private final LlamaCppKVCacheManager kvCacheManager;
    private final LlamaCppTokenSampler tokenSampler;
    private final LlamaCppMetricsRecorder metricsRecorder;
    private final ModelManifest manifest;

    InferenceLogicExecutor(LlamaCppBinding binding, GGUFProviderConfig providerConfig,
            GGUFChatTemplateService templateService, MemorySegment model, MemorySegment context,
            int contextSize, int vocabSize, int eosToken, int bosToken, int runtimeBatchSize,
            String chatTemplate, LlamaCppKVCacheManager kvCacheManager, LlamaCppTokenSampler tokenSampler,
            LlamaCppMetricsRecorder metricsRecorder, ModelManifest manifest) {
        this.binding = binding; this.providerConfig = providerConfig; this.templateService = templateService;
        this.model = model; this.context = context; this.contextSize = contextSize;
        this.vocabSize = vocabSize; this.eosToken = eosToken; this.bosToken = bosToken;
        this.runtimeBatchSize = runtimeBatchSize; this.chatTemplate = chatTemplate;
        this.kvCacheManager = kvCacheManager; this.tokenSampler = tokenSampler; this.metricsRecorder = metricsRecorder; this.manifest = manifest;
    }

    InferenceResponse execute(InferenceRequest request, Consumer<String> onTokenPiece) {
        String prompt = resolvePrompt(request);
        if (prompt == null || prompt.isBlank()) return createEmptyResponse(request);
        long requestStart = System.nanoTime();
        kvCacheManager.loadSessionIfExists(context, request);
        boolean hasChatSpecial = prompt.contains(CHAT_TOKEN) || prompt.contains(HEADER_START) || prompt.contains(ASSISTANT);
        int[] promptTokens = kvCacheManager.tokenizeWithCache(model, prompt, !hasChatSpecial);
        int nTokens = promptTokens.length;
        if (nTokens == 0) return createEmptyResponse(request);
        int maxContext = providerConfig.maxContextTokens();
        if (maxContext > 0 && nTokens > maxContext) {
            int[] truncated = new int[maxContext];
            System.arraycopy(promptTokens, nTokens - maxContext, truncated, 0, maxContext);
            promptTokens = truncated; nTokens = maxContext;
        }
        int reusePrefix = kvCacheManager.computeReusePrefix(promptTokens, nTokens);
        if (reusePrefix == 0) kvCacheManager.resetKvCache(context);
        float temperature = ((Number) request.getParameters().getOrDefault("temperature", 0.8f)).floatValue();
        int topK = ((Number) request.getParameters().getOrDefault("top_k", 40)).intValue();
        float topP = ((Number) request.getParameters().getOrDefault("top_p", 0.95f)).floatValue();
        float minP = ((Number) request.getParameters().getOrDefault("min_p", 0.05f)).floatValue();
        float repeatPenalty = ((Number) request.getParameters().getOrDefault("repeat_penalty", 1.1f)).floatValue();
        float frequencyPenalty = ((Number) request.getParameters().getOrDefault("frequency_penalty", 0.0f)).floatValue();
        float presencePenalty = ((Number) request.getParameters().getOrDefault("presence_penalty", 0.0f)).floatValue();
        int repeatLastN = ((Number) request.getParameters().getOrDefault("repeat_last_n", 64)).intValue();
        int seed = ((Number) request.getParameters().getOrDefault("seed", -1)).intValue();
        Random random = seed == -1 ? java.util.concurrent.ThreadLocalRandom.current() : new Random(seed);
        int maxTokens = ((Number) request.getParameters().getOrDefault("max_tokens", 128)).intValue();
        long timeoutMs = Math.max(1000L, ((Number) request.getParameters().getOrDefault("inference_timeout_ms", 120000L)).longValue());
        Instant deadline = Instant.now().plusMillis(timeoutMs);
        List<String> stopSequences = resolveStopSequences(request);
        int maxStopLength = maxStopSequenceLength(stopSequences);
        boolean usePenalties = repeatPenalty > 1.0f || presencePenalty != 0.0f || frequencyPenalty != 0.0f;
        int effectiveRepeatLastN = usePenalties ? Math.max(0, repeatLastN) : 0;
        int[] recentRing = effectiveRepeatLastN > 0 ? new int[effectiveRepeatLastN] : null;
        int recentRingSize = 0, recentRingIndex = 0;
        int[] recentTokenCounts = effectiveRepeatLastN > 0 ? new int[Math.max(1, vocabSize)] : null;
        int maxBatch = Math.max(1, runtimeBatchSize);
        MemorySegment batch = binding.batchInit(maxBatch, 0, 1);
        StringBuilder result = new StringBuilder();
        int tokensGenerated = 0;
        long promptStartNanos = requestStart, promptEndNanos = 0L, firstTokenNanos = 0L;
        try {
            // Check for multimodal data (images, etc.)
            MultimodalData multimodalData = extractMultimodalData(request);
            int processed = reusePrefix;
            
            // If multimodal, use special batch setting with embeddings
            if (multimodalData != null && multimodalData.hasEmbeddings()) {
                processed = processMultimodalBatch(batch, multimodalData, promptTokens, nTokens, reusePrefix, maxBatch, deadline);
            } else {
                // Text-only processing
                while (processed < nTokens) {
                    if (Instant.now().isAfter(deadline)) throw new RuntimeException("Prompt timed out");
                    int chunk = Math.min(maxBatch, nTokens - processed);
                    binding.setBatchSize(batch, chunk);
                    for (int i = 0; i < chunk; i++) binding.setBatchToken(batch, i, promptTokens[processed + i], processed + i, 0, i == chunk - 1);
                    if (binding.decode(context, batch) != 0) throw new RuntimeException("Prompt evaluation failed");
                    processed += chunk;
                }
            }
            promptEndNanos = System.nanoTime();
            kvCacheManager.updateAfterPrompt(promptTokens, nTokens);
            int currentPos = nTokens;
            LlamaCppTokenSampler.SamplingConfig config = new LlamaCppTokenSampler.SamplingConfig(temperature, topK, topP, minP, repeatPenalty, frequencyPenalty, presencePenalty, recentTokenCounts);
            while (tokensGenerated < maxTokens) {
                if (Instant.now().isAfter(deadline)) throw new RuntimeException("Generation timed out");
                int newToken = tokenSampler.sampleNextToken(context, 0, config, random);
                if (isEndToken(newToken)) break;
                String piece = binding.tokenToPiece(model, newToken);
                if (tokensGenerated == 0) firstTokenNanos = System.nanoTime();
                result.append(piece);
                if (onTokenPiece != null && piece != null) onTokenPiece.accept(piece);
                tokensGenerated++;
                kvCacheManager.updateAfterGeneration(newToken);
                if (!stopSequences.isEmpty() && maxStopLength > 0) {
                    String matched = checkStopSequence(result.toString(), stopSequences, maxStopLength);
                    if (matched != null) { int cut = result.indexOf(matched, Math.max(0, result.length() - maxStopLength)); if (cut >= 0) { result.setLength(cut); break; } }
                }
                if (effectiveRepeatLastN > 0) { int[] state = kvCacheManager.pushRecentToken(newToken, recentRing, recentRingSize, recentRingIndex, recentTokenCounts, effectiveRepeatLastN); recentRingSize = state[0]; recentRingIndex = state[1]; }
                binding.setBatchSize(batch, 1);
                binding.setBatchToken(batch, 0, newToken, currentPos++, 0, true);
                if (binding.decode(context, batch) != 0) { log.error("Decode failed"); break; }
            }
            kvCacheManager.saveSessionIfExists(context, request);
            metricsRecorder.recordInferenceMetrics(requestStart, promptStartNanos, promptEndNanos, promptEndNanos, firstTokenNanos, nTokens, tokensGenerated);
            return InferenceResponse.builder().requestId(request.getRequestId()).model(manifest.modelId()).content(result.toString()).inputTokens(nTokens).outputTokens(tokensGenerated).tokensUsed(nTokens + tokensGenerated).build();
        } finally { binding.batchFree(batch); }
    }

    private String resolvePrompt(InferenceRequest request) {
        String prompt = (String) request.getParameters().getOrDefault("prompt", "");
        if (request.getMessages() == null || request.getMessages().isEmpty()) return prompt;
        if (isNativeImageRuntime() && (chatTemplate == null || chatTemplate.isBlank())) return buildNativeSafePrompt(request.getMessages());
        String rendered = templateService.render(chatTemplate, request.getMessages());
        return (rendered == null || rendered.isBlank()) ? buildNativeSafePrompt(request.getMessages()) : rendered;
    }
    private boolean isNativeImageRuntime() { return System.getProperty("org.graalvm.nativeimage.imagecode") != null; }
    private String buildNativeSafePrompt(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            String role = msg.getRole() == null ? "user" : msg.getRole().name().toLowerCase();
            sb.append("<|").append(role).append("|>\n").append(msg.getContent() == null ? "" : msg.getContent()).append("\n");
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }
    private String checkStopSequence(String text, List<String> stops, int maxLen) { for (String stop : stops) if (text.contains(stop)) return stop; return null; }
    private List<String> resolveStopSequences(InferenceRequest request) {
        Object stop = request.getParameters().get("stop");
        if (stop == null) return List.of();
        if (stop instanceof String s) return s.isBlank() ? List.of() : List.of(s);
        if (stop instanceof List<?> list) return list.stream().filter(o -> o != null && !o.toString().isBlank()).map(Object::toString).toList();
        return List.of();
    }
    private int maxStopSequenceLength(List<String> stops) { if (stops.isEmpty()) return 0; return stops.stream().mapToInt(s -> s == null ? 0 : s.length()).max().orElse(0); }
    private boolean isEndToken(int tokenId) {
        if (tokenId < 0) return true;
        try { if (binding.isEndOfGeneration(model, tokenId)) return true; } catch (RuntimeException e) { log.debug("EOG check failed: " + e.getMessage()); }
        return tokenId == eosToken;
    }
    private InferenceResponse createEmptyResponse(InferenceRequest request) {
        return InferenceResponse.builder().requestId(request.getRequestId()).model(manifest.modelId()).content("").tokensUsed(0).build();
    }

    private MultimodalData extractMultimodalData(InferenceRequest request) {
        Object multimodal = request.getParameters().get("multimodal");
        if (multimodal instanceof MultimodalData) {
            return (MultimodalData) multimodal;
        }
        return null;
    }

    private int processMultimodalBatch(MemorySegment batch, MultimodalData multimodalData, 
            int[] promptTokens, int nTokens, int reusePrefix, int maxBatch, Instant deadline) {
        int processed = reusePrefix;
        int embdIndex = 0;
        float[][] embeddings = multimodalData.getEmbeddings();
        int[] embdPos = multimodalData.getEmbeddingPositions();
        
        // Process multimodal embeddings first
        while (embdIndex < embeddings.length && processed < nTokens) {
            if (Instant.now().isAfter(deadline)) throw new RuntimeException("Multimodal prompt timed out");
            
            // Check if current position matches embedding position
            if (embdPos != null && embdIndex < embdPos.length && processed == embdPos[embdIndex]) {
                // Set batch with multimodal embedding
                MemorySegment embdSegment = createEmbeddingSegment(embeddings[embdIndex]);
                binding.setBatchMultimodalEmbd(batch, 0, embdSegment, processed, 0, true);
                binding.setBatchSize(batch, 1);
                if (binding.decode(context, batch) != 0) throw new RuntimeException("Multimodal decode failed");
                embdIndex++;
                processed++;
            } else {
                // Process regular token
                int chunk = Math.min(maxBatch, nTokens - processed);
                binding.setBatchSize(batch, chunk);
                for (int i = 0; i < chunk; i++) {
                    binding.setBatchToken(batch, i, promptTokens[processed + i], processed + i, 0, i == chunk - 1);
                }
                if (binding.decode(context, batch) != 0) throw new RuntimeException("Prompt evaluation failed");
                processed += chunk;
            }
        }
        
        // Process remaining tokens
        while (processed < nTokens) {
            if (Instant.now().isAfter(deadline)) throw new RuntimeException("Prompt timed out");
            int chunk = Math.min(maxBatch, nTokens - processed);
            binding.setBatchSize(batch, chunk);
            for (int i = 0; i < chunk; i++) binding.setBatchToken(batch, i, promptTokens[processed + i], processed + i, 0, i == chunk - 1);
            if (binding.decode(context, batch) != 0) throw new RuntimeException("Prompt evaluation failed");
            processed += chunk;
        }
        
        return processed;
    }

    private MemorySegment createEmbeddingSegment(float[] embedding) {
        int nEmbd = embedding.length;
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(nEmbd * 4); // 4 bytes per float
        for (int i = 0; i < nEmbd; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, embedding[i]);
        }
        return segment;
    }
}
