package tech.kayys.wayang.engine;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;

import io.smallrye.mutiny.Multi;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.jboss.resteasy.reactive.RestStreamElementType;
import java.util.Map;

public class LlamaInferenceService {
    
    private static final Logger LOG = Logger.getLogger(LlamaInferenceService.class);
    private static final int MAX_CONTEXTS = 100;
    private static final long CONTEXT_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
    
    @Inject
    LlamaConfig config;
    
    private LlamaCppBinding binding;
    private Arena arena;
    private MemorySegment model;
    private MemorySegment context;
    private int vocabSize;
    private int bosToken;
    private int eosToken;
    private int nlToken;
    private String modelDesc;
    
    private final Map<String, ConversationContext> conversations = new ConcurrentHashMap<>();
    private final AtomicInteger requestCounter = new AtomicInteger(0);
    
    @PostConstruct
    void initialize() {
        LOG.info("Initializing Llama.cpp inference service...");
        
        try {
            arena = Arena.ofShared();
            binding = new LlamaCppBinding(config.libraryPath());
            binding.backendInit();
            
            LOG.infof("Loading model: %s", config.modelPath());
            model = binding.loadModel(arena, config.modelPath(), config.gpuLayers());
            
            if (model.address() == 0) {
                throw new RuntimeException("Failed to load model");
            }
            
            vocabSize = binding.nVocab(model);
            bosToken = binding.tokenBos(model);
            eosToken = binding.tokenEos(model);
            nlToken = binding.tokenNl(model);
            modelDesc = binding.modelDesc(arena, model);
            
            LOG.infof("Model loaded: %s, vocab_size=%d", modelDesc, vocabSize);
            
            context = binding.createContext(arena, model, 
                config.contextSize(), config.batchSize(), config.threads(),
                config.seed(), config.ropeFreqBase(), config.ropeFreqScale());
            
            if (context.address() == 0) {
                throw new RuntimeException("Failed to create context");
            }
            
            LOG.infof("Context created: ctx_size=%d, batch=%d, threads=%d", 
                config.contextSize(), config.batchSize(), config.threads());
            
            // Start cleanup task
            startCleanupTask();
            
            LOG.info("Llama.cpp inference service initialized successfully");
            
        } catch (Throwable e) {
            LOG.error("Failed to initialize inference service", e);
            throw new RuntimeException(e);
        }
    }
    
    @PreDestroy
    void cleanup() {
        LOG.info("Shutting down Llama.cpp inference service...");
        
        try {
            conversations.clear();
            
            if (context != null && context.address() != 0) {
                binding.freeContext(context);
            }
            
            if (model != null && model.address() != 0) {
                binding.freeModel(model);
            }
            
            binding.backendFree();
            
            if (arena != null) {
                arena.close();
            }
            
            LOG.info("Shutdown complete");
        } catch (Throwable e) {
            LOG.error("Error during shutdown", e);
        }
    }
    
    public ChatResponse generateChat(ChatRequest request) {
        String id = "chatcmpl-" + requestCounter.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;
        
        try {
            String prompt = formatChatPrompt(request.messages());
            SamplingParams samplingParams = createSamplingParams(request);
            
            int[] promptTokens = binding.tokenize(arena, model, prompt, true, false);
            LOG.infof("Request %s: prompt_tokens=%d", id, promptTokens.length);
            
            List<String> pieces = new ArrayList<>();
            int completionTokens = generateTokens(promptTokens, request.maxTokens(), 
                samplingParams, request.stop(), pieces::add);
            
            String content = String.join("", pieces);
            
            return new ChatResponse(
                id, "chat.completion", created, modelDesc,
                new ChatResponse.Choice(0, 
                    new ChatMessage("assistant", content),
                    "stop"),
                new ChatResponse.Usage(promptTokens.length, completionTokens, 
                    promptTokens.length + completionTokens)
            );
            
        } catch (Throwable e) {
            LOG.error("Generation failed", e);
            throw new RuntimeException("Generation failed: " + e.getMessage(), e);
        }
    }
    
    public Multi<String> generateChatStream(ChatRequest request) {
        String id = "chatcmpl-" + requestCounter.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;
        
        return Multi.createFrom().emitter(emitter -> {
            try {
                String prompt = formatChatPrompt(request.messages());
                SamplingParams samplingParams = createSamplingParams(request);
                
                int[] promptTokens = binding.tokenize(arena, model, prompt, true, false);
                LOG.infof("Stream request %s: prompt_tokens=%d", id, promptTokens.length);
                
                // Send initial chunk
                emitter.emit(formatStreamChunk(id, created, "assistant", null, null));
                
                generateTokens(promptTokens, request.maxTokens(), samplingParams, 
                    request.stop(), piece -> {
                        emitter.emit(formatStreamChunk(id, created, null, piece, null));
                    });
                
                // Send final chunk
                emitter.emit(formatStreamChunk(id, created, null, null, "stop"));
                emitter.emit("data: [DONE]\n\n");
                emitter.complete();
                
            } catch (Throwable e) {
                LOG.error("Stream generation failed", e);
                emitter.fail(e);
            }
        });
    }
    
    private synchronized int generateTokens(int[] promptTokens, int maxTokens,
                                           SamplingParams samplingParams, List<String> stopStrings,
                                           java.util.function.Consumer<String> callback) throws Throwable {
        
        Arena sessionArena = Arena.ofConfined();
        
        try {
            // Clear KV cache
            binding.kvCacheClear(context);
            
            // Process prompt
            int nPrompt = promptTokens.length;
            for (int i = 0; i < nPrompt; i += config.batchSize()) {
                int batchSize = Math.min(config.batchSize(), nPrompt - i);
                int[] batchTokens = Arrays.copyOfRange(promptTokens, i, i + batchSize);
                int[] positions = new int[batchSize];
                boolean[] logits = new boolean[batchSize];
                
                for (int j = 0; j < batchSize; j++) {
                    positions[j] = i + j;
                    logits[j] = (i + j == nPrompt - 1); // Only last token needs logits
                }
                
                MemorySegment batch = LlamaStructs.createBatch(sessionArena, batchTokens, positions, logits);
                
                int ret = binding.decode(context, batch);
                if (ret != 0) {
                    throw new RuntimeException("Decode failed: " + ret);
                }
            }
            
            // Generate tokens
            Sampler sampler = new Sampler(samplingParams);
            Set<Integer> stopTokens = new HashSet<>();
            stopTokens.add(eosToken);
            
            StringBuilder currentText = new StringBuilder();
            int nGenerated = 0;
            int pos = nPrompt;
            
            for (int i = 0; i < maxTokens; i++) {
                MemorySegment logits = binding.getLogitsIth(context, 0);
                int nextToken = sampler.sample(sessionArena, logits, vocabSize, stopTokens);
                
                if (nextToken == eosToken) {
                    break;
                }
                
                String piece = binding.tokenToString(sessionArena, model, nextToken);
                currentText.append(piece);
                nGenerated++;
                
                // Check stop strings
                boolean shouldStop = false;
                if (stopStrings != null && !stopStrings.isEmpty()) {
                    String fullText = currentText.toString();
                    for (String stop : stopStrings) {
                        if (fullText.endsWith(stop)) {
                            piece = piece.substring(0, piece.length() - stop.length());
                            shouldStop = true;
                            break;
                        }
                    }
                }
                
                if (!piece.isEmpty()) {
                    callback.accept(piece);
                }
                
                if (shouldStop) {
                    break;
                }
                
                // Decode next token
                int[] nextTokens = {nextToken};
                int[] positions = {pos++};
                boolean[] logitsFlags = {true};
                
                MemorySegment batch = LlamaStructs.createBatch(sessionArena, nextTokens, positions, logitsFlags);
                int ret = binding.decode(context, batch);
                if (ret != 0) {
                    throw new RuntimeException("Decode failed: " + ret);
                }
            }
            
            return nGenerated;
            
        } finally {
            sessionArena.close();
        }
    }
    
    private String formatChatPrompt(List<ChatMessage> messages) {
        // Llama-2-chat format
        StringBuilder prompt = new StringBuilder();
        prompt.append("<s>");
        
        for (ChatMessage msg : messages) {
            if ("system".equals(msg.role())) {
                prompt.append("[INST] <<SYS>>\n")
                      .append(msg.content())
                      .append("\n<</SYS>>\n\n");
            } else if ("user".equals(msg.role())) {
                prompt.append("[INST] ")
                      .append(msg.content())
                      .append(" [/INST]");
            } else if ("assistant".equals(msg.role())) {
                prompt.append(" ")
                      .append(msg.content())
                      .append(" </s><s>");
            }
        }
        
        return prompt.toString();
    }
    
    private SamplingParams createSamplingParams(ChatRequest request) {
        var defaultSampling = config.sampling();
        
        return new SamplingParams(
            request.temperature() != null ? request.temperature() : defaultSampling.temperature(),
            request.topK() != null ? request.topK() : defaultSampling.topK(),
            request.topP() != null ? request.topP() : defaultSampling.topP(),
            request.minP() != null ? request.minP() : defaultSampling.minP(),
            request.repeatPenalty() != null ? request.repeatPenalty() : defaultSampling.repeatPenalty(),
            defaultSampling.repeatLastN(),
            request.presencePenalty() != null ? request.presencePenalty() : defaultSampling.presencePenalty(),
            request.frequencyPenalty() != null ? request.frequencyPenalty() : defaultSampling.frequencyPenalty(),
            config.seed()
        );
    }
    
    private String formatStreamChunk(String id, long created, String role, String content, String finishReason) {
        var delta = role != null ? 
            new StreamChunk.Delta(role, null) : 
            new StreamChunk.Delta(null, content);
        
        var chunk = new StreamChunk(id, "chat.completion.chunk", created, modelDesc, delta, finishReason);
        
        try {
            return "data: " + new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(chunk) + "\n\n";
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    private void startCleanupTask() {
        new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(60000); // Check every minute
                    cleanupOldContexts();
                } catch (InterruptedException e) {
                    break;
                }
            }
        }, "context-cleanup").start();
    }
    
    private void cleanupOldContexts() {
        long now = System.currentTimeMillis();
        conversations.entrySet().removeIf(entry -> 
            now - entry.getValue().getLastAccessedAt() > CONTEXT_TIMEOUT_MS
        );
    }
}
