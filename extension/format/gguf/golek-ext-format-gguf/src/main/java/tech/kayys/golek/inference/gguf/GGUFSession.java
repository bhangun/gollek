package tech.kayys.golek.inference.gguf;

import org.jboss.logging.Logger;
import java.lang.foreign.MemorySegment;
import java.time.Instant;
import java.util.Arrays;

/**
 * Represents a stateful GGUF inference session with KV cache reuse.
 * Enables efficient multi-turn conversations by retaining the native context.
 */
public class GGUFSession {

    private static final Logger log = Logger.getLogger(GGUFSession.class);

    private final String sessionId;
    private final String modelPath;
    private final LlamaCppBinding binding;

    // Native handles (owned by this session)
    private MemorySegment model;
    private MemorySegment context;

    // State tracking
    private int currentPosition = 0;
    private int[] tokenHistory = new int[0];
    private Instant lastAccessTime;
    private volatile boolean closed = false;

    // Model metadata
    private int eosToken;
    private int bosToken;
    private String chatTemplate;

    public GGUFSession(String sessionId, String modelPath, LlamaCppBinding binding) {
        this.sessionId = sessionId;
        this.modelPath = modelPath;
        this.binding = binding;
        this.lastAccessTime = Instant.now();
    }

    /**
     * Initialize the native model and context for this session.
     */
    public void initialize() {
        if (model != null) {
            return; // Already initialized
        }

        log.debugf("Initializing session %s for model %s", sessionId, modelPath);

        MemorySegment modelParams = binding.getDefaultModelParams();
        this.model = binding.loadModel(modelPath, modelParams);

        MemorySegment contextParams = binding.getDefaultContextParams();
        this.context = binding.createContext(model, contextParams);

        this.eosToken = binding.getEosToken(model);
        this.bosToken = binding.getBosToken(model);
        this.chatTemplate = binding.getModelMetadata(model, "tokenizer.chat_template");

        log.debugf("Session %s initialized, context size: %d", sessionId, binding.getContextSize(context));
    }

    /**
     * Calculate which tokens are new (not already in KV cache).
     * Returns delta tokens that need to be evaluated.
     */
    public int[] getDeltaTokens(int[] newTokens) {
        touch();

        // Find common prefix length
        int commonLen = 0;
        int minLen = Math.min(tokenHistory.length, newTokens.length);
        for (int i = 0; i < minLen; i++) {
            if (tokenHistory[i] == newTokens[i]) {
                commonLen++;
            } else {
                break;
            }
        }

        // If we need to backtrack, reset KV cache position
        if (commonLen < tokenHistory.length) {
            log.debugf("Session %s: backtracking from %d to %d", sessionId, currentPosition, commonLen);
            currentPosition = commonLen;
            tokenHistory = Arrays.copyOf(tokenHistory, commonLen);
        }

        // Return only the new tokens
        if (commonLen < newTokens.length) {
            return Arrays.copyOfRange(newTokens, commonLen, newTokens.length);
        }
        return new int[0];
    }

    /**
     * Add generated tokens to the session history.
     */
    public void addGeneratedTokens(int[] tokens) {
        touch();
        int newLen = tokenHistory.length + tokens.length;
        int[] newHistory = Arrays.copyOf(tokenHistory, newLen);
        System.arraycopy(tokens, 0, newHistory, tokenHistory.length, tokens.length);
        tokenHistory = newHistory;
        currentPosition = newLen;
    }

    /**
     * Get the current position in KV cache.
     */
    public int getCurrentPosition() {
        return currentPosition;
    }

    /**
     * Reset the session (clear KV cache).
     */
    public void reset() {
        log.debugf("Resetting session %s", sessionId);
        currentPosition = 0;
        tokenHistory = new int[0];
        // Note: We don't free the context here, just reset position
        // llama.cpp can reuse the same context by setting batch position to 0
    }

    /**
     * Check if session has expired based on timeout.
     */
    public boolean isExpired(long timeoutMinutes) {
        return Instant.now().isAfter(lastAccessTime.plusSeconds(timeoutMinutes * 60));
    }

    /**
     * Update last access time.
     */
    public void touch() {
        lastAccessTime = Instant.now();
    }

    /**
     * Close and release native resources.
     */
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        log.debugf("Closing session %s", sessionId);

        if (context != null) {
            binding.freeContext(context);
            context = null;
        }
        if (model != null) {
            binding.freeModel(model);
            model = null;
        }
    }

    // Getters
    public String getSessionId() {
        return sessionId;
    }

    public String getModelPath() {
        return modelPath;
    }

    public MemorySegment getModel() {
        return model;
    }

    public MemorySegment getContext() {
        return context;
    }

    public int getEosToken() {
        return eosToken;
    }

    public int getBosToken() {
        return bosToken;
    }

    public String getChatTemplate() {
        return chatTemplate;
    }

    public Instant getLastAccessTime() {
        return lastAccessTime;
    }

    public boolean isClosed() {
        return closed;
    }

    public int[] getTokenHistory() {
        return tokenHistory.clone();
    }
}
