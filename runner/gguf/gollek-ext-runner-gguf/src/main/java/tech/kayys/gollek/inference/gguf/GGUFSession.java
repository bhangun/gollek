package tech.kayys.gollek.inference.gguf;

import org.jboss.logging.Logger;
import java.lang.foreign.MemorySegment;
import java.time.Instant;
import java.util.Arrays;

/**
 * Stateful GGUF inference session backed by a native llama.cpp context.
 *
 * <p>Maintains a KV-cache across multiple turns of a conversation, so only
 * the <em>delta</em> tokens (those not already in the cache) need to be
 * evaluated on each call. This significantly reduces latency for multi-turn
 * chat workloads.
 *
 * <p>Instances are created and pooled by {@link GGUFSessionManager}. Callers
 * must not close a session directly — return it to the manager via
 * {@code releaseSession} instead.
 *
 * <p>Thread-safety: a session is not thread-safe and must be used by at most
 * one inference request at a time. The session manager enforces this via its
 * pool.
 *
 * @see GGUFSessionManager
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

    /**
     * Creates a new session. The session is not usable until {@link #initialize()} is called.
     *
     * @param sessionId  unique identifier for this session
     * @param modelPath  absolute path to the {@code .gguf} model file
     * @param binding    FFM binding to the native llama.cpp library
     */
    public GGUFSession(String sessionId, String modelPath, LlamaCppBinding binding) {
        this.sessionId = sessionId;
        this.modelPath = modelPath;
        this.binding = binding;
        this.lastAccessTime = Instant.now();
    }

    /**
     * Loads the model and creates the native llama.cpp context.
     * Idempotent — safe to call multiple times; subsequent calls are no-ops.
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
     * Computes the delta tokens that are not yet present in the KV cache.
     *
     * <p>Finds the longest common prefix between the current token history and
     * {@code newTokens}. If the new sequence diverges before the end of the
     * history, the KV cache position is rolled back to the divergence point.
     *
     * @param newTokens the full token sequence for the new request
     * @return the suffix of {@code newTokens} that must be evaluated (may be empty)
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
     * Appends generated tokens to the session's token history and advances the KV cache position.
     *
     * @param tokens the tokens produced by the last generation step
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
     * Returns the current position in the KV cache (number of tokens evaluated so far).
     *
     * @return current KV cache position
     */
    public int getCurrentPosition() {
        return currentPosition;
    }

    /**
     * Resets the session's token history and KV cache position to zero.
     * The native context is retained and reused — only the logical position is cleared.
     */
    public void reset() {
        log.debugf("Resetting session %s", sessionId);
        currentPosition = 0;
        tokenHistory = new int[0];
        // Note: We don't free the context here, just reset position
        // llama.cpp can reuse the same context by setting batch position to 0
    }

    /**
     * Returns {@code true} if this session has not been accessed within the given timeout.
     *
     * @param timeoutMinutes idle timeout in minutes
     * @return {@code true} if the session should be evicted
     */
    public boolean isExpired(long timeoutMinutes) {
        return Instant.now().isAfter(lastAccessTime.plusSeconds(timeoutMinutes * 60));
    }

    /**
     * Updates the last-access timestamp to the current time.
     * Called automatically by {@link #getDeltaTokens} and {@link #addGeneratedTokens}.
     */
    public void touch() {
        lastAccessTime = Instant.now();
    }

    /**
     * Releases the native llama.cpp context and model, freeing all associated memory.
     * Idempotent — safe to call multiple times.
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

    // ── Getters ───────────────────────────────────────────────────────────────

    /** @return unique session identifier */
    public String getSessionId() { return sessionId; }

    /** @return absolute path to the {@code .gguf} model file */
    public String getModelPath() { return modelPath; }

    /** @return the native llama.cpp model handle, or {@code null} if not initialized */
    public MemorySegment getModel() { return model; }

    /** @return the native llama.cpp context handle, or {@code null} if not initialized */
    public MemorySegment getContext() { return context; }

    /** @return the EOS (end-of-sequence) token ID for this model */
    public int getEosToken() { return eosToken; }

    /** @return the BOS (beginning-of-sequence) token ID for this model */
    public int getBosToken() { return bosToken; }

    /** @return the chat template string from model metadata, or {@code null} if absent */
    public String getChatTemplate() { return chatTemplate; }

    /** @return the timestamp of the last access to this session */
    public Instant getLastAccessTime() { return lastAccessTime; }

    /** @return {@code true} if this session has been closed */
    public boolean isClosed() { return closed; }

    /** @return a defensive copy of the current token history */
    public int[] getTokenHistory() { return tokenHistory.clone(); }
}
