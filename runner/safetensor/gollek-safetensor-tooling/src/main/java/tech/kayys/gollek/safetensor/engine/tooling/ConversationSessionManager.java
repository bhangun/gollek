/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ConversationSessionManager.java
 * ─────────────────────────────────
 * Persists per-conversation KV cache state across multiple HTTP requests.
 *
 * Problem
 * ═══════
 * Without session persistence, every /v1/chat/completions request re-prefills
 * the entire conversation history from scratch.  For a 10-turn chat with an
 * average of 200 tokens per turn, turn 10 requires a 1800-token prefill —
 * 9× the work of just processing the new message.
 *
 * Solution
 * ════════
 * Store the encoded token IDs AND the corresponding KV cache blocks for each
 * conversation.  On the next turn:
 *   1. Encode only the NEW messages (assistant response + new user message).
 *   2. Re-use the existing KV blocks for the prefix — skip prefill for them.
 *   3. Run prefill ONLY for the new tokens.
 *   4. Decode as normal.
 *
 * This reduces prefill cost from O(total_tokens) to O(new_tokens_this_turn).
 *
 * Session lifecycle
 * ═════════════════
 *   CREATE  — on first request with a session_id (or auto-generated)
 *   UPDATE  — after each turn: append new tokens + KV blocks
 *   EXPIRE  — after idleTimeoutMinutes of inactivity (default: 30)
 *   DELETE  — on explicit /v1/sessions/{id} DELETE, or on model unload
 *
 * KV block ownership
 * ══════════════════
 * The session owns a reference to the KV blocks from the PagedKVCache pool.
 * When the session expires, the blocks are returned to the pool.
 * The PagedKVSessionRegistry tracks these for eviction under memory pressure.
 *
 * Session ID
 * ══════════
 * Passed as request.sessionId or X-Session-Id HTTP header.
 * If absent, a new session is created and the ID is returned in the response
 * metadata as "session_id".
 *
 * Conversation reconstruction
 * ═══════════════════════════
 * When a session is found, we don't re-encode the full message history.
 * Instead we track cumulative token IDs and KV block offsets, and only
 * process the delta since last turn.
 */
package tech.kayys.gollek.safetensor.engine.tooling;

import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager.KVCacheSession;
import tech.kayys.gollek.safetensor.engine.generation.paged.PagedKVSessionRegistry;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Per-conversation KV cache session manager for multi-turn inference.
 *
 * <p>
 * Usage in {@code DirectInferenceEngine.generate()}:
 * 
 * <pre>{@code
 * // Look up existing session
 * Optional<ConversationSession> session = sessionMgr.find(sessionId, modelKey);
 *
 * if (session.isPresent()) {
 *     // Only encode new messages (delta since last turn)
 *     int[] deltaIds = tokenizer.encode(newMessagesOnly, true);
 *     // Prepend session's accumulated IDs for context
 *     int[] allIds = session.get().extendWith(deltaIds);
 *     // Run forward pass starting from the cached position
 *     logits = forwardPass.prefill(deltaIds, weights, config, arch, session.get().kvCache());
 *     session.get().advance(deltaIds.length);
 * } else {
 *     // Fresh start — prefill everything
 *     int[] allIds = tokenizer.encode(allMessages, true);
 *     logits = forwardPass.prefill(allIds, weights, config, arch, newKvCache);
 *     session = sessionMgr.create(sessionId, modelKey, allIds, newKvCache);
 * }
 * }</pre>
 */
@ApplicationScoped
public class ConversationSessionManager {

    private static final Logger log = Logger.getLogger(ConversationSessionManager.class);

    @ConfigProperty(name = "gollek.session.idle-timeout-minutes", defaultValue = "30")
    int idleTimeoutMinutes;

    @ConfigProperty(name = "gollek.session.max-sessions", defaultValue = "1000")
    int maxSessions;

    @ConfigProperty(name = "gollek.session.max-tokens-per-session", defaultValue = "32768")
    int maxTokensPerSession;

    @Inject
    KVCacheManager kvCacheManager;
    @Inject
    PagedKVSessionRegistry sessionRegistry;

    /** sessionId → ConversationSession. */
    private final ConcurrentHashMap<String, ConversationSession> sessions = new ConcurrentHashMap<>();

    private final ScheduledExecutorService cleaner = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "gollek-session-cleaner");
        t.setDaemon(true);
        return t;
    });

    private final AtomicInteger totalCreated = new AtomicInteger(0);
    private final AtomicInteger totalExpired = new AtomicInteger(0);

    // ─────────────────────────────────────────────────────────────────────────

    @jakarta.annotation.PostConstruct
    void start() {
        cleaner.scheduleAtFixedRate(this::evictExpired,
                idleTimeoutMinutes, idleTimeoutMinutes, TimeUnit.MINUTES);
        log.infof("ConversationSessionManager: started (timeout=%dm, max=%d)",
                idleTimeoutMinutes, maxSessions);
    }

    @PreDestroy
    void stop() {
        cleaner.shutdownNow();
        sessions.values().forEach(ConversationSession::close);
        sessions.clear();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Session API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Find an existing session by ID and model key.
     *
     * @param sessionId unique session identifier
     * @param modelKey  key identifying which model is loaded
     * @return the session if found, valid, and for the same model
     */
    public Optional<ConversationSession> find(String sessionId, String modelKey) {
        if (sessionId == null || sessionId.isBlank())
            return Optional.empty();
        ConversationSession s = sessions.get(sessionId);
        if (s == null)
            return Optional.empty();
        if (!s.modelKey().equals(modelKey)) {
            log.debugf("Session %s found but model changed — discarding", sessionId);
            sessions.remove(sessionId);
            s.close();
            return Optional.empty();
        }
        if (isExpired(s)) {
            sessions.remove(sessionId);
            s.close();
            return Optional.empty();
        }
        s.touch();
        return Optional.of(s);
    }

    /**
     * Create a new conversation session after completing a fresh prefill.
     *
     * @param sessionId    session identifier (from request or auto-generated)
     * @param modelKey     model key from DirectInferenceEngine
     * @param prefixTokens all token IDs that have been prefilled
     * @param kvCache      the KV cache session populated by prefill
     * @return the created session
     */
    public ConversationSession create(String sessionId, String modelKey,
            int[] prefixTokens,
            KVCacheSession kvCache) {
        if (sessions.size() >= maxSessions) {
            evictOldest(sessions.size() - maxSessions + 1);
        }

        ConversationSession session = new ConversationSession(
                sessionId, modelKey, prefixTokens, kvCache, Instant.now());
        sessions.put(sessionId, session);
        sessionRegistry.register(sessionId, null,
                PagedKVSessionRegistry.SessionPriority.NORMAL);
        totalCreated.incrementAndGet();

        log.debugf("ConversationSession created: id=%s model=%s tokens=%d",
                sessionId, modelKey, prefixTokens.length);
        return session;
    }

    /**
     * Extend an existing session with new token IDs after a completed turn.
     *
     * @param sessionId session to update
     * @param newTokens token IDs generated this turn (assistant + user combined)
     */
    public void extend(String sessionId, int[] newTokens) {
        ConversationSession s = sessions.get(sessionId);
        if (s == null)
            return;

        if (s.totalTokens() + newTokens.length > maxTokensPerSession) {
            log.infof("Session %s exceeded max tokens (%d) — truncating prefix",
                    sessionId, maxTokensPerSession);
            // Sliding window: drop oldest half of the context
            s.truncateToHalf();
        }

        s.extend(newTokens);
        s.touch();
    }

    /**
     * Explicitly close and remove a session.
     */
    public void close(String sessionId) {
        ConversationSession s = sessions.remove(sessionId);
        if (s != null) {
            sessionRegistry.deregister(sessionId);
            s.close();
        }
    }

    /** Number of active sessions. */
    public int activeSessions() {
        return sessions.size();
    }

    /** Total sessions created since startup. */
    public int totalCreated() {
        return totalCreated.get();
    }

    /** Total sessions expired/evicted since startup. */
    public int totalExpired() {
        return totalExpired.get();
    }

    // ─────────────────────────────────────────────────────────────────────────

    private void evictExpired() {
        Instant cutoff = Instant.now().minusSeconds(idleTimeoutMinutes * 60L);
        int evicted = 0;
        Iterator<Map.Entry<String, ConversationSession>> it = sessions.entrySet().iterator();
        while (it.hasNext()) {
            ConversationSession s = it.next().getValue();
            if (s.lastAccessTime().isBefore(cutoff)) {
                it.remove();
                sessionRegistry.deregister(s.sessionId());
                s.close();
                evicted++;
            }
        }
        if (evicted > 0) {
            totalExpired.addAndGet(evicted);
            log.infof("ConversationSessionManager: evicted %d expired sessions", evicted);
        }
    }

    private void evictOldest(int count) {
        sessions.entrySet().stream()
                .sorted(Map.Entry.comparingByValue(
                        Comparator.comparing(ConversationSession::lastAccessTime)))
                .limit(count)
                .forEach(e -> {
                    sessions.remove(e.getKey());
                    sessionRegistry.deregister(e.getKey());
                    e.getValue().close();
                });
        totalExpired.addAndGet(count);
    }

    private boolean isExpired(ConversationSession s) {
        return s.lastAccessTime().isBefore(
                Instant.now().minusSeconds(idleTimeoutMinutes * 60L));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ConversationSession value type
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * A single multi-turn conversation session.
     *
     * <p>
     * Holds the accumulated token IDs and the associated KV cache session.
     */
    public static final class ConversationSession implements AutoCloseable {

        private final String sessionId;
        private final String modelKey;
        private volatile KVCacheSession kvCache;
        private volatile Instant lastAccess;
        private volatile int[] tokenIds; // cumulative token IDs
        private volatile boolean closed = false;

        ConversationSession(String sessionId, String modelKey,
                int[] initialTokens,
                KVCacheSession kvCache,
                Instant createdAt) {
            this.sessionId = sessionId;
            this.modelKey = modelKey;
            this.kvCache = kvCache;
            this.tokenIds = Arrays.copyOf(initialTokens, initialTokens.length);
            this.lastAccess = createdAt;
        }

        public String sessionId() {
            return sessionId;
        }

        public String modelKey() {
            return modelKey;
        }

        public KVCacheSession kvCache() {
            return kvCache;
        }

        public Instant lastAccessTime() {
            return lastAccess;
        }

        public int[] tokenIds() {
            return Arrays.copyOf(tokenIds, tokenIds.length);
        }

        public int totalTokens() {
            return tokenIds.length;
        }

        void touch() {
            this.lastAccess = Instant.now();
        }

        /**
         * Extend the session with new token IDs (from a completed decode step).
         */
        void extend(int[] newTokens) {
            int[] combined = new int[tokenIds.length + newTokens.length];
            System.arraycopy(tokenIds, 0, combined, 0, tokenIds.length);
            System.arraycopy(newTokens, 0, combined, tokenIds.length, newTokens.length);
            this.tokenIds = combined;
        }

        /**
         * Truncate the accumulated token IDs to half their current length,
         * keeping the most recent tokens (sliding window context management).
         */
        void truncateToHalf() {
            int half = tokenIds.length / 2;
            this.tokenIds = Arrays.copyOfRange(tokenIds, half, tokenIds.length);
            // Note: KV cache blocks for dropped tokens will be reclaimed on next eviction
        }

        /**
         * Build the combined token array for a new turn:
         * existing prefix + new delta tokens.
         */
        public int[] extendWith(int[] deltaTokens) {
            int[] combined = new int[tokenIds.length + deltaTokens.length];
            System.arraycopy(tokenIds, 0, combined, 0, tokenIds.length);
            System.arraycopy(deltaTokens, 0, combined, tokenIds.length, deltaTokens.length);
            return combined;
        }

        @Override
        public synchronized void close() {
            if (!closed) {
                closed = true;
                if (kvCache != null) {
                    try {
                        kvCache.close();
                    } catch (Exception ignored) {
                    }
                    kvCache = null;
                }
            }
        }

        public boolean isClosed() {
            return closed;
        }
    }
}
