package tech.kayys.gollek.inference.gguf;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.model.ModelManifest;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.nio.file.Files;
import java.io.IOException;
import java.util.Arrays;

/**
 * Manages KV cache reuse across requests, session persistence, and token history tracking.
 * Supports loading/saving session state for conversation continuity.
 */
public class LlamaCppKVCacheManager {

    private static final Logger log = Logger.getLogger(LlamaCppKVCacheManager.class);

    private final LlamaCppBinding binding;
    private final GGUFProviderConfig providerConfig;
    private final ModelManifest manifest;

    private int[] kvTokenHistory = new int[0];
    private int kvTokenCount = 0;

    private final java.util.Map<String, int[]> tokenCache = Collections.synchronizedMap(
            new java.util.LinkedHashMap<>(64, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(java.util.Map.Entry<String, int[]> eldest) {
                    return size() > 64;
                }
            });

    public LlamaCppKVCacheManager(LlamaCppBinding binding, GGUFProviderConfig providerConfig, ModelManifest manifest) {
        this.binding = binding;
        this.providerConfig = providerConfig;
        this.manifest = manifest;
    }

    /**
     * Compute how many tokens from the current prompt can reuse existing KV cache.
     */
    public int computeReusePrefix(int[] promptTokens, int nTokens) {
        if (kvTokenCount == 0 || kvTokenHistory.length == 0) {
            return 0;
        }
        int minLen = Math.min(kvTokenCount, nTokens);
        int common = 0;
        for (int i = 0; i < minLen; i++) {
            if (kvTokenHistory[i] == promptTokens[i]) {
                common++;
            } else {
                break;
            }
        }
        if (common == kvTokenCount && nTokens > kvTokenCount) {
            return common;
        }
        return 0;
    }

    /**
     * Reset the KV cache and token history.
     */
    public void resetKvCache(MemorySegment context) {
        if (context != null && !context.equals(MemorySegment.NULL)) {
            binding.kvCacheClear(context);
        }
        kvTokenHistory = new int[0];
        kvTokenCount = 0;
    }

    /**
     * Update token history with newly generated tokens.
     */
    public int[] appendToken(int[] history, int count, int token) {
        if (count < history.length) {
            history[count] = token;
            return history;
        }
        int newSize = Math.max(8, history.length * 2);
        if (newSize <= count) {
            newSize = count + 1;
        }
        int[] expanded = Arrays.copyOf(history, newSize);
        expanded[count] = token;
        return expanded;
    }

    /**
     * Tokenize with LRU caching for repeated prompts.
     */
    public int[] tokenizeWithCache(MemorySegment model, String prompt, boolean addBos) {
        if (prompt == null) {
            return new int[0];
        }
        if (prompt.length() > 8192) {
            return binding.tokenize(model, prompt, addBos, true);
        }
        String key = addBos + "|" + prompt;
        int[] cached = tokenCache.get(key);
        if (cached != null) {
            return cached;
        }
        int[] tokens = binding.tokenize(model, prompt, addBos, true);
        tokenCache.put(key, tokens);
        return tokens;
    }

    /**
     * Load session state from disk if session persistence is enabled.
     */
    public void loadSessionIfExists(MemorySegment context, InferenceRequest request) {
        boolean sessionPersist = Boolean.parseBoolean(String.valueOf(
                request.getParameters().getOrDefault("gguf.session.persist", "false")));
        if (!sessionPersist) {
            return;
        }

        Path sessionPath = resolveSessionPath(request);
        if (sessionPath == null || !Files.exists(sessionPath)) {
            return;
        }

        try {
            int[] loadedTokens = binding.loadSession(context, sessionPath, providerConfig.maxContextTokens());
            if (loadedTokens.length > 0) {
                kvTokenHistory = loadedTokens;
                kvTokenCount = loadedTokens.length;
                log.debugf("Loaded session from %s with %d tokens", sessionPath, kvTokenCount);
            }
        } catch (Exception e) {
            log.warnf("Failed to load session from %s: %s", sessionPath, e.getMessage());
        }
    }

    /**
     * Save session state to disk if session persistence is enabled.
     */
    public void saveSessionIfExists(MemorySegment context, InferenceRequest request) {
        boolean sessionPersist = Boolean.parseBoolean(String.valueOf(
                request.getParameters().getOrDefault("gguf.session.persist", "false")));
        if (!sessionPersist) {
            return;
        }

        Path sessionPath = resolveSessionPath(request);
        if (sessionPath == null) {
            return;
        }

        try {
            Files.createDirectories(sessionPath.getParent());
            binding.saveSession(context, sessionPath, kvTokenHistory, kvTokenCount);
            log.debugf("Saved session to %s with %d tokens", sessionPath, kvTokenCount);
        } catch (IOException e) {
            log.warnf("Failed to persist session to %s: %s", sessionPath, e.getMessage());
        }
    }

    /**
     * Update token history with prompt tokens after prompt evaluation.
     */
    public void updateAfterPrompt(int[] promptTokens, int nTokens) {
        kvTokenHistory = Arrays.copyOf(promptTokens, nTokens);
        kvTokenCount = nTokens;
    }

    /**
     * Update token history with a generated token.
     */
    public void updateAfterGeneration(int tokenId) {
        if (tokenId >= 0) {
            kvTokenHistory = appendToken(kvTokenHistory, kvTokenCount++, tokenId);
        }
    }

    /**
     * Get current token history length.
     */
    public int getTokenCount() {
        return kvTokenCount;
    }

    /**
     * Get current token history.
     */
    public int[] getTokenHistory() {
        return Arrays.copyOf(kvTokenHistory, kvTokenCount);
    }

    /**
     * Push a token to the recent token ring buffer for penalty calculations.
     */
    public int[] pushRecentToken(
            int tokenId,
            int[] recentRing,
            int ringSize,
            int ringIndex,
            int[] recentTokenCounts,
            int repeatLastN) {

        if (repeatLastN <= 0 || recentRing == null) {
            return new int[]{ringSize, ringIndex};
        }

        if (ringSize < repeatLastN) {
            recentRing[ringSize] = tokenId;
            ringSize++;
        } else {
            int evicted = recentRing[ringIndex];
            if (recentTokenCounts != null && evicted >= 0 && evicted < recentTokenCounts.length) {
                if (recentTokenCounts[evicted] > 0) {
                    recentTokenCounts[evicted] -= 1;
                }
            }
            recentRing[ringIndex] = tokenId;
            ringIndex = (ringIndex + 1) % repeatLastN;
        }

        if (recentTokenCounts != null && tokenId >= 0 && tokenId < recentTokenCounts.length) {
            recentTokenCounts[tokenId] += 1;
        }

        return new int[]{ringSize, ringIndex};
    }

    private Path resolveSessionPath(InferenceRequest request) {
        Object explicit = request.getParameters().get("gguf.session.path");
        if (explicit != null) {
            return Path.of(explicit.toString());
        }
        java.util.Optional<String> sessionId = request.getSessionId();
        if (sessionId.isEmpty() || sessionId.get().isBlank()) {
            return null;
        }
        String baseDir = String.valueOf(request.getParameters().getOrDefault(
                "gguf.session.cache_dir",
                System.getProperty("user.home") + "/.gollek/cache/gguf/sessions"));
        String safeModel = manifest == null ? "unknown" : manifest.modelId().replace('/', '_');
        return Path.of(baseDir, safeModel, sessionId.get() + ".bin");
    }

    public void clear() {
        kvTokenHistory = new int[0];
        kvTokenCount = 0;
        tokenCache.clear();
    }

    private static class Collections {
        static <K, V> java.util.Map<K, V> synchronizedMap(java.util.Map<K, V> m) {
            return java.util.Collections.synchronizedMap(m);
        }
    }
}
