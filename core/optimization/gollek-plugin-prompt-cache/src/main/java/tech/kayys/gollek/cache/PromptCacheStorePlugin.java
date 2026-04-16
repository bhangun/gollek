package tech.kayys.gollek.cache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.plugin.PluginContext;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Inference phase plugin that runs at
 * {@link InferencePhase#PROMPT_CACHE_STORE}.
 *
 * <h3>What it does</h3>
 * On a cache miss (after the prefill phase has computed fresh KV blocks), this
 * plugin stores the resulting block metadata for future reuse.
 *
 * <p>
 * It stores an entry for <em>every block boundary</em> in the prompt, not just
 * the full prefix length. This enables partial-prefix reuse on future requests
 * that
 * share only the first N blocks of this prompt (e.g. a shared system prompt
 * followed
 * by different user messages).
 *
 * <h3>Async write path</h3>
 * When {@code gollek.cache.prompt.async-store=true} (default), the store
 * operation
 * runs on a dedicated single-thread executor so it never blocks the inference
 * response streaming back to the client. The KV data is safe to reference
 * because the physical blocks are registered to the request and will not be
 * freed until {@code freeRequest()} is called at the end of the response
 * lifecycle.
 *
 * <h3>Unpin on request completion</h3>
 * This plugin also calls {@link PagedKVCacheManagerExtension#unpinBlocks} for
 * any
 * previously-pinned prefix blocks from a cache hit, ensuring they are returned
 * to
 * the free pool when the request finishes.
 *
 * <h3>ExecutionContext contract</h3>
 * <b>Reads:</b>
 * <ul>
 * <li>{@code tokenIds} — {@code int[]} (required)</li>
 * <li>{@code modelId} — {@code String} (required)</li>
 * <li>{@code sessionId} — {@code String} (optional)</li>
 * <li>{@code cache.hit} — {@code Boolean} (required; skip store if true)</li>
 * <li>{@code prefillBlockIds} — {@code List<Integer>} (required on miss; set by
 * PREFILL)</li>
 * <li>{@code cache.prefixBlocks} — {@code List<Integer>} (required on hit; to
 * unpin)</li>
 * </ul>
 */
@ApplicationScoped
public class PromptCacheStorePlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(PromptCacheStorePlugin.class);

    @Inject
    PromptCacheStore store;
    @Inject
    PromptPrefixHasher hasher;
    @Inject
    PagedKVCacheManagerExtension kvExt;
    @Inject
    PromptCacheConfig config;
    @Inject
    PromptCacheMetrics metrics;
    @Inject
    KVCacheConfig kvConfig;

    private volatile boolean enabled = true;

    /**
     * Single background thread for async stores — keeps inference hot path clean.
     */
    private final ExecutorService asyncWriter = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "gollek-prompt-cache-writer");
        t.setDaemon(true);
        return t;
    });

    @Override
    public String id() {
        return "prompt-cache-store";
    }

    @Override
    public int order() {
        return 200;
    } // after PREFILL

    @Override
    public InferencePhase phase() {
        return InferencePhase.POST_PROCESSING;
    }

    @Override
    public void initialize(PluginContext ctx) {
        this.enabled = Boolean.parseBoolean(ctx.getConfig("enabled", "true"));
        LOG.infof("[PromptCacheStore] initialized, asyncStore=%s", config.asyncStore());
    }

    @Override
    public boolean shouldExecute(ExecutionContext ctx) {
        return enabled && config.enabled();
    }

    @Override
    public void execute(ExecutionContext ctx, EngineContext engine) {
        // --- Step 1: Unpin prefix blocks from a prior cache hit (if any) ---
        unpinPrefixBlocksIfPresent(ctx);

        // --- Step 2: Skip store if this was already a hit ---
        Boolean wasHit = ctx.getVariable(PromptCacheLookupPlugin.VAR_CACHE_HIT, Boolean.class)
                .orElse(false);
        if (Boolean.TRUE.equals(wasHit)) {
            LOG.debugf("[PromptCacheStore] skip store — was a cache hit");
            return;
        }

        // --- Step 3: Collect required data ---
        int[] tokenIds = ctx.getVariable("tokenIds", int[].class).orElse(null);
        @SuppressWarnings("unchecked")
        List<Integer> blockIds = ctx.getVariable("prefillBlockIds", List.class).orElse(null);
        String modelId = ctx.getVariable("modelId", String.class).orElse("default");
        String sessionId = ctx.getVariable("sessionId", String.class).orElse(null);

        if (tokenIds == null || blockIds == null || blockIds.isEmpty()) {
            LOG.debugf("[PromptCacheStore] skip: missing tokenIds or prefillBlockIds");
            return;
        }

        if (tokenIds.length < config.minCacheableTokens()) {
            LOG.debugf("[PromptCacheStore] skip: prompt too short (%d < %d)",
                    tokenIds.length, config.minCacheableTokens());
            return;
        }

        String scope = resolveScope(modelId, sessionId);

        // Capture snapshot for async write (ExecutionContext must not be held across
        // threads)
        final int[] tokenIdsSnapshot = tokenIds.clone();
        final List<Integer> blockIdsSnapshot = List.copyOf(blockIds);
        final String modelIdSnapshot = modelId;
        final String scopeSnapshot = scope;

        Runnable storeTask = () -> storeBoundaries(
                tokenIdsSnapshot, blockIdsSnapshot, modelIdSnapshot, scopeSnapshot);

        if (config.asyncStore()) {
            asyncWriter.submit(storeTask);
        } else {
            storeTask.run();
        }
    }

    @Override
    public void onConfigUpdate(Map<String, Object> cfg) {
        this.enabled = (Boolean) cfg.getOrDefault("enabled", true);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of("enabled", enabled, "asyncStore", config.asyncStore());
    }

    // -------------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------------

    /**
     * Store a {@link CachedKVEntry} for every block boundary in the token sequence.
     *
     * <p>
     * Example: 48-token prompt with blockSize=16 stores three entries:
     * 
     * <pre>
     *   prefixLength=16 → blockIds=[0]
     *   prefixLength=32 → blockIds=[0,1]
     *   prefixLength=48 → blockIds=[0,1,2]
     * </pre>
     * 
     * This way a future request that shares only the first 16 tokens still gets a
     * hit.
     */
    private void storeBoundaries(int[] tokenIds, List<Integer> blockIds,
            String modelId, String scope) {
        Map<Integer, Long> boundaries = hasher.hashBoundaries(tokenIds);
        int blockSize = kvConfig.getBlockSize();

        for (Map.Entry<Integer, Long> boundary : boundaries.entrySet()) {
            int prefixLen = boundary.getKey();
            long prefixHash = boundary.getValue();

            if (prefixLen < config.minCacheableTokens())
                continue;

            // Take only the blocks that cover this prefix length
            int blocksForPrefix = (prefixLen + blockSize - 1) / blockSize;
            if (blocksForPrefix > blockIds.size())
                continue;

            List<Integer> prefixBlockIds = blockIds.subList(0, blocksForPrefix);
            PrefixHash key = new PrefixHash(modelId, prefixHash, prefixLen, scope);
            CachedKVEntry e = CachedKVEntry.of(key, prefixBlockIds, prefixLen);

            try {
                store.store(e);
                LOG.debugf("[PromptCacheStore] stored key=%s blocks=%d",
                        key.storageKey(), prefixBlockIds.size());
            } catch (Exception ex) {
                LOG.warnf("[PromptCacheStore] store failed for key=%s: %s",
                        key.storageKey(), ex.getMessage());
            }
        }

        LOG.infof("[PromptCacheStore] stored %d boundary entries for model=%s tokens=%d",
                boundaries.size(), modelId, tokenIds.length);
    }

    @SuppressWarnings("unchecked")
    private void unpinPrefixBlocksIfPresent(ExecutionContext ctx) {
        ctx.getVariable(PromptCacheLookupPlugin.VAR_CACHE_PREFIX_BLOCKS, List.class)
                .ifPresent(blocks -> {
                    kvExt.unpinBlocks((List<Integer>) blocks);
                    LOG.debugf("[PromptCacheStore] unpinned %d prefix blocks", blocks.size());
                });
    }

    private String resolveScope(String modelId, String sessionId) {
        return switch (config.scope()) {
            case "session" -> sessionId != null ? "session:" + sessionId : "global";
            case "model" -> "model:" + modelId;
            default -> "global";
        };
    }
}
