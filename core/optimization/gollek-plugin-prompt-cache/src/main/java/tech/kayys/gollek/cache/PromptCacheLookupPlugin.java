package tech.kayys.gollek.cache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.plugin.PluginContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Inference phase plugin that runs at
 * {@link InferencePhase#PROMPT_CACHE_LOOKUP}.
 *
 * <h3>What it does</h3>
 * <ol>
 * <li>Reads the tokenized prompt from
 * {@code ExecutionContext.variables["tokenIds"]}.</li>
 * <li>Computes rolling boundary hashes via {@link PromptPrefixHasher}.</li>
 * <li>Probes {@link PromptCacheStore} from longest prefix to shortest, stopping
 * on the first hit (greedy longest-prefix matching).</li>
 * <li>On hit: pins the prefix blocks and performs a CoW to give this request
 * its own writable suffix blocks.</li>
 * <li>Writes hit metadata into {@code ExecutionContext} for consumption by
 * the prefill plugin and {@link PromptCacheStorePlugin}.</li>
 * </ol>
 *
 * <h3>ExecutionContext contract</h3>
 * <b>Reads:</b>
 * <ul>
 * <li>{@code tokenIds} — {@code int[]} (required, set by TOKENIZE phase)</li>
 * <li>{@code modelId} — {@code String} (required)</li>
 * <li>{@code sessionId} — {@code String} (optional; used for
 * session-scope)</li>
 * </ul>
 * <b>Writes on HIT:</b>
 * <ul>
 * <li>{@code cache.hit} — {@code Boolean} true</li>
 * <li>{@code cache.entry} — {@code CachedKVEntry}</li>
 * <li>{@code cache.prefixTokens} — {@code Integer} token count reused</li>
 * <li>{@code cache.prefixBlocks} — {@code List<Integer>} pinned block IDs
 * (read-only)</li>
 * <li>{@code cache.cowBlocks} — {@code List<Integer>} CoW block IDs (writable
 * suffix)</li>
 * </ul>
 * <b>Writes on MISS:</b>
 * <ul>
 * <li>{@code cache.hit} — {@code Boolean} false</li>
 * </ul>
 */
@ApplicationScoped
public class PromptCacheLookupPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(PromptCacheLookupPlugin.class);

    // Variable keys — also used by PromptCacheStorePlugin and prefill phase
    public static final String VAR_CACHE_HIT = "cache.hit";
    public static final String VAR_CACHE_ENTRY = "cache.entry";
    public static final String VAR_CACHE_PREFIX_TOKENS = "cache.prefixTokens";
    public static final String VAR_CACHE_PREFIX_BLOCKS = "cache.prefixBlocks";
    public static final String VAR_CACHE_COW_BLOCKS = "cache.cowBlocks";

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

    private volatile boolean enabled = true;

    @Override
    public String id() {
        return "prompt-cache-lookup";
    }

    @Override
    public int order() {
        return 100;
    } // before PREFILL

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public void initialize(PluginContext ctx) {
        this.enabled = Boolean.parseBoolean(ctx.getConfig("enabled", "true"));
        LOG.infof("[PromptCacheLookup] initialized, enabled=%s strategy=%s",
                enabled, config.strategy());
    }

    @Override
    public boolean shouldExecute(ExecutionContext ctx) {
        return enabled && config.enabled();
    }

    @Override
    public void execute(ExecutionContext ctx, EngineContext engine) {
        int[] tokenIds = ctx.getVariable("tokenIds", int[].class).orElse(null);
        if (tokenIds == null || tokenIds.length < config.minCacheableTokens()) {
            ctx.putVariable(VAR_CACHE_HIT, false);
            LOG.debugf("[PromptCacheLookup] skip: tokenIds null or < minCacheable(%d)",
                    config.minCacheableTokens());
            return;
        }

        String modelId = ctx.getVariable("modelId", String.class).orElse("default");
        String sessionId = ctx.getVariable("sessionId", String.class).orElse(null);
        String scope = resolveScope(modelId, sessionId);

        // Compute hashes at every block boundary
        Map<Integer, Long> boundaries = hasher.hashBoundaries(tokenIds);
        if (boundaries.isEmpty()) {
            ctx.putVariable(VAR_CACHE_HIT, false);
            return;
        }

        // Greedy longest-prefix search: iterate boundaries from largest to smallest
        List<Map.Entry<Integer, Long>> sortedDesc = new ArrayList<>(boundaries.entrySet());
        sortedDesc.sort((a, b) -> Integer.compare(b.getKey(), a.getKey()));

        try (var sample = metrics.startLookupTimer()) {
            for (var boundary : sortedDesc) {
                int prefixLen = boundary.getKey();
                long prefixHash = boundary.getValue();

                PrefixHash key = new PrefixHash(modelId, prefixHash, prefixLen, scope);
                Optional<CachedKVEntry> hit = store.lookup(key);

                if (hit.isPresent()) {
                    CachedKVEntry entry = hit.get();
                    sample.stop(true);

                    // Pin the shared prefix blocks — they must not be evicted mid-request
                    List<Integer> prefixBlocks = kvExt.pinBlocks(entry.blockIds());

                    // CoW: allocate writable copies for this request's suffix
                    List<Integer> cowBlocks;
                    try {
                        cowBlocks = kvExt.cowBlocks(prefixBlocks);
                    } catch (Exception e) {
                        // Pool exhausted — fall back to full prefill, unpin immediately
                        LOG.warnf("[PromptCacheLookup] CoW failed (%s), falling back to full prefill", e.getMessage());
                        kvExt.unpinBlocks(prefixBlocks);
                        ctx.putVariable(VAR_CACHE_HIT, false);
                        return;
                    }

                    // Populate ExecutionContext for prefill phase and store plugin
                    ctx.putVariable(VAR_CACHE_HIT, true);
                    ctx.putVariable(VAR_CACHE_ENTRY, entry);
                    ctx.putVariable(VAR_CACHE_PREFIX_TOKENS, prefixLen);
                    ctx.putVariable(VAR_CACHE_PREFIX_BLOCKS, prefixBlocks);
                    ctx.putVariable(VAR_CACHE_COW_BLOCKS, cowBlocks);
                    ctx.putMetadata("cache.savedTokens", prefixLen);
                    ctx.putMetadata("cache.strategy", config.strategy());

                    LOG.infof("[PromptCacheLookup] HIT model=%s scope=%s prefix=%d tokens saved=%d blocks=%d",
                            modelId, scope, prefixLen, prefixLen, prefixBlocks.size());
                    return;
                }
            }

            // No matching prefix found at any boundary
            sample.stop(false);
            ctx.putVariable(VAR_CACHE_HIT, false);
            LOG.debugf("[PromptCacheLookup] MISS model=%s tokens=%d", modelId, tokenIds.length);
        }
    }

    @Override
    public void onConfigUpdate(Map<String, Object> cfg) {
        this.enabled = (Boolean) cfg.getOrDefault("enabled", true);
        LOG.infof("[PromptCacheLookup] config updated, enabled=%s", enabled);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of("enabled", enabled);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private String resolveScope(String modelId, String sessionId) {
        return switch (config.scope()) {
            case "session" -> sessionId != null ? "session:" + sessionId : "global";
            case "model" -> "model:" + modelId;
            default -> "global";
        };
    }
}
