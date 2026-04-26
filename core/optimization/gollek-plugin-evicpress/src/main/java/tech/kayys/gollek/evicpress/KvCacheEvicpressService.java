package tech.kayys.gollek.evicpress;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.evicpress.binding.EvicpressBinding;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.kvcache.PhysicalBlockPool;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * EVICPRESS KV-Cache Eviction/Compression Service for Gollek.
 *
 * <p>Intercepts Gollek's {@link PagedKVCacheManager} block pool and applies
 * the EVICPRESS joint KEEP/COMPRESS/EVICT/DROP policy via {@link EvicpressBinding}.
 *
 * <h2>Papers</h2>
 * <ul>
 *   <li>EVICPRESS (arXiv:2512.14946) — joint 4-action eviction policy</li>
 *   <li>GVote (arXiv:2509.03136, ICLR 2026) — query-guided budget-free eviction</li>
 *   <li>Token-Route (arXiv:2603.01426) — attention-chain connectivity protection</li>
 * </ul>
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.kvcache.evicpress.enabled=false
 *   gollek.kvcache.evicpress.library-path=/opt/gollek/lib/libgollek_evicpress.so
 *   gollek.kvcache.evicpress.eviction-threshold=0.85
 *   gollek.kvcache.evicpress.retention-target=0.6
 *   gollek.kvcache.evicpress.compress-dtype=int4
 *   gollek.kvcache.evicpress.gvote.enabled=true
 *   gollek.kvcache.evicpress.gvote.num-samples=16
 *   gollek.kvcache.evicpress.route-protection=true
 *   gollek.kvcache.evicpress.evict-pool-blocks=512
 * </pre>
 *
 * <h3>REST</h3>
 * <pre>
 *   GET  /v1/kvcache/stats
 *   PUT  /v1/kvcache/retention?target=0.6
 *   POST /v1/kvcache/evict-now
 * </pre>
 */
@ApplicationScoped
@Path("/v1/kvcache")
@Produces(MediaType.APPLICATION_JSON)
public class KvCacheEvicpressService {

    private static final Logger LOG = Logger.getLogger(KvCacheEvicpressService.class);

    @ConfigProperty(name = "gollek.kvcache.evicpress.enabled",            defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.kvcache.evicpress.library-path",
                    defaultValue = "/opt/gollek/lib/libgollek_evicpress.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.kvcache.evicpress.eviction-threshold", defaultValue = "0.85")
    double evictionThreshold;

    @ConfigProperty(name = "gollek.kvcache.evicpress.retention-target",   defaultValue = "0.6")
    double retentionTarget;

    @ConfigProperty(name = "gollek.kvcache.evicpress.compress-dtype",     defaultValue = "int4")
    String compressDtype;

    @ConfigProperty(name = "gollek.kvcache.evicpress.gvote.enabled",      defaultValue = "true")
    boolean gvoteEnabled;

    @ConfigProperty(name = "gollek.kvcache.evicpress.gvote.num-samples",  defaultValue = "16")
    int gvoteNumSamples;

    @ConfigProperty(name = "gollek.kvcache.evicpress.route-protection",   defaultValue = "true")
    boolean routeProtection;

    @ConfigProperty(name = "gollek.kvcache.evicpress.evict-pool-blocks",  defaultValue = "512")
    int evictPoolBlocks;

    @Inject PagedKVCacheManager kvCacheManager;

    private EvicpressBinding evicpress;
    private MemorySegment    cpuEvictPool;
    private Arena            evictArena;

    private final ConcurrentHashMap<Integer, Long> evictedBlocks = new ConcurrentHashMap<>();
    private final AtomicLong evictPoolCursor = new AtomicLong();

    // Counters
    private final AtomicLong totalKept       = new AtomicLong();
    private final AtomicLong totalCompressed = new AtomicLong();
    private final AtomicLong totalEvicted    = new AtomicLong();
    private final AtomicLong totalDropped    = new AtomicLong();
    private final AtomicLong evictCycles     = new AtomicLong();

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @PostConstruct
    public void init() {
        if (!enabled) {
            LOG.info("[EVICPRESS] Disabled (gollek.kvcache.evicpress.enabled=false)");
            return;
        }

        EvicpressBinding.initialize(java.nio.file.Path.of(libraryPath));
        evicpress = EvicpressBinding.getInstance();

        // Pre-allocate CPU eviction pool (pinned off-heap)
        evictArena = Arena.ofAuto();
        long poolBytes = (long) evictPoolBlocks
                * kvCacheManager.getBlockPool().getBytesPerBlock()
                * kvCacheManager.getConfig().getNumLayers() * 2L; // K + V
        cpuEvictPool = evictArena.allocate(poolBytes, 64);

        LOG.infof("[EVICPRESS] Ready — threshold=%.0f%% retention=%.0f%% gvote=%s native=%s",
                evictionThreshold * 100, retentionTarget * 100,
                gvoteEnabled, evicpress.isNativeAvailable());
    }

    // ── Core eviction API ─────────────────────────────────────────────────────

    /**
     * Score KV blocks and apply KEEP/COMPRESS/EVICT/DROP actions.
     *
     * <p>Called by the inference pipeline when KV pool utilisation exceeds
     * {@code evictionThreshold}. Uses {@link EvicpressBinding#score} for
     * importance scoring and {@link EvicpressBinding#compress} for in-place
     * INT4/INT8/FP8 quantisation.
     *
     * @param requestId  active inference request
     * @param attnScores cumulative attention weights per block position
     * @return number of GPU blocks freed
     */
    public int applyEvictionPolicy(String requestId, float[] attnScores) {
        if (!enabled || evicpress == null) return 0;

        double util = kvCacheManager.getStats().utilization();
        if (util < evictionThreshold) return 0;

        List<Integer> blockTable = kvCacheManager.getBlockTable(requestId);
        if (blockTable.isEmpty()) return 0;

        int numBlocks  = blockTable.size();
        int blockSize  = kvCacheManager.getConfig().getBlockSize();
        int numLayers  = kvCacheManager.getConfig().getNumLayers();
        int freed      = 0;

        try (Arena arena = Arena.ofConfined()) {
            // ── Score blocks via EvicpressBinding ─────────────────────────────
            MemorySegment scoresSeg = arena.allocate((long) numBlocks * 4L, 4);
            MemorySegment attnSeg   = MemorySegment.ofArray(attnScores);

            evicpress.score(scoresSeg, attnSeg, numBlocks, blockSize, numLayers,
                    gvoteEnabled ? gvoteNumSamples : 0);

            // ── Compute route-protection mask ─────────────────────────────────
            long maskWords = (numBlocks + 63) / 64;
            MemorySegment mask = arena.allocate(maskWords * 8L, 8);
            if (routeProtection) {
                evicpress.routeProtect(mask, attnSeg, numBlocks, blockSize, numLayers);
            }

            // ── Compute action thresholds ──────────────────────────────────────
            float[] scores = new float[numBlocks];
            for (int i = 0; i < numBlocks; i++)
                scores[i] = scoresSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float[] thresholds = computeThresholds(scores, retentionTarget);

            PhysicalBlockPool pool = kvCacheManager.getBlockPool();

            // ── Apply actions ─────────────────────────────────────────────────
            for (int i = 0; i < numBlocks; i++) {
                int   blockId  = blockTable.get(i);
                float score    = scores[i];
                boolean route  = routeProtection && isProtected(mask, i);

                if (score >= thresholds[2]) {
                    // KEEP
                    totalKept.incrementAndGet();

                } else if (score >= thresholds[1]) {
                    // COMPRESS — quantise in-place via EvicpressBinding
                    for (int layer = 0; layer < numLayers; layer++) {
                        MemorySegment kBlk = pool.getKBlock(blockId, layer);
                        MemorySegment vBlk = pool.getVBlock(blockId, layer);
                        evicpress.compress(kBlk, kBlk.byteSize(), compressDtype);
                        evicpress.compress(vBlk, vBlk.byteSize(), compressDtype);
                    }
                    totalCompressed.incrementAndGet();

                } else if (score >= thresholds[0]) {
                    // EVICT — copy to CPU off-heap pool
                    evictToCpu(pool, blockId, numLayers);
                    totalEvicted.incrementAndGet();
                    freed++;

                } else if (!route) {
                    // DROP — zero out and free
                    for (int layer = 0; layer < numLayers; layer++) {
                        pool.getKBlock(blockId, layer).fill((byte) 0);
                        pool.getVBlock(blockId, layer).fill((byte) 0);
                    }
                    totalDropped.incrementAndGet();
                    freed++;

                } else {
                    // Route-protected — keep anyway
                    totalKept.incrementAndGet();
                }
            }
        }

        evictCycles.incrementAndGet();
        LOG.debugf("[EVICPRESS] Request %s: freed %d blocks (util %.0f%%)",
                requestId, freed, util * 100);
        return freed;
    }

    // ── REST management ───────────────────────────────────────────────────────

    @GET @Path("/stats")
    public Response stats() {
        var s = kvCacheManager.getStats();
        return Response.ok(Map.of(
                "kv_total_blocks",   s.totalBlocks(),
                "kv_free_blocks",    s.freeBlocks(),
                "kv_utilization",    s.utilization(),
                "eviction_threshold", evictionThreshold,
                "retention_target",  retentionTarget,
                "total_kept",        totalKept.get(),
                "total_compressed",  totalCompressed.get(),
                "total_evicted",     totalEvicted.get(),
                "total_dropped",     totalDropped.get(),
                "evict_cycles",      evictCycles.get()
        )).build();
    }

    @PUT @Path("/retention")
    public Response setRetention(@QueryParam("target") double target) {
        if (target <= 0.0 || target > 1.0)
            return Response.status(400).entity(Map.of("error", "target ∈ (0,1]")).build();
        double old = retentionTarget;
        retentionTarget = target;
        return Response.ok(Map.of("previous", old, "current", target)).build();
    }

    @POST @Path("/evict-now")
    public Response evictNow() {
        return Response.ok(Map.of(
                "message",      "Call applyEvictionPolicy(requestId, attnScores) programmatically",
                "utilization",  kvCacheManager.getStats().utilization()
        )).build();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void evictToCpu(PhysicalBlockPool pool, int blockId, int numLayers) {
        long blockBytes = pool.getBytesPerBlock();
        long stride     = blockBytes * numLayers * 2L;
        long base       = evictPoolCursor.getAndAdd(stride);
        if (base + stride > cpuEvictPool.byteSize()) {
            evictPoolCursor.addAndGet(-stride);
            return;
        }

        for (int layer = 0; layer < numLayers; layer++) {
            long kOff = base + (long) layer * blockBytes;
            long vOff = kOff + (long) numLayers * blockBytes;
            cpuEvictPool.asSlice(kOff, blockBytes).copyFrom(pool.getKBlock(blockId, layer));
            cpuEvictPool.asSlice(vOff, blockBytes).copyFrom(pool.getVBlock(blockId, layer));
        }
        evictedBlocks.put(blockId, base);
    }

    private float[] computeThresholds(float[] scores, double keepFraction) {
        float[] sorted = scores.clone();
        java.util.Arrays.sort(sorted);
        int n = sorted.length;
        return new float[]{
            sorted[Math.max(0, (int)(n * 0.10))],
            sorted[Math.max(0, (int)(n * (1 - keepFraction) * 0.5))],
            sorted[Math.max(0, (int)(n * (1 - keepFraction)))]
        };
    }

    private boolean isProtected(MemorySegment mask, int blockIndex) {
        long word = blockIndex / 64;
        long bit  = blockIndex % 64;
        return (mask.getAtIndex(ValueLayout.JAVA_LONG, word) & (1L << bit)) != 0;
    }
}
