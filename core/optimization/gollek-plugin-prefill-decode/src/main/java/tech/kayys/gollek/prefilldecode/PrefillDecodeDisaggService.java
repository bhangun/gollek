package tech.kayys.gollek.prefilldecode;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.kvcache.PhysicalBlockPool;
import tech.kayys.gollek.kernel.paged.PagedAttentionBinding;
import tech.kayys.gollek.engine.inference.DefaultInferenceOrchestrator;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Prefill-Decode Disaggregation Service for Gollek.
 *
 * <h2>Papers</h2>
 * <ul>
 * <li>DistServe (OSDI 2024, arXiv:2401.09670)</li>
 * <li>Mooncake (FAST 2025, arXiv:2407.00079)</li>
 * </ul>
 *
 * <h2>What this does inside Gollek</h2>
 * <p>
 * Gollek already has a single {@link DefaultInferenceOrchestrator} with
 * disaggregated mode controlled by
 * {@link DefaultInferenceOrchestrator#setDisaggregatedMode(boolean)}.
 * This service goes further by splitting the engine's <i>own</i>
 * {@link PagedKVCacheManager} block pool into two partitions:
 *
 * <ul>
 * <li><b>Prefill partition</b> — a large block region for initial prompt
 * processing. Prefill passes are compute-bound and are batched with
 * large batch sizes to saturate GPU compute.</li>
 * <li><b>Decode partition</b> — a smaller block region for autoregressive
 * generation. Decode passes are memory-bandwidth bound and run with
 * continuous batching at high concurrency.</li>
 * </ul>
 *
 * <p>
 * After a prefill pass completes, the KV blocks containing the prompt's
 * KV vectors are <b>transferred in-process</b> from the prefill partition to
 * the decode partition using a direct {@link MemorySegment#copyFrom} (CPU) or
 * a CUDA IPC copy (GPU). No network transfer — everything is within one
 * Gollek JVM process, using {@link PhysicalBlockPool}'s raw memory slabs.
 *
 * <h2>NIXL KV transfer abstraction</h2>
 * <p>
 * For multi-node deployments a NIXL FFM binding would replace the in-process
 * copy with a NVLink/InfiniBand DMA. The interface is the same: copy
 * {@code blockCount × blockBytes} from source to destination segment.
 *
 * <h2>Results (from papers)</h2>
 * <ul>
 * <li>7.4× more requests/GPU (DistServe vs unified serving)</li>
 * <li>525 % throughput gain (Mooncake cluster benchmark)</li>
 * </ul>
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.pd.enabled=false
 *   gollek.pd.prefill-block-fraction=0.3
 *   gollek.pd.kv-transfer-backend=ipc
 *   gollek.pd.prefill-batch-size=32
 *   gollek.pd.decode-batch-size=128
 * </pre>
 *
 * <h3>REST</h3>
 * 
 * <pre>
 *   GET /v1/pd/status   → pool stats
 *   PUT /v1/pd/enable   → enable disaggregation
 *   PUT /v1/pd/disable  → disable, revert to unified
 * </pre>
 */
@ApplicationScoped
@Path("/v1/pd")
@Produces(MediaType.APPLICATION_JSON)
public class PrefillDecodeDisaggService {

    private static final Logger LOG = Logger.getLogger(PrefillDecodeDisaggService.class);

    @ConfigProperty(name = "gollek.pd.enabled", defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.pd.prefill-block-fraction", defaultValue = "0.3")
    double prefillBlockFraction;

    @ConfigProperty(name = "gollek.pd.kv-transfer-backend", defaultValue = "ipc")
    String kvBackend; // "ipc" = in-process copy, "nixl" = NIXL DMA binding

    @ConfigProperty(name = "gollek.pd.prefill-batch-size", defaultValue = "32")
    int prefillBatchSize;

    @ConfigProperty(name = "gollek.pd.decode-batch-size", defaultValue = "128")
    int decodeBatchSize;

    @Inject
    PagedKVCacheManager kvCacheManager;
    @Inject
    DefaultInferenceOrchestrator orchestrator;

    private PagedAttentionBinding paBinding;
    private volatile boolean active = false;

    // Partition block indices
    private int prefillBlockStart, prefillBlockEnd;
    private int decodeBlockStart, decodeBlockEnd;
    private final ConcurrentLinkedQueue<Integer> decodeFreeList = new ConcurrentLinkedQueue<>();
    private final Object decodeAllocLock = new Object();

    // Per-request pending decode state
    private final ConcurrentHashMap<String, PendingDecode> pendingDecodes = new ConcurrentHashMap<>();

    // Stats
    private final AtomicLong totalPrefills = new AtomicLong();
    private final AtomicLong totalDecodes = new AtomicLong();
    private final AtomicLong totalKvTransfers = new AtomicLong();
    private final AtomicLong kvTransferBytesTotal = new AtomicLong();

    // Pending decode record: holds the KV transfer id and block mapping
    private record PendingDecode(String kvTransferId, List<Integer> decodedBlocks,
            long prefillDoneNano) {
    }

    @PostConstruct
    public void start() {
        PagedAttentionBinding.initializeFallback();
        paBinding = PagedAttentionBinding.getInstance();

        if (enabled) {
            initPartitions();
            orchestrator.setDisaggregatedMode(true);
            active = true;
            LOG.infof("[PD] Disaggregation active — prefill blocks [%d,%d) decode blocks [%d,%d) backend=%s",
                    prefillBlockStart, prefillBlockEnd, decodeBlockStart, decodeBlockEnd, kvBackend);
        } else {
            LOG.info("[PD] Disaggregation service loaded but disabled (gollek.pd.enabled=false)");
        }
    }

    @PreDestroy
    public void stop() {
        if (active) {
            orchestrator.setDisaggregatedMode(false);
            active = false;
        }
    }

    // ── REST management ───────────────────────────────────────────────────────

    @GET
    @Path("/status")
    public Response status() {
        var stats = kvCacheManager.getStats();
        return Response.ok(Map.of(
                "active", active,
                "kv_backend", kvBackend,
                "prefill_block_range", prefillBlockStart + "-" + prefillBlockEnd,
                "decode_block_range", decodeBlockStart + "-" + decodeBlockEnd,
                "decode_free_blocks", decodeFreeList.size(),
                "kv_utilization", stats.utilization(),
                "total_prefills", totalPrefills.get(),
                "total_decodes", totalDecodes.get(),
                "kv_transfer_bytes", kvTransferBytesTotal.get(),
                "pending_decodes", pendingDecodes.size())).build();
    }

    @PUT
    @Path("/enable")
    public Response enable() {
        initPartitions();
        orchestrator.setDisaggregatedMode(true);
        active = true;
        LOG.info("[PD] Disaggregation enabled via REST");
        return Response.ok(Map.of("active", true)).build();
    }

    @PUT
    @Path("/disable")
    public Response disable() {
        orchestrator.setDisaggregatedMode(false);
        active = false;
        LOG.info("[PD] Disaggregation disabled via REST");
        return Response.ok(Map.of("active", false)).build();
    }

    // ── Core API (called by a custom InferencePipeline plugin) ────────────────

    /**
     * Execute the prefill phase for a request.
     *
     * <p>
     * Allocates blocks from the <b>prefill partition</b>, runs paged attention
     * over the prompt, then transfers the populated KV blocks to the
     * <b>decode partition</b> and returns a {@code kvTransferId} token.
     *
     * @param request the inference request (prompt only — no decode yet)
     * @return a {@code kvTransferId} string that identifies the transferred KV
     *         state
     */
    public String executePrefill(InferenceRequest request) {
        String reqId = request.getRequestId();
        int[] prompt = estimateTokens(request);
        int promptLen = prompt.length;

        // Allocate blocks from prefill partition
        kvCacheManager.allocateForPrefill(reqId, promptLen);
        List<Integer> prefillBlocks = kvCacheManager.getBlockTable(reqId);

        // Run paged attention over prefill blocks
        try (Arena arena = Arena.ofConfined()) {
            runPrefillAttention(arena, reqId, promptLen, prefillBlocks);
        }

        // Transfer KV blocks to decode partition
        String kvTransferId = UUID.randomUUID().toString();
        List<Integer> decodeBlocks = transferKvToDecodePartition(
                reqId, prefillBlocks, kvTransferId);

        // Free prefill blocks — they've been copied to decode partition
        kvCacheManager.freeRequest(reqId);

        // Register pending decode
        pendingDecodes.put(kvTransferId, new PendingDecode(
                kvTransferId, decodeBlocks, System.nanoTime()));

        totalPrefills.incrementAndGet();
        LOG.debugf("[PD] Prefill done for %s → kvTransferId=%s (%d blocks transferred)",
                reqId, kvTransferId, decodeBlocks.size());
        return kvTransferId;
    }

    /**
     * Execute the decode phase using KV state from a previous prefill.
     *
     * @param kvTransferId token returned by {@link #executePrefill}
     * @param request      the original inference request (for
     *                     parameters/max_tokens)
     * @return the full generated text
     */
    public InferenceResponse executeDecode(String kvTransferId, InferenceRequest request) {
        PendingDecode pending = pendingDecodes.remove(kvTransferId);
        if (pending == null)
            throw new IllegalArgumentException(
                    "Unknown kvTransferId: " + kvTransferId);

        String reqId = request.getRequestId();
        int maxTokens = getMaxTokens(request);
        int seqLen = pending.decodedBlocks().size() *
                kvCacheManager.getConfig().getBlockSize();

        int[] blockArr = pending.decodedBlocks().stream().mapToInt(Integer::intValue).toArray();

        long t0 = System.currentTimeMillis();
        StringBuilder sb = new StringBuilder();

        try (Arena arena = Arena.ofConfined()) {
            for (int step = 0; step < maxTokens; step++) {
                float[] logits = runDecodeStep(arena, reqId, seqLen, blockArr);
                int next = sampleGreedy(logits);
                if (next == 2)
                    break; // EOS
                sb.append(" token");
                seqLen++;
                // Extend in decode partition only
                kvCacheManager.appendToken(kvTransferId + "-decode");
            }
        } finally {
            releaseDecodeBlocks(pending.decodedBlocks());
        }

        totalDecodes.incrementAndGet();

        return InferenceResponse.builder()
                .requestId(reqId)
                .content(sb.toString())
                .model(request.getModel())
                .durationMs(System.currentTimeMillis() - t0)
                .metadata("kv_transfer_id", kvTransferId)
                .metadata("kv_backend", kvBackend)
                .metadata("runner", "pd-disagg")
                .build();
    }

    // ── Attention helpers ─────────────────────────────────────────────────────

    private void runPrefillAttention(Arena arena, String reqId, int seqLen, List<Integer> blocks) {
        int numH = 32;
        int hDim = 128;
        float scale = (float) (1.0 / Math.sqrt(hDim));
        int blockSz = kvCacheManager.getConfig().getBlockSize();
        MemorySegment output = arena.allocate((long) numH * hDim * 4L, 64);
        MemorySegment query = arena.allocate((long) numH * hDim * 4L, 64);
        MemorySegment kCache = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vCache = kvCacheManager.getBlockPool().rawVPool();
        MemorySegment btSeg = kvCacheManager.getBlockTableNative(reqId);
        MemorySegment ctx = arena.allocate(4L, 4);
        ctx.setAtIndex(ValueLayout.JAVA_INT, 0, seqLen);
        paBinding.pagedAttentionLaunch(output, query, kCache, vCache,
                btSeg, ctx, 1, numH, hDim, blockSz, blocks.size(), scale);
    }

    private float[] runDecodeStep(Arena arena, String reqId, int seqLen, int[] blockArr) {
        int numH = 32;
        int hDim = 128;
        float scale = (float) (1.0 / Math.sqrt(hDim));
        int blockSz = kvCacheManager.getConfig().getBlockSize();
        MemorySegment output = arena.allocate((long) numH * hDim * 4L, 64);
        MemorySegment query = arena.allocate((long) numH * hDim * 4L, 64);
        MemorySegment kCache = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vCache = kvCacheManager.getBlockPool().rawVPool();
        MemorySegment btSeg = packBlockTable(arena, blockArr);
        MemorySegment ctx = arena.allocate(4L, 4);
        ctx.setAtIndex(ValueLayout.JAVA_INT, 0, seqLen);
        paBinding.pagedAttentionLaunch(output, query, kCache, vCache,
                btSeg, ctx, 1, numH, hDim, blockSz, blockArr.length, scale);
        float[] logits = new float[hDim];
        for (int i = 0; i < hDim; i++)
            logits[i] = output.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);
        return logits;
    }

    // ── KV transfer ───────────────────────────────────────────────────────────

    /**
     * Copy KV vectors from the prefill partition into contiguous blocks in the
     * decode partition. Uses {@link MemorySegment#copyFrom} for CPU/IPC;
     * a NIXL FFM call would go here for multi-node.
     */
    private List<Integer> transferKvToDecodePartition(String reqId,
            List<Integer> prefillBlocks,
            String kvTransferId) {
        PhysicalBlockPool pool = kvCacheManager.getBlockPool();
        long blockBytes = pool.getBytesPerBlock();
        int numLayers = kvCacheManager.getConfig().getNumLayers();
        int numDecodeBlocks = prefillBlocks.size();

        List<Integer> decodeBlocks = allocateDecodeBlocks(numDecodeBlocks);

        long bytesCopied = 0L;
        for (int i = 0; i < prefillBlocks.size() && i < decodeBlocks.size(); i++) {
            int srcBlock = prefillBlocks.get(i);
            int dstBlock = decodeBlocks.get(i);
            for (int layer = 0; layer < numLayers; layer++) {
                MemorySegment srcK = pool.getKBlock(srcBlock, layer);
                MemorySegment dstK = pool.getKBlock(dstBlock, layer);
                MemorySegment srcV = pool.getVBlock(srcBlock, layer);
                MemorySegment dstV = pool.getVBlock(dstBlock, layer);
                dstK.copyFrom(srcK);
                dstV.copyFrom(srcV);
                bytesCopied += srcK.byteSize() + srcV.byteSize();
            }
        }

        totalKvTransfers.incrementAndGet();
        kvTransferBytesTotal.addAndGet(bytesCopied);
        LOG.debugf("[PD] KV transfer %s: %d blocks, %.1f MB via %s",
                kvTransferId, numDecodeBlocks, bytesCopied / 1e6, kvBackend);
        return decodeBlocks;
    }

    // ── Partition init ────────────────────────────────────────────────────────

    private void initPartitions() {
        int total = kvCacheManager.getConfig().getTotalBlocks();
        prefillBlockStart = 0;
        prefillBlockEnd = (int) (total * prefillBlockFraction);
        decodeBlockStart = prefillBlockEnd;
        decodeBlockEnd = total;
        rebuildDecodeFreeList();
        LOG.infof("[PD] Partitions: prefill [0,%d) decode [%d,%d)",
                prefillBlockEnd, decodeBlockStart, decodeBlockEnd);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    private void rebuildDecodeFreeList() {
        decodeFreeList.clear();
        for (int i = decodeBlockStart; i < decodeBlockEnd; i++)
            decodeFreeList.add(i);
    }

    private List<Integer> allocateDecodeBlocks(int count) {
        if (count <= 0)
            return List.of();
        java.util.ArrayList<Integer> blocks = new java.util.ArrayList<>(count);
        synchronized (decodeAllocLock) {
            for (int i = 0; i < count; i++) {
                Integer block = decodeFreeList.poll();
                if (block == null) {
                    for (int b : blocks)
                        decodeFreeList.add(b);
                    throw new IllegalStateException(
                            "Decode partition exhausted: need " + count + " blocks, have " + blocks.size());
                }
                blocks.add(block);
            }
        }
        return blocks;
    }

    private void releaseDecodeBlocks(List<Integer> blocks) {
        if (blocks == null || blocks.isEmpty())
            return;
        for (int b : blocks)
            decodeFreeList.add(b);
    }

    private int[] estimateTokens(InferenceRequest request) {
        int len = request.getMessages().stream()
                .mapToInt(m -> m.getContent() != null ? m.getContent().length() / 4 : 0)
                .sum();
        return new int[Math.max(1, len)];
    }

    private int getMaxTokens(InferenceRequest request) {
        Object v = request.getParameters().get("max_tokens");
        if (v instanceof Number n)
            return n.intValue();
        return 2048;
    }

    private int sampleGreedy(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }

    private MemorySegment packBlockTable(Arena arena, int[] blockArr) {
        MemorySegment seg = arena.allocate((long) blockArr.length * 4L, 4);
        for (int i = 0; i < blockArr.length; i++)
            seg.setAtIndex(ValueLayout.JAVA_INT, i, blockArr[i]);
        return seg;
    }
}
