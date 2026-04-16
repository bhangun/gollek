package tech.kayys.gollek.cache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.kvcache.KVCacheExhaustedException;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Extends {@link PagedKVCacheManager} with prompt-cache-specific operations:
 * block pinning, unpinning, and copy-on-write (CoW) for shared prefix blocks.
 *
 * <h3>Why a separate class?</h3>
 * The existing {@code PagedKVCacheManager} is stable and well-tested.
 * Rather than modifying it directly, this class wraps/composes it
 * to add the three operations needed for prompt caching, following the
 * Open/Closed Principle.
 *
 * <h3>Pinning protocol</h3>
 * When a cache lookup finds a matching prefix entry, the lookup plugin:
 * <ol>
 * <li>Calls {@link #pinBlocks(List)} to increment the ref-count on each
 * block.</li>
 * <li>Calls {@link #cowBlocks(List)} to get a writable copy for the
 * suffix.</li>
 * <li>Passes prefix blocks (read-only) + suffix blocks (writable) to the
 * prefill kernel.</li>
 * <li>Calls {@link #unpinBlocks(List)} after decode finishes (or on
 * error).</li>
 * </ol>
 *
 * <p>
 * Blocks with a ref-count &gt; 0 are excluded from the free list during
 * normal request deallocation — they belong to the shared prefix and must
 * not be reclaimed until all requests using them have completed.
 */
@ApplicationScoped
public class PagedKVCacheManagerExtension {

    private static final Logger LOG = Logger.getLogger(PagedKVCacheManagerExtension.class);

    private final PagedKVCacheManager kvManager;
    private final KVCacheConfig config;

    /**
     * Per-block reference counts.
     * Absent key means ref-count == 0 (block is freely evictable).
     */
    private final ConcurrentHashMap<Integer, AtomicInteger> pinCounts = new ConcurrentHashMap<>();

    @Inject
    public PagedKVCacheManagerExtension(PagedKVCacheManager kvManager, KVCacheConfig config) {
        this.kvManager = kvManager;
        this.config = config;
    }

    // -------------------------------------------------------------------------
    // Pinning
    // -------------------------------------------------------------------------

    /**
     * Increment the reference count of each block in {@code blockIds}.
     * Pinned blocks will not be returned to the free pool by
     * {@link PagedKVCacheManager#freeRequest(String)} until unpinned.
     *
     * @param blockIds the block IDs to pin (from a {@code CachedKVEntry})
     * @return the same list (for chaining)
     */
    public List<Integer> pinBlocks(List<Integer> blockIds) {
        if (blockIds == null || blockIds.isEmpty())
            return Collections.emptyList();
        for (int id : blockIds) {
            pinCounts.computeIfAbsent(id, k -> new AtomicInteger(0)).incrementAndGet();
        }
        LOG.debugf("[KVExt] pinned %d blocks: %s", blockIds.size(), blockIds);
        return blockIds;
    }

    /**
     * Decrement the reference count of each block.
     * When the count reaches zero the block becomes eligible for normal eviction.
     *
     * @param blockIds the block IDs to unpin
     */
    public void unpinBlocks(List<Integer> blockIds) {
        if (blockIds == null || blockIds.isEmpty())
            return;
        for (int id : blockIds) {
            AtomicInteger count = pinCounts.get(id);
            if (count != null) {
                int remaining = count.decrementAndGet();
                if (remaining <= 0) {
                    pinCounts.remove(id);
                }
            }
        }
        LOG.debugf("[KVExt] unpinned %d blocks", blockIds.size());
    }

    /**
     * Returns {@code true} if the block has a non-zero pin count.
     * Used by a patched {@code freeRequest} to skip pinned blocks.
     */
    public boolean isPinned(int blockId) {
        AtomicInteger count = pinCounts.get(blockId);
        return count != null && count.get() > 0;
    }

    /**
     * Return the current pin count for a block (0 = free, &gt;0 = pinned).
     */
    public int pinCount(int blockId) {
        AtomicInteger count = pinCounts.get(blockId);
        return count == null ? 0 : count.get();
    }

    // -------------------------------------------------------------------------
    // Copy-on-Write
    // -------------------------------------------------------------------------

    /**
     * Allocate a fresh set of blocks and copy the KV tensors from
     * {@code srcBlockIds} into them. The source blocks are not modified.
     *
     * <p>
     * This is called when a cache-hit prefix must be "forked" for a new request:
     * the prefix blocks are shared (pinned, read-only), but the suffix decode
     * must write new tokens into its own private blocks.
     *
     * <p>
     * Memory layout copied:
     * 
     * <pre>
     *   For each (blockId, layer):
     *     dest_K[blockId][layer] ← src_K[blockId][layer]
     *     dest_V[blockId][layer] ← src_V[blockId][layer]
     * </pre>
     *
     * @param srcBlockIds the read-only pinned prefix block IDs
     * @return new writable block IDs containing a full copy of the prefix KV data
     * @throws KVCacheExhaustedException if there are not enough free blocks
     */
    public List<Integer> cowBlocks(List<Integer> srcBlockIds) {
        if (srcBlockIds == null || srcBlockIds.isEmpty())
            return Collections.emptyList();

        // Allocate fresh blocks via the manager's internal allocator.
        // We use a synthetic requestId to borrow the allocation API.
        String cowReqId = "__cow__" + Thread.currentThread().getId()
                + "_" + System.nanoTime();

        List<Integer> dstBlockIds;
        try {
            dstBlockIds = new ArrayList<>(
                    kvManager.allocateForPrefill(cowReqId, srcBlockIds.size() * config.getBlockSize()));
        } catch (KVCacheExhaustedException e) {
            LOG.warnf("[KVExt] CoW failed — pool exhausted. srcBlocks=%d", srcBlockIds.size());
            throw e;
        }

        // Copy KV tensors layer by layer
        var blockPool = kvManager.getBlockPool();
        for (int i = 0; i < srcBlockIds.size(); i++) {
            int src = srcBlockIds.get(i);
            int dst = dstBlockIds.get(i);
            for (int layer = 0; layer < config.getNumLayers(); layer++) {
                var srcK = blockPool.getKBlock(src, layer);
                var dstK = blockPool.getKBlock(dst, layer);
                dstK.copyFrom(srcK);

                var srcV = blockPool.getVBlock(src, layer);
                var dstV = blockPool.getVBlock(dst, layer);
                dstV.copyFrom(srcV);
            }
        }

        // Detach the CoW blocks from the synthetic request so the caller manages them
        kvManager.freeRequest(cowReqId);
        // Re-register them under the caller's control by returning the IDs.
        // The caller must register these with its own requestId.

        LOG.debugf("[KVExt] CoW: cloned %d blocks from prefix → %s", dstBlockIds.size(), dstBlockIds);
        return Collections.unmodifiableList(dstBlockIds);
    }

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    public int pinnedBlockCount() {
        return pinCounts.size();
    }

    public long totalPinCount() {
        return pinCounts.values().stream().mapToLong(AtomicInteger::get).sum();
    }
}
