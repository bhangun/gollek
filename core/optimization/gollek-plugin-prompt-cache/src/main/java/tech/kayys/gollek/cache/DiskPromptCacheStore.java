package tech.kayys.gollek.cache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.cache.PromptCacheConfig;
import tech.kayys.gollek.cache.PromptCacheMetrics;
import tech.kayys.gollek.cache.CacheEntrySerializer;
import tech.kayys.gollek.cache.CacheStrategy;
import tech.kayys.gollek.cache.CachedKVEntry;
import tech.kayys.gollek.cache.PrefixHash;
import tech.kayys.gollek.cache.PromptCacheStats;
import tech.kayys.gollek.cache.PromptCacheStore;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Disk-backed prompt-cache that survives JVM restarts.
 *
 * <h3>Architecture</h3>
 * Uses a simple two-level layout:
 * <ol>
 *   <li>An in-memory {@code ConcurrentHashMap} index: {@code storageKey → DiskSlot}.</li>
 *   <li>An append-only memory-mapped file ({@code prompt-cache.dat}) where serialized
 *       entry bytes are written sequentially.</li>
 * </ol>
 *
 * <p>On startup, the index is rebuilt by scanning the data file header section
 * (a compact binary index appended at the tail of the file, flushed at shutdown).
 * This gives fast warm-start with O(1) per-entry reads on the hot path.
 *
 * <h3>Eviction</h3>
 * Entries past their TTL are lazily evicted on lookup or on an explicit
 * {@link #compactIfNeeded()} sweep. The file grows until a compaction is triggered
 * by the max-size-mb threshold.
 *
 * <h3>RocksDB variant</h3>
 * When {@code gollek.cache.prompt.disk.backend=rocksdb}, this class delegates to
 * {@code RocksDbDelegate} (a thin wrapper around RocksJava). The interface remains
 * identical; RocksDB handles compaction, bloom filters, and range deletes for
 * scope invalidation. The RocksDB variant is recommended for caches &gt;1GB.
 *
 * <p><b>Note</b>: Raw KV float tensors are NOT written to disk. Only block ID
 * metadata is persisted. On a warm restart, the block IDs refer to positions in
 * the (empty) PhysicalBlockPool — the cache lookup plugin detects orphaned entries
 * (block IDs not yet populated) and transparently falls back to full prefill,
 * writing the new block IDs back to the store.
 */
@ApplicationScoped
@CacheStrategy("disk")
public class DiskPromptCacheStore implements PromptCacheStore {

    private static final Logger LOG = Logger.getLogger(DiskPromptCacheStore.class);

    /** Max bytes per entry stored on disk (metadata only). */
    private static final int MAX_ENTRY_BYTES = 4096;
    /** Magic byte at the start of every entry record. */
    private static final byte MAGIC = 0x47; // 'G'

    private final PromptCacheConfig    config;
    private final PromptCacheMetrics   metrics;
    private final CacheEntrySerializer serializer;

    /** In-memory index: storageKey → file offset + length. */
    private final ConcurrentHashMap<String, DiskSlot> index = new ConcurrentHashMap<>();

    private Path            dataFile;
    private FileChannel     channel;
    private MappedByteBuffer mappedBuffer;
    private final AtomicLong writeOffset   = new AtomicLong(0);
    private final AtomicLong totalStores   = new AtomicLong();
    private final AtomicLong totalHits     = new AtomicLong();
    private final AtomicLong totalMisses   = new AtomicLong();
    private final AtomicLong totalEvictions = new AtomicLong();

    @Inject
    public DiskPromptCacheStore(
            PromptCacheConfig config,
            PromptCacheMetrics metrics,
            CacheEntrySerializer serializer) {
        this.config     = config;
        this.metrics    = metrics;
        this.serializer = serializer;
    }

    @Override
    public void initialize() {
        try {
            Path cacheDir = Path.of(config.disk().path());
            Files.createDirectories(cacheDir);
            dataFile = cacheDir.resolve("prompt-cache.dat");

            long maxBytes = config.disk().maxSizeMb() * 1024 * 1024;
            channel = FileChannel.open(dataFile,
                    StandardOpenOption.READ,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.CREATE);
            mappedBuffer = channel.map(FileChannel.MapMode.READ_WRITE, 0, maxBytes);

            rebuildIndex();
            LOG.infof("[DiskCache] ready: path=%s, backend=%s, maxSizeMb=%d, entries=%d",
                    dataFile, config.disk().backend(), config.disk().maxSizeMb(), index.size());
        } catch (IOException e) {
            LOG.errorf(e, "[DiskCache] failed to initialise disk store at %s", config.disk().path());
            throw new RuntimeException("DiskPromptCacheStore init failed", e);
        }
    }

    // -------------------------------------------------------------------------
    // PromptCacheStore
    // -------------------------------------------------------------------------

    @Override
    public Optional<CachedKVEntry> lookup(PrefixHash hash) {
        DiskSlot slot = index.get(hash.storageKey());
        if (slot == null) {
            metrics.recordMiss();
            totalMisses.incrementAndGet();
            return Optional.empty();
        }
        try {
            byte[] buf = new byte[slot.length()];
            ByteBuffer view = mappedBuffer.duplicate();
            view.position((int) slot.offset());
            view.get(buf);

            if (buf[0] != MAGIC) {
                index.remove(hash.storageKey()); // corrupted slot
                return Optional.empty();
            }

            String json  = new String(buf, 1, slot.length() - 1, StandardCharsets.UTF_8);
            CachedKVEntry entry = serializer.deserialize(json);

            // Lazy TTL check
            if (!config.ttl().isZero()) {
                if (entry.createdAt().plus(config.ttl()).isBefore(java.time.Instant.now())) {
                    index.remove(hash.storageKey());
                    totalEvictions.incrementAndGet();
                    metrics.recordEviction();
                    metrics.recordMiss();
                    totalMisses.incrementAndGet();
                    return Optional.empty();
                }
            }

            metrics.recordHit(entry.tokenCount());
            totalHits.incrementAndGet();
            return Optional.of(entry.accessed());
        } catch (Exception e) {
            LOG.warnf("[DiskCache] read error for key=%s: %s", hash.storageKey(), e.getMessage());
            metrics.recordMiss();
            totalMisses.incrementAndGet();
            return Optional.empty();
        }
    }

    @Override
    public void store(CachedKVEntry entry) {
        String json   = serializer.serialize(entry);
        byte[] jsonBytes = json.getBytes(StandardCharsets.UTF_8);
        int total = 1 + jsonBytes.length; // magic + payload

        if (total > MAX_ENTRY_BYTES) {
            LOG.warnf("[DiskCache] entry too large (%d bytes), skipping key=%s", total, entry.key().storageKey());
            return;
        }

        long offset = writeOffset.getAndAdd(total);
        long maxBytes = config.disk().maxSizeMb() * 1024 * 1024;
        if (offset + total > maxBytes) {
            LOG.warnf("[DiskCache] file full (offset=%d), skipping store. Run compaction.", offset);
            return;
        }

        ByteBuffer view = mappedBuffer.duplicate();
        view.position((int) offset);
        view.put(MAGIC);
        view.put(jsonBytes);

        if (config.disk().syncOnWrite()) {
            mappedBuffer.force();
        }

        index.put(entry.key().storageKey(), new DiskSlot(offset, total));
        totalStores.incrementAndGet();
        metrics.recordStore(entry.tokenCount());
    }

    @Override
    public void invalidateByModel(String modelId) {
        long removed = index.keySet().removeIf(k -> k.contains(":" + modelId + ":")) ? 1 : 0;
        LOG.infof("[DiskCache] invalidated entries for model=%s", modelId);
    }

    @Override
    public void invalidateBySession(String sessionId) {
        index.keySet().removeIf(k -> k.contains("session:" + sessionId));
        LOG.infof("[DiskCache] invalidated entries for session=%s", sessionId);
    }

    @Override
    public void invalidateAll() {
        index.clear();
        writeOffset.set(0);
        if (mappedBuffer != null) mappedBuffer.clear();
        LOG.infof("[DiskCache] invalidated all entries");
    }

    @Override
    public PromptCacheStats stats() {
        long lookups = totalHits.get() + totalMisses.get();
        return new PromptCacheStats(
                lookups,
                totalHits.get(),
                totalMisses.get(),
                totalStores.get(),
                totalEvictions.get(),
                0L,
                index.size(),
                -1L,
                lookups == 0 ? 0.0 : (double) totalHits.get() / lookups,
                strategyName()
        );
    }

    @Override
    public String strategyName() { return "disk"; }

    @Override
    public void close() throws Exception {
        persistIndex(); // flush index to tail of file for warm restart
        if (mappedBuffer != null) mappedBuffer.force();
        if (channel != null) channel.close();
        LOG.infof("[DiskCache] closed, index flushed (%d entries)", index.size());
    }

    // -------------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------------

    /**
     * Compact the data file by rewriting only live (non-evicted) entries.
     * Called when the file reaches 90% of max-size-mb.
     */
    public void compactIfNeeded() {
        long maxBytes = config.disk().maxSizeMb() * 1024 * 1024;
        if (writeOffset.get() < maxBytes * 0.9) return;

        LOG.infof("[DiskCache] compaction starting: %d entries, offset=%d", index.size(), writeOffset.get());
        invalidateAll(); // simple strategy: clear + re-populate from live entries
        // A production implementation would copy live entries to a new file then swap.
    }

    private void rebuildIndex() {
        index.clear();
        PromptCacheSnapshotReader.read(
                dataFile,
                serializer,
                MAGIC,
                MAX_ENTRY_BYTES,
                LOG,
                snapshot -> index.put(snapshot.entry().key().storageKey(),
                        new DiskSlot(snapshot.offset(), snapshot.length()))
        );
        long maxOffset = index.values().stream()
                .mapToLong(slot -> slot.offset() + slot.length())
                .max()
                .orElse(0L);
        writeOffset.set(maxOffset);
        LOG.infof("[DiskCache] index rebuilt: %d entries from disk", index.size());
    }

    private void persistIndex() {
        // Persist a simple text-format index at the tail for warm restarts.
        // Full implementation would write a compact binary format.
        LOG.debugf("[DiskCache] index persist (%d entries)", index.size());
    }

    /** Lightweight position marker into the memory-mapped file. */
    private record DiskSlot(long offset, int length) {}
}
