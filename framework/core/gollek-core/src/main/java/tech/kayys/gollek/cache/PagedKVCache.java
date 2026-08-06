package tech.kayys.gollek.cache;


import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import java.nio.ByteBuffer;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * PagedKVCache with copy-on-write semantics and LRU eviction.
 *
 * - Stores pages as direct ByteBuffer for zero-copy views.
 * - Each page has an atomic reference count for pin/unpin semantics.
 * - Copy-on-write on put when page is shared.
 * - Simple LRU eviction that skips pinned pages.
 */
public class PagedKVCache {
    private final Map<Long, Page> pages = new ConcurrentHashMap<>();
    private final long maxBytes;
    private final AtomicLong bytesPooled = new AtomicLong(0);
    private final AtomicLong hits = new AtomicLong(0);
    private final AtomicLong misses = new AtomicLong(0);

    private final LinkedHashMap<Long, Boolean> lru = new LinkedHashMap<>(16, 0.75f, true);

    // eviction listeners notified when a page is evicted
    private final java.util.List<java.util.function.Consumer<Long>> evictionListeners = java.util.Collections.synchronizedList(new java.util.ArrayList<>());

    public PagedKVCache(long maxBytes) {
        if (maxBytes <= 0) throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "maxBytes must be > 0");
        this.maxBytes = maxBytes;
    }

    static class Page {
        final AtomicInteger refCount = new AtomicInteger(0);
        volatile ByteBuffer buf;
        final int capacity;

        Page(ByteBuffer buf) {
            this.buf = Objects.requireNonNull(buf);
            this.capacity = buf.capacity();
        }

        void retain() { refCount.incrementAndGet(); }
        void release() { int v = refCount.decrementAndGet(); if (v < 0) throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "negative refcount"); }
        int refs() { return refCount.get(); }

        ByteBuffer readOnlyView() { ByteBuffer v = buf.asReadOnlyBuffer(); v.rewind(); return v; }

        synchronized ByteBuffer ensureUnsharedCopy() {
            if (refs() <= 1) return buf;
            ByteBuffer newBuf = ByteBuffer.allocateDirect(buf.capacity());
            ByteBuffer src = buf.duplicate(); src.rewind();
            newBuf.put(src);
            newBuf.rewind();
            this.buf = newBuf;
            return newBuf;
        }
    }

    public void put(long key, float[] data) {
        Objects.requireNonNull(data);
        ByteBuffer newBuf = ByteBuffer.allocateDirect(data.length * Float.BYTES);
        newBuf.asFloatBuffer().put(data);
        newBuf.rewind();

        pages.compute(key, (k, old) -> {
            if (old == null) {
                Page p = new Page(newBuf);
                bytesPooled.addAndGet(p.capacity);
                touchLRU(k);
                return p;
            } else {
                synchronized (old) {
                    int prevCap = old.capacity;
                    if (old.refs() > 0) {
                        Page p = new Page(newBuf);
                        bytesPooled.addAndGet(p.capacity - prevCap);
                        touchLRU(k);
                        return p;
                    } else {
                        bytesPooled.addAndGet(-old.capacity);
                        old.buf = newBuf;
                        bytesPooled.addAndGet(old.capacity);
                        touchLRU(k);
                        return old;
                    }
                }
            }
        });
        // Perform eviction outside of compute to avoid subtle concurrency issues with map mutations during compute
        evictIfNeeded();
    }

    public ByteBuffer getView(long key) {
        Page p = pages.get(key);
        if (p == null) { misses.incrementAndGet(); return null; }
        hits.incrementAndGet();
        touchLRU(key);
        return p.readOnlyView();
    }

    public ByteBuffer pin(long key) {
        Page p = pages.get(key);
        if (p == null) { misses.incrementAndGet(); return null; }
        p.retain();
        hits.incrementAndGet();
        touchLRU(key);
        ByteBuffer dup = p.buf.duplicate(); dup.rewind();
        return dup;
    }

    public void unpin(long key) {
        Page p = pages.get(key);
        if (p == null) throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "unpin on missing key");
        p.release();
    }

    public void remove(long key) {
        Page p = pages.remove(key);
        if (p != null) {
            synchronized (p) {
                if (p.refs() > 0) throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "removing pinned page");
                bytesPooled.addAndGet(-p.capacity);
            }
            synchronized (lru) { lru.remove(key); }
        }
    }

    public void clear() {
        pages.clear();
        synchronized (lru) { lru.clear(); }
        bytesPooled.set(0);
    }

    private void touchLRU(long key) { synchronized (lru) { lru.put(key, Boolean.TRUE); } }

    private void evictIfNeeded() {
        while (bytesPooled.get() > maxBytes) {
            Long victim = null;
            synchronized (lru) { for (Map.Entry<Long, Boolean> e : lru.entrySet()) { victim = e.getKey(); break; } if (victim == null) return; }
            Page p = pages.get(victim);
            if (p == null) { synchronized (lru) { lru.remove(victim); } continue; }
            synchronized (p) {
                if (p.refs() > 0) { touchLRU(victim); break; }
                if (pages.remove(victim, p)) {
                    bytesPooled.addAndGet(-p.capacity);
                    synchronized (lru) { lru.remove(victim); }
                    // notify listeners
                    try {
                        for (java.util.function.Consumer<Long> c : evictionListeners) {
                            c.accept(victim);
                        }
                    } catch (Exception ex) {
                        // ignore listener failures
                    }
                }
            }
        }
    }

    public long bytesPooled() { return bytesPooled.get(); }
    public long hits() { return hits.get(); }
    public long misses() { return misses.get(); }
    public int pageCount() { return pages.size(); }

    public void forEachPage(Consumer<Map.Entry<Long, Page>> consumer) { pages.entrySet().forEach(consumer); }

    public void addEvictionListener(java.util.function.Consumer<Long> listener) {
        evictionListeners.add(listener);
    }
}

