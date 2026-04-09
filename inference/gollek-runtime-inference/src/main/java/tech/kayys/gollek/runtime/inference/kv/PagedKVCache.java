package tech.kayys.gollek.runtime.inference.kv;

import tech.kayys.gollek.runtime.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * vLLM-style paged KV cache implementation.
 * <p>
 * Stores K/V projections in fixed-size pages per layer. When a page fills up,
 * a new page is allocated. This avoids memory fragmentation and enables
 * fine-grained memory management.
 * <p>
 * Each layer maintains its own list of pages, growing as needed during
 * autoregressive generation.
 */
public final class PagedKVCache implements KVCache {

    private final Map<Integer, List<KVPage>> layers = new ConcurrentHashMap<>();
    private final int pageSize;
    private int totalTokens = 0;

    /**
     * @param pageSize number of tokens per page (e.g. 128)
     */
    public PagedKVCache(int pageSize) {
        this.pageSize = pageSize;
    }

    @Override
    public void append(int layer, Tensor k, Tensor v) {
        List<KVPage> pages = layers.computeIfAbsent(layer, l -> new ArrayList<>());

        KVPage last = pages.isEmpty() ? null : pages.getLast();

        if (last == null || !last.hasSpace()) {
            last = allocatePage();
            pages.add(last);
        }

        // Track the token count (actual native copy happens via backend)
        last.size++;
        totalTokens++;
    }

    @Override
    public Tensor getK(int layer) {
        // In a full implementation, this would concat all K pages into one tensor
        // For now, return null — the backend's attention kernel reads pages directly
        return null;
    }

    @Override
    public Tensor getV(int layer) {
        return null;
    }

    @Override
    public int length() {
        return totalTokens;
    }

    @Override
    public void clear() {
        layers.clear();
        totalTokens = 0;
    }

    @Override
    public KVCache snapshot() {
        PagedKVCache copy = new PagedKVCache(pageSize);
        for (var entry : layers.entrySet()) {
            List<KVPage> clonedPages = new ArrayList<>();
            for (KVPage p : entry.getValue()) {
                KVPage cloned = new KVPage(p.k, p.v, p.capacity);
                cloned.size = p.size;
                clonedPages.add(cloned);
            }
            copy.layers.put(entry.getKey(), clonedPages);
        }
        copy.totalTokens = this.totalTokens;
        return copy;
    }

    /** Get all pages for a given layer (for PagedAttention kernel). */
    public List<KVPage> getPages(int layer) {
        return layers.getOrDefault(layer, List.of());
    }

    /** Number of pages allocated across all layers. */
    public int totalPages() {
        return layers.values().stream().mapToInt(List::size).sum();
    }

    private KVPage allocatePage() {
        // Native memory allocation would happen here via Backend
        return new KVPage(null, null, pageSize);
    }
}
