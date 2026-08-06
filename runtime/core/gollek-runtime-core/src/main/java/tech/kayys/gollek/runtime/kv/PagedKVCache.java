package tech.kayys.gollek.runtime.kv;

import java.util.*;

/**
 * PAGED KV CACHE
 * Problem:
 * KV grows linearly → OOM ❌
 * Solution:
 * Paged KV → reuse memory blocks ✅
 */
public final class PagedKVCache {
    private final List<KVPage> pages = new ArrayList<>();

    public KVPage acquire() {
        for (KVPage p : pages) {
            if (!p.used) {
                p.used = true;
                return p;
            }
        }
        KVPage p = new KVPage(null, null);
        p.used = true;
        pages.add(p);
        return p;
    }

    public void release(KVPage page) {
        page.used = false;
    }
}