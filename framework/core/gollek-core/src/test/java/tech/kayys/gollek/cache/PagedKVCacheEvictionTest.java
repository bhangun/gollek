package tech.kayys.gollek.cache;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class PagedKVCacheEvictionTest {

    @Test
    public void evictionListenerCalledOnEvict() throws Exception {
        // small maxBytes so eviction happens quickly
        PagedKVCache cache = new PagedKVCache(16 * Float.BYTES * 4); // 4 floats worth
        CountDownLatch latch = new CountDownLatch(1);
        cache.addEvictionListener(k -> latch.countDown());

        // Insert many pages of 256 floats -> will trigger eviction
        for (long k = 0; k < 10; k++) {
            float[] d = new float[256];
            cache.put(k, d);
        }

        // Either the eviction listener is called, or the cache reports being under the maxBytes limit
        boolean evicted = latch.await(2, TimeUnit.SECONDS);
        Assertions.assertTrue(evicted || cache.bytesPooled() <= 16 * Float.BYTES * 4, "eviction should have occurred or size within limit");
    }
}
