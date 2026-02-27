package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link KVCacheManager}.
 */
class KVCacheManagerTest {

    private KVCacheManager manager;

    @BeforeEach
    void setUp() {
        manager = new KVCacheManager(4); // max 4 caches
    }

    @Test
    void createsCacheForNewSession() {
        KVCacheManager.KVCache cache = manager.getOrCreate("session-1", 2, 4, 64);
        assertNotNull(cache);
        assertEquals(0, cache.getSeqLen());
        assertEquals(2, cache.getNumLayers());
        assertEquals(1, manager.size());
    }

    @Test
    void returnsSameCacheForSameSession() {
        KVCacheManager.KVCache first = manager.getOrCreate("session-1", 2, 4, 64);
        KVCacheManager.KVCache second = manager.getOrCreate("session-1", 2, 4, 64);
        assertSame(first, second, "Should return cached instance");
    }

    @Test
    void createsDifferentCachesForDifferentSessions() {
        KVCacheManager.KVCache cache1 = manager.getOrCreate("session-1", 2, 4, 64);
        KVCacheManager.KVCache cache2 = manager.getOrCreate("session-2", 2, 4, 64);
        assertNotSame(cache1, cache2);
        assertEquals(2, manager.size());
    }

    @Test
    void invalidateRemovesCache() {
        manager.getOrCreate("session-1", 2, 4, 64);
        assertEquals(1, manager.size());

        manager.invalidate("session-1");
        assertEquals(0, manager.size());
    }

    @Test
    void invalidateNonExistentDoesNothing() {
        manager.invalidate("nonexistent");
        assertEquals(0, manager.size());
    }

    @Test
    void clearAllRemovesEverything() {
        manager.getOrCreate("s1", 2, 4, 64);
        manager.getOrCreate("s2", 2, 4, 64);
        manager.getOrCreate("s3", 2, 4, 64);
        assertEquals(3, manager.size());

        manager.clearAll();
        assertEquals(0, manager.size());
    }

    @Test
    void evictsOldestWhenFull() throws InterruptedException {
        // Max 4 caches
        manager.getOrCreate("s1", 1, 1, 1);
        Thread.sleep(5);
        manager.getOrCreate("s2", 1, 1, 1);
        Thread.sleep(5);
        manager.getOrCreate("s3", 1, 1, 1);
        Thread.sleep(5);
        manager.getOrCreate("s4", 1, 1, 1);
        assertEquals(4, manager.size());

        // Adding 5th should evict oldest (s1)
        manager.getOrCreate("s5", 1, 1, 1);
        assertTrue(manager.size() <= 4, "Should not exceed max cache size");
    }

    // ── KVCache inner class tests ────────────────────────────────────

    @Test
    void cacheAppendIncreasesSeqLen() {
        KVCacheManager.KVCache cache = manager.getOrCreate("s1", 2, 2, 4);
        assertEquals(0, cache.getSeqLen());

        float[][] keys = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f },
                { 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f } };
        float[][] values = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f },
                { 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f } };
        cache.append(keys, values);

        assertEquals(1, cache.getSeqLen());
    }

    @Test
    void cacheResetClearsSequence() {
        KVCacheManager.KVCache cache = manager.getOrCreate("s1", 2, 2, 4);
        float[][] keys = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f } };
        float[][] values = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f } };
        cache.append(keys, values);
        assertEquals(1, cache.getSeqLen());

        cache.reset();
        assertEquals(0, cache.getSeqLen());
    }

    @Test
    void cacheCloseResetsState() {
        KVCacheManager.KVCache cache = manager.getOrCreate("s1", 2, 2, 4);
        float[][] keys = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f } };
        float[][] values = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f } };
        cache.append(keys, values);

        cache.close();
        assertEquals(0, cache.getSeqLen());
    }

    @Test
    void multipleAppendsGrowSequence() {
        KVCacheManager.KVCache cache = manager.getOrCreate("s1", 1, 2, 4);
        float[][] keys = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f } };
        float[][] values = new float[][] { { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f } };

        cache.append(keys, values);
        cache.append(keys, values);
        cache.append(keys, values);
        assertEquals(3, cache.getSeqLen());
    }
}
