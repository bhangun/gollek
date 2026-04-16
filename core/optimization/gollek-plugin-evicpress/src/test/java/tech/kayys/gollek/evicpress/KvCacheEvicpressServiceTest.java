package tech.kayys.gollek.evicpress;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.kvcache.PhysicalBlockPool;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

class KvCacheEvicpressServiceTest {

    @Test
    void evictToCpuUsesMonotonicCursor() throws Exception {
        KvCacheEvicpressService svc = new KvCacheEvicpressService();
        PagedKVCacheManager kv = new PagedKVCacheManager(KVCacheConfig.builder()
                .blockSize(4)
                .totalBlocks(8)
                .numLayers(1)
                .numHeads(1)
                .headDim(1)
                .useGpu(false)
                .build());

        setField(svc, "kvCacheManager", kv);
        setField(svc, "evictPoolBlocks", 4);
        setField(svc, "enabled", true);
        setField(svc, "libraryPath", "/tmp/libgollek_evicpress.so");
        invokeInit(svc);

        PhysicalBlockPool pool = kv.getBlockPool();
        long bytesPerBlock = pool.getBytesPerBlock();
        long stride = bytesPerBlock * kv.getConfig().getNumLayers() * 2L;

        long first = getCursor(svc);
        invokeEvict(svc, pool, 0, kv.getConfig().getNumLayers());
        long second = getCursor(svc);
        invokeEvict(svc, pool, 1, kv.getConfig().getNumLayers());
        long third = getCursor(svc);

        assertEquals(first + stride, second);
        assertEquals(second + stride, third);
    }

    private static void invokeInit(KvCacheEvicpressService svc) throws Exception {
        var m = KvCacheEvicpressService.class.getDeclaredMethod("init");
        m.setAccessible(true);
        m.invoke(svc);
    }

    private static void invokeEvict(KvCacheEvicpressService svc, PhysicalBlockPool pool, int blockId, int numLayers)
            throws Exception {
        var m = KvCacheEvicpressService.class.getDeclaredMethod("evictToCpu",
                PhysicalBlockPool.class, int.class, int.class);
        m.setAccessible(true);
        m.invoke(svc, pool, blockId, numLayers);
    }

    private static long getCursor(KvCacheEvicpressService svc) throws Exception {
        Field f = KvCacheEvicpressService.class.getDeclaredField("evictPoolCursor");
        f.setAccessible(true);
        return ((java.util.concurrent.atomic.AtomicLong) f.get(svc)).get();
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field f = target.getClass().getDeclaredField(name);
        f.setAccessible(true);
        f.set(target, value);
    }
}
