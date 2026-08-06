package tech.kayys.gollek.prefilldecode;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PrefillDecodeDisaggServiceTest {

    @Test
    void allocatesAndReleasesDecodeBlocks() throws Exception {
        PrefillDecodeDisaggService svc = new PrefillDecodeDisaggService();
        setField(svc, "kvCacheManager", new PagedKVCacheManager(testConfig(10)));
        setField(svc, "prefillBlockFraction", 0.5d);

        invoke(svc, "initPartitions");

        @SuppressWarnings("unchecked")
        List<Integer> blocks = (List<Integer>) invoke(svc, "allocateDecodeBlocks",
                new Class<?>[]{int.class}, 3);
        assertEquals(3, blocks.size());
        assertEquals(3, new HashSet<>(blocks).size(), "blocks should be unique");

        invoke(svc, "releaseDecodeBlocks", new Class<?>[]{List.class}, blocks);

        @SuppressWarnings("unchecked")
        List<Integer> all = (List<Integer>) invoke(svc, "allocateDecodeBlocks",
                new Class<?>[]{int.class}, 5);
        assertEquals(5, all.size());
        invoke(svc, "releaseDecodeBlocks", new Class<?>[]{List.class}, all);
    }

    @Test
    void exhaustsDecodeBlocks() throws Exception {
        PrefillDecodeDisaggService svc = new PrefillDecodeDisaggService();
        setField(svc, "kvCacheManager", new PagedKVCacheManager(testConfig(10)));
        setField(svc, "prefillBlockFraction", 0.5d);

        invoke(svc, "initPartitions");

        Exception thrown = assertThrows(Exception.class, () ->
                invoke(svc, "allocateDecodeBlocks", new Class<?>[]{int.class}, 6));
        Throwable cause = thrown.getCause() != null ? thrown.getCause() : thrown;
        assertTrue(cause instanceof IllegalStateException,
                "expected IllegalStateException, got " + cause.getClass().getName());

        @SuppressWarnings("unchecked")
        List<Integer> all = (List<Integer>) invoke(svc, "allocateDecodeBlocks",
                new Class<?>[]{int.class}, 5);
        assertEquals(5, all.size());
    }

    private static KVCacheConfig testConfig(int totalBlocks) {
        return KVCacheConfig.builder()
                .blockSize(4)
                .totalBlocks(totalBlocks)
                .numLayers(1)
                .numHeads(1)
                .headDim(1)
                .useGpu(false)
                .build();
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static Object invoke(Object target, String name, Object... args) throws Exception {
        return invoke(target, name, new Class<?>[0], args);
    }

    private static Object invoke(Object target, String name, Class<?>[] paramTypes, Object... args) throws Exception {
        Method method = target.getClass().getDeclaredMethod(name, paramTypes);
        method.setAccessible(true);
        return method.invoke(target, args);
    }
}
