package tech.kayys.gollek.waitscheduler;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingDeque;

import static org.junit.jupiter.api.Assertions.*;

class FluidInferenceSchedulerTest {

    @Test
    void bypassesHeadWhenLaterFits() throws Exception {
        FluidInferenceScheduler scheduler = new FluidInferenceScheduler();
        setField(scheduler, "kvCacheManager", new PagedKVCacheManager(testConfig(10)));
        setField(scheduler, "kvCapacityFraction", 0.5d); // ceiling = 5 blocks
        setField(scheduler, "scanDepth", 4);

        LinkedBlockingDeque<Object> queue = new LinkedBlockingDeque<>();
        queue.add(pendingRequest("head", 8)); // does not fit
        Object smaller = pendingRequest("small", 2); // fits
        queue.add(smaller);
        setField(scheduler, "waitQueue", queue);

        Object candidate = scheduler.findAdmissibleForTest();
        assertSame(smaller, candidate, "expected smaller request to bypass head");
    }

    @Test
    void respectsScanDepth() throws Exception {
        FluidInferenceScheduler scheduler = new FluidInferenceScheduler();
        setField(scheduler, "kvCacheManager", new PagedKVCacheManager(testConfig(10)));
        setField(scheduler, "kvCapacityFraction", 0.5d);
        setField(scheduler, "scanDepth", 1);

        LinkedBlockingDeque<Object> queue = new LinkedBlockingDeque<>();
        queue.add(pendingRequest("head", 8)); // does not fit
        queue.add(pendingRequest("small", 2)); // would fit but scanDepth=1
        setField(scheduler, "waitQueue", queue);

        Object candidate = scheduler.findAdmissibleForTest();
        assertNull(candidate, "expected no bypass when scanDepth=1");
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

    private static Object pendingRequest(String id, int predictedBlocks) throws Exception {
        Class<?> pending = Class.forName(
                "tech.kayys.gollek.waitscheduler.FluidInferenceScheduler$PendingRequest");
        Constructor<?> ctor = pending.getDeclaredConstructors()[0];
        ctor.setAccessible(true);

        InferenceRequest req = InferenceRequest.builder()
                .requestId(id)
                .model("model")
                .messages(List.of(Message.user("hi")))
                .build();

        return ctor.newInstance(
                id,
                req,
                new CompletableFuture<>(),
                predictedBlocks,
                System.nanoTime()
        );
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field f = target.getClass().getDeclaredField(name);
        f.setAccessible(true);
        f.set(target, value);
    }
}
