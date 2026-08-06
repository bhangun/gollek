package tech.kayys.gollek.core.util;


import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public final class ThreadPools {
    private static final ForkJoinPool COMMON_PARALLEL = ForkJoinPool.commonPool();
    private static ExecutorService sharedExecutor;
    private static ScheduledExecutorService scheduledExecutor;

    static {
        int cores = Runtime.getRuntime().availableProcessors();
        sharedExecutor = Executors.newFixedThreadPool(
                Math.max(4, cores),
                new GollekThreadFactory());
        scheduledExecutor = Executors.newScheduledThreadPool(
                Math.max(2, cores / 2),
                new GollekThreadFactory());
    }

    public static ExecutorService getSharedPool() {
        return sharedExecutor;
    }

    public static ScheduledExecutorService getScheduledPool() {
        return scheduledExecutor;
    }

    public static ForkJoinPool getParallelPool() {
        return COMMON_PARALLEL;
    }

    public static void shutdown() {
        sharedExecutor.shutdown();
        scheduledExecutor.shutdown();
    }

    private static class GollekThreadFactory implements ThreadFactory {
        private final AtomicInteger counter = new AtomicInteger(0);

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setName("gollek-worker-" + counter.incrementAndGet());
            t.setDaemon(true);
            return t;
        }
    }
}