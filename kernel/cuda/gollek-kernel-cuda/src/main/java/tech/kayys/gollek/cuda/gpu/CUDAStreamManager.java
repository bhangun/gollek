package tech.kayys.gollek.multimodal.gpu;

import org.jboss.logging.Logger;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CUDA stream manager for parallel GPU execution.
 * Manages multiple CUDA streams for concurrent kernel execution.
 */
public class CUDAStreamManager {

    private static final Logger log = Logger.getLogger(CUDAStreamManager.class);
    private static final int DEFAULT_STREAM_COUNT = 4;

    private final int streamCount;
    private final AtomicInteger currentStreamIndex;
    private final Map<Integer, StreamStats> streamStats;
    private final ExecutorService streamExecutor;
    private final AtomicLong totalKernelsLaunched;
    private final AtomicLong totalKernelsCompleted;
    private volatile boolean initialized;

    /**
     * Create CUDA stream manager.
     */
    public CUDAStreamManager() {
        this(DEFAULT_STREAM_COUNT);
    }

    /**
     * Create CUDA stream manager with custom stream count.
     *
     * @param streamCount Number of CUDA streams
     */
    public CUDAStreamManager(int streamCount) {
        this.streamCount = streamCount;
        this.currentStreamIndex = new AtomicInteger(0);
        this.streamStats = new ConcurrentHashMap<>();
        this.streamExecutor = Executors.newFixedThreadPool(
            streamCount,
            r -> {
                Thread t = new Thread(r, "cuda-stream-worker");
                t.setDaemon(true);
                return t;
            }
        );
        this.totalKernelsLaunched = new AtomicLong(0);
        this.totalKernelsCompleted = new AtomicLong(0);
        this.initialized = false;

        // Initialize stream stats
        for (int i = 0; i < streamCount; i++) {
            streamStats.put(i, new StreamStats(i));
        }

        log.infof("CUDA Stream Manager created with %d streams", streamCount);
    }

    /**
     * Initialize CUDA streams.
     */
    public void initialize() {
        if (initialized) {
            log.warn("CUDA Stream Manager already initialized");
            return;
        }

        // In production, this would create actual CUDA streams
        // cudaStreamCreate() for each stream
        for (int i = 0; i < streamCount; i++) {
            log.infof("Initialized CUDA stream %d", i);
        }

        initialized = true;
        log.info("CUDA Stream Manager initialized");
    }

    /**
     * Get next available stream (round-robin).
     *
     * @return Stream index
     */
    public int getNextStream() {
        int index = currentStreamIndex.getAndIncrement() % streamCount;
        streamStats.get(index).incrementAssignments();
        return index;
    }

    /**
     * Launch kernel on stream.
     *
     * @param streamIndex Stream index
     * @param kernelName Kernel name
     * @param task Kernel execution task
     */
    public void launchKernel(int streamIndex, String kernelName, Runnable task) {
        if (streamIndex < 0 || streamIndex >= streamCount) {
            throw new IllegalArgumentException("Invalid stream index: " + streamIndex);
        }

        StreamStats stats = streamStats.get(streamIndex);
        stats.incrementKernelsLaunched();
        totalKernelsLaunched.incrementAndGet();

        // Execute kernel asynchronously
        streamExecutor.submit(() -> {
            long startTime = System.currentTimeMillis();
            try {
                task.run();
                long duration = System.currentTimeMillis() - startTime;
                stats.recordKernelCompletion(duration);
                totalKernelsCompleted.incrementAndGet();
                log.debugf("Kernel %s completed on stream %d in %dms", 
                          kernelName, streamIndex, duration);
            } catch (Exception e) {
                stats.incrementErrors();
                log.errorf("Kernel %s failed on stream %d: %s", 
                          kernelName, streamIndex, e.getMessage());
            }
        });
    }

    /**
     * Launch kernel on default stream (round-robin).
     */
    public void launchKernel(String kernelName, Runnable task) {
        int streamIndex = getNextStream();
        launchKernel(streamIndex, kernelName, task);
    }

    /**
     * Synchronize stream (wait for completion).
     *
     * @param streamIndex Stream index
     */
    public void synchronize(int streamIndex) {
        if (streamIndex < 0 || streamIndex >= streamCount) {
            throw new IllegalArgumentException("Invalid stream index: " + streamIndex);
        }

        // In production, this would call cudaStreamSynchronize()
        log.debugf("Synchronizing stream %d", streamIndex);
    }

    /**
     * Synchronize all streams.
     */
    public void synchronizeAll() {
        log.info("Synchronizing all streams");
        for (int i = 0; i < streamCount; i++) {
            synchronize(i);
        }
    }

    /**
     * Get stream statistics.
     */
    public Map<Integer, StreamStats> getStreamStats() {
        return new ConcurrentHashMap<>(streamStats);
    }

    /**
     * Get overall statistics.
     */
    public CUDAStreamStats getOverallStats() {
        long totalKernels = 0;
        long totalErrors = 0;
        long totalActive = 0;
        double avgDuration = 0.0;

        for (StreamStats stats : streamStats.values()) {
            totalKernels += stats.kernelsLaunched;
            totalErrors += stats.errors;
            totalActive += stats.activeKernels;
            avgDuration += stats.averageDuration;
        }

        avgDuration /= streamCount;

        return new CUDAStreamStats(
            streamCount,
            totalKernels,
            totalKernelsLaunched.get(),
            totalKernelsCompleted.get(),
            totalErrors,
            totalActive,
            avgDuration
        );
    }

    /**
     * Check if manager is initialized.
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get stream count.
     */
    public int getStreamCount() {
        return streamCount;
    }

    /**
     * Shutdown stream manager.
     */
    public void shutdown() {
        log.info("Shutting down CUDA Stream Manager");
        
        synchronizeAll();
        streamExecutor.shutdown();
        
        try {
            if (!streamExecutor.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                streamExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            streamExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // In production, this would destroy CUDA streams
        // cudaStreamDestroy() for each stream

        initialized = false;
        log.info("CUDA Stream Manager shut down");
    }

    /**
     * Statistics for individual stream.
     */
    public static class StreamStats {
        public final int streamIndex;
        public long assignments;
        public long kernelsLaunched;
        public long kernelsCompleted;
        public long errors;
        public long activeKernels;
        public double averageDuration;
        private long totalDuration;
        private long durationCount;

        public StreamStats(int streamIndex) {
            this.streamIndex = streamIndex;
            this.assignments = 0;
            this.kernelsLaunched = 0;
            this.kernelsCompleted = 0;
            this.errors = 0;
            this.activeKernels = 0;
            this.averageDuration = 0.0;
            this.totalDuration = 0;
            this.durationCount = 0;
        }

        public synchronized void incrementAssignments() {
            assignments++;
        }

        public synchronized void incrementKernelsLaunched() {
            kernelsLaunched++;
            activeKernels++;
        }

        public synchronized void recordKernelCompletion(long duration) {
            kernelsCompleted++;
            activeKernels--;
            totalDuration += duration;
            durationCount++;
            averageDuration = (double) totalDuration / durationCount;
        }

        public synchronized void incrementErrors() {
            errors++;
            activeKernels--;
        }

        @Override
        public String toString() {
            return String.format(
                "StreamStats{index=%d, assignments=%d, launched=%d, completed=%d, errors=%d, active=%d, avgDuration=%.2fms}",
                streamIndex, assignments, kernelsLaunched, kernelsCompleted, errors, activeKernels, averageDuration
            );
        }
    }

    /**
     * Overall CUDA stream statistics.
     */
    public static class CUDAStreamStats {
        public final int streamCount;
        public final long totalKernels;
        public final long launched;
        public final long completed;
        public final long errors;
        public final long active;
        public final double avgDuration;

        public CUDAStreamStats(int streamCount, long totalKernels, long launched,
                              long completed, long errors, long active, double avgDuration) {
            this.streamCount = streamCount;
            this.totalKernels = totalKernels;
            this.launched = launched;
            this.completed = completed;
            this.errors = errors;
            this.active = active;
            this.avgDuration = avgDuration;
        }

        @Override
        public String toString() {
            return String.format(
                "CUDAStreamStats{streams=%d, total=%d, launched=%d, completed=%d, errors=%d, active=%d, avgDuration=%.2fms}",
                streamCount, totalKernels, launched, completed, errors, active, avgDuration
            );
        }
    }
}
