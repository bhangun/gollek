package tech.kayys.gollek.multimodal.gpu;

import org.jboss.logging.Logger;

/**
 * GPU accelerator manager.
 * Coordinates GPU optimizations for multimodal inference.
 */
public class GPUAccelerator {

    private static final Logger log = Logger.getLogger(GPUAccelerator.class);

    private final GPUMemoryPool memoryPool;
    private final CUDAStreamManager streamManager;
    private final int gpuId;
    private volatile boolean initialized;

    /**
     * Create GPU accelerator.
     */
    public GPUAccelerator() {
        this(0);
    }

    /**
     * Create GPU accelerator for specific GPU.
     *
     * @param gpuId GPU device ID
     */
    public GPUAccelerator(int gpuId) {
        this.gpuId = gpuId;
        // Memory pool: 256MB blocks, max 100 blocks
        this.memoryPool = new GPUMemoryPool(256 * 1024 * 1024, 100);
        // 4 CUDA streams for parallelism
        this.streamManager = new CUDAStreamManager(4);
        this.initialized = false;

        log.infof("GPU Accelerator created for GPU %d", gpuId);
    }

    /**
     * Initialize GPU accelerator.
     */
    public void initialize() {
        if (initialized) {
            log.warn("GPU Accelerator already initialized");
            return;
        }

        log.infof("Initializing GPU Accelerator for GPU %d", gpuId);

        // Initialize CUDA streams
        streamManager.initialize();

        // In production, this would:
        // 1. cudaSetDevice(gpuId)
        // 2. Check GPU capabilities
        // 3. Configure compute mode
        // 4. Initialize cuBLAS/cuDNN handles

        initialized = true;
        log.info("GPU Accelerator initialized");
    }

    /**
     * Get memory pool.
     */
    public GPUMemoryPool getMemoryPool() {
        return memoryPool;
    }

    /**
     * Get stream manager.
     */
    public CUDAStreamManager getStreamManager() {
        return streamManager;
    }

    /**
     * Get GPU statistics.
     */
    public GPUStats getStats() {
        GPUMemoryPool.PoolStats poolStats = memoryPool.getStats();
        CUDAStreamManager.CUDAStreamStats streamStats = streamManager.getOverallStats();

        return new GPUStats(
            gpuId,
            initialized,
            poolStats,
            streamStats
        );
    }

    /**
     * Check if accelerator is initialized.
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get GPU ID.
     */
    public int getGpuId() {
        return gpuId;
    }

    /**
     * Shutdown accelerator.
     */
    public void shutdown() {
        log.info("Shutting down GPU Accelerator");

        streamManager.shutdown();
        memoryPool.clear();
        memoryPool.close();

        initialized = false;
        log.info("GPU Accelerator shut down");
    }

    /**
     * GPU statistics.
     */
    public static class GPUStats {
        public final int gpuId;
        public final boolean initialized;
        public final GPUMemoryPool.PoolStats memoryPool;
        public final CUDAStreamManager.CUDAStreamStats streams;

        public GPUStats(int gpuId, boolean initialized,
                       GPUMemoryPool.PoolStats memoryPool,
                       CUDAStreamManager.CUDAStreamStats streams) {
            this.gpuId = gpuId;
            this.initialized = initialized;
            this.memoryPool = memoryPool;
            this.streams = streams;
        }

        @Override
        public String toString() {
            return String.format(
                "GPUStats{gpuId=%d, initialized=%s, memory=%s, streams=%s}",
                gpuId, initialized, memoryPool, streams
            );
        }
    }
}
