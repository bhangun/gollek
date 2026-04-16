package tech.kayys.gollek.multimodal.metal;

import org.jboss.logging.Logger;

/**
 * Metal accelerator manager for Apple Silicon.
 * Coordinates Metal GPU optimizations for multimodal inference.
 */
public class MetalAccelerator {

    private static final Logger log = Logger.getLogger(MetalAccelerator.class);

    private final MetalMemoryPool memoryPool;
    private final MetalCommandQueueManager commandQueueManager;
    private final int deviceId;
    private volatile boolean initialized;
    private final boolean isUnifiedMemory;

    /**
     * Create Metal accelerator.
     */
    public MetalAccelerator() {
        this(0);
    }

    /**
     * Create Metal accelerator for specific device.
     *
     * @param deviceId Metal device ID (0 for default)
     */
    public MetalAccelerator(int deviceId) {
        this.deviceId = deviceId;
        this.isUnifiedMemory = isAppleSilicon();
        
        // Memory pool: 256MB blocks, max 100 blocks
        // Unified memory architecture allows more efficient pooling
        this.memoryPool = new MetalMemoryPool(256 * 1024 * 1024, 100, isUnifiedMemory);
        
        // 4 command queues for parallelism (matches M-series GPU cores)
        this.commandQueueManager = new MetalCommandQueueManager(4);
        
        this.initialized = false;

        log.infof("Metal Accelerator created for device %d (unified memory: %s)", 
                 deviceId, isUnifiedMemory);
    }

    /**
     * Initialize Metal accelerator.
     */
    public void initialize() {
        if (initialized) {
            log.warn("Metal Accelerator already initialized");
            return;
        }

        log.infof("Initializing Metal Accelerator for device %d", deviceId);

        // Initialize command queues
        commandQueueManager.initialize();

        // In production, this would:
        // 1. MTLCreateSystemDefaultDevice()
        // 2. Check device capabilities
        // 3. Create command queues
        // 4. Initialize MPS (Metal Performance Shaders) kernels

        initialized = true;
        log.info("Metal Accelerator initialized");
    }

    /**
     * Get memory pool.
     */
    public MetalMemoryPool getMemoryPool() {
        return memoryPool;
    }

    /**
     * Get command queue manager.
     */
    public MetalCommandQueueManager getCommandQueueManager() {
        return commandQueueManager;
    }

    /**
     * Get Metal statistics.
     */
    public MetalStats getStats() {
        MetalMemoryPool.PoolStats poolStats = memoryPool.getStats();
        MetalCommandQueueManager.MetalQueueStats queueStats = 
            commandQueueManager.getOverallStats();

        return new MetalStats(
            deviceId,
            initialized,
            poolStats,
            queueStats,
            isUnifiedMemory
        );
    }

    /**
     * Check if accelerator is initialized.
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get device ID.
     */
    public int getDeviceId() {
        return deviceId;
    }

    /**
     * Check if running on Apple Silicon.
     */
    public boolean isAppleSilicon() {
        String arch = System.getProperty("os.arch");
        return "aarch64".equals(arch) || "arm64".equals(arch);
    }

    /**
     * Shutdown accelerator.
     */
    public void shutdown() {
        log.info("Shutting down Metal Accelerator");

        commandQueueManager.shutdown();
        memoryPool.clear();
        memoryPool.close();

        initialized = false;
        log.info("Metal Accelerator shut down");
    }

    /**
     * Metal statistics.
     */
    public static class MetalStats {
        public final int deviceId;
        public final boolean initialized;
        public final MetalMemoryPool.PoolStats memoryPool;
        public final MetalCommandQueueManager.MetalQueueStats queues;
        public final boolean unifiedMemory;

        public MetalStats(int deviceId, boolean initialized,
                         MetalMemoryPool.PoolStats memoryPool,
                         MetalCommandQueueManager.MetalQueueStats queues,
                         boolean unifiedMemory) {
            this.deviceId = deviceId;
            this.initialized = initialized;
            this.memoryPool = memoryPool;
            this.queues = queues;
            this.unifiedMemory = unifiedMemory;
        }

        @Override
        public String toString() {
            return String.format(
                "MetalStats{deviceId=%d, initialized=%s, memory=%s, queues=%s, unified=%s}",
                deviceId, initialized, memoryPool, queues, unifiedMemory
            );
        }
    }
}
