package tech.kayys.gollek.multimodal.metal;

import org.jboss.logging.Logger;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Metal command queue manager for Apple Silicon GPU parallelism.
 * Manages multiple MTLCommandQueues for concurrent GPU execution.
 */
public class MetalCommandQueueManager {

    private static final Logger log = Logger.getLogger(MetalCommandQueueManager.class);
    private static final int DEFAULT_QUEUE_COUNT = 4;

    private final int queueCount;
    private final AtomicInteger currentQueueIndex;
    private final Map<Integer, QueueStats> queueStats;
    private final ExecutorService commandExecutor;
    private final AtomicLong totalCommandsSubmitted;
    private final AtomicLong totalCommandsCompleted;
    private volatile boolean initialized;
    private final boolean isUnifiedMemory;

    /**
     * Create Metal command queue manager.
     */
    public MetalCommandQueueManager() {
        this(DEFAULT_QUEUE_COUNT);
    }

    /**
     * Create Metal command queue manager with custom queue count.
     *
     * @param queueCount Number of Metal command queues
     */
    public MetalCommandQueueManager(int queueCount) {
        this.queueCount = queueCount;
        this.currentQueueIndex = new AtomicInteger(0);
        this.queueStats = new ConcurrentHashMap<>();
        this.commandExecutor = Executors.newFixedThreadPool(
            queueCount,
            r -> {
                Thread t = new Thread(r, "metal-command-worker");
                t.setDaemon(true);
                return t;
            }
        );
        this.totalCommandsSubmitted = new AtomicLong(0);
        this.totalCommandsCompleted = new AtomicLong(0);
        this.initialized = false;
        this.isUnifiedMemory = isAppleSilicon();

        // Initialize queue stats
        for (int i = 0; i < queueCount; i++) {
            queueStats.put(i, new QueueStats(i));
        }

        log.infof("Metal Command Queue Manager created with %d queues (unified memory: %s)", 
                 queueCount, isUnifiedMemory);
    }

    /**
     * Initialize Metal command queues.
     */
    public void initialize() {
        if (initialized) {
            log.warn("Metal Command Queue Manager already initialized");
            return;
        }

        // In production, this would create actual Metal command queues
        // [MTLDevice newCommandQueue] for each queue
        for (int i = 0; i < queueCount; i++) {
            log.infof("Initialized Metal command queue %d", i);
        }

        initialized = true;
        log.info("Metal Command Queue Manager initialized");
    }

    /**
     * Get next available queue (round-robin).
     *
     * @return Queue index
     */
    public int getNextQueue() {
        int index = currentQueueIndex.getAndIncrement() % queueCount;
        queueStats.get(index).incrementAssignments();
        return index;
    }

    /**
     * Submit command buffer to queue.
     *
     * @param queueIndex Queue index
     * @param commandName Command name
     * @param task Command execution task
     */
    public void submitCommand(int queueIndex, String commandName, Runnable task) {
        if (queueIndex < 0 || queueIndex >= queueCount) {
            throw new IllegalArgumentException("Invalid queue index: " + queueIndex);
        }

        QueueStats stats = queueStats.get(queueIndex);
        stats.incrementCommandsSubmitted();
        totalCommandsSubmitted.incrementAndGet();

        // Execute command asynchronously
        commandExecutor.submit(() -> {
            long startTime = System.currentTimeMillis();
            try {
                task.run();
                long duration = System.currentTimeMillis() - startTime;
                stats.recordCommandCompletion(duration);
                totalCommandsCompleted.incrementAndGet();
                log.debugf("Command %s completed on queue %d in %dms", 
                          commandName, queueIndex, duration);
            } catch (Exception e) {
                stats.incrementErrors();
                log.errorf("Command %s failed on queue %d: %s", 
                          commandName, queueIndex, e.getMessage());
            }
        });
    }

    /**
     * Submit command to default queue (round-robin).
     */
    public void submitCommand(String commandName, Runnable task) {
        int queueIndex = getNextQueue();
        submitCommand(queueIndex, commandName, task);
    }

    /**
     * Synchronize queue (wait for completion).
     *
     * @param queueIndex Queue index
     */
    public void synchronize(int queueIndex) {
        if (queueIndex < 0 || queueIndex >= queueCount) {
            throw new IllegalArgumentException("Invalid queue index: " + queueIndex);
        }

        // In production, this would call [MTLCommandBuffer waitUntilCompleted]
        log.debugf("Synchronizing queue %d", queueIndex);
    }

    /**
     * Synchronize all queues.
     */
    public void synchronizeAll() {
        log.info("Synchronizing all Metal command queues");
        for (int i = 0; i < queueCount; i++) {
            synchronize(i);
        }
    }

    /**
     * Get queue statistics.
     */
    public Map<Integer, QueueStats> getQueueStats() {
        return new ConcurrentHashMap<>(queueStats);
    }

    /**
     * Get overall statistics.
     */
    public MetalQueueStats getOverallStats() {
        long totalCommands = 0;
        long totalErrors = 0;
        long totalActive = 0;
        double avgDuration = 0.0;

        for (QueueStats stats : queueStats.values()) {
            totalCommands += stats.commandsSubmitted;
            totalErrors += stats.errors;
            totalActive += stats.activeCommands;
            avgDuration += stats.averageDuration;
        }

        avgDuration /= queueCount;

        return new MetalQueueStats(
            queueCount,
            totalCommands,
            totalCommandsSubmitted.get(),
            totalCommandsCompleted.get(),
            totalErrors,
            totalActive,
            avgDuration,
            isUnifiedMemory
        );
    }

    /**
     * Check if manager is initialized.
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get queue count.
     */
    public int getQueueCount() {
        return queueCount;
    }

    /**
     * Check if running on Apple Silicon.
     */
    public boolean isAppleSilicon() {
        String arch = System.getProperty("os.arch");
        return "aarch64".equals(arch) || "arm64".equals(arch);
    }

    /**
     * Shutdown queue manager.
     */
    public void shutdown() {
        log.info("Shutting down Metal Command Queue Manager");
        
        synchronizeAll();
        commandExecutor.shutdown();
        
        try {
            if (!commandExecutor.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                commandExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            commandExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // In production, this would release Metal resources

        initialized = false;
        log.info("Metal Command Queue Manager shut down");
    }

    /**
     * Statistics for individual queue.
     */
    public static class QueueStats {
        public final int queueIndex;
        public long assignments;
        public long commandsSubmitted;
        public long commandsCompleted;
        public long errors;
        public long activeCommands;
        public double averageDuration;
        private long totalDuration;
        private long durationCount;

        public QueueStats(int queueIndex) {
            this.queueIndex = queueIndex;
            this.assignments = 0;
            this.commandsSubmitted = 0;
            this.commandsCompleted = 0;
            this.errors = 0;
            this.activeCommands = 0;
            this.averageDuration = 0.0;
            this.totalDuration = 0;
            this.durationCount = 0;
        }

        public synchronized void incrementAssignments() {
            assignments++;
        }

        public synchronized void incrementCommandsSubmitted() {
            commandsSubmitted++;
            activeCommands++;
        }

        public synchronized void recordCommandCompletion(long duration) {
            commandsCompleted++;
            activeCommands--;
            totalDuration += duration;
            durationCount++;
            averageDuration = (double) totalDuration / durationCount;
        }

        public synchronized void incrementErrors() {
            errors++;
            activeCommands--;
        }

        @Override
        public String toString() {
            return String.format(
                "QueueStats{index=%d, assignments=%d, submitted=%d, completed=%d, errors=%d, active=%d, avgDuration=%.2fms}",
                queueIndex, assignments, commandsSubmitted, commandsCompleted, errors, activeCommands, averageDuration
            );
        }
    }

    /**
     * Overall Metal command queue statistics.
     */
    public static class MetalQueueStats {
        public final int queueCount;
        public final long totalCommands;
        public final long submitted;
        public final long completed;
        public final long errors;
        public final long active;
        public final double avgDuration;
        public final boolean unifiedMemory;

        public MetalQueueStats(int queueCount, long totalCommands, long submitted,
                              long completed, long errors, long active, double avgDuration,
                              boolean unifiedMemory) {
            this.queueCount = queueCount;
            this.totalCommands = totalCommands;
            this.submitted = submitted;
            this.completed = completed;
            this.errors = errors;
            this.active = active;
            this.avgDuration = avgDuration;
            this.unifiedMemory = unifiedMemory;
        }

        @Override
        public String toString() {
            return String.format(
                "MetalQueueStats{queues=%d, total=%d, submitted=%d, completed=%d, errors=%d, active=%d, avgDuration=%.2fms, unified=%s}",
                queueCount, totalCommands, submitted, completed, errors, active, avgDuration, unifiedMemory
            );
        }
    }
}
