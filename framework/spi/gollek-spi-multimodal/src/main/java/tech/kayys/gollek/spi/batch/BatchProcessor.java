package tech.kayys.gollek.spi.batch;

import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.spi.model.MultimodalRequest;
import tech.kayys.alkhawarizm.spi.model.MultimodalResponse;
import tech.kayys.gollek.spi.processor.MultimodalProcessor;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Batch processor for multimodal inference.
 * Efficiently processes multiple requests together.
 */
public class BatchProcessor {

    private static final Logger log = Logger.getLogger(BatchProcessor.class);

    private static final int DEFAULT_MAX_BATCH_SIZE = 32;
    private static final Duration DEFAULT_BATCH_TIMEOUT = Duration.ofMillis(100);
    private static final Duration DEFAULT_REQUEST_TIMEOUT = Duration.ofMinutes(2);

    private final int maxBatchSize;
    private final Duration batchTimeout;
    private final Duration requestTimeout;
    private final BlockingQueue<BatchRequest> queue;
    private final ExecutorService executorService;
    private final AtomicBoolean running;
    private final AtomicInteger batchSizeGauge;
    private final AtomicInteger queueSizeGauge;

    private MultimodalProcessor processor;

    public BatchProcessor() {
        this(DEFAULT_MAX_BATCH_SIZE, DEFAULT_BATCH_TIMEOUT, DEFAULT_REQUEST_TIMEOUT);
    }

    public BatchProcessor(int maxBatchSize, Duration batchTimeout, Duration requestTimeout) {
        this.maxBatchSize = maxBatchSize;
        this.batchTimeout = batchTimeout;
        this.requestTimeout = requestTimeout;
        this.queue = new PriorityBlockingQueue<>(100, Comparator.comparingInt(BatchRequest::getPriority));
        this.executorService = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "multimodal-batch-processor");
            t.setDaemon(true);
            return t;
        });
        this.running = new AtomicBoolean(false);
        this.batchSizeGauge = new AtomicInteger(0);
        this.queueSizeGauge = new AtomicInteger(0);
    }

    /**
     * Start the batch processor.
     */
    public void start() {
        if (running.compareAndSet(false, true)) {
            executorService.submit(this::processLoop);
            log.infof("Batch processor started (maxBatchSize=%d, timeout=%dms)", 
                     maxBatchSize, batchTimeout.toMillis());
        }
    }

    /**
     * Stop the batch processor.
     */
    public void stop() {
        if (running.compareAndSet(true, false)) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
            
            // Fail remaining requests
            BatchRequest request;
            while ((request = queue.poll()) != null) {
                request.completeExceptionally(new TimeoutException("Batch processor stopped"));
            }
            
            log.info("Batch processor stopped");
        }
    }

    /**
     * Submit a request for batch processing.
     */
    public Uni<MultimodalResponse> submit(MultimodalRequest request) {
        return Uni.createFrom().emitter(emitter -> {
            String requestId = request.getRequestId();
            if (requestId == null || requestId.isBlank()) {
                requestId = "batch-" + UUID.randomUUID();
            }

            BatchRequest batchRequest = BatchRequest.builder()
                .id(requestId)
                .request(request)
                .future(new CompletableFuture<>())
                .priority(getPriority(request))
                .build();

            queue.offer(batchRequest);
            queueSizeGauge.incrementAndGet();

            // Handle completion
            batchRequest.getFuture().whenComplete((response, error) -> {
                queueSizeGauge.decrementAndGet();
                if (error != null) {
                    emitter.fail(error);
                } else {
                    emitter.complete((MultimodalResponse) response);
                }
            });

            // Handle timeout
            ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
            scheduler.schedule(() -> {
                if (!batchRequest.getFuture().isDone()) {
                    batchRequest.completeExceptionally(
                        new TimeoutException("Request timeout after " + requestTimeout.getSeconds() + "s")
                    );
                }
            }, requestTimeout.toMillis(), TimeUnit.MILLISECONDS);

        });
    }

    /**
     * Main processing loop.
     */
    private void processLoop() {
        while (running.get()) {
            try {
                // Collect batch
                List<BatchRequest> batch = collectBatch();
                
                if (batch.isEmpty()) {
                    Thread.sleep(10);
                    continue;
                }

                batchSizeGauge.set(batch.size());
                log.infof("Processing batch of %d requests", batch.size());

                // Process batch
                processBatch(batch);

                batchSizeGauge.set(0);

            } catch (Exception e) {
                log.errorf("Batch processing error: %s", e.getMessage());
            }
        }
    }

    /**
     * Collect requests into a batch.
     */
    private List<BatchRequest> collectBatch() throws InterruptedException {
        List<BatchRequest> batch = new ArrayList<>(maxBatchSize);

        // Get first request (wait if queue is empty)
        BatchRequest first = queue.poll(batchTimeout.toMillis(), TimeUnit.MILLISECONDS);
        if (first == null) {
            return batch;
        }
        
        batch.add(first);

        // Collect more requests up to max batch size
        while (batch.size() < maxBatchSize) {
            BatchRequest request = queue.poll();
            if (request == null) {
                break;
            }
            
            // Check if request is expired
            if (request.isExpired(requestTimeout.toMillis())) {
                request.completeExceptionally(
                    new TimeoutException("Request expired in queue")
                );
                continue;
            }
            
            batch.add(request);
        }

        return batch;
    }

    /**
     * Process a batch of requests.
     */
    private void processBatch(List<BatchRequest> batch) {
        long batchStartTime = System.currentTimeMillis();

        try {
            // Process each request in the batch
            for (BatchRequest batchRequest : batch) {
                try {
                    MultimodalRequest request = batchRequest.getRequest();
                    
                    // Process with processor
                    MultimodalResponse response = processor.process(request)
                        .await().atMost(requestTimeout);
                    
                    // Calculate additional metrics
                    long totalDuration = System.currentTimeMillis() - batchStartTime;
                    long queueTime = batchRequest.getQueueTime();
                    
                    // Add batch metrics to response
                    Map<String, Object> metadata = new HashMap<>(response.getMetadata());
                    metadata.put("batch_size", batch.size());
                    metadata.put("batch_duration_ms", totalDuration);
                    metadata.put("queue_time_ms", queueTime);
                    
                    MultimodalResponse responseWithMetrics = MultimodalResponse.builder()
                        .from(response)
                        .metadata(metadata)
                        .build();
                    
                    batchRequest.complete(responseWithMetrics);
                    
                } catch (Exception e) {
                    log.errorf("Request %s failed: %s", batchRequest.getId(), e.getMessage());
                    batchRequest.completeExceptionally(e);
                }
            }
            
            log.infof("Batch completed: %d requests in %dms", batch.size(), 
                     System.currentTimeMillis() - batchStartTime);
            
        } catch (Exception e) {
            log.errorf("Batch failed: %s", e.getMessage());
            // Fail all remaining requests in batch
            for (BatchRequest batchRequest : batch) {
                if (!batchRequest.getFuture().isDone()) {
                    batchRequest.completeExceptionally(e);
                }
            }
        }
    }

    /**
     * Set the processor for batch execution.
     */
    public void setProcessor(MultimodalProcessor processor) {
        this.processor = processor;
    }

    /**
     * Get current queue size.
     */
    public int getQueueSize() {
        return queue.size();
    }

    /**
     * Get current batch size (while processing).
     */
    public int getCurrentBatchSize() {
        return batchSizeGauge.get();
    }

    /**
     * Get pending request count.
     */
    public int getPendingCount() {
        return queue.size();
    }

    /**
     * Get batch processing statistics.
     */
    public BatchStats getStats() {
        return new BatchStats(
            maxBatchSize,
            batchTimeout.toMillis(),
            queue.size(),
            batchSizeGauge.get(),
            running.get()
        );
    }

    /**
     * Get priority for a request.
     * Higher priority = processed first.
     */
    private int getPriority(MultimodalRequest request) {
        // Default priority is 0
        // Can be overridden based on request parameters
        if (request.getParameters() != null && 
            request.getParameters().containsKey("priority")) {
            Object priority = request.getParameters().get("priority");
            if (priority instanceof Number) {
                return ((Number) priority).intValue();
            }
        }
        return 0;
    }

    /**
     * Batch statistics.
     */
    public static class BatchStats {
        public final int maxBatchSize;
        public final long batchTimeoutMs;
        public final int queueSize;
        public final int currentBatchSize;
        public final boolean running;

        public BatchStats(int maxBatchSize, long batchTimeoutMs, int queueSize, 
                         int currentBatchSize, boolean running) {
            this.maxBatchSize = maxBatchSize;
            this.batchTimeoutMs = batchTimeoutMs;
            this.queueSize = queueSize;
            this.currentBatchSize = currentBatchSize;
            this.running = running;
        }

        @Override
        public String toString() {
            return String.format(
                "BatchStats{maxSize=%d, timeout=%dms, queue=%d, currentBatch=%d, running=%s}",
                maxBatchSize, batchTimeoutMs, queueSize, currentBatchSize, running
            );
        }
    }
}
