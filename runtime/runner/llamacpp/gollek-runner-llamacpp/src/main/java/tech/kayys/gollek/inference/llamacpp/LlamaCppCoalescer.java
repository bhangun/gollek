package tech.kayys.gollek.inference.llamacpp;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Handles request coalescing and batching for improved throughput.
 * Supports both single-sequence and multi-sequence batch execution.
 */
public class LlamaCppCoalescer {

    private static final Logger log = Logger.getLogger(LlamaCppCoalescer.class);

    private final LlamaCppBinding binding;
    private final LlamaCppProviderConfig providerConfig;
    private final LlamaCppMetricsRecorder metricsRecorder;
    private final InferenceExecutor inferenceExecutor;

    private final BlockingQueue<InferenceTask> coalesceQueue;
    private final ExecutorService coalesceExecutor;
    private final boolean coalesceEnabled;
    private final int coalesceWindowMs;
    private final int coalesceMaxBatch;
    private final int coalesceMaxQueue;
    private final int coalesceSeqMax;

    private volatile boolean coalesceShutdown;
    private volatile boolean coalesceStarted;

    public LlamaCppCoalescer(
            LlamaCppBinding binding,
            LlamaCppProviderConfig providerConfig,
            LlamaCppMetricsRecorder metricsRecorder,
            InferenceExecutor inferenceExecutor) {

        this.binding = binding;
        this.providerConfig = providerConfig;
        this.metricsRecorder = metricsRecorder;
        this.inferenceExecutor = inferenceExecutor;

        this.coalesceEnabled = providerConfig.coalesceEnabled();
        this.coalesceWindowMs = Math.max(0, providerConfig.coalesceWindowMs());
        this.coalesceMaxBatch = Math.max(1, providerConfig.coalesceMaxBatch());
        this.coalesceMaxQueue = Math.max(1, providerConfig.coalesceMaxQueue());
        this.coalesceSeqMax = providerConfig.coalesceSeqMax();

        if (coalesceEnabled) {
            this.coalesceQueue = new ArrayBlockingQueue<>(coalesceMaxQueue);
            this.coalesceExecutor = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "gguf-coalesce");
                t.setDaemon(true);
                return t;
            });
        } else {
            this.coalesceQueue = null;
            this.coalesceExecutor = null;
        }

        this.coalesceShutdown = false;
        this.coalesceStarted = false;
    }

    /**
     * Start the coalescer background thread.
     */
    public void start() {
        if (!coalesceEnabled || coalesceExecutor == null || coalesceQueue == null || coalesceShutdown) {
            return;
        }

        coalesceStarted = true;
        coalesceExecutor.execute(() -> {
            while (!coalesceShutdown && !Thread.currentThread().isInterrupted()) {
                try {
                    InferenceTask first = coalesceQueue.take();
                    List<InferenceTask> batch = new ArrayList<>(coalesceMaxBatch);
                    batch.add(first);

                    long windowNanos = TimeUnit.MILLISECONDS.toNanos(coalesceWindowMs);
                    long start = System.nanoTime();
                    while (batch.size() < coalesceMaxBatch) {
                        long elapsed = System.nanoTime() - start;
                        long remaining = windowNanos - elapsed;
                        if (remaining <= 0) {
                            break;
                        }
                        InferenceTask extra = coalesceQueue.poll(remaining, TimeUnit.NANOSECONDS);
                        if (extra == null) {
                            break;
                        }
                        batch.add(extra);
                    }

                    metricsRecorder.recordCoalesceBatch(batch.size());
                    processBatch(batch);

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (Exception e) {
                    log.error("Coalesce worker failed", e);
                }
            }
        });
    }

    /**
     * Submit a request for coalesced execution.
     */
    public InferenceResponse submit(InferenceRequest request, Consumer<String> onTokenPiece,
            java.util.function.Supplier<InferenceResponse> fallbackExecutor) {
        if (!coalesceEnabled || !coalesceStarted || coalesceExecutor == null || coalesceQueue == null
                || shouldBypassCoalesce(request)) {
            return fallbackExecutor.get();
        }

        InferenceTask task = new InferenceTask(request, onTokenPiece, null);
        boolean queued = coalesceQueue.offer(task);
        if (!queued) {
            metricsRecorder.recordCoalesceDrop();
            log.warn("Coalesce queue full; running request immediately");
            return fallbackExecutor.get();
        }

        try {
            return task.future.get();
        } catch (Exception e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            throw new RuntimeException("Coalesced inference failed", cause);
        }
    }

    /**
     * Shutdown the coalescer and fail pending tasks.
     */
    public void shutdown() {
        if (!coalesceEnabled || coalesceExecutor == null || coalesceQueue == null) {
            return;
        }

        coalesceShutdown = true;
        coalesceExecutor.shutdownNow();

        InferenceTask pending;
        while ((pending = coalesceQueue.poll()) != null) {
            pending.future.completeExceptionally(new RuntimeException("Runner closed"));
        }
    }

    /**
     * Check if multi-sequence execution is supported.
     */
    public boolean supportsMultiSequence() {
        return coalesceSeqMax > 1;
    }

    private void processBatch(List<InferenceTask> batch) {
        if (supportsMultiSequence()) {
            processMultiSequenceBatch(batch);
        } else {
            for (InferenceTask task : batch) {
                try {
                    InferenceResponse response = inferenceExecutor.execute(task.request, task.onTokenPiece);
                    task.future.complete(response);
                } catch (Exception e) {
                    task.future.completeExceptionally(e);
                }
            }
        }
    }

    private void processMultiSequenceBatch(List<InferenceTask> batch) {
        List<InferenceTask> eligible = new ArrayList<>(batch.size());
        int eligibleCount = 0;

        for (InferenceTask task : batch) {
            if (isMultiSequenceEligible(task)) {
                eligible.add(task);
                eligibleCount++;
            } else {
                executeSingleTask(task);
            }
        }

        if (eligible.size() <= 1) {
            metricsRecorder.recordCoalesceSequences(eligibleCount);
            for (InferenceTask task : eligible) {
                executeSingleTask(task);
            }
            return;
        }

        if (eligible.size() > coalesceSeqMax) {
            for (int i = coalesceSeqMax; i < eligible.size(); i++) {
                executeSingleTask(eligible.get(i));
            }
            eligible = eligible.subList(0, coalesceSeqMax);
        }

        metricsRecorder.recordCoalesceSequences(eligible.size());
        inferenceExecutor.executeMultiSequence(eligible);
    }

    private void executeSingleTask(InferenceTask task) {
        try {
            InferenceResponse response = inferenceExecutor.execute(task.request, task.onTokenPiece);
            task.future.complete(response);
        } catch (Exception e) {
            task.future.completeExceptionally(e);
        }
    }

    public boolean shouldBypassCoalesce(InferenceRequest request) {
        if (request == null) {
            return false;
        }
        Object coalesce = request.getParameters().getOrDefault("gguf.coalesce", "true");
        if (!Boolean.parseBoolean(String.valueOf(coalesce))) {
            return true;
        }
        Object persist = request.getParameters().getOrDefault("gguf.session.persist", "false");
        return Boolean.parseBoolean(String.valueOf(persist));
    }

    private boolean isMultiSequenceEligible(InferenceTask task) {
        Object persist = task.request.getParameters().getOrDefault("gguf.session.persist", "false");
        return !Boolean.parseBoolean(String.valueOf(persist));
    }

    /**
     * Internal task representation for coalescing.
     */
    public static final class InferenceTask {
        public final InferenceRequest request;
        public final Consumer<String> onTokenPiece;
        public final String prompt;
        public final CompletableFuture<InferenceResponse> future = new CompletableFuture<>();

        public InferenceTask(InferenceRequest request, Consumer<String> onTokenPiece, String prompt) {
            this.request = request;
            this.onTokenPiece = onTokenPiece;
            this.prompt = prompt;
        }
    }

    /**
     * Interface for executing inference requests.
     */
    public interface InferenceExecutor {
        InferenceResponse execute(InferenceRequest request, Consumer<String> onTokenPiece);

        void executeMultiSequence(List<InferenceTask> tasks);
    }
}
