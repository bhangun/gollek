package tech.kayys.golek.mcp;

import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.golek.model.GenerationResult;

public class InferenceScheduler {
    private static final Logger log = LoggerFactory.getLogger(InferenceScheduler.class);
    
    private final PriorityBlockingQueue<InferenceTask> taskQueue;
    private final ExecutorService executorService;
    private final int maxConcurrent;
    private final AtomicInteger activeRequests = new AtomicInteger(0);
    private volatile boolean shutdown = false;
    
    public InferenceScheduler(int maxConcurrent) {
        this.maxConcurrent = maxConcurrent;
        this.taskQueue = new PriorityBlockingQueue<>(100, 
            (a, b) -> Integer.compare(b.priority, a.priority));
        this.executorService = Executors.newFixedThreadPool(
            maxConcurrent,
            new ThreadFactory() {
                private final AtomicInteger counter = new AtomicInteger(0);
                @Override
                public Thread newThread(Runnable r) {
                    Thread t = new Thread(r, "inference-worker-" + counter.incrementAndGet());
                    t.setDaemon(false);
                    return t;
                }
            }
        );
        
        startWorkers();
    }
    
    private void startWorkers() {
        for (int i = 0; i < maxConcurrent; i++) {
            executorService.submit(() -> {
                while (!shutdown) {
                    try {
                        InferenceTask task = taskQueue.poll(1, TimeUnit.SECONDS);
                        if (task != null) {
                            processTask(task);
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            });
        }
    }
    
    private void processTask(InferenceTask task) {
        activeRequests.incrementAndGet();
        try {
            GenerationResult result = task.operation.call();
            task.future.complete(result);
        } catch (Exception e) {
            task.future.completeExceptionally(e);
        } finally {
            activeRequests.decrementAndGet();
        }
    }
    
    public CompletableFuture<GenerationResult> submit(
            Callable<GenerationResult> operation, 
            int priority) {
        
        CompletableFuture<GenerationResult> future = new CompletableFuture<>();
        InferenceTask task = new InferenceTask(operation, priority, future);
        
        if (!taskQueue.offer(task)) {
            future.completeExceptionally(new RejectedExecutionException("Queue full"));
        }
        
        return future;
    }
    
    public SchedulerStats getStats() {
        return new SchedulerStats(
            taskQueue.size(),
            activeRequests.get(),
            maxConcurrent
        );
    }
    
    public void shutdown() {
        shutdown = true;
        executorService.shutdown();
        try {
            executorService.awaitTermination(30, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }
    }
    
    private record InferenceTask(
        Callable<GenerationResult> operation,
        int priority,
        CompletableFuture<GenerationResult> future
    ) {}
    
    public record SchedulerStats(
        int queuedTasks,
        int activeTasks,
        int maxConcurrent
    ) {}
}
