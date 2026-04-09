package tech.kayys.gollek.ml.distributed;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.List;
import java.util.concurrent.*;

/**
 * Data Parallel training across multiple workers (threads or processes).
 *
 * <p>Equivalent to {@code torch.nn.DataParallel} for single-node multi-worker.
 *
 * <p>Strategy: each worker computes gradients on its shard of the batch,
 * then gradients are all-reduced (averaged) across workers before the optimizer step.
 *
 * <p>Uses JDK 25 virtual threads for worker coordination.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * var ddp = new DataParallel(model, numWorkers = 4);
 *
 * // Each training step:
 * ddp.zeroGrad();
 * List<GradTensor> shards = ddp.splitBatch(inputs, 4);
 * ddp.forwardBackwardAll(shards, labels, lossFn);
 * ddp.allReduceGradients();   // average grads across workers
 * optimizer.step();
 * }</pre>
 */
public class DataParallel {

    private final NNModule model;
    private final int numWorkers;
    private final ExecutorService executor;

    public DataParallel(NNModule model, int numWorkers) {
        this.model      = model;
        this.numWorkers = numWorkers;
        // Virtual thread executor — lightweight, no OS thread pool needed
        this.executor   = Executors.newVirtualThreadPerTaskExecutor();
    }

    /**
     * Split a batch tensor along dim 0 into {@code numWorkers} shards.
     */
    public List<GradTensor> splitBatch(GradTensor batch, int workers) {
        long[] s = batch.shape();
        int N = (int) s[0];
        int shard = N / workers;
        java.util.List<GradTensor> shards = new java.util.ArrayList<>(workers);
        for (int w = 0; w < workers; w++) {
            int start = w * shard;
            int end   = (w == workers - 1) ? N : start + shard;
            int len   = end - start;
            long[] newShape = s.clone(); newShape[0] = len;
            int elemSize = (int) (batch.numel() / N);
            float[] d = new float[len * elemSize];
            System.arraycopy(batch.data(), start * elemSize, d, 0, len * elemSize);
            shards.add(GradTensor.of(d, newShape));
        }
        return shards;
    }

    /**
     * All-reduce gradients: average across all workers using Vector API.
     * In single-node mode this averages the accumulated gradients by numWorkers.
     */
    public void allReduceGradients() {
        float scale = 1.0f / numWorkers;
        for (Parameter p : model.parameters()) {
            GradTensor g = p.data().grad();
            if (g == null) continue;
            float[] gd = g.data();
            // Scale by 1/N using VectorOps (SIMD)
            VectorOps.mulScalar(gd, scale, gd);
        }
    }

    /**
     * Run forward + backward on multiple shards in parallel using virtual threads.
     *
     * @param shards   list of input shards (one per worker)
     * @param labels   list of label shards (one per worker)
     * @param lossFn   loss function: (pred, label) → scalar GradTensor
     */
    public void forwardBackwardAll(
            List<GradTensor> shards,
            List<GradTensor> labels,
            java.util.function.BiFunction<GradTensor, GradTensor, GradTensor> lossFn) {

        List<Future<?>> futures = new java.util.ArrayList<>(shards.size());
        for (int w = 0; w < shards.size(); w++) {
            final GradTensor x = shards.get(w);
            final GradTensor y = labels.get(w);
            futures.add(executor.submit(() -> {
                GradTensor pred = model.forward(x);
                GradTensor loss = lossFn.apply(pred, y);
                loss.backward();
            }));
        }
        // Wait for all workers
        for (Future<?> f : futures) {
            try { f.get(); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            catch (ExecutionException e)   { throw new RuntimeException("Worker failed", e.getCause()); }
        }
    }

    public void shutdown() { executor.shutdown(); }
}
