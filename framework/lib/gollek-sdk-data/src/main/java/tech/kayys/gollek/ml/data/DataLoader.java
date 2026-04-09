package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Multi-threaded DataLoader with prefetching — uses JDK 25 virtual threads.
 *
 * <p>Equivalent to {@code torch.utils.data.DataLoader}.
 *
 * @param <T> The type of individual items in the dataset.
 */
public class DataLoader<T> implements Iterable<List<T>> {

    private final Dataset<T> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final int numWorkers;
    private final int prefetchFactor;
    private final List<Function<T, T>> transforms = new ArrayList<>();

    /**
     * Standard constructor used for basic batching.
     */
    public DataLoader(Dataset<T> dataset, int batchSize) {
        this(dataset, batchSize, false, false);
    }

    /**
     * Standard constructor used for basic batching.
     */
    public DataLoader(Dataset<T> dataset, int batchSize, boolean shuffle, boolean dropLast) {
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.numWorkers = 1;
        this.prefetchFactor = 1;
    }

    /**
     * Extended constructor used by the Builder.
     */
    private DataLoader(Builder<T> b) {
        this.dataset = b.dataset;
        this.batchSize = b.batchSize;
        this.shuffle = b.shuffle;
        this.dropLast = b.dropLast;
        this.numWorkers = b.numWorkers;
        this.prefetchFactor = b.prefetchFactor;
        if (b.transform != null) {
            // Unsafe cast but necessary for internal builder compatibility with UnaryOperator<GradTensor>
            this.transforms.add((Function<T, T>) b.transform);
        }
    }

    public static <T> Builder<T> builder(Dataset<T> dataset) { return new Builder<>(dataset); }

    public int numBatches() {
        if (dropLast) return dataset.size() / batchSize;
        return (int) Math.ceil((double) dataset.size() / batchSize);
    }

    /**
     * Basic map function to transform data elements.
     */
    public <R> DataLoader<R> map(Function<T, R> mapper) {
        // This is a simplified implementation for the test's requirements.
        // In a real scenario, this might return a new DataLoader wrapping a MappedDataset.
        return new DataLoader<>(new Dataset<R>() {
            @Override public R get(int index) { return mapper.apply(dataset.get(index)); }
            @Override public int size() { return dataset.size(); }
        }, batchSize, shuffle, dropLast);
    }

    @Override
    public Iterator<List<T>> iterator() {
        int[] indices = buildIndices();
        BlockingQueue<List<T>> queue = new LinkedBlockingQueue<>(prefetchFactor * numWorkers);

        // Producer: virtual threads load batches into queue
        Thread.ofVirtual().start(() -> {
            try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
                for (int b = 0; b < numBatches(); b++) {
                    final int batchIdx = b;
                    exec.submit(() -> {
                        List<T> batch = loadBatch(indices, batchIdx);
                        try { queue.put(batch); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                    });
                }
                exec.shutdown();
                exec.awaitTermination(1, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            // Poison pill
            try { queue.put(Collections.emptyList()); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        });

        return new Iterator<>() {
            List<T> nextBatch = null;
            @Override public boolean hasNext() {
                if (nextBatch == null) nextBatch = poll();
                return !nextBatch.isEmpty();
            }
            @Override public List<T> next() {
                List<T> b = nextBatch; nextBatch = null; return b;
            }
            private List<T> poll() {
                try { return queue.take(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); return Collections.emptyList(); }
            }
        };
    }

    private List<T> loadBatch(int[] indices, int batchIdx) {
        int start = batchIdx * batchSize;
        int end = Math.min(start + batchSize, dataset.size());
        List<T> batch = new ArrayList<>(end - start);

        for (int i = start; i < end; i++) {
            T item = dataset.get(indices[i]);
            for (Function<T, T> t : transforms) {
                item = t.apply(item);
            }
            batch.add(item);
        }
        return batch;
    }

    private int[] buildIndices() {
        int[] idx = new int[dataset.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        if (shuffle) {
            Random rng = new Random();
            for (int i = idx.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1); int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
        return idx;
    }

    // ── Batch (Specific for NN) ──────────────────────────────────────────

    /**
     * NN-specific batch record. Used when T is Dataset.Sample.
     */
    public record Batch(GradTensor inputs, GradTensor labels) {
        public static final Batch END = new Batch(null, null);
    }

    /**
     * Utility to collate a list of samples into a GradTensor Batch.
     */
    public static Batch collate(List<Dataset.Sample> samples) {
        if (samples.isEmpty()) return Batch.END;
        List<GradTensor> inputs = samples.stream().map(Dataset.Sample::input).toList();
        List<GradTensor> labels = samples.stream().map(Dataset.Sample::label).toList();
        return new Batch(stack(inputs), stack(labels));
    }

    private static GradTensor stack(List<GradTensor> tensors) {
        if (tensors.isEmpty()) return GradTensor.zeros(0);
        long[] shape = tensors.get(0).shape();
        int N = tensors.size();
        int elemSize = (int) GradTensor.numelFor(shape);
        float[] out = new float[N * elemSize];
        for (int i = 0; i < N; i++)
            System.arraycopy(tensors.get(i).data(), 0, out, i * elemSize, elemSize);
        long[] batchShape = new long[shape.length + 1];
        batchShape[0] = N;
        System.arraycopy(shape, 0, batchShape, 1, shape.length);
        return GradTensor.of(out, batchShape);
    }

    // ── Builder ───────────────────────────────────────────────────────────

    public static class Builder<T> {
        private final Dataset<T> dataset;
        private int batchSize = 32;
        private boolean shuffle = false;
        private boolean dropLast = false;
        private int numWorkers = 2;
        private int prefetchFactor = 2;
        private UnaryOperator<T> transform;

        Builder(Dataset<T> dataset) { this.dataset = dataset; }
        public Builder<T> batchSize(int n)          { this.batchSize = n; return this; }
        public Builder<T> shuffle(boolean s)         { this.shuffle = s; return this; }
        public Builder<T> dropLast(boolean d)        { this.dropLast = d; return this; }
        public Builder<T> numWorkers(int n)          { this.numWorkers = n; return this; }
        public Builder<T> prefetchFactor(int n)      { this.prefetchFactor = n; return this; }
        public Builder<T> transform(UnaryOperator<T> t) { this.transform = t; return this; }
        public DataLoader<T> build()                 { return new DataLoader<>(this); }
    }
}
