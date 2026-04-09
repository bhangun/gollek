package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.optim.Optimizer;

import java.util.List;
import java.util.function.BiFunction;

/**
 * Training loop abstraction for neural network models.
 * <p>
 * Handles the epoch loop, gradient computation, and optimizer steps.
 * Provides callbacks for monitoring training progress.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var trainer = Trainer.builder()
 *     .model(model)
 *     .optimizer(new AdamW(model.parameters(), 1e-4f))
 *     .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
 *     .epochs(10)
 *     .build();
 * 
 * trainer.fit(trainingData, trainingLabels, batchSize);
 * }</pre>
 */
public class Trainer {

    private final NNModule model;
    private final Optimizer optimizer;
    private final BiFunction<GradTensor, GradTensor, GradTensor> lossFunction;
    private final int epochs;
    private final TrainingCallback callback;

    private Trainer(Builder builder) {
        this.model = builder.model;
        this.optimizer = builder.optimizer;
        this.lossFunction = builder.lossFunction;
        this.epochs = builder.epochs;
        this.callback = builder.callback != null ? builder.callback : TrainingCallback.NOOP;
    }

    /**
     * Train the model on batched data.
     *
     * @param inputs list of input batches
     * @param targets list of target batches (matching inputs)
     */
    public TrainingResult fit(List<GradTensor> inputs, List<GradTensor> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("inputs and targets must have the same number of batches");
        }

        model.train();
        float[] epochLosses = new float[epochs];

        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0;
            int batchCount = inputs.size();

            callback.onEpochStart(epoch, epochs);

            for (int batch = 0; batch < batchCount; batch++) {
                // Zero gradients
                optimizer.zeroGrad();

                // Forward pass
                GradTensor output = model.forward(inputs.get(batch));

                // Compute loss
                GradTensor loss = lossFunction.apply(output, targets.get(batch));
                float lossVal = loss.item();
                epochLoss += lossVal;

                // Backward pass
                loss.backward();

                // Update parameters
                optimizer.step();

                callback.onBatchEnd(epoch, batch, batchCount, lossVal);
            }

            epochLosses[epoch] = epochLoss / batchCount;
            callback.onEpochEnd(epoch, epochs, epochLosses[epoch]);
        }

        return new TrainingResult(epochLosses);
    }

    /**
     * Train on a single dataset (auto-batching).
     */
    public TrainingResult fit(GradTensor input, GradTensor target, int batchSize) {
        int n = (int) input.shape()[0];
        int numBatches = (n + batchSize - 1) / batchSize;
        int dim = (int) (input.numel() / n);
        int targetDim = (int) (target.numel() / n);

        var inputs = new java.util.ArrayList<GradTensor>();
        var targets = new java.util.ArrayList<GradTensor>();

        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, n);
            int bs = end - start;

            float[] batchIn = new float[bs * dim];
            System.arraycopy(input.data(), start * dim, batchIn, 0, bs * dim);
            inputs.add(GradTensor.of(batchIn, bs, dim));

            float[] batchTarget = new float[bs * targetDim];
            System.arraycopy(target.data(), start * targetDim, batchTarget, 0, bs * targetDim);
            long[] targetShape = targetDim == 1 ? new long[]{bs} : new long[]{bs, targetDim};
            targets.add(GradTensor.of(batchTarget, targetShape));
        }

        return fit(inputs, targets);
    }

    // ── Builder ──────────────────────────────────────────────────────────

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private NNModule model;
        private Optimizer optimizer;
        private BiFunction<GradTensor, GradTensor, GradTensor> lossFunction;
        private int epochs = 1;
        private TrainingCallback callback;

        public Builder model(NNModule model) { this.model = model; return this; }
        public Builder optimizer(Optimizer optimizer) { this.optimizer = optimizer; return this; }
        public Builder lossFunction(BiFunction<GradTensor, GradTensor, GradTensor> loss) { this.lossFunction = loss; return this; }
        public Builder epochs(int epochs) { this.epochs = epochs; return this; }
        public Builder callback(TrainingCallback callback) { this.callback = callback; return this; }

        public Trainer build() {
            if (model == null) throw new IllegalStateException("model is required");
            if (optimizer == null) throw new IllegalStateException("optimizer is required");
            if (lossFunction == null) throw new IllegalStateException("lossFunction is required");
            return new Trainer(this);
        }
    }

    // ── Callback ─────────────────────────────────────────────────────────

    public interface TrainingCallback {
        TrainingCallback NOOP = new TrainingCallback() {};

        default void onEpochStart(int epoch, int totalEpochs) {}
        default void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {}
        default void onBatchEnd(int epoch, int batch, int totalBatches, float loss) {}
    }

    /** Simple callback that prints progress. */
    public static TrainingCallback printCallback() {
        return new TrainingCallback() {
            @Override
            public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                System.out.printf("Epoch %d/%d — loss: %.6f%n", epoch + 1, totalEpochs, avgLoss);
            }
        };
    }

    // ── Training Result ──────────────────────────────────────────────────

    public record TrainingResult(float[] epochLosses) {
        public float finalLoss() {
            return epochLosses[epochLosses.length - 1];
        }

        public boolean converged(float threshold) {
            if (epochLosses.length < 2) return false;
            float delta = Math.abs(epochLosses[epochLosses.length - 1] - epochLosses[epochLosses.length - 2]);
            return delta < threshold;
        }

        @Override
        public String toString() {
            return "TrainingResult(epochs=" + epochLosses.length + ", finalLoss=" + finalLoss() + ")";
        }
    }
}
