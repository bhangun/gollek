package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Cross-entropy loss for multi-class classification tasks.
 * <p>
 * Cross-entropy is the de facto standard loss function for classification problems.
 * It combines log-softmax normalization and negative log-likelihood in a single
 * numerically stable computation. It penalizes confident incorrect predictions heavily.
 * <p>
 * The loss expects raw logits (not probabilities) as input and handles softmax
 * internally for numerical stability using the log-sum-exp trick.
 * <p>
 * Equivalent to {@code torch.nn.CrossEntropyLoss}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * For each sample:
 *   LogSoftmax(x_c) = x_c - log(Σ_i exp(x_i))
 *   NLL = -LogSoftmax(x_target)
 *
 * Loss = (1/batch) Σ NLL
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Logits:</b> [batch, numClasses] - raw model outputs (NOT probabilities)</li>
 *   <li><b>Targets:</b> [batch] - class indices (0 to numClasses-1) as float values</li>
 *   <li><b>Output:</b> scalar loss value</li>
 * </ul>
 *
 * <h3>Example: Binary Classification</h3>
 * <pre>{@code
 * var loss = new CrossEntropyLoss();
 *
 * // Binary classification: 2 classes
 * var logits = GradTensor.of(new float[][]{
 *     {2.0f, -1.0f},    // sample 1: high score for class 0
 *     {-1.0f, 2.0f},    // sample 2: high score for class 1
 * }, 2, 2);
 *
 * var targets = GradTensor.of(new float[]{0, 1}, 2);  // true classes
 * var lossValue = loss.compute(logits, targets);      // ~0.126 (low loss = good)
 * }</pre>
 *
 * <h3>Example: Multi-class (MNIST-like)</h3>
 * <pre>{@code
 * var loss = new CrossEntropyLoss();
 *
 * // 10-class classification (MNIST)
 * var model = ...;  // neural network returning [batch, 10] logits
 * var predictions = model.forward(images);  // [32, 10] logits
 * var targets = GradTensor.of(new float[]{3, 7, 2, ...}, 32);  // 32 class labels
 *
 * var lossValue = loss.compute(predictions, targets);
 * lossValue.backward();
 * optimizer.step();
 * }</pre>
 *
 * <h3>Example: ImageNet Classification</h3>
 * <pre>{@code
 * // 1000-class classification
 * var logits = model.forward(images);  // [batch, 1000]
 * var loss = new CrossEntropyLoss();
 * var lossValue = loss.compute(logits, labels);
 * }</pre>
 *
 * <h3>Numerical Stability Features</h3>
 * <ul>
 *   <li><b>Log-sum-exp trick:</b> Subtracts max logit before exp to prevent overflow</li>
 *   <li><b>Epsilon:</b> Adds 1e-8 to log argument to prevent log(0)</li>
 *   <li><b>Stable backward:</b> Gradients computed from softmax outputs, not raw logits</li>
 * </ul>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Optimal for categorical probability distributions</li>
 *   <li>Penalizes confident incorrect predictions heavily</li>
 *   <li>Expected class probability targets: one class has prob 1.0, others 0.0</li>
 *   <li>Well-defined gradients everywhere</li>
 * </ul>
 *
 * <h3>Important Notes</h3>
 * <ul>
 *   <li><b>Input:</b> Use raw logits, NOT softmax probabilities</li>
 *   <li><b>Target format:</b> Integer class indices (as float), not one-hot vectors</li>
 *   <li><b>Batch dimension:</b> Always required, even for single samples</li>
 * </ul>
 *
 * <h3>Gradient Interpretation</h3>
 * The gradient w.r.t logits equals: softmax(logits) - one_hot(target)
 * This means the model learns by moving the softmax distribution towards one-hot.
 *
 * <h3>Alternatives</h3>
 * <ul>
 *   <li><b>BCEWithLogitsLoss:</b> For binary classification (2 classes)</li>
 *   <li><b>FocalLoss:</b> For imbalanced datasets</li>
 *   <li><b>SoftmaxCrossEntropyLoss:</b> For soft labels/label smoothing</li>
 * </ul>
 */
public class CrossEntropyLoss {

    /**
     * Compute cross-entropy loss for classification.
     *
     * @param logits  raw model scores [batch, numClasses] (NOT probabilities)
     * @param targets class indices [batch] (integer values as floats, range: 0 to numClasses-1)
     * @return scalar loss tensor
     *
     * @throws IllegalArgumentException if logits shape is not 2D
     * @throws IllegalArgumentException if targets batch size doesn't match logits batch size
     * @throws IndexOutOfBoundsException if any target index is out of range [0, numClasses)
     */
    public GradTensor compute(GradTensor logits, GradTensor targets) {
        long[] s = logits.shape();
        if (s.length != 2) {
            throw new IllegalArgumentException(
                "logits must be 2D [batch, numClasses], got shape: " + java.util.Arrays.toString(s));
        }

        int batch = (int) s[0];
        int numClasses = (int) s[1];
        float[] logitsData = logits.data();
        float[] targetsData = targets.data();

        if (targetsData.length != batch) {
            throw new IllegalArgumentException(
                "targets batch size must match logits batch size, got: " + targetsData.length + " vs " + batch);
        }

        // Compute log-softmax and NLL
        float totalLoss = 0;
        float[] softmaxData = new float[logitsData.length];

        // Process each sample in batch
        for (int b = 0; b < batch; b++) {
            int off = b * numClasses;

            // Log-sum-exp trick: subtract max for numerical stability
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < numClasses; c++) {
                max = Math.max(max, logitsData[off + c]);
            }

            // Compute softmax with numerical stability
            float sumExp = 0;
            for (int c = 0; c < numClasses; c++) {
                softmaxData[off + c] = (float) Math.exp(logitsData[off + c] - max);
                sumExp += softmaxData[off + c];
            }
            for (int c = 0; c < numClasses; c++) {
                softmaxData[off + c] /= sumExp;
            }

            // Get target class and compute NLL
            int target = (int) targetsData[b];
            if (target < 0 || target >= numClasses) {
                throw new IndexOutOfBoundsException(
                    "target class index " + target + " out of range [0, " + (numClasses - 1) + "]");
            }
            // NLL = -log(softmax[target])
            totalLoss -= (float) Math.log(softmaxData[off + target] + 1e-8f);
        }

        float meanLoss = totalLoss / batch;
        GradTensor out = GradTensor.scalar(meanLoss);

        if (logits.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("CrossEntropyLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / batch;
                    float[] grad = new float[logitsData.length];

                    // Gradient = softmax - one_hot(target)
                    for (int b = 0; b < batch; b++) {
                        int off = b * numClasses;
                        int target = (int) targetsData[b];
                        for (int c = 0; c < numClasses; c++) {
                            grad[off + c] = softmaxData[off + c] * scale;
                        }
                        // Subtract 1 for target class (one_hot encoding)
                        grad[off + target] -= scale;
                    }
                    logits.backward(GradTensor.of(grad, logits.shape()));
                }
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "CrossEntropyLoss()";
    }
}
