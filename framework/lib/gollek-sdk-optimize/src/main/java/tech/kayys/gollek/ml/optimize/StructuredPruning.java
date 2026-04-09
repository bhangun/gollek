package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.*;

/**
 * Structured pruning — removes entire filters/channels from convolutional layers
 * or neurons from linear layers, producing a smaller dense model.
 *
 * <p>Unlike unstructured pruning (which creates sparse weights), structured pruning
 * produces models that run faster on standard hardware without sparse kernels.
 *
 * <p>Strategy: rank filters/neurons by their L1 norm and remove the lowest-ranked ones.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var pruner = new StructuredPruning();
 *
 * // Prune 30% of neurons from each Linear layer
 * pruner.pruneLinear(model, sparsity = 0.3f);
 *
 * // Get pruning report
 * StructuredPruning.Report report = pruner.report(model);
 * System.out.printf("Remaining params: %d (%.1f%% of original)%n",
 *     report.remainingParams(), report.retentionRate() * 100);
 * }</pre>
 */
public final class StructuredPruning {

    /**
     * Pruning report summarizing the effect on a model.
     *
     * @param originalParams  parameter count before pruning
     * @param remainingParams parameter count after pruning (zeroed weights)
     * @param retentionRate   fraction of parameters retained (0–1)
     */
    public record Report(long originalParams, long remainingParams, float retentionRate) {}

    /**
     * Applies magnitude-based structured pruning to all {@code Linear} parameters.
     *
     * <p>For each weight matrix, computes the L1 norm of each row (output neuron),
     * ranks them, and zeros out the bottom {@code sparsity} fraction.
     *
     * @param model    model to prune (in-place)
     * @param sparsity fraction of neurons to remove, e.g. {@code 0.3f} = 30%
     */
    public void pruneLinear(NNModule model, float sparsity) {
        for (Parameter p : model.parameters()) {
            float[] data = p.data().data();
            long[] shape = p.data().shape();
            if (shape.length != 2) continue; // only 2D weight matrices

            int rows = (int) shape[0];
            int cols = (int) shape[1];
            int numPrune = (int) (rows * sparsity);

            // Compute L1 norm per row
            float[] rowNorms = new float[rows];
            for (int r = 0; r < rows; r++) {
                float norm = 0f;
                for (int c = 0; c < cols; c++) norm += Math.abs(data[r * cols + c]);
                rowNorms[r] = norm;
            }

            // Find threshold: the numPrune-th smallest norm
            float[] sorted = rowNorms.clone();
            Arrays.sort(sorted);
            float threshold = numPrune > 0 ? sorted[numPrune - 1] : -1f;

            // Zero out rows below threshold
            for (int r = 0; r < rows; r++) {
                if (rowNorms[r] <= threshold) {
                    Arrays.fill(data, r * cols, r * cols + cols, 0f);
                }
            }
        }
    }

    /**
     * Applies magnitude-based structured pruning to all parameters globally.
     *
     * <p>Computes a global threshold across all parameters and zeros out
     * individual weights below it (unstructured global pruning).
     *
     * @param model    model to prune (in-place)
     * @param sparsity fraction of total weights to zero out
     */
    public void pruneGlobal(NNModule model, float sparsity) {
        // Collect all weights
        List<float[]> allData = new ArrayList<>();
        for (Parameter p : model.parameters()) allData.add(p.data().data());

        int total = allData.stream().mapToInt(d -> d.length).sum();
        float[] allWeights = new float[total];
        int offset = 0;
        for (float[] d : allData) { System.arraycopy(d, 0, allWeights, offset, d.length); offset += d.length; }

        // Find global threshold using VectorOps for abs values
        float[] absWeights = new float[total];
        for (int i = 0; i < total; i++) absWeights[i] = Math.abs(allWeights[i]);
        Arrays.sort(absWeights);
        float threshold = absWeights[(int) (total * sparsity)];

        // Apply threshold
        for (Parameter p : model.parameters()) {
            float[] d = p.data().data();
            for (int i = 0; i < d.length; i++) if (Math.abs(d[i]) <= threshold) d[i] = 0f;
        }
    }

    /**
     * Generates a pruning report for the given model.
     *
     * <p>Counts zero vs non-zero parameters to measure effective sparsity.
     *
     * @param model the (possibly pruned) model
     * @return {@link Report} with original/remaining parameter counts
     */
    public Report report(NNModule model) {
        long total = 0, nonZero = 0;
        for (Parameter p : model.parameters()) {
            float[] d = p.data().data();
            total += d.length;
            for (float v : d) if (v != 0f) nonZero++;
        }
        return new Report(total, nonZero, total > 0 ? (float) nonZero / total : 1f);
    }
}
