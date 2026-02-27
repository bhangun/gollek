package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * Flatten layer — reshapes input to [batch_size, -1].
 * Mirrors {@code torch::nn::Flatten}.
 */
public class Flatten extends Module {

    private final long startDim;
    private final long endDim;

    /**
     * @param startDim first dimension to flatten (inclusive)
     * @param endDim   last dimension to flatten (inclusive, -1 for last)
     */
    public Flatten(long startDim, long endDim) {
        this.startDim = startDim;
        this.endDim = endDim;
    }

    /** Default: flatten all dimensions except batch (dim 0). */
    public Flatten() {
        this(1, -1);
    }

    @Override
    public Tensor forward(Tensor input) {
        long[] shape = input.shape();
        int ndim = shape.length;

        // Resolve negative dims
        long start = startDim >= 0 ? startDim : ndim + startDim;
        long end = endDim >= 0 ? endDim : ndim + endDim;

        // Calculate flattened size
        long flatSize = 1;
        for (long d = start; d <= end; d++) {
            flatSize *= shape[(int) d];
        }

        // Build new shape
        long[] newShape = new long[(int) (start + 1 + (ndim - end - 1))];
        for (int i = 0; i < start; i++) {
            newShape[i] = shape[i];
        }
        newShape[(int) start] = flatSize;
        for (int i = (int) end + 1; i < ndim; i++) {
            newShape[(int) (start + 1 + (i - end - 1))] = shape[i];
        }

        return input.reshape(newShape);
    }

    @Override
    public String toString() {
        return String.format("Flatten(start_dim=%d, end_dim=%d)", startDim, endDim);
    }
}
