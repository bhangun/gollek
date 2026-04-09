package tech.kayys.gollek.sdk.core;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Advanced tensor operations for indexing, slicing, gathering, and scattering.
 *
 * <p>Provides PyTorch-like advanced indexing operations:
 * <ul>
 *   <li>Slicing: {@link #slice(int, long, long)}, {@link #index(int, long)}</li>
 *   <li>Stacking: {@link #cat(int, List)}, {@link #stack(int, List)}</li>
 *   <li>Gathering: {@link #gather(int, Tensor)}, {@link #scatter(int, Tensor, Tensor)}</li>
 *   <li>Boolean indexing: {@link #maskedSelect(Tensor)}, {@link #maskedFill(Tensor, float)}</li>
 *   <li>Advanced indexing: {@link #indexPut(int, long, Tensor)}</li>
 * </ul>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * Tensor x = Tensor.randn(2, 10, 20);
 *
 * // Slicing along dimension
 * Tensor slice = TensorOps.slice(x, 2, 5, 15);  // x[..., 5:15]
 *
 * // Concatenation
 * List<Tensor> tensors = List.of(x, y, z);
 * Tensor combined = TensorOps.cat(0, tensors);  // [x; y; z]
 *
 * // Gathering with indices
 * Tensor indices = Tensor.of(new float[]{0, 2, 4}, 3);
 * Tensor gathered = TensorOps.gather(1, x, indices);
 *
 * // Boolean masking
 * Tensor mask = Tensor.randn(2, 10, 20).gt(0);
 * Tensor masked = TensorOps.maskedSelect(x, mask);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class TensorOps {

    private TensorOps() {}

    // ─────────────────────────────────────────────────────────────────────────
    // Slicing and Indexing
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Slice tensor along a specific dimension.
     *
     * @param tensor input tensor
     * @param dim dimension to slice
     * @param start start index (inclusive)
     * @param end end index (exclusive)
     * @return sliced tensor
     * @throws IndexOutOfBoundsException if indices are out of range
     *
     * <p>Example: {@code tensor.slice(1, 5, 15)} selects indices 5 to 14 along dimension 1.
     */
    public static Tensor slice(Tensor tensor, int dim, long start, long end) {
        validateDimension(tensor, dim);
        if (start < 0 || end > tensor.size(dim) || start > end) {
            throw new IndexOutOfBoundsException(
                String.format("Invalid slice indices [%d:%d] for dimension %d (size=%d)",
                    start, end, dim, tensor.size(dim)));
        }

        GradTensor gt = tensor.gradTensor();
        long[] shape = tensor.shape().clone();
        long sliceSize = end - start;

        if (sliceSize == 0) {
            shape[dim] = 0;
            return Tensor.of(new float[0], shape);
        }

        // Compute strides for multi-dimensional indexing
        float[] srcData = gt.data();
        long[] strides = computeStrides(shape);
        long srcStride = strides[dim];

        List<Float> result = new ArrayList<>();
        sliceRecursive(srcData, shape, strides, dim, start, end, new long[shape.length], 0, result);

        shape[dim] = sliceSize;
        float[] resultData = new float[result.size()];
        for (int i = 0; i < result.size(); i++) {
            resultData[i] = result.get(i);
        }
        return Tensor.of(resultData, shape)
            .requiresGrad(tensor.requiresGrad());
    }

    /**
     * Get single element at index along dimension.
     *
     * @param tensor input tensor
     * @param dim dimension
     * @param index index value
     * @return 1D tensor with selected elements
     */
    public static Tensor index(Tensor tensor, int dim, long index) {
        return slice(tensor, dim, index, index + 1).squeeze();
    }

    /**
     * Advanced indexing with multiple indices.
     *
     * @param tensor input tensor
     * @param indices indices tensor
     * @return indexed elements
     *
     * <p>Selects elements using advanced boolean or integer indexing.
     */
    public static Tensor indexSelect(Tensor tensor, int dim, Tensor indices) {
        validateDimension(tensor, dim);
        if (indices.ndim() != 1) {
            throw new IllegalArgumentException("Indices must be 1D tensor");
        }

        long[] shape = tensor.shape().clone();
        long numIndices = indices.size(0);
        shape[dim] = numIndices;

        float[] indexData = indices.data();
        float[] srcData = tensor.data();
        long numel = 1;
        for (long s : shape) {
            numel *= s;
        }
        float[] result = new float[(int) numel];

        long[] strides = computeStrides(tensor.shape());
        long srcStride = strides[dim];
        long dstStride = strides[dim];

        int idx = 0;
        for (int i = 0; i < numIndices; i++) {
            int srcIndex = (int) (indexData[i] * srcStride);
            System.arraycopy(srcData, srcIndex, result, idx, (int) dstStride);
            idx += (int) dstStride;
        }

        return Tensor.of(result, shape).requiresGrad(tensor.requiresGrad());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Concatenation and Stacking
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Concatenate tensors along existing dimension.
     *
     * @param dim dimension to concatenate along
     * @param tensors list of tensors to concatenate
     * @return concatenated tensor
     *
     * <p>All tensors must have same shape except in dimension {@code dim}.
     * Example: {@code cat(1, [x, y, z])} with shapes (2, 3), (2, 5), (2, 4)
     * produces (2, 12).
     */
    public static Tensor cat(int dim, List<Tensor> tensors) {
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("Must provide at least one tensor");
        }

        // Validate all tensors have compatible shapes
        long[] baseShape = tensors.get(0).shape().clone();
        long catSize = baseShape[dim];
        float[] baseData = tensors.get(0).data();

        for (int i = 1; i < tensors.size(); i++) {
            Tensor t = tensors.get(i);
            if (t.ndim() != baseShape.length) {
                throw new IllegalArgumentException(
                    String.format("All tensors must have same number of dimensions (ndim=%d vs %d)",
                        baseShape.length, t.ndim()));
            }

            for (int d = 0; d < baseShape.length; d++) {
                if (d != dim && t.size(d) != baseShape[d]) {
                    throw new IllegalArgumentException(
                        String.format("Shape mismatch at dimension %d: %d vs %d", d, baseShape[d], t.size(d)));
                }
            }
            catSize += t.size(dim);
        }

        baseShape[dim] = catSize;
        long numel = 1;
        for (long s : baseShape) {
            numel *= s;
        }
        float[] result = new float[(int) numel];

        int offset = 0;
        for (Tensor t : tensors) {
            int tSize = (int) t.numel();
            System.arraycopy(t.data(), 0, result, offset, tSize);
            offset += tSize;
        }

        return Tensor.of(result, baseShape).requiresGrad(tensors.get(0).requiresGrad());
    }

    /**
     * Stack tensors into new dimension.
     *
     * @param dim dimension to create
     * @param tensors list of tensors to stack
     * @return stacked tensor with new dimension
     *
     * <p>All tensors must have same shape.
     * Example: {@code stack(0, [x, y, z])} with shapes (3, 4) each
     * produces (3, 3, 4).
     */
    public static Tensor stack(int dim, List<Tensor> tensors) {
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("Must provide at least one tensor");
        }

        // Validate all tensors have identical shapes
        long[] baseShape = tensors.get(0).shape().clone();
        for (int i = 1; i < tensors.size(); i++) {
            long[] otherShape = tensors.get(i).shape();
            if (!shapeEquals(baseShape, otherShape)) {
                throw new IllegalArgumentException(
                    String.format("All tensors must have same shape for stacking"));
            }
        }

        // Insert new dimension
        long[] resultShape = new long[baseShape.length + 1];
        System.arraycopy(baseShape, 0, resultShape, 0, dim);
        resultShape[dim] = tensors.size();
        System.arraycopy(baseShape, dim, resultShape, dim + 1, baseShape.length - dim);

        long numel = 1;
        for (long s : resultShape) {
            numel *= s;
        }
        float[] result = new float[(int) numel];

        int offset = 0;
        for (Tensor t : tensors) {
            int tSize = (int) t.numel();
            System.arraycopy(t.data(), 0, result, offset, tSize);
            offset += tSize;
        }

        return Tensor.of(result, resultShape).requiresGrad(tensors.get(0).requiresGrad());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gathering and Scattering
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Gather elements from input tensor using index tensor.
     *
     * @param dim dimension to gather along
     * @param tensor input tensor to gather from
     * @param indices indices to gather (same shape as tensor except in dim)
     * @return gathered tensor
     *
     * <p>Equivalent to PyTorch's gather operation.
     * Example: {@code gather(1, x, idx)} where x.shape=(2,10) and idx.shape=(2,3)
     * selects 3 elements along dimension 1 for each batch.
     */
    public static Tensor gather(int dim, Tensor tensor, Tensor indices) {
        validateDimension(tensor, dim);
        if (tensor.ndim() != indices.ndim()) {
            throw new IllegalArgumentException("Tensor and indices must have same number of dimensions");
        }

        long[] resultShape = indices.shape().clone();
        long numel = 1;
        for (long s : resultShape) {
            numel *= s;
        }
        float[] result = new float[(int) numel];
        float[] srcData = tensor.data();
        float[] idxData = indices.data();

        long[] strides = computeStrides(tensor.shape());
        long stride = strides[dim];

        int resultIdx = 0;
        for (int i = 0; i < idxData.length; i++) {
            int srcIndex = (int) (i * stride + (int) idxData[i]);
            if (srcIndex >= 0 && srcIndex < srcData.length) {
                result[resultIdx++] = srcData[srcIndex];
            }
        }

        return Tensor.of(result, resultShape).requiresGrad(tensor.requiresGrad());
    }

    /**
     * Scatter updates into input tensor using index tensor.
     *
     * @param dim dimension to scatter along
     * @param indices indices where to scatter
     * @param updates values to scatter
     * @return output tensor (clone of input with updates)
     *
     * <p>Inverse operation of gather.
     */
    public static Tensor scatter(int dim, Tensor tensor, Tensor indices, Tensor updates) {
        validateDimension(tensor, dim);
        long[] shape = tensor.shape().clone();

        float[] result = tensor.data().clone();
        float[] idxData = indices.data();
        float[] updateData = updates.data();

        long[] strides = computeStrides(shape);
        long stride = strides[dim];

        int updateIdx = 0;
        for (int i = 0; i < idxData.length; i++) {
            int dstIndex = (int) (i * stride + (int) idxData[i]);
            if (dstIndex >= 0 && dstIndex < result.length) {
                result[dstIndex] = updateData[updateIdx++];
            }
        }

        return Tensor.of(result, shape).requiresGrad(tensor.requiresGrad());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Boolean Indexing and Masking
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Select elements where mask is true.
     *
     * @param tensor input tensor
     * @param mask boolean mask (same shape as tensor)
     * @return 1D tensor with selected elements
     *
     * <p>Example: {@code maskedSelect(x, x > 0)} selects positive elements.
     */
    public static Tensor maskedSelect(Tensor tensor, Tensor mask) {
        if (!shapeEquals(tensor.shape(), mask.shape())) {
            throw new IllegalArgumentException("Tensor and mask must have same shape");
        }

        float[] data = tensor.data();
        float[] maskData = mask.data();
        List<Float> result = new ArrayList<>();

        for (int i = 0; i < data.length; i++) {
            if (maskData[i] != 0f) {  // Non-zero = true
                result.add(data[i]);
            }
        }

        if (result.isEmpty()) {
            return Tensor.zeros(0).requiresGrad(tensor.requiresGrad());
        }

        float[] resultData = new float[result.size()];
        for (int i = 0; i < result.size(); i++) {
            resultData[i] = result.get(i);
        }
        return Tensor.of(resultData)
            .requiresGrad(tensor.requiresGrad());
    }

    /**
     * Fill elements where mask is true with value.
     *
     * @param tensor input tensor
     * @param mask boolean mask
     * @param value value to fill
     * @return modified tensor
     *
     * <p>Example: {@code maskedFill(x, x < 0, 0)} replaces negative values with 0.
     */
    public static Tensor maskedFill(Tensor tensor, Tensor mask, float value) {
        if (!shapeEquals(tensor.shape(), mask.shape())) {
            throw new IllegalArgumentException("Tensor and mask must have same shape");
        }

        float[] result = tensor.data().clone();
        float[] maskData = mask.data();

        for (int i = 0; i < result.length; i++) {
            if (maskData[i] != 0f) {
                result[i] = value;
            }
        }

        return Tensor.of(result, tensor.shape()).requiresGrad(tensor.requiresGrad());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Comparison Operations
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Element-wise greater than comparison.
     *
     * @param tensor input tensor
     * @param other other tensor or scalar
     * @return boolean tensor (1.0 for true, 0.0 for false)
     */
    public static Tensor gt(Tensor tensor, float other) {
        float[] data = tensor.data();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] > other ? 1f : 0f;
        }
        return Tensor.of(result, tensor.shape());
    }

    /**
     * Element-wise less than comparison.
     */
    public static Tensor lt(Tensor tensor, float other) {
        float[] data = tensor.data();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] < other ? 1f : 0f;
        }
        return Tensor.of(result, tensor.shape());
    }

    /**
     * Element-wise greater than or equal comparison.
     */
    public static Tensor ge(Tensor tensor, float other) {
        float[] data = tensor.data();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] >= other ? 1f : 0f;
        }
        return Tensor.of(result, tensor.shape());
    }

    /**
     * Element-wise less than or equal comparison.
     */
    public static Tensor le(Tensor tensor, float other) {
        float[] data = tensor.data();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] <= other ? 1f : 0f;
        }
        return Tensor.of(result, tensor.shape());
    }

    /**
     * Element-wise equality comparison.
     */
    public static Tensor eq(Tensor tensor, float other) {
        float[] data = tensor.data();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] == other ? 1f : 0f;
        }
        return Tensor.of(result, tensor.shape());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helper Methods
    // ─────────────────────────────────────────────────────────────────────────

    private static long[] computeStrides(long[] shape) {
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static void sliceRecursive(float[] src, long[] shape, long[] strides, int dim,
                                       long start, long end, long[] indices, int currentDim,
                                       List<Float> result) {
        if (currentDim == shape.length) {
            long index = 0;
            for (int i = 0; i < indices.length; i++) {
                index += indices[i] * strides[i];
            }
            result.add(src[(int) index]);
            return;
        }

        long limit = (currentDim == dim) ? end : shape[currentDim];
        long st = (currentDim == dim) ? start : 0;

        for (long i = st; i < limit; i++) {
            indices[currentDim] = i;
            sliceRecursive(src, shape, strides, dim, start, end, indices, currentDim + 1, result);
        }
    }

    private static void validateDimension(Tensor tensor, int dim) {
        if (dim < 0 || dim >= tensor.ndim()) {
            throw new IndexOutOfBoundsException(
                String.format("Dimension %d out of range for tensor with %d dimensions", dim, tensor.ndim()));
        }
    }

    private static boolean shapeEquals(long[] shape1, long[] shape2) {
        if (shape1.length != shape2.length) return false;
        for (int i = 0; i < shape1.length; i++) {
            if (shape1[i] != shape2[i]) return false;
        }
        return true;
    }
}
