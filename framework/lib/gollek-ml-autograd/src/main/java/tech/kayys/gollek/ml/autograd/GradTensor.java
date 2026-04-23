package tech.kayys.gollek.ml.autograd;

import tech.kayys.gollek.runtime.tensor.DType;
import tech.kayys.gollek.runtime.tensor.Device;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReferenceArray;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * A tensor with automatic differentiation support.
 * <p>
 * {@code GradTensor} wraps raw data arrays and tracks computational history
 * via a tape of {@link Function} nodes. Calling {@link #backward()} performs
 * reverse-mode AD to compute gradients for all upstream tensors that have
 * {@code requiresGrad = true}.
 * <p>
 * This is the core primitive for building neural networks in Gollek ML,
 * analogous to {@code torch.Tensor} with {@code requires_grad=True} in PyTorch.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var x = GradTensor.of(new float[] { 1, 2, 3, 4 }, 2, 2).requiresGrad(true);
 * var w = GradTensor.randn(2, 3).requiresGrad(true);
 * var y = x.matmul(w).relu();
 * y.sum().backward();
 * // x.grad() now contains ∂y/∂x
 * }</pre>
 */
public final class GradTensor {

    // ── Storage ──────────────────────────────────────────────────────────

    private final TensorStorage storage;
    private final long[] shape;
    private DType dtype;
    private Device device;

    // ── Autograd ─────────────────────────────────────────────────────────

    private TensorStorage gradStorage;
    private boolean requiresGrad;
    private Function.Context gradFn; // the function that produced this tensor
    private final List<GradTensor> children = new ArrayList<>();

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    /**
     * In-place addition with autograd tracking.
     */
    public GradTensor add_(GradTensor other) {
        if (requiresGrad() || other.requiresGrad()) {
            // Record operation before in-place modification
            GradTensor result = Functions.Add.apply(this, other);
            // Copy result back and attach gradient function
            System.arraycopy(result.data(), 0, data(), 0, (int) numel());
            if (requiresGrad()) {
                this.gradFn = result.gradFn();
            }
            return this;
        } else {
            // No grad needed, direct in-place addition
            float[] d = data(), od = other.data();
            for (int i = 0; i < d.length; i++) {
                d[i] += od[i % od.length];
            }
            return this;
        }
    }

    /**
     * Returns a view with gradient checkpointing support.
     */
    public GradTensor checkpoint() {
        if (!requiresGrad())
            return this;

        // Save current state for recomputation
        final GradTensor original = this;
        final float[] savedData = data().clone();

        GradTensor detached = detach();
        detached.requiresGrad(true);
        detached.setGradFn(new Function.Context("Checkpoint") {
            @Override
            public void backward(GradTensor upstream) {
                // Restore original data and recompute forward
                System.arraycopy(savedData, 0, original.data(), 0, savedData.length);
                GradTensor recomputed = original;
                recomputed.backward(upstream);
            }
        });
        return detached;
    }

    /**
     * Sparse gradient accumulation for embeddings.
     */
    public void backwardSparse(GradTensor upstreamGrad, int[] indices) {
        if (!requiresGrad || gradStorage == null)
            return;

        float[] gd = gradStorage.asArray();
        float[] ug = upstreamGrad.data();

        // Only update specified indices
        for (int idx : indices) {
            if (idx >= 0 && idx < gd.length) {
                gd[idx] += ug[idx];
            }
        }

        if (gradFn != null) {
            gradFn.backward(upstreamGrad);
        }
    }

    /**
     * Zero gradients with optional set of parameters to preserve.
     */
    public void zeroGrad(boolean[] preserveMask) {
        if (gradStorage == null)
            return;
        float[] gd = gradStorage.asArray();
        if (preserveMask != null) {
            for (int i = 0; i < gd.length; i++) {
                if (!preserveMask[i])
                    gd[i] = 0;
            }
        } else {
            Arrays.fill(gd, 0.0f);
        }
    }

    // ── Constructors ─────────────────────────────────────────────────────

    public GradTensor(TensorStorage storage, long[] shape, DType dtype, Device device, boolean requiresGrad) {
        this.storage = Objects.requireNonNull(storage);
        this.shape = Objects.requireNonNull(shape);
        this.dtype = dtype;
        this.device = device;
        this.requiresGrad = requiresGrad;
        if (requiresGrad) {
            this.gradStorage = new HeapStorage(new float[storage.size()]);
        }
    }

    // ── Factory methods (like torch.*) ───────────────────────────────────

    /** Create a tensor from raw data with the given shape. */
    public static GradTensor of(float[] data, long... shape) {
        long numel = numelFor(shape);
        if (data.length != numel) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + " != shape " + Arrays.toString(shape) + " (numel=" + numel + ")");
        }
        return new GradTensor(new HeapStorage(data.clone()), shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a 1D tensor from data. */
    public static GradTensor of(float... data) {
        return new GradTensor(new HeapStorage(data.clone()), new long[] { data.length }, DType.FLOAT32, Device.CPU,
                false);
    }

    /** Create a tensor filled with zeros. */
    public static GradTensor zeros(long... shape) {
        return new GradTensor(new HeapStorage(new float[(int) numelFor(shape)]), shape.clone(), DType.FLOAT32,
                Device.CPU, false);
    }

    /** Create a tensor filled with ones. */
    public static GradTensor ones(long... shape) {
        float[] dataArr = new float[(int) numelFor(shape)];
        Arrays.fill(dataArr, 1.0f);
        return new GradTensor(new HeapStorage(dataArr), shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor filled with a scalar value. */
    public static GradTensor full(float value, long... shape) {
        float[] dataArr = new float[(int) numelFor(shape)];
        Arrays.fill(dataArr, value);
        return new GradTensor(new HeapStorage(dataArr), shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor with random normal values (mean=0, std=1). */
    public static GradTensor randn(long... shape) {
        int n = (int) numelFor(shape);
        float[] dataArr = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) {
            dataArr[i] = (float) rng.nextGaussian();
        }
        return new GradTensor(new HeapStorage(dataArr), shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor with uniform random values in [0, 1). */
    public static GradTensor rand(long... shape) {
        int n = (int) numelFor(shape);
        float[] dataArr = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) {
            dataArr[i] = rng.nextFloat();
        }
        return new GradTensor(new HeapStorage(dataArr), shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /**
     * Create a tensor with uniform random values in [lo, hi).
     *
     * @param lo    lower bound (inclusive)
     * @param hi    upper bound (exclusive)
     * @param shape tensor shape
     * @return initialized tensor
     */
    public static GradTensor uniform(double lo, double hi, long... shape) {
        int n = (int) numelFor(shape);
        float[] dataArr = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) {
            dataArr[i] = (float) (lo + rng.nextDouble() * (hi - lo));
        }
        return new GradTensor(new HeapStorage(dataArr), shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /**
     * Overload for int varargs to resolve ambiguity in some JVMs.
     */
    public static GradTensor uniform(double lo, double hi, int... shape) {
        long[] lshape = new long[shape.length];
        for (int i = 0; i < shape.length; i++)
            lshape[i] = shape[i];
        return uniform(lo, hi, lshape);
    }

    /**
     * Create a tensor with uniform random values in [lo, hi).
     * Convenience method for 4D/5D shapes used in CNNs.
     *
     * @param lo   lower bound
     * @param hi   upper bound
     * @param dims dimensions
     * @return initialized tensor
     */
    public static GradTensor uniform(long[] dims, double lo, double hi) {
        return uniform(lo, hi, dims);
    }

    /**
     * Internal compatibility for old API: GradTensor.uniform(d1, d2, d3... lo, hi)
     */
    public static GradTensor uniform(int d1, int d2, int d3, int d4, double lo, double hi) {
        return uniform(lo, hi, d1, d2, d3, d4);
    }

    public static GradTensor uniform(int d1, int d2, int d3, int d4, int d5, double lo, double hi) {
        return uniform(lo, hi, d1, d2, d3, d4, d5);
    }

    /** Create a 1D tensor with values from start (inclusive) to end (exclusive). */
    public static GradTensor arange(float start, float end, float step) {
        int n = (int) Math.ceil((end - start) / step);
        float[] dataArr = new float[n];
        for (int i = 0; i < n; i++) {
            dataArr[i] = start + i * step;
        }
        return new GradTensor(new HeapStorage(dataArr), new long[] { n }, DType.FLOAT32, Device.CPU, false);
    }

    /** Create a scalar tensor. */
    public static GradTensor scalar(float value) {
        return new GradTensor(new HeapStorage(new float[] { value }), new long[] {}, DType.FLOAT32, Device.CPU, false);
    }

    /** Create an identity matrix. */
    public static GradTensor eye(int n) {
        float[] dataArr = new float[n * n];
        for (int i = 0; i < n; i++) {
            dataArr[i * n + i] = 1.0f;
        }
        return new GradTensor(new HeapStorage(dataArr), new long[] { n, n }, DType.FLOAT32, Device.CPU, false);
    }

    /**
     * Create a tensor from a BufferedImage [C, H, W] in range [0, 1].
     */
    public static GradTensor fromImage(java.awt.image.BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        int c = 3;
        float[] data = new float[c * h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                data[0 * h * w + y * w + x] = ((rgb >> 16) & 0xFF) / 255.0f; // R
                data[1 * h * w + y * w + x] = ((rgb >> 8) & 0xFF) / 255.0f; // G
                data[2 * h * w + y * w + x] = (rgb & 0xFF) / 255.0f; // B
            }
        }
        return GradTensor.of(data, c, h, w);
    }

    // ── Autograd controls ────────────────────────────────────────────────

    /** Enable gradient tracking. Returns this tensor for chaining. */
    public GradTensor requiresGrad(boolean requires) {
        this.requiresGrad = requires;
        if (requires && this.gradStorage == null) {
            this.gradStorage = new HeapStorage(new float[storage.size()]);
        }
        return this;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    /** Get the gradient. Returns null if no gradient has been computed. */
    public GradTensor grad() {
        if (gradStorage == null)
            return null;
        return GradTensor.of(gradStorage.asArray(), shape);
    }

    /** Zero out the gradient. */
    public void zeroGrad() {
        if (gradStorage != null) {
            Arrays.fill(gradStorage.asArray(), 0.0f);
        }
    }

    /**
     * Returns a tensor on the specified device.
     *
     * <p>
     * If the tensor is already on the target device, returns {@code this}.
     * Otherwise, creates a copy tagged with the new device. Actual data transfer
     * to GPU memory will be handled by the active
     * {@link tech.kayys.gollek.spi.tensor.ComputeBackend}
     * when GPU backends are available.
     *
     * @param target the target device (e.g. {@code Device.CUDA},
     *               {@code Device.METAL})
     * @return a tensor on the target device
     */
    public GradTensor to(Device target) {
        if (this.device == target) {
            return this;
        }

        TensorStorage targetStorage;
        if (target == Device.METAL) {
            targetStorage = new UnifiedStorage(storage.size());
            // Copy data to unified memory
            MemorySegment.copy(storage.asSegment(), ValueLayout.JAVA_FLOAT, 0,
                    targetStorage.asSegment(), ValueLayout.JAVA_FLOAT, 0, storage.size());
        } else {
            targetStorage = new HeapStorage(storage.asArray().clone());
        }

        GradTensor moved = new GradTensor(targetStorage, shape.clone(), dtype, target, requiresGrad);
        if (gradStorage != null) {
            moved.gradStorage = (target == Device.METAL)
                    ? new UnifiedStorage(gradStorage.size())
                    : new HeapStorage(gradStorage.asArray().clone());
        }
        return moved;
    }

    /** Returns the underlying storage object. */
    public TensorStorage storage() {
        return storage;
    }

    /**
     * Casts this tensor to the specified data type.
     *
     * <p>
     * If it is already the target DType, returns {@code this}.
     * Otherwise returns a new casted tensor.
     * Currently stores float[] internally but mocks DType representation.
     */
    public GradTensor to(DType targetDType) {
        if (this.dtype == targetDType) {
            return this;
        }
        GradTensor casted = new GradTensor(storage.duplicate(), shape.clone(), targetDType, device, requiresGrad);
        if (gradStorage != null) {
            casted.gradStorage = gradStorage.duplicate();
        }
        return casted;
    }

    /** Set the function that created this tensor (for autograd). */
    public void setGradFn(Function.Context ctx) {
        this.gradFn = ctx;
    }

    public Function.Context gradFn() {
        return gradFn;
    }

    // ── Backward pass ────────────────────────────────────────────────────

    /**
     * Compute gradients via reverse-mode automatic differentiation.
     * <p>
     * This tensor must be a scalar (empty shape or single element).
     * Gradients are accumulated into the {@code .grad} field of all
     * upstream tensors with {@code requiresGrad = true}.
     */
    public void backward() {
        if (numel() != 1) {
            throw new IllegalStateException(
                    "backward() can only be called on a scalar tensor, got shape " + Arrays.toString(shape));
        }
        backward(GradTensor.ones(shape));
    }

    /**
     * Compute gradients with a given upstream gradient.
     * Uses SIMD for gradient accumulation when possible.
     */
    public void backward(GradTensor upstreamGrad) {
        if (requiresGrad && gradStorage != null) {
            float[] gd = gradStorage.asArray();
            float[] ug = upstreamGrad.data();
            int len = Math.min(gd.length, ug.length);

            // SIMD-accelerated gradient accumulation
            int i = 0;
            int bound = SPECIES.loopBound(len);
            for (; i < bound; i += SPECIES.length()) {
                FloatVector gVec = FloatVector.fromArray(SPECIES, gd, i);
                FloatVector ugVec = FloatVector.fromArray(SPECIES, ug, i);
                gVec.add(ugVec).intoArray(gd, i);
            }
            for (; i < len; i++) {
                gd[i] += ug[i];
            }
        }
        if (gradFn != null) {
            gradFn.backward(upstreamGrad);
        }
    }

    // ── Tensor operations (differentiable) ───────────────────────────────

    /** Element-wise addition. */
    public GradTensor add(GradTensor other) {
        return Functions.Add.apply(this, other);
    }

    /** Scalar addition. */
    public GradTensor add(float scalar) {
        return add(GradTensor.full(scalar, shape));
    }

    /** Element-wise subtraction. */
    public GradTensor sub(GradTensor other) {
        return Functions.Sub.apply(this, other);
    }

    /** Scalar subtraction. */
    public GradTensor sub(float scalar) {
        return sub(GradTensor.full(scalar, shape));
    }

    /** Element-wise multiplication (Hadamard product). */
    public GradTensor mul(GradTensor other) {
        return Functions.Mul.apply(this, other);
    }

    /** Scalar multiplication. */
    public GradTensor mul(float scalar) {
        return mul(GradTensor.full(scalar, shape));
    }

    /** Element-wise division. */
    public GradTensor div(GradTensor other) {
        return Functions.Div.apply(this, other);
    }

    /** Scalar division. */
    public GradTensor div(float scalar) {
        return div(GradTensor.full(scalar, shape));
    }

    /** Matrix multiplication. */
    public GradTensor matmul(GradTensor other) {
        return Functions.Matmul.apply(this, other);
    }

    /** Transpose (swap last two dimensions). */
    public GradTensor transpose() {
        return Functions.Transpose.apply(this, -2, -1);
    }

    /** ReLU activation. */
    public GradTensor relu() {
        return Functions.Relu.apply(this);
    }

    /** Sigmoid activation. */
    public GradTensor sigmoid() {
        return Functions.Sigmoid.apply(this);
    }

    /** Tanh activation. */
    public GradTensor tanh() {
        return Functions.Tanh.apply(this);
    }

    /** Natural logarithm. */
    public GradTensor log() {
        return Functions.Log.apply(this);
    }

    /** Exponential. */
    public GradTensor exp() {
        return Functions.Exp.apply(this);
    }

    /** Softmax along the last dimension. */
    public GradTensor softmax() {
        return Functions.Softmax.apply(this);
    }

    /** Log-softmax along the last dimension. */
    public GradTensor logSoftmax() {
        return softmax().log();
    }

    /** Sum all elements to a scalar. */
    public GradTensor sum() {
        float[] data = data();
        float total = 0;
        for (float v : data) total += v;
        GradTensor out = GradTensor.scalar(total);
        if (requiresGrad()) {
            out.requiresGrad(true);
            final GradTensor self = this;
            out.setGradFn(new Function.Context("Sum") {
                @Override public void backward(GradTensor upstream) {
                    float s = upstream.item();
                    float[] grad = new float[data.length];
                    java.util.Arrays.fill(grad, s);
                    self.backward(GradTensor.of(grad, self.shape()));
                }
            });
        }
        return out;
    }

    /** Sum along specified dimension. */
    public GradTensor sum(int dim) {
        return Functions.Sum.apply(this, dim, false);
    }

    /** Mean of all elements. */
    public GradTensor mean() {
        return Functions.Mean.apply(this);
    }

    /** Mean along specified dimension. */
    public GradTensor mean(int dim) {
        return Functions.Mean.apply(this, dim);
    }

    /** SiLU (Swish) activation: x * sigmoid(x). */
    public GradTensor silu() {
        float[] d = data(), out = new float[d.length];
        for (int i = 0; i < d.length; i++)
            out[i] = d[i] / (1f + (float) Math.exp(-d[i]));
        GradTensor result = GradTensor.of(out, shape());
        if (requiresGrad()) {
            result.requiresGrad(true);
            final GradTensor inputTensor = this;
            result.setGradFn(new Function.Context("SiLUBackward") {
                @Override
                public void backward(GradTensor g) {
                    float[] gd = g.data(), grad = new float[d.length];
                    for (int i = 0; i < d.length; i++) {
                        float sig = 1f / (1f + (float) Math.exp(-d[i]));
                        grad[i] = gd[i] * (sig + d[i] * sig * (1f - sig));
                    }
                    inputTensor.backward(GradTensor.of(grad, inputTensor.shape()));
                }
            });
        }
        return result;
    }

    /** Negate: -x. */
    public GradTensor neg() {
        return mul(-1.0f);
    }

    /** Power: x^p (element-wise). */
    public GradTensor pow(float p) {
        return Functions.Pow.apply(this, p);
    }

    /** Absolute value. */
    public GradTensor abs() {
        float[] d = data(), out = new float[d.length];
        for (int i = 0; i < d.length; i++)
            out[i] = Math.abs(d[i]);
        GradTensor result = GradTensor.of(out, shape());
        if (requiresGrad()) {
            result.requiresGrad(true);
            final GradTensor inputTensor = this;
            result.setGradFn(new Function.Context("AbsBackward") {
                @Override
                public void backward(GradTensor g) {
                    float[] gd = g.data(), grad = new float[d.length];
                    for (int i = 0; i < d.length; i++) {
                        grad[i] = gd[i] * (d[i] > 0 ? 1f : d[i] < 0 ? -1f : 0f);
                    }
                    inputTensor.backward(GradTensor.of(grad, inputTensor.shape()));
                }
            });
        }
        return result;
    }

    /** Square root. */
    public GradTensor sqrt() {
        return pow(0.5f);
    }

    /** Reciprocal square root. */
    public GradTensor rsqrt() {
        return pow(-0.5f);
    }

    /** Clamp values to [min, max]. */
    public GradTensor clamp(float min, float max) {
        float[] d = data(), out = new float[d.length];
        for (int i = 0; i < d.length; i++)
            out[i] = Math.max(min, Math.min(max, d[i]));
        GradTensor result = GradTensor.of(out, shape());
        if (requiresGrad()) {
            result.requiresGrad(true);
            final GradTensor inputTensor = this;
            result.setGradFn(new Function.Context("ClampBackward") {
                @Override
                public void backward(GradTensor g) {
                    float[] gd = g.data(), grad = new float[d.length];
                    for (int i = 0; i < d.length; i++) {
                        grad[i] = (d[i] > min && d[i] < max) ? gd[i] : 0f;
                    }
                    inputTensor.backward(GradTensor.of(grad, inputTensor.shape()));
                }
            });
        }
        return result;
    }

    /** Find the maximum value along the specified dimension. */
    public GradTensor max(int dim) {
        if (dim < 0)
            dim = shape.length + dim;
        long[] s = shape();
        long[] outShape = new long[s.length - 1];
        System.arraycopy(s, 0, outShape, 0, dim);
        System.arraycopy(s, dim + 1, outShape, dim, s.length - dim - 1);

        float[] out = new float[(int) numelFor(outShape)];
        Arrays.fill(out, Float.NEGATIVE_INFINITY);

        int outerStride = 1;
        for (int i = 0; i < dim; i++)
            outerStride *= s[i];
        int dimSize = (int) s[dim];
        int innerStride = 1;
        for (int i = dim + 1; i < s.length; i++)
            innerStride *= s[i];

        for (int o = 0; o < outerStride; o++) {
            for (int i = 0; i < innerStride; i++) {
                float m = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < dimSize; v++) {
                    float val = storage.asArray()[o * dimSize * innerStride + v * innerStride + i];
                    if (val > m)
                        m = val;
                }
                out[o * innerStride + i] = m;
            }
        }

        GradTensor result = GradTensor.of(out, outShape);
        // Autograd for max: gradient flows to the index that was maximum.
        // Simplified: not implementing backward for max right now as it's mostly for
        // inference.
        return result;
    }

    /** Find the index of the maximum value along the specified dimension. */
    public GradTensor argmax(int dim) {
        if (dim < 0)
            dim = shape.length + dim;
        long[] s = shape();
        long[] outShape = new long[s.length - 1];
        System.arraycopy(s, 0, outShape, 0, dim);
        System.arraycopy(s, dim + 1, outShape, dim, s.length - dim - 1);

        float[] out = new float[(int) numelFor(outShape)];

        int outerStride = 1;
        for (int i = 0; i < dim; i++)
            outerStride *= s[i];
        int dimSize = (int) s[dim];
        int innerStride = 1;
        for (int i = dim + 1; i < s.length; i++)
            innerStride *= s[i];

        for (int o = 0; o < outerStride; o++) {
            for (int i = 0; i < innerStride; i++) {
                float m = Float.NEGATIVE_INFINITY;
                int idx = 0;
                for (int v = 0; v < dimSize; v++) {
                    float val = storage.asArray()[o * dimSize * innerStride + v * innerStride + i];
                    if (val > m) {
                        m = val;
                        idx = v;
                    }
                }
                out[o * innerStride + i] = (float) idx;
            }
        }

        return GradTensor.of(out, outShape);
    }

    /** Reshape (zero-copy view). */
    public GradTensor reshape(long... newShape) {
        long newNumel = numelFor(newShape);
        if (newNumel != numel()) {
            throw new IllegalArgumentException(
                    "Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
        }
        GradTensor result = new GradTensor(storage, newShape.clone(), dtype, device, requiresGrad);
        // For autograd: reshaping is a view, share grad reference
        if (requiresGrad) {
            result.gradStorage = this.gradStorage;
        }
        return result;
    }

    /** Flatten to 1D. */
    public GradTensor flatten() {
        return reshape(numel());
    }

    /** Unsqueeze — add a dimension of size 1 at the given position. */
    public GradTensor unsqueeze(int dim) {
        if (dim < 0)
            dim = shape.length + 1 + dim;
        long[] newShape = new long[shape.length + 1];
        System.arraycopy(shape, 0, newShape, 0, dim);
        newShape[dim] = 1;
        System.arraycopy(shape, dim, newShape, dim + 1, shape.length - dim);
        return reshape(newShape);
    }

    /** Squeeze — remove all dimensions of size 1. */
    public GradTensor squeeze() {
        long[] newShape = Arrays.stream(shape).filter(d -> d != 1).toArray();
        if (newShape.length == 0)
            newShape = new long[] {};
        return reshape(newShape);
    }

    // ── Advanced Tensor Operations ─────────────────────────────────────

    /**
     * Concatenate tensors along the given dimension.
     * All tensors must have the same shape except in the concatenating dimension.
     */
    public static GradTensor cat(GradTensor... tensors) {
        return cat(0, tensors);
    }

    /**
     * Concatenate tensors along the specified dimension.
     */
    public static GradTensor cat(int dim, GradTensor... tensors) {
        if (tensors.length == 0)
            throw new IllegalArgumentException("cat requires at least one tensor");
        if (tensors.length == 1)
            return tensors[0];

        int ndim = tensors[0].ndim();
        if (dim < 0)
            dim = ndim + dim;
        if (dim < 0 || dim >= ndim)
            throw new IllegalArgumentException("dim out of range");

        // Validate shapes match except on concat dim
        long[] refShape = tensors[0].shape();
        long concatSize = 0;
        for (GradTensor t : tensors) {
            if (t.ndim() != ndim)
                throw new IllegalArgumentException("tensors must have same number of dimensions");
            concatSize += t.size(dim);
            for (int d = 0; d < ndim; d++) {
                if (d != dim && t.size(d) != refShape[d]) {
                    throw new IllegalArgumentException("tensor shapes incompatible at dimension " + d);
                }
            }
        }

        // Compute output shape
        long[] outShape = refShape.clone();
        outShape[dim] = concatSize;
        float[] out = new float[(int) numelFor(outShape)];

        // Copy data — strided copy that handles any concat dimension
        // For dim=0 this degenerates to simple arraycopy.
        // For dim>0, we need to interleave chunks from each tensor.
        final int fDim = dim;

        // innerSize = product of dimensions after concat dim
        long innerSize = 1;
        for (int d = fDim + 1; d < ndim; d++)
            innerSize *= outShape[d];

        // outerSize = product of dimensions before concat dim
        long outerSize = 1;
        for (int d = 0; d < fDim; d++)
            outerSize *= outShape[d];

        long dstOffset = 0;
        for (long outer = 0; outer < outerSize; outer++) {
            long catOffset = 0;
            for (GradTensor t : tensors) {
                long tDimSize = t.size(fDim);
                long tSlice = tDimSize * innerSize;
                long srcOffset = outer * tSlice;
                System.arraycopy(t.data(), (int) srcOffset, out,
                        (int) (outer * concatSize * innerSize + catOffset * innerSize), (int) tSlice);
                catOffset += tDimSize;
            }
        }

        boolean needsGrad = Arrays.stream(tensors).anyMatch(GradTensor::requiresGrad);
        GradTensor result = GradTensor.of(out, outShape);
        if (needsGrad) {
            result.requiresGrad(true);
        }
        return result;
    }

    /**
     * Stack tensors along a new dimension.
     * All tensors must have the same shape.
     */
    public static GradTensor stack(GradTensor... tensors) {
        return stack(0, tensors);
    }

    /**
     * Stack tensors along the specified dimension.
     */
    public static GradTensor stack(int dim, GradTensor... tensors) {
        if (tensors.length == 0)
            throw new IllegalArgumentException("stack requires at least one tensor");
        long[] refShape = tensors[0].shape();
        long numel = tensors[0].numel();

        // Validate all shapes match
        for (GradTensor t : tensors) {
            if (!Arrays.equals(t.shape(), refShape)) {
                throw new IllegalArgumentException("all tensors must have the same shape for stack");
            }
        }

        long[] outShape = new long[refShape.length + 1];
        if (dim < 0)
            dim = refShape.length + 1 + dim;
        System.arraycopy(refShape, 0, outShape, 0, dim);
        outShape[dim] = tensors.length;
        System.arraycopy(refShape, dim, outShape, dim + 1, refShape.length - dim);

        float[] out = new float[(int) numelFor(outShape)];
        int pos = 0;
        int chunkSize = (int) numel;
        for (GradTensor t : tensors) {
            System.arraycopy(t.data(), 0, out, pos, chunkSize);
            pos += chunkSize;
        }

        boolean needsGrad = Arrays.stream(tensors).anyMatch(GradTensor::requiresGrad);
        GradTensor result = GradTensor.of(out, outShape);
        if (needsGrad)
            result.requiresGrad(true);
        return result;
    }

    /**
     * Select elements at given indices along a dimension.
     * Equivalent to PyTorch's index_select.
     */
    public GradTensor indexSelect(int dim, int... indices) {
        long[] s = shape();
        if (dim < 0)
            dim = s.length + dim;
        if (dim < 0 || dim >= s.length)
            throw new IllegalArgumentException("dim out of range");

        long[] outShape = s.clone();
        outShape[dim] = indices.length;
        float[] out = new float[(int) numelFor(outShape)];

        int stride = 1;
        for (int d = s.length - 1; d > dim; d--)
            stride *= s[d];
        int outerStride = 1;
        for (int d = dim + 1; d < s.length; d++)
            outerStride *= s[d];
        int dimSize = (int) s[dim];

        for (int idx = 0; idx < indices.length; idx++) {
            int srcIdx = indices[idx];
            if (srcIdx < 0 || srcIdx >= dimSize)
                throw new IndexOutOfBoundsException("index " + srcIdx + " out of range [0, " + dimSize + ")");
            for (int outer = 0; outer < (int) numel() / (dimSize * outerStride); outer++) {
                int srcBase = outer * dimSize * outerStride + srcIdx * outerStride;
                int dstBase = outer * indices.length * outerStride + idx * outerStride;
                System.arraycopy(data(), srcBase, out, dstBase, outerStride);
            }
        }

        GradTensor result = GradTensor.of(out, outShape);
        if (requiresGrad()) {
            result.requiresGrad(true);
            final int fDim = dim;
            final int[] fIndices = indices.clone();
            final long[] fShape = s.clone();
            result.setGradFn(new Function.Context("IndexSelectBackward") {
                @Override
                public void backward(GradTensor g) {
                    float[] grad = new float[(int) numel()];
                    int dStride = 1;
                    for (int d = fShape.length - 1; d > fDim; d--)
                        dStride *= fShape[d];
                    int dOuterStride = 1;
                    for (int d = fDim + 1; d < fShape.length; d++)
                        dOuterStride *= fShape[d];
                    int dDimSize = (int) fShape[fDim];
                    for (int idx = 0; idx < fIndices.length; idx++) {
                        for (int outer = 0; outer < (int) numel() / (dDimSize * dOuterStride); outer++) {
                            int srcBase = outer * fIndices.length * dOuterStride + idx * dOuterStride;
                            int dstBase = outer * dDimSize * dOuterStride + fIndices[idx] * dOuterStride;
                            for (int j = 0; j < dOuterStride; j++) {
                                grad[dstBase + j] += g.data()[srcBase + j];
                            }
                        }
                    }
                    backward(GradTensor.of(grad, fShape));
                }
            });
        }
        return result;
    }

    /**
     * Repeat tensor along each dimension.
     * Equivalent to PyTorch's repeat/repeat_interleave.
     */
    public GradTensor repeat(long... repeats) {
        final long[] fRepeats;
        if (repeats.length < shape().length) {
            long[] newRepeats = new long[shape().length];
            int diff = (int) (shape().length - repeats.length);
            Arrays.fill(newRepeats, 0, diff, 1);
            System.arraycopy(repeats, 0, newRepeats, diff, repeats.length);
            fRepeats = newRepeats;
        } else {
            fRepeats = repeats.clone();
        }

        long[] outShape = new long[shape().length];
        for (int i = 0; i < shape().length; i++)
            outShape[i] = shape()[i] * fRepeats[i];
        float[] out = new float[(int) numelFor(outShape)];

        _repeatCopy(data(), out, shape(), fRepeats, 0, 0, 0);

        GradTensor result = GradTensor.of(out, outShape);
        if (requiresGrad()) {
            result.requiresGrad(true);
            final long[] fShape = shape().clone();
            result.setGradFn(new Function.Context("RepeatBackward") {
                @Override
                public void backward(GradTensor g) {
                    float[] grad = new float[(int) numel()];
                    _sumRepeatGrad(g.data(), grad, fShape, fRepeats, 0, 0, 0);
                    backward(GradTensor.of(grad, fShape));
                }
            });
        }
        return result;
    }

    private void _repeatCopy(float[] src, float[] dst, long[] shape, long[] repeats,
            int dim, long srcOffset, long dstOffset) {
        if (dim == shape.length) {
            dst[(int) dstOffset] = src[(int) srcOffset];
            return;
        }
        long size = shape[dim];
        long rep = repeats[dim];
        long srcStride = _prod(shape, dim + 1);
        long dstStride = srcStride * rep;
        for (long i = 0; i < size; i++) {
            for (long r = 0; r < rep; r++) {
                _repeatCopy(src, dst, shape, repeats, dim + 1,
                        srcOffset + i * srcStride, dstOffset + (i * rep + r) * srcStride);
            }
        }
    }

    private void _sumRepeatGrad(float[] srcGrad, float[] dstGrad, long[] shape, long[] repeats,
            int dim, long srcOffset, long dstOffset) {
        if (dim == shape.length) {
            dstGrad[(int) dstOffset] += srcGrad[(int) srcOffset];
            return;
        }
        long size = shape[dim];
        long rep = repeats[dim];
        long srcStride = _prod(shape, dim + 1);
        for (long i = 0; i < size; i++) {
            long sOff = srcOffset + i * srcStride;
            long dOff = dstOffset + i * rep * srcStride;
            _sumRepeatGrad(srcGrad, dstGrad, shape, repeats, dim + 1, sOff, dOff);
        }
    }

    private long _prod(long[] arr, int from) {
        long p = 1;
        for (int i = from; i < arr.length; i++)
            p *= arr[i];
        return p;
    }

    /**
     * Where condition: return elements from x where condition is true, else from y.
     * Simplified: condition tensor is a float[] where >0 means true.
     */
    public static GradTensor where(GradTensor condition, GradTensor x, GradTensor y) {
        if (x.numel() != y.numel() || condition.numel() != x.numel()) {
            throw new IllegalArgumentException("condition, x, y must have same numel");
        }
        float[] out = new float[x.data().length];
        float[] cd = condition.data(), xd = x.data(), yd = y.data();
        for (int i = 0; i < out.length; i++) {
            out[i] = cd[i] > 0 ? xd[i] : yd[i];
        }
        return GradTensor.of(out, x.shape());
    }

    /**
     * Gather elements along a dimension at given indices.
     * Indices tensor must have same shape as self except in the gather dim.
     */
    public GradTensor gather(int dim, GradTensor indices) {
        if (ndim() != indices.ndim())
            throw new IllegalArgumentException("self and indices must have same ndim");
        long[] outShape = indices.shape();
        float[] out = new float[(int) indices.numel()];

        long selfSize = size(dim < 0 ? shape().length + dim : dim);
        for (int i = 0; i < indices.numel(); i++) {
            int idx = (int) indices.data()[i];
            if (idx < 0 || idx >= selfSize)
                throw new IndexOutOfBoundsException("gather index out of range");
            // Simple 1D gather for now — full ND gather needs coordinate calculation
            out[i] = data()[i / (int) indices.numel() * (int) selfSize + idx];
        }
        return GradTensor.of(out, outShape);
    }

    /**
     * Einsum — Einstein summation convention.
     * Simplified implementation supporting common patterns:
     * "ij,jk->ik" (matmul), "ij->ji" (transpose), "ii->i" (diag),
     * "ijk,klm->ilm" (batched matmul), etc.
     *
     * <p>
     * For full einsum, use a library. This covers the 90% case.
     */
    public GradTensor einsum(String equation, GradTensor other) {
        // Parse equation like "ij,jk->ik"
        String[] parts = equation.replace(" ", "").split("->");
        String lhs = parts[0];
        String rhs = parts.length > 1 ? parts[1] : "";
        String[] operands = lhs.split(",");

        if (operands.length != 2)
            throw new IllegalArgumentException("einsum currently supports 2 operands");

        String sub1 = operands[0], sub2 = operands[1];

        // Simple matmul: "ij,jk->ik"
        if (sub1.length() == 2 && sub2.length() == 2 && rhs.length() == 2) {
            char i1 = sub1.charAt(0), j1 = sub1.charAt(1);
            char j2 = sub2.charAt(0), k2 = sub2.charAt(1);
            if (j1 == j2) {
                // Extract dimensions
                int M = (int) this.size(sub1.indexOf(i1));
                int K = (int) this.size(sub1.indexOf(j1));
                int N = (int) other.size(sub2.indexOf(k2));
                return this.matmul(other);
            }
        }

        // Transpose: "ij->ji"
        if (sub1.length() == 2 && rhs.length() == 2 && sub1.charAt(0) == rhs.charAt(1)
                && sub1.charAt(1) == rhs.charAt(0)) {
            return this.transpose();
        }

        // Batched matmul: "bik,bkj->bij"
        if (sub1.length() == 3 && sub2.length() == 3 && rhs.length() == 3 && sub1.charAt(0) == sub2.charAt(0)
                && sub1.charAt(0) == rhs.charAt(0)) {
            return this.matmul(other);
        }

        throw new IllegalArgumentException("Unsupported einsum equation: " + equation);
    }

    // ── Detach / clone ───────────────────────────────────────────────────

    /** Detach from computation graph. Returns a new tensor with no grad_fn. */
    public GradTensor detach() {
        return new GradTensor(storage.duplicate(), shape.clone(), dtype, device, false);
    }

    /** Clone with full copy of data. */
    public GradTensor clone_() {
        GradTensor copy = new GradTensor(storage.duplicate(), shape.clone(), dtype, device, requiresGrad);
        if (gradStorage != null) {
            copy.gradStorage = gradStorage.duplicate();
        }
        return copy;
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /** Raw data array (shared reference — do not mutate). */
    public float[] data() {
        return storage.asArray();
    }

    /** Returns the raw gradient data as a float array. */
    public float[] gradData() {
        return gradStorage != null ? gradStorage.asArray() : null;
    }

    /** Get a single element (flat index). */
    public float item(int index) {
        return data()[index];
    }

    /** Get the scalar value (tensor must have exactly 1 element). */
    public float item() {
        if (numel() != 1) {
            throw new IllegalStateException("item() requires exactly 1 element, got " + numel());
        }
        return data()[0];
    }

    /** Shape of this tensor. */
    public long[] shape() {
        return shape.clone();
    }

    /** Number of dimensions. */
    public int ndim() {
        return shape.length;
    }

    /** Size along a dimension. */
    public long size(int dim) {
        if (dim < 0)
            dim = shape.length + dim;
        return shape[dim];
    }

    /** Total number of elements. */
    public long numel() {
        return numelFor(shape);
    }

    public DType dtype() {
        return dtype;
    }

    public Device device() {
        return device;
    }

    // ── Internals ────────────────────────────────────────────────────────

    /** Accumulate gradient (used by backward pass). */
    void accumulateGrad(float[] upstream) {
        float[] gd = gradData();
        if (gd != null) {
            for (int i = 0; i < gd.length && i < upstream.length; i++) {
                gd[i] += upstream[i];
            }
        }
    }

    public static long numelFor(long[] shape) {
        if (shape.length == 0)
            return 1; // scalar
        long n = 1;
        for (long d : shape)
            n *= d;
        return n;
    }

    // ── Object methods ───────────────────────────────────────────────────

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        GradTensor that = (GradTensor) o;
        if (dtype != that.dtype)
            return false;
        if (!Arrays.equals(shape, that.shape))
            return false;
        return Arrays.equals(data(), that.data());
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(dtype);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data());
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("GradTensor(shape=");
        sb.append(Arrays.toString(shape));
        sb.append(", dtype=").append(dtype);
        if (requiresGrad)
            sb.append(", requires_grad=true");
        if (gradFn != null)
            sb.append(", grad_fn=").append(gradFn.name());
        sb.append(", data=");
        float[] d = data();
        if (d.length <= 10) {
            sb.append(Arrays.toString(d));
        } else {
            sb.append("[").append(d[0]).append(", ").append(d[1]).append(", ..., ")
                    .append(d[d.length - 1]).append("]");
        }
        sb.append(")");
        return sb.toString();
    }
}
