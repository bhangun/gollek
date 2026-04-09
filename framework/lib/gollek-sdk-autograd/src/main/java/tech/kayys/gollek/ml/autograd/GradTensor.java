package tech.kayys.gollek.ml.autograd;

import tech.kayys.gollek.runtime.tensor.DType;
import tech.kayys.gollek.runtime.tensor.Device;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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
 * <pre>{@code
 * var x = GradTensor.of(new float[]{1, 2, 3, 4}, 2, 2).requiresGrad(true);
 * var w = GradTensor.randn(2, 3).requiresGrad(true);
 * var y = x.matmul(w).relu();
 * y.sum().backward();
 * // x.grad() now contains ∂y/∂x
 * }</pre>
 */
public final class GradTensor {

    // ── Storage ──────────────────────────────────────────────────────────

    private final float[] data;
    private final long[] shape;
    private DType dtype;
    private Device device;

    // ── Autograd ─────────────────────────────────────────────────────────

    private float[] grad;
    private boolean requiresGrad;
    private Function.Context gradFn;  // the function that produced this tensor
    private final List<GradTensor> children = new ArrayList<>();

    // ── Constructors ─────────────────────────────────────────────────────

    private GradTensor(float[] data, long[] shape, DType dtype, Device device, boolean requiresGrad) {
        this.data = Objects.requireNonNull(data);
        this.shape = Objects.requireNonNull(shape);
        this.dtype = dtype;
        this.device = device;
        this.requiresGrad = requiresGrad;
        if (requiresGrad) {
            this.grad = new float[data.length];
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
        return new GradTensor(data.clone(), shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a 1D tensor from data. */
    public static GradTensor of(float... data) {
        return new GradTensor(data.clone(), new long[]{data.length}, DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor filled with zeros. */
    public static GradTensor zeros(long... shape) {
        return new GradTensor(new float[(int) numelFor(shape)], shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor filled with ones. */
    public static GradTensor ones(long... shape) {
        float[] data = new float[(int) numelFor(shape)];
        Arrays.fill(data, 1.0f);
        return new GradTensor(data, shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor filled with a scalar value. */
    public static GradTensor full(float value, long... shape) {
        float[] data = new float[(int) numelFor(shape)];
        Arrays.fill(data, value);
        return new GradTensor(data, shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor with random normal values (mean=0, std=1). */
    public static GradTensor randn(long... shape) {
        int n = (int) numelFor(shape);
        float[] data = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) {
            data[i] = (float) rng.nextGaussian();
        }
        return new GradTensor(data, shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /** Create a tensor with uniform random values in [0, 1). */
    public static GradTensor rand(long... shape) {
        int n = (int) numelFor(shape);
        float[] data = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) {
            data[i] = rng.nextFloat();
        }
        return new GradTensor(data, shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /**
     * Create a tensor with uniform random values in [lo, hi).
     *
     * @param lo lower bound (inclusive)
     * @param hi upper bound (exclusive)
     * @param shape tensor shape
     * @return initialized tensor
     */
    public static GradTensor uniform(double lo, double hi, long... shape) {
        int n = (int) numelFor(shape);
        float[] data = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) {
            data[i] = (float) (lo + rng.nextDouble() * (hi - lo));
        }
        return new GradTensor(data, shape.clone(), DType.FLOAT32, Device.CPU, false);
    }

    /**
     * Create a tensor with uniform random values in [lo, hi).
     * Convenience method for 4D/5D shapes used in CNNs.
     *
     * @param lo lower bound
     * @param hi upper bound
     * @param dims dimensions
     * @return initialized tensor
     */
    public static GradTensor uniform(long[] dims, double lo, double hi) {
        return uniform(lo, hi, dims);
    }

    /**
     * Internal compatibility for old API: GradTensor.uniform(d1, d2, d3... lo, hi)
     */
    public static GradTensor uniform(long d1, long d2, long d3, long d4, double lo, double hi) {
        return uniform(lo, hi, d1, d2, d3, d4);
    }

    public static GradTensor uniform(long d1, long d2, long d3, long d4, long d5, double lo, double hi) {
        return uniform(lo, hi, d1, d2, d3, d4, d5);
    }

    /** Create a 1D tensor with values from start (inclusive) to end (exclusive). */
    public static GradTensor arange(float start, float end, float step) {
        int n = (int) Math.ceil((end - start) / step);
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = start + i * step;
        }
        return new GradTensor(data, new long[]{n}, DType.FLOAT32, Device.CPU, false);
    }

    /** Create a scalar tensor. */
    public static GradTensor scalar(float value) {
        return new GradTensor(new float[]{value}, new long[]{}, DType.FLOAT32, Device.CPU, false);
    }

    /** Create an identity matrix. */
    public static GradTensor eye(int n) {
        float[] data = new float[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0f;
        }
        return new GradTensor(data, new long[]{n, n}, DType.FLOAT32, Device.CPU, false);
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
                data[1 * h * w + y * w + x] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
                data[2 * h * w + y * w + x] = (rgb & 0xFF) / 255.0f;         // B
            }
        }
        return GradTensor.of(data, c, h, w);
    }

    // ── Autograd controls ────────────────────────────────────────────────

    /** Enable gradient tracking. Returns this tensor for chaining. */
    public GradTensor requiresGrad(boolean requires) {
        this.requiresGrad = requires;
        if (requires && this.grad == null) {
            this.grad = new float[data.length];
        }
        return this;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    /** Get the gradient. Returns null if no gradient has been computed. */
    public GradTensor grad() {
        if (grad == null) return null;
        return GradTensor.of(grad, shape);
    }

    /** Zero out the gradient. */
    public void zeroGrad() {
        if (grad != null) {
            Arrays.fill(grad, 0.0f);
        }
    }

    /**
     * Returns a tensor on the specified device.
     *
     * <p>If the tensor is already on the target device, returns {@code this}.
     * Otherwise, creates a copy tagged with the new device. Actual data transfer
     * to GPU memory will be handled by the active {@link tech.kayys.gollek.spi.tensor.ComputeBackend}
     * when GPU backends are available.
     *
     * @param target the target device (e.g. {@code Device.CUDA}, {@code Device.METAL})
     * @return a tensor on the target device
     */
    public GradTensor to(Device target) {
        if (this.device == target) {
            return this;
        }
        GradTensor moved = new GradTensor(data.clone(), shape.clone(), dtype, target, requiresGrad);
        if (grad != null) {
            moved.grad = grad.clone();
        }
        return moved;
    }

    /**
     * Casts this tensor to the specified data type.
     *
     * <p>If it is already the target DType, returns {@code this}.
     * Otherwise returns a new casted tensor.
     * Currently stores float[] internally but mocks DType representation.
     */
    public GradTensor to(DType targetDType) {
        if (this.dtype == targetDType) {
            return this;
        }
        GradTensor casted = new GradTensor(data.clone(), shape.clone(), targetDType, device, requiresGrad);
        if (grad != null) {
            casted.grad = grad.clone();
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
     */
    public void backward(GradTensor upstreamGrad) {
        if (requiresGrad && grad != null) {
            float[] ug = upstreamGrad.data();
            for (int i = 0; i < grad.length && i < ug.length; i++) {
                grad[i] += ug[i];
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

    /** Matrix multiplication. */
    public GradTensor matmul(GradTensor other) {
        return Functions.Matmul.apply(this, other);
    }

    /** Transpose (swap last two dimensions). */
    public GradTensor transpose() {
        return Functions.Transpose.apply(this);
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
    public GradTensor tanh_() {
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
        return Functions.Sum.apply(this);
    }

    /** Mean of all elements. */
    public GradTensor mean() {
        return Functions.Mean.apply(this);
    }

    /** SiLU (Swish) activation: x * sigmoid(x). */
    public GradTensor silu() {
        float[] d = data(), out = new float[d.length];
        for (int i = 0; i < d.length; i++) out[i] = d[i] / (1f + (float)Math.exp(-d[i]));
        GradTensor result = GradTensor.of(out, shape());
        if (requiresGrad()) {
            result.requiresGrad(true);
            result.setGradFn(new Function.Context("SiLUBackward") {
                @Override public void backward(GradTensor g) {
                    float[] gd = g.data(), grad = new float[d.length];
                    for (int i = 0; i < d.length; i++) {
                        float sig = 1f / (1f + (float)Math.exp(-d[i]));
                        grad[i] = gd[i] * (sig + d[i] * sig * (1f - sig));
                    }
                    backward(GradTensor.of(grad, shape()));
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

    /** Reshape (zero-copy view). */
    public GradTensor reshape(long... newShape) {
        long newNumel = numelFor(newShape);
        if (newNumel != numel()) {
            throw new IllegalArgumentException(
                "Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
        }
        GradTensor result = new GradTensor(data, newShape.clone(), dtype, device, requiresGrad);
        // For autograd: reshaping is a view, share grad reference
        if (requiresGrad) {
            result.grad = this.grad;
        }
        return result;
    }

    /** Flatten to 1D. */
    public GradTensor flatten() {
        return reshape(numel());
    }

    /** Unsqueeze — add a dimension of size 1 at the given position. */
    public GradTensor unsqueeze(int dim) {
        if (dim < 0) dim = shape.length + 1 + dim;
        long[] newShape = new long[shape.length + 1];
        System.arraycopy(shape, 0, newShape, 0, dim);
        newShape[dim] = 1;
        System.arraycopy(shape, dim, newShape, dim + 1, shape.length - dim);
        return reshape(newShape);
    }

    /** Squeeze — remove all dimensions of size 1. */
    public GradTensor squeeze() {
        long[] newShape = Arrays.stream(shape).filter(d -> d != 1).toArray();
        if (newShape.length == 0) newShape = new long[]{};
        return reshape(newShape);
    }

    // ── Detach / clone ───────────────────────────────────────────────────

    /** Detach from computation graph. Returns a new tensor with no grad_fn. */
    public GradTensor detach() {
        return new GradTensor(data.clone(), shape.clone(), dtype, device, false);
    }

    /** Clone with full copy of data. */
    public GradTensor clone_() {
        GradTensor copy = new GradTensor(data.clone(), shape.clone(), dtype, device, requiresGrad);
        if (grad != null) {
            copy.grad = grad.clone();
        }
        return copy;
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /** Raw data array (shared reference — do not mutate). */
    public float[] data() {
        return data;
    }

    /** Get a single element (flat index). */
    public float item(int index) {
        return data[index];
    }

    /** Get the scalar value (tensor must have exactly 1 element). */
    public float item() {
        if (numel() != 1) {
            throw new IllegalStateException("item() requires exactly 1 element, got " + numel());
        }
        return data[0];
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
        if (dim < 0) dim = shape.length + dim;
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
        if (grad != null) {
            for (int i = 0; i < grad.length && i < upstream.length; i++) {
                grad[i] += upstream[i];
            }
        }
    }

    public static long numelFor(long[] shape) {
        if (shape.length == 0) return 1; // scalar
        long n = 1;
        for (long d : shape) n *= d;
        return n;
    }

    // ── Object methods ───────────────────────────────────────────────────

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("GradTensor(shape=");
        sb.append(Arrays.toString(shape));
        sb.append(", dtype=").append(dtype);
        if (requiresGrad) sb.append(", requires_grad=true");
        if (gradFn != null) sb.append(", grad_fn=").append(gradFn.name());
        sb.append(", data=");
        if (data.length <= 10) {
            sb.append(Arrays.toString(data));
        } else {
            sb.append("[").append(data[0]).append(", ").append(data[1]).append(", ..., ")
              .append(data[data.length - 1]).append("]");
        }
        sb.append(")");
        return sb.toString();
    }
}
