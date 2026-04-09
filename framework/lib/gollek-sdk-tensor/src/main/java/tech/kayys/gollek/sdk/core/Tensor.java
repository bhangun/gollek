package tech.kayys.gollek.sdk.core;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Arrays;

/**
 * Unified Tensor abstraction with PyTorch-like API.
 *
 * <p>Wraps {@link GradTensor} with device placement, operator overloading,
 * and convenient factory methods for a more intuitive API.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // Factory methods
 * Tensor x = Tensor.randn(2, 3, 224, 224).to(Device.CUDA);
 * Tensor y = Tensor.zeros(2, 1000);
 *
 * // Operator overloading
 * Tensor z = x.add(y).relu();
 *
 * // Device placement
 * Tensor cpu_tensor = z.to(Device.CPU);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class Tensor {

    private final GradTensor tensor;
    private Device device;

    private Tensor(GradTensor tensor, Device device) {
        this.tensor = tensor;
        this.device = device;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Factory Methods (like torch.*)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a tensor from raw data.
     */
    public static Tensor of(float[] data, long... shape) {
        return new Tensor(GradTensor.of(data, shape), Device.CPU);
    }

    /**
     * Create a tensor filled with zeros.
     */
    public static Tensor zeros(long... shape) {
        return new Tensor(GradTensor.zeros(shape), Device.CPU);
    }

    /**
     * Create a tensor filled with ones.
     */
    public static Tensor ones(long... shape) {
        return new Tensor(GradTensor.ones(shape), Device.CPU);
    }

    /**
     * Create a tensor filled with a scalar value.
     */
    public static Tensor full(float value, long... shape) {
        return new Tensor(GradTensor.full(value, shape), Device.CPU);
    }

    /**
     * Create a tensor with random normal values (mean=0, std=1).
     */
    public static Tensor randn(long... shape) {
        return new Tensor(GradTensor.randn(shape), Device.CPU);
    }

    /**
     * Create a tensor with random uniform values [0, 1).
     */
    public static Tensor rand(long... shape) {
        return new Tensor(GradTensor.rand(shape), Device.CPU);
    }

    /**
     * Create an identity matrix.
     */
    public static Tensor eye(int n) {
        return new Tensor(GradTensor.eye(n), Device.CPU);
    }

    /**
     * Create a tensor with values from a range.
     */
    public static Tensor arange(float start, float end, float step) {
        return new Tensor(GradTensor.arange(start, end, step), Device.CPU);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Device Placement
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Move tensor to a device.
     *
     * @param device target device
     * @return tensor on target device
     */
    public Tensor to(Device device) {
        // Move underlying GradTensor to the target device
        return new Tensor(tensor.to(
            switch (device.type()) {
                case "cuda"  -> tech.kayys.gollek.runtime.tensor.Device.CUDA;
                case "metal" -> tech.kayys.gollek.runtime.tensor.Device.METAL;
                default      -> tech.kayys.gollek.runtime.tensor.Device.CPU;
            }), device);
    }

    /**
     * Get the device this tensor is on.
     */
    public Device device() {
        return device;
    }

    /**
     * Check if tensor is on CUDA device.
     */
    public boolean isCuda() {
        return device.isCuda();
    }

    /**
     * Check if tensor is on CPU.
     */
    public boolean isCpu() {
        return device.isCpu();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Operations (Operator Overloading)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Element-wise addition.
     */
    public Tensor add(Tensor other) {
        return new Tensor(tensor.add(other.tensor), device);
    }

    /**
     * Element-wise addition with scalar.
     */
    public Tensor add(float scalar) {
        return new Tensor(tensor.add(scalar), device);
    }

    /**
     * Element-wise subtraction.
     */
    public Tensor sub(Tensor other) {
        return new Tensor(tensor.sub(other.tensor), device);
    }

    /**
     * Element-wise multiplication.
     */
    public Tensor mul(Tensor other) {
        return new Tensor(tensor.mul(other.tensor), device);
    }

    /**
     * Element-wise multiplication with scalar.
     */
    public Tensor mul(float scalar) {
        return new Tensor(tensor.mul(scalar), device);
    }

    /**
     * Matrix multiplication.
     */
    public Tensor matmul(Tensor other) {
        return new Tensor(tensor.matmul(other.tensor), device);
    }

    /**
     * Element-wise division.
     */
    public Tensor div(Tensor other) {
        return new Tensor(tensor.div(other.tensor), device);
    }

    /**
     * ReLU activation.
     */
    public Tensor relu() {
        return new Tensor(tensor.relu(), device);
    }

    /**
     * Sigmoid activation.
     */
    public Tensor sigmoid() {
        return new Tensor(tensor.sigmoid(), device);
    }

    /**
     * Softmax activation.
     */
    public Tensor softmax() {
        return new Tensor(tensor.softmax(), device);
    }

    /**
     * Sum all elements.
     */
    public Tensor sum() {
        return new Tensor(tensor.sum(), device);
    }

    /**
     * Mean of all elements.
     */
    public Tensor mean() {
        return new Tensor(tensor.mean(), device);
    }

    /**
     * Reshape tensor.
     */
    public Tensor reshape(long... shape) {
        return new Tensor(tensor.reshape(shape), device);
    }

    /**
     * Flatten tensor.
     */
    public Tensor flatten() {
        return new Tensor(tensor.flatten(), device);
    }

    /**
     * Transpose last two dimensions.
     */
    public Tensor transpose() {
        return new Tensor(tensor.transpose(), device);
    }

    /**
     * Add a dimension of size 1.
     */
    public Tensor unsqueeze(int dim) {
        return new Tensor(tensor.unsqueeze(dim), device);
    }

    /**
     * Remove dimensions of size 1.
     */
    public Tensor squeeze() {
        return new Tensor(tensor.squeeze(), device);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get the underlying GradTensor.
     */
    public GradTensor gradTensor() {
        return tensor;
    }

    /**
     * Get raw data array.
     */
    public float[] data() {
        return tensor.data();
    }

    /**
     * Get tensor shape.
     */
    public long[] shape() {
        return tensor.shape();
    }

    /**
     * Get number of dimensions.
     */
    public int ndim() {
        return tensor.ndim();
    }

    /**
     * Get size of a dimension.
     */
    public long size(int dim) {
        return tensor.size(dim);
    }

    /**
     * Get total number of elements.
     */
    public long numel() {
        return tensor.numel();
    }

    /**
     * Get scalar value (tensor must have 1 element).
     */
    public float item() {
        return tensor.item();
    }

    /**
     * Enable gradient computation.
     */
    public Tensor requiresGrad(boolean requires) {
        return new Tensor(tensor.requiresGrad(requires), device);
    }

    /**
     * Check if gradients are required.
     */
    public boolean requiresGrad() {
        return tensor.requiresGrad();
    }

    /**
     * Get gradient tensor.
     */
    public Tensor grad() {
        if (tensor.grad() == null) return null;
        return new Tensor(GradTensor.of(tensor.grad().data(), tensor.shape()), device);
    }

    /**
     * Clear gradients.
     */
    public void zeroGrad() {
        tensor.zeroGrad();
    }

    /**
     * Perform backward pass.
     */
    public void backward() {
        tensor.backward();
    }

    /**
     * Detach from computation graph.
     */
    public Tensor detach() {
        return new Tensor(tensor.detach(), device);
    }

    /**
     * Clone tensor with full copy.
     */
    public Tensor clone_() {
        return new Tensor(tensor.clone_(), device);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Object Methods
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return String.format("Tensor(shape=%s, device=%s, requires_grad=%b)",
                Arrays.toString(shape()), device, requiresGrad());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor tensor1)) return false;
        return device.equals(tensor1.device) && tensor.equals(tensor1.tensor);
    }

    @Override
    public int hashCode() {
        return 31 * tensor.hashCode() + device.hashCode();
    }
}
