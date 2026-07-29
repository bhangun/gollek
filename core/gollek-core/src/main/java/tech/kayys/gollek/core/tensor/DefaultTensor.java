package tech.kayys.alkhawarizm.core.tensor;

import tech.kayys.gollek.core.memory.Buffer;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;

public final class DefaultTensor implements Tensor {
    private final Shape shape;
    private final DType dtype;
    private final DeviceType device;
    private final Buffer buffer;
    private final ComputeBackend backend;

    public DefaultTensor(
            Shape shape,
            DType dtype,
            DeviceType device,
            Buffer buffer,
            ComputeBackend backend) {
        this.shape = shape;
        this.dtype = dtype;
        this.device = device;
        this.buffer = buffer;
        this.backend = backend;
    }

    @Override
    public Shape shape() {
        return shape;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public DeviceType device() {
        return device;
    }

    @Override
    public ComputeBackend backend() {
        return backend;
    }

    public Buffer buffer() {
        return buffer;
    }

    @Override
    public Tensor add(Tensor other) {
        return backend.add(this, other);
    }

    @Override
    public Tensor sub(Tensor other) {
        // Implement via backend or simple fallback if backend doesn't have sub yet
        throw new UnsupportedOperationException("sub not implemented in backend yet");
    }

    @Override
    public Tensor mul(Tensor other) {
        throw new UnsupportedOperationException("T x T mul not implemented yet");
    }

    @Override
    public Tensor mul(float scalar) {
        return backend.mul(this, scalar);
    }

    @Override
    public Tensor div(float scalar) {
        return backend.mul(this, 1.0f / scalar);
    }

    @Override
    public Tensor matmul(Tensor other) {
        return backend.matmul(this, other);
    }

    @Override
    public Tensor reshape(long... newShape) {
        return backend.reshape(this, newShape);
    }

    private Tensor grad;
    private boolean requiresGrad;

    @Override
    public Tensor grad() {
        return grad;
    }

    @Override
    public void setGrad(Tensor grad) {
        this.grad = grad;
    }

    @Override
    public boolean requiresGrad() {
        return requiresGrad;
    }

    @Override
    public void setRequiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
    }

    @Override
    public float item() {
        // Implement item access (read from buffer)
        return 0; // Placeholder for now
    }

    @Override
    public void backward() {
        // This will be implemented via AutogradEngine
    }

    @Override
    public Tensor softmax() {
        return backend.softmax(this);
    }

    @Override
    public Tensor slice(long[] offsets, long[] sizes) {
        return backend.slice(this, offsets, sizes);
    }

    @Override
    public Tensor pow(float exponent) {
        return backend.pow(this, exponent);
    }

    @Override
    public Tensor mean() {
        return backend.mean(this);
    }

    @Override
    public Tensor abs() {
        return backend.abs(this);
    }

    @Override
    public Tensor crossEntropy(Tensor target) {
        return backend.crossEntropy(this, target);
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor target) {
        return backend.binaryCrossEntropy(this, target);
    }

    @Override
    public Tensor cast(DType dtype) {
        return backend.cast(this, dtype);
    }

    @Override
    public Tensor to(DeviceType device) {
        return backend.to(this, device);
    }

    @Override
    public Tensor div(Tensor other) {
        return backend.div(this, other);
    }

    @Override
    public Tensor add(float scalar) {
        return backend.addScalar(this, scalar);
    }

    @Override
    public Tensor zerosLike() {
        return backend.zerosLike(this);
    }

    @Override
    public Tensor sqrt() {
        return backend.sqrt(this);
    }

    @Override
    public Tensor relu() {
        return backend.relu(this);
    }

    @Override
    public Tensor sigmoid() {
        return backend.sigmoid(this);
    }

    @Override
    public Tensor tanh() {
        return backend.tanh(this);
    }

    @Override
    public Tensor log() {
        return backend.log(this);
    }

    @Override
    public Tensor exp() {
        return backend.exp(this);
    }

    @Override
    public Tensor silu() {
        return backend.silu(this);
    }

    @Override
    public Tensor flatten() {
        return backend.flatten(this);
    }

    @Override
    public Tensor unsqueeze(int dim) {
        return backend.unsqueeze(this, dim);
    }

    @Override
    public Tensor squeeze() {
        return backend.squeeze(this);
    }

    @Override
    public Tensor transpose() {
        return backend.transpose(this);
    }

    @Override
    public Tensor transpose(int dim0, int dim1) {
        return backend.transpose(this, dim0, dim1);
    }

    @Override
    public long numel() {
        return shape.numel();
    }

    @Override
    public float[] toFloatArray() {
        return buffer.segment().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
    }

    @Override
    public void release() {
        if (buffer != null) {
            buffer.release();
        }
    }

    @Override
    public Tensor eval() {
        return this;
    }

    @Override
    public Tensor gelu() {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor softmax(int dim) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor logSoftmax(int dim) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor sum() {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public java.util.List<Tensor> split(int axis, int parts) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor layerNorm(long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor rmsNorm(Tensor weight, float eps) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor batchNorm(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training,
            float momentum, float eps) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor maxPool2d(int kernelSize, int stride, int padding) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor adaptiveAvgPool2d(int outputH, int outputW) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor dropout(float p, boolean training) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor attention(Tensor K, Tensor V) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor embedding(Tensor weight, long paddingIdx) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public Tensor applyRoPE(int dim, float base, boolean useNeox) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
