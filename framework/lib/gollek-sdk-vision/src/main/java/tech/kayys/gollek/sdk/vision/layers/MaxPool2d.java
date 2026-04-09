package tech.kayys.gollek.sdk.vision.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;

/**
 * 2D Max Pooling layer.
 *
 * <p>Applies a 2D max pooling over an input signal composed of several input planes.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * MaxPool2d pool = new MaxPool2d(2, 2);  // kernel=2x2, stride=2
 * GradTensor x = GradTensor.randn(1, 64, 224, 224);
 * GradTensor y = pool.forward(x);  // Shape: [1, 64, 112, 112]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class MaxPool2d extends NNModule {

    private final int kernelSize;
    private final int stride;
    private final int padding;

    /**
     * Create a 2D max pooling layer.
     *
     * @param kernelSize size of the pooling window
     * @param stride     stride of the pooling operation
     * @param padding    zero-padding added to both sides of the input
     */
    public MaxPool2d(int kernelSize, int stride, int padding) {
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }

    /**
     * Create MaxPool2d with specified kernelSize and stride, and no padding.
     */
    public MaxPool2d(int kernelSize, int stride) {
        this(kernelSize, stride, 0);
    }

    /**
     * Create MaxPool2d with stride = kernelSize and no padding.
     */
    public MaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize, 0);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // Get the appropriate backend for execution
        VisionBackendProvider backend = VisionBackendRegistry.getDefault();
        
        // Delegate to backend
        return backend.maxPool2d(input, kernelSize, stride, padding);
    }

    /**
     * Get kernel size.
     */
    public int getKernelSize() {
        return kernelSize;
    }

    /**
     * Get stride.
     */
    public int getStride() {
        return stride;
    }

    /**
     * Get padding.
     */
    public int getPadding() {
        return padding;
    }

    @Override
    public String toString() {
        return String.format("MaxPool2d(kernel_size=%d, stride=%d, padding=%d)",
                kernelSize, stride, padding);
    }
}
