package tech.kayys.gollek.sdk.vision.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;

/**
 * Adaptive Average Pooling layer.
 *
 * <p>Applies a 2D adaptive average pooling over an input signal.
 * The output size is fixed, and the kernel/stride are computed automatically.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(1);  // Global average pooling
 * GradTensor x = GradTensor.randn(1, 512, 7, 7);
 * GradTensor y = pool.forward(x);  // Shape: [1, 512, 1, 1]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class AdaptiveAvgPool2d extends NNModule {

    private final int outputSize;

    /**
     * Create adaptive average pooling.
     *
     * @param outputSize output spatial size (square)
     */
    public AdaptiveAvgPool2d(int outputSize) {
        this.outputSize = outputSize;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // Get the appropriate backend for execution
        VisionBackendProvider backend = VisionBackendRegistry.getDefault();
        
        // Delegate to backend with output size
        return backend.adaptiveAvgPool2d(input, new int[]{outputSize, outputSize});
    }

    /**
     * Get output size.
     */
    public int getOutputSize() {
        return outputSize;
    }

    @Override
    public String toString() {
        return String.format("AdaptiveAvgPool2d(output_size=%d)", outputSize);
    }
}
