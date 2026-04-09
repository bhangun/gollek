package tech.kayys.gollek.sdk.cnn.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;

/**
 * 3D Convolutional Layer.
 *
 * <p>Applies a 3D convolution over volumetric input of shape [N, C_in, D, H, W].</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Conv3d conv = new Conv3d(1, 32, 3, 1, 1);  // in_channels=1, out_channels=32, kernel=3, stride=1, padding=1
 * GradTensor x = GradTensor.randn(2, 1, 10, 28, 28);  // batch_size=2, channels=1, depth=10, height=28, width=28
 * GradTensor y = conv.forward(x);  // Shape: [2, 32, 10, 28, 28]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class Conv3d {

    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final int dilation;
    private final boolean bias;

    // Weight and bias parameters
    private GradTensor weight;
    private GradTensor biasParam;

    /**
     * Create a 3D convolution layer.
     *
     * @param inChannels    number of input channels
     * @param outChannels   number of output channels
     * @param kernelSize    size of the convolution kernel (cubic)
     * @param stride        stride of the convolution (default 1)
     * @param padding       padding added to input (default 0)
     * @param dilation      spacing between kernel elements (default 1)
     * @param bias          whether to use bias (default true)
     */
    public Conv3d(int inChannels, int outChannels, int kernelSize,
                  int stride, int padding, int dilation, boolean bias) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.bias = bias;

        // Initialize weight: [out_channels, in_channels, kernel_size, kernel_size, kernel_size]
        double limit = Math.sqrt(6.0 / (inChannels * kernelSize * kernelSize * kernelSize + outChannels));
        this.weight = GradTensor.uniform(outChannels, inChannels, kernelSize, kernelSize, kernelSize, -limit, limit);
        this.weight.requiresGrad(true);

        // Initialize bias: [out_channels]
        if (bias) {
            this.biasParam = GradTensor.zeros(outChannels);
            this.biasParam.requiresGrad(true);
        }
    }

    /**
     * Create Conv3d with default dilation=1 and bias=true.
     */
    public Conv3d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, 1, true);
    }

    /**
     * Forward pass.
     *
     * @param input input tensor of shape [N, C_in, D, H, W]
     * @return output tensor of shape [N, C_out, D_out, H_out, W_out]
     */
    public GradTensor forward(GradTensor input) {
        VisionBackendProvider backend = VisionBackendRegistry.getDefault();
        return backend.conv3d(input, weight, bias ? biasParam : null, stride, padding);
    }

    /**
     * Get weight parameter.
     */
    public GradTensor getWeight() {
        return weight;
    }

    /**
     * Get bias parameter.
     */
    public GradTensor getBias() {
        return biasParam;
    }

    /**
     * Get input channels.
     */
    public int getInChannels() {
        return inChannels;
    }

    /**
     * Get output channels.
     */
    public int getOutChannels() {
        return outChannels;
    }

    /**
     * Get kernel size.
     */
    public int getKernelSize() {
        return kernelSize;
    }

    @Override
    public String toString() {
        return String.format("Conv3d(%d, %d, kernel_size=%d, stride=%d, padding=%d, dilation=%d)",
                inChannels, outChannels, kernelSize, stride, padding, dilation);
    }
}
