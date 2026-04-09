package tech.kayys.gollek.sdk.cnn.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;

/**
 * 2D Transposed Convolution Layer (also known as Deconvolution).
 *
 * <p>Applies a 2D transposed convolution over input of shape [N, C_in, H, W].
 * Used for upsampling in networks like FCN, U-Net, and GANs.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * ConvTranspose2d transpose = new ConvTranspose2d(64, 32, 4, 2, 1);
 * GradTensor x = GradTensor.randn(2, 64, 16, 16);
 * GradTensor y = transpose.forward(x);  // Shape: [2, 32, 32, 32]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class ConvTranspose2d extends NNModule {

    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final int outputPadding;
    private final int dilation;
    private final boolean bias;

    // Weight and bias parameters
    private final Parameter weight;
    private final Parameter biasParam;

    /**
     * Create a 2D transposed convolution layer.
     *
     * @param inChannels      number of input channels
     * @param outChannels     number of output channels
     * @param kernelSize      size of the convolution kernel
     * @param stride          stride of the convolution (default 1)
     * @param padding         zero-padding added to input (default 0)
     * @param outputPadding   additional size added to output (default 0)
     * @param dilation        spacing between kernel elements (default 1)
     * @param bias            whether to use bias (default true)
     */
    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize,
                          int stride, int padding, int outputPadding, int dilation, boolean bias) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.outputPadding = outputPadding;
        this.dilation = dilation;
        this.bias = bias;

        // Initialize weight: [in_channels, out_channels, kernel_size, kernel_size]
        double limit = Math.sqrt(6.0 / (inChannels * kernelSize * kernelSize + outChannels));
        this.weight = registerParameter("weight", 
            GradTensor.uniform(-limit, limit, inChannels, outChannels, kernelSize, kernelSize));

        // Initialize bias: [out_channels]
        if (bias) {
            this.biasParam = registerParameter("bias", GradTensor.zeros(outChannels));
        } else {
            this.biasParam = null;
        }
    }

    /**
     * Create ConvTranspose2d with default outputPadding=0, dilation=1, and bias=true.
     */
    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, 0, 1, true);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        VisionBackendProvider backend = VisionBackendRegistry.getDefault();
        return backend.convTranspose2d(input, weight.data(), bias ? biasParam.data() : null, stride, padding, outputPadding);
    }

    /**
     * Get weight parameter.
     */
    public GradTensor getWeight() {
        return weight.data();
    }

    /**
     * Get bias parameter.
     */
    public GradTensor getBias() {
        return biasParam != null ? biasParam.data() : null;
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
        return String.format("ConvTranspose2d(%d, %d, kernel_size=%d, stride=%d, padding=%d, output_padding=%d)",
                inChannels, outChannels, kernelSize, stride, padding, outputPadding);
    }
}
