package tech.kayys.gollek.sdk.vision.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;

import java.util.Arrays;
import java.util.Random;

/**
 * 2D Convolution layer.
 *
 * <p>Applies a 2D convolution over an input signal composed of several input planes.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Conv2d conv = new Conv2d(3, 64, 3, 1, 1);  // in_channels=3, out_channels=64, kernel=3x3
 * GradTensor x = GradTensor.randn(1, 3, 224, 224);
 * GradTensor y = conv.forward(x);  // Shape: [1, 64, 224, 224]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class Conv2d extends NNModule {

    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final boolean bias;

    // Learnable parameters
    private final Parameter weight;
    private final Parameter biasParam;

    /**
     * Create a 2D convolution layer.
     *
     * @param inChannels  number of input channels
     * @param outChannels number of output channels
     * @param kernelSize  size of the convolving kernel
     * @param stride      stride of the convolution
     * @param padding     zero-padding added to both sides of the input
     * @param bias        if true, adds a learnable bias to the output
     */
    public Conv2d(int inChannels, int outChannels, int kernelSize, int stride, int padding, boolean bias) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.bias = bias;

        // Initialize weights with Kaiming uniform initialization
        this.weight = registerParameter("weight", kaimingUniform(outChannels, inChannels, kernelSize, kernelSize));
        
        if (bias) {
            float[] biasData = new float[outChannels];
            // Initialize bias to zeros
            this.biasParam = registerParameter("bias", GradTensor.of(biasData, outChannels));
        } else {
            this.biasParam = null;
        }
    }

    /**
     * Create Conv2d with bias enabled.
     */
    public Conv2d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, true);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // Get the appropriate backend for execution
        VisionBackendProvider backend = VisionBackendRegistry.getDefault();
        
        // Delegate to backend
        return backend.conv2d(input, weight.data(), biasParam != null ? biasParam.data() : null, stride, padding);
    }

    /**
     * im2col: Extract patches from input image with blocking for cache efficiency.
     *
     * @param input  input data [N, C, H, W]
     * @param N      batch size
     * @param C      channels
     * @param H      height
     * @param W      width
     * @param H_out  output height
     * @param W_out  output width
     * @return patches [N * H_out * W_out, C * kernelSize * kernelSize]
     */
    private float[] im2col(float[] input, int N, int C, int H, int W, int H_out, int W_out) {
        int patchSize = C * kernelSize * kernelSize;
        int numPatches = N * H_out * W_out;
        float[] patches = new float[numPatches * patchSize];

        // Block size for cache efficiency
        final int BLOCK_H = 16;
        final int BLOCK_W = 16;

        for (int n = 0; n < N; n++) {
            for (int h_block = 0; h_block < H_out; h_block += BLOCK_H) {
                int h_end = Math.min(h_block + BLOCK_H, H_out);

                for (int w_block = 0; w_block < W_out; w_block += BLOCK_W) {
                    int w_end = Math.min(w_block + BLOCK_W, W_out);

                    // Process block
                    for (int h = h_block; h < h_end; h++) {
                        for (int w = w_block; w < w_end; w++) {
                            int patchIdx = (n * H_out * W_out + h * W_out + w) * patchSize;
                            int colIdx = 0;

                            for (int c = 0; c < C; c++) {
                                for (int kh = 0; kh < kernelSize; kh++) {
                                    for (int kw = 0; kw < kernelSize; kw++) {
                                        int h_in = h * stride - padding + kh;
                                        int w_in = w * stride - padding + kw;

                                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                            int inputIdx = ((n * C + c) * H + h_in) * W + w_in;
                                            patches[patchIdx + colIdx] = input[inputIdx];
                                        } else {
                                            patches[patchIdx + colIdx] = 0.0f;  // Zero padding
                                        }
                                        colIdx++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return patches;
    }

    /**
     * Matrix multiplication.
     *
     * @param A       matrix A [M, K]
     * @param M       rows of A
     * @param K       cols of A / rows of B
     * @param B       matrix B [K, N] (if transposeB, then [N, K])
     * @param N       cols of B
     * @param transB  if true, transpose B before multiply
     * @return C = A @ B.T [M, N]
     */
    private float[] matmul(float[] A, int M, int K, float[] B, int N, boolean transB) {
        float[] C = new float[M * N];

        if (transB) {
            // C = A @ B.T
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[m * K + k] * B[n * K + k];
                    }
                    C[m * N + n] = sum;
                }
            }
        } else {
            // C = A @ B
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[m * K + k] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        return C;
    }

    /**
     * Get the weight tensor.
     *
     * @return weight tensor of shape [out_channels, in_channels, kernel_size, kernel_size]
     */
    public GradTensor getWeight() {
        return weight.data();
    }

    /**
     * Get the bias tensor.
     *
     * @return bias tensor of shape [out_channels], or null if bias is disabled
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

    /**
     * Kaiming uniform initialization.
     *
     * @param outCh output channels
     * @param inCh  input channels
     * @param kH    kernel height
     * @param kW    kernel width
     * @return initialized tensor
     */
    private static GradTensor kaimingUniform(int outCh, int inCh, int kH, int kW) {
        int fanIn = inCh * kH * kW;
        double std = Math.sqrt(2.0 / fanIn);
        double bound = std * Math.sqrt(3.0);

        int size = outCh * inCh * kH * kW;
        float[] data = new float[size];
        Random rng = new Random(42);

        for (int i = 0; i < size; i++) {
            data[i] = (float) (rng.nextDouble() * 2 * bound - bound);
        }

        return GradTensor.of(data, outCh, inCh, kH, kW);
    }

    @Override
    public String toString() {
        return String.format("Conv2d(%d, %d, kernel_size=%d, stride=%d, padding=%d, bias=%b)",
                inChannels, outChannels, kernelSize, stride, padding, bias);
    }
}
