/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.sdk.vision.backend;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Pluggable backend provider for vision operations.
 *
 * <p>This interface defines the contract for vision-specific accelerated operations.
 * Different implementations can provide hardware acceleration via CUDA, Metal, ROCm,
 * or CPU fallback.</p>
 *
 * <h2>Backend Types</h2>
 * <ul>
 *   <li><strong>CPU:</strong> Pure Java implementation with optional BLAS</li>
 *   <li><strong>CUDA:</strong> NVIDIA GPU acceleration via CUDA/cuDNN</li>
 *   <li><strong>Metal:</strong> Apple GPU acceleration via Metal Performance Shaders</li>
 *   <li><strong>ROCm:</strong> AMD GPU acceleration via ROCm/MIOpen</li>
 * </ul>
 *
 * <h2>Implementation Requirements</h2>
 * <p>Implementations must:</p>
 * <ul>
 *   <li>Be thread-safe for concurrent operation execution</li>
 *   <li>Handle device-specific memory management</li>
 *   <li>Support automatic fallback to CPU if accelerated path unavailable</li>
 *   <li>Register themselves with {@link VisionBackendRegistry}</li>
 * </ul>
 *
 * @author Gollek Team
 * @since 0.2.0
 */
public interface VisionBackendProvider {

    /**
     * Get backend identifier.
     *
     * @return backend type identifier
     */
    String getBackendId();

    /**
     * Check if backend is available on this system.
     *
     * @return true if backend can be used
     */
    boolean isAvailable();

    /**
     * Get backend priority (0=highest priority).
     *
     * @return priority value
     */
    int getPriority();

    // ─────────────────────── Convolution Operations ───────────────────────

    /**
     * Perform 2D convolution.
     *
     * @param input input tensor [N, C_in, H, W]
     * @param weight weight tensor [C_out, C_in, K_h, K_w]
     * @param bias optional bias [C_out]
     * @param stride convolution stride
     * @param padding zero-padding
     * @return output tensor [N, C_out, H_out, W_out]
     */
    GradTensor conv2d(GradTensor input, GradTensor weight, GradTensor bias,
                      int stride, int padding);

    /**
     * Perform 1D convolution.
     *
     * @param input input tensor [N, C_in, L]
     * @param weight weight tensor [C_out, C_in, K]
     * @param bias optional bias [C_out]
     * @param stride convolution stride
     * @param padding zero-padding
     * @return output tensor [N, C_out, L_out]
     */
    GradTensor conv1d(GradTensor input, GradTensor weight, GradTensor bias,
                      int stride, int padding);

    /**
     * Perform 3D convolution.
     *
     * @param input input tensor [N, C_in, D, H, W]
     * @param weight weight tensor [C_out, C_in, K_d, K_h, K_w]
     * @param bias optional bias [C_out]
     * @param stride convolution stride
     * @param padding zero-padding
     * @return output tensor [N, C_out, D_out, H_out, W_out]
     */
    GradTensor conv3d(GradTensor input, GradTensor weight, GradTensor bias,
                      int stride, int padding);

    /**
     * Perform 2D transposed convolution (deconvolution).
     *
     * @param input input tensor [N, C_in, H, W]
     * @param weight weight tensor [C_in, C_out, K_h, K_w]
     * @param bias optional bias [C_out]
     * @param stride convolution stride
     * @param padding zero-padding
     * @param outputPadding additional size added to output
     * @return output tensor [N, C_out, H_out, W_out]
     */
    GradTensor convTranspose2d(GradTensor input, GradTensor weight, GradTensor bias,
                               int stride, int padding, int outputPadding);

    /**
     * Perform depthwise 2D convolution.
     *
     * @param input input tensor [N, C, H, W]
     * @param weight weight tensor [C, 1, K_h, K_w]
     * @param bias optional bias [C]
     * @param stride convolution stride
     * @param padding zero-padding
     * @return output tensor [N, C, H_out, W_out]
     */
    GradTensor depthwiseConv2d(GradTensor input, GradTensor weight, GradTensor bias,
                               int stride, int padding);

    // ─────────────────────── Fully Connected Operations ───────────────────────

    /**
     * Perform linear (fully connected) transformation.
     *
     * @param input input tensor [N, inFeatures]
     * @param weight weight tensor [outFeatures, inFeatures]
     * @param bias optional bias [outFeatures]
     * @return output tensor [N, outFeatures]
     */
    GradTensor linear(GradTensor input, GradTensor weight, GradTensor bias);

    // ─────────────────────── Pooling Operations ───────────────────────

    /**
     * Perform max pooling.
     *
     * @param input input tensor [N, C, H, W]
     * @param kernelSize pooling kernel size
     * @param stride pooling stride
     * @param padding zero-padding
     * @return output tensor [N, C, H_out, W_out]
     */
    GradTensor maxPool2d(GradTensor input, int kernelSize, int stride, int padding);

    /**
     * Perform average pooling.
     *
     * @param input input tensor [N, C, H, W]
     * @param kernelSize pooling kernel size
     * @param stride pooling stride
     * @param padding zero-padding
     * @return output tensor [N, C, H_out, W_out]
     */
    GradTensor avgPool2d(GradTensor input, int kernelSize, int stride, int padding);

    /**
     * Perform adaptive average pooling.
     *
     * @param input input tensor [N, C, H, W]
     * @param outputSize target output size [H_out, W_out]
     * @return output tensor [N, C, H_out, W_out]
     */
    GradTensor adaptiveAvgPool2d(GradTensor input, int[] outputSize);

    // ─────────────────────── Normalization Operations ───────────────────────

    /**
     * Perform batch normalization.
     *
     * @param input input tensor [N, C, H, W] or [N, C]
     * @param weight scale parameters [C]
     * @param bias shift parameters [C]
     * @param runningMean running mean [C]
     * @param runningVar running variance [C]
     * @param training whether in training mode
     * @param momentum momentum for running stats
     * @param eps small constant for numerical stability
     * @return normalized output tensor
     */
    GradTensor batchNorm(GradTensor input, GradTensor weight, GradTensor bias,
                         GradTensor runningMean, GradTensor runningVar,
                         boolean training, float momentum, float eps);

    /**
     * Perform layer normalization.
     *
     * @param input input tensor
     * @param normalizedShape shape of normalized dimensions
     * @param weight scale parameters
     * @param bias shift parameters
     * @param eps small constant for numerical stability
     * @return normalized output tensor
     */
    GradTensor layerNorm(GradTensor input, int[] normalizedShape,
                         GradTensor weight, GradTensor bias, float eps);

    // ─────────────────────── Image Processing Operations ───────────────────────

    /**
     * Resize image to target size.
     *
     * @param input input tensor [N, C, H, W]
     * @param height target height
     * @param width target width
     * @param mode interpolation mode ("bilinear", "nearest", "bicubic")
     * @return resized tensor [N, C, height, width]
     */
    GradTensor resize(GradTensor input, int height, int width, String mode);

    /**
     * Crop image region.
     *
     * @param input input tensor [N, C, H, W]
     * @param top top coordinate
     * @param left left coordinate
     * @param height crop height
     * @param width crop width
     * @return cropped tensor [N, C, height, width]
     */
    GradTensor crop(GradTensor input, int top, int left, int height, int width);

    /**
     * Normalize image with mean and std.
     *
     * @param input input tensor [N, C, H, W]
     * @param mean mean values per channel [C]
     * @param std standard deviation per channel [C]
     * @return normalized tensor
     */
    GradTensor normalize(GradTensor input, float[] mean, float[] std);

    // ─────────────────────── Attention Operations ───────────────────────

    /**
     * Perform scaled dot-product attention.
     *
     * @param query query tensor [N, ..., L, D]
     * @param key key tensor [N, ..., S, D]
     * @param value value tensor [N, ..., S, D_v]
     * @param mask optional attention mask
     * @param scale scaling factor (typically 1/sqrt(D))
     * @return attention output [N, ..., L, D_v]
     */
    GradTensor attention(GradTensor query, GradTensor key, GradTensor value,
                         GradTensor mask, float scale);

    /**
     * Perform multi-head attention.
     *
     * @param query query tensor [N, L, D]
     * @param key key tensor [N, S, D]
     * @param value value tensor [N, S, D]
     * @param numHeads number of attention heads
     * @param mask optional attention mask
     * @return attention output [N, L, D]
     */
    GradTensor multiHeadAttention(GradTensor query, GradTensor key, GradTensor value,
                                   int numHeads, GradTensor mask);

    // ─────────────────────── Optimization Methods ───────────────────────

    /**
     * Enable kernel fusion for optimized execution.
     *
     * @param enable true to enable fusion
     */
    void setKernelFusionEnabled(boolean enable);

    /**
     * Check if kernel fusion is enabled.
     *
     * @return true if fusion is enabled
     */
    boolean isKernelFusionEnabled();

    /**
     * Set memory allocation strategy.
     *
     * @param strategy "pool" for memory pool, "malloc" for direct allocation
     */
    void setMemoryStrategy(String strategy);

    /**
     * Get current memory allocation strategy.
     *
     * @return memory strategy name
     */
    String getMemoryStrategy();

    /**
     * Get memory usage in bytes.
     *
     * @return allocated memory in bytes
     */
    long getMemoryUsage();

    /**
     * Clear memory pools and cached resources.
     */
    void clearMemory();
}
