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

package tech.kayys.gollek.plugin.optimization;

import java.util.*;

/**
 * Optimization context for applying optimizations.
 *
 * @since 2.0.0
 */
public interface OptimizationContext {

    /**
     * Get model parameter.
     *
     * @param key parameter key
     * @param type parameter type
     * @param <T> parameter type
     * @return parameter value
     */
    <T> T getParameter(String key, Class<T> type);

    /**
     * Get model parameter with default.
     *
     * @param key parameter key
     * @param type parameter type
     * @param defaultValue default value
     * @param <T> parameter type
     * @return parameter value or default
     */
    default <T> T getParameter(String key, Class<T> type, T defaultValue) {
        T value = getParameter(key, type);
        return value != null ? value : defaultValue;
    }

    /**
     * Get GPU device ID.
     *
     * @return device ID
     */
    int getDeviceId();

    /**
     * Check if running on GPU.
     *
     * @return true if GPU
     */
    boolean isGpu();

    /**
     * Get GPU architecture.
     *
     * @return architecture (e.g., "hopper", "blackwell", "ampere")
     */
    String getGpuArch();

    /**
     * Get compute capability (NVIDIA).
     *
     * @return compute capability (e.g., 90 for H100)
     */
    default int getComputeCapability() {
        return 0;
    }

    /**
     * Get operation ID.
     *
     * @return operation ID
     */
    String getOperationId();

    /**
     * Get optimization config.
     *
     * @return config
     */
    OptimizationConfig getConfig();

    /**
     * Get runner ID.
     *
     * @return runner ID
     */
    String getRunnerId();

    /**
     * Get model path.
     *
     * @return model path
     */
    String getModelPath();
}
