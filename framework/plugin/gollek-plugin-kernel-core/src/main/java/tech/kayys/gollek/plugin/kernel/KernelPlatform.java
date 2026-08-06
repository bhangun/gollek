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

package tech.kayys.gollek.plugin.kernel;

/**
 * Kernel platform enumeration.
 *
 * @since 2.0.0
 */
public enum KernelPlatform {
    /**
     * Apple Metal (Apple Silicon M1/M2/M3/M4)
     */
    METAL("Metal", "Apple Silicon GPU acceleration"),

    /**
     * NVIDIA CUDA (NVIDIA GPUs)
     */
    CUDA("CUDA", "NVIDIA GPU acceleration"),

    /**
     * AMD ROCm (AMD GPUs)
     */
    ROCM("ROCm", "AMD GPU acceleration"),

    /**
     * Microsoft DirectML (Windows DirectX)
     */
    DIRECTML("DirectML", "Windows DirectX acceleration"),

    /**
     * CPU (Fallback)
     */
    CPU("CPU", "CPU-only execution");

    private final String displayName;
    private final String description;

    KernelPlatform(String displayName, String description) {
        this.displayName = displayName;
        this.description = description;
    }

    /**
     * Get display name.
     */
    public String getDisplayName() {
        return displayName;
    }

    /**
     * Get description.
     */
    public String getDescription() {
        return description;
    }

    /**
     * Check if platform is GPU-based.
     */
    public boolean isGpu() {
        return this != CPU;
    }

    /**
     * Check if platform is CPU-based.
     */
    public boolean isCpu() {
        return this == CPU;
    }
}
