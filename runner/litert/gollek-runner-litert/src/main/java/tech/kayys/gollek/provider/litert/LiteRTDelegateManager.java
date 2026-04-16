package tech.kayys.gollek.provider.litert;

import lombok.extern.slf4j.Slf4j;

/**
 * Hardware Accelerator Manager for LiteRT 2.0.
 *
 * <p>In LiteRT 2.0, the old TfLite Delegate API is replaced by the
 * {@code LiteRtSetOptionsHardwareAccelerators} API. Hardware selection is now
 * declarative through bitmask flags on the {@code LiteRtOptions} object:
 * <ul>
 *   <li>{@code kLiteRtHwAcceleratorCpu} (1) — CPU with XNNPACK</li>
 *   <li>{@code kLiteRtHwAcceleratorGpu} (2) — GPU (Metal on macOS/iOS, OpenCL on Android)</li>
 *   <li>{@code kLiteRtHwAcceleratorNpu} (4) — NPU (Hexagon, NNAPI, etc.)</li>
 * </ul>
 *
 * <p>The environment auto-registers available accelerators at creation time.
 * This class provides a compatibility layer for the existing configuration system.
 */
@Slf4j
public class LiteRTDelegateManager {

    /**
     * Delegate types supported by LiteRT.
     * Maps to hardware accelerator flags in LiteRT 2.0.
     */
    public enum DelegateType {
        GPU_OPENCL,
        GPU_VULKAN,
        GPU_METAL,
        NPU_HEXAGON,
        NPU_NNAPI,
        NPU_ETHOS,
        CUSTOM
    }

    /**
     * GPU backend types.
     */
    public enum GpuBackend {
        OPENCL,
        VULKAN,
        METAL,
        CUDA
    }

    /**
     * NPU types.
     */
    public enum NpuType {
        HEXAGON,
        NNAPI,
        ETHOS,
        NEURON
    }

    /**
     * Resolves the LiteRT 2.0 hardware accelerator bitmask from configuration.
     *
     * @param useCpu  enable CPU (XNNPACK)
     * @param useGpu  enable GPU
     * @param useNpu  enable NPU
     * @return bitmask of accelerators
     */
    public static int resolveAccelerators(boolean useCpu, boolean useGpu, boolean useNpu) {
        int accel = 0;
        if (useCpu) accel |= LiteRTNativeBindings.kLiteRtHwAcceleratorCpu;
        if (useGpu) accel |= LiteRTNativeBindings.kLiteRtHwAcceleratorGpu;
        if (useNpu) accel |= LiteRTNativeBindings.kLiteRtHwAcceleratorNpu;
        return accel;
    }

    /**
     * @deprecated Use {@link LiteRTNativeBindings#setOptionsHardwareAccelerators} directly.
     * In LiteRT 2.0, delegates are auto-managed by the environment.
     */
    @Deprecated
    public void autoDetectAndInitializeDelegates() {
        log.info("LiteRT 2.0: Hardware accelerators are auto-registered by the environment.");
    }

    /**
     * @deprecated Use {@link #resolveAccelerators(boolean, boolean, boolean)} instead.
     */
    @Deprecated
    public boolean tryInitializeGpuDelegate(GpuBackend backend, String name) {
        log.info("LiteRT 2.0: GPU delegates are replaced by kLiteRtHwAcceleratorGpu flag.");
        return true;
    }

    /**
     * @deprecated Use {@link #resolveAccelerators(boolean, boolean, boolean)} instead.
     */
    @Deprecated
    public boolean tryInitializeNpuDelegate(NpuType type, String name) {
        log.info("LiteRT 2.0: NPU delegates are replaced by kLiteRtHwAcceleratorNpu flag.");
        return true;
    }

    /**
     * @deprecated No-op in LiteRT 2.0.
     */
    @Deprecated
    public DelegateType getBestAvailableDelegate() {
        return null;
    }

    /**
     * @deprecated No-op in LiteRT 2.0.
     */
    @Deprecated
    public void addDelegateToOptions(java.lang.foreign.MemorySegment options, DelegateType type) {
        // No-op: In LiteRT 2.0, use LiteRtSetOptionsHardwareAccelerators instead
    }

    /**
     * @deprecated No-op in LiteRT 2.0.
     */
    @Deprecated
    public void cleanup() {
        // No-op: LiteRT 2.0 environment manages accelerator lifecycle
    }
}
