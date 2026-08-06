package tech.kayys.gollek.provider.litert;

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
 * This class provides a utility to resolve accelerator bitmasks.
 */
public class LiteRTDelegateManager {

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
}
