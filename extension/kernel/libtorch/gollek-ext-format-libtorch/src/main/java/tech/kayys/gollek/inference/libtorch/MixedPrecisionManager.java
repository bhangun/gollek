package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.lang.invoke.MethodHandle;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Mixed precision inference manager for LibTorch.
 * <p>
 * Provides utilities to run inference in FP16 or BF16 mode,
 * reducing memory usage and increasing throughput on supported
 * hardware (GPU with Tensor Cores, or Apple Silicon).
 * <p>
 * Falls back gracefully to FP32 if the target precision is
 * not available on the current device.
 */
@ApplicationScoped
public class MixedPrecisionManager {

    private static final Logger log = Logger.getLogger(MixedPrecisionManager.class);

    /**
     * Supported precision modes.
     */
    public enum Precision {
        FP32("float32"),
        FP16("float16"),
        BF16("bfloat16");

        private final String torchDtype;

        Precision(String torchDtype) {
            this.torchDtype = torchDtype;
        }

        public String getTorchDtype() {
            return torchDtype;
        }

        public static Precision fromString(String s) {
            if (s == null)
                return FP32;
            return switch (s.toLowerCase().trim()) {
                case "fp16", "float16", "half" -> FP16;
                case "bf16", "bfloat16" -> BF16;
                default -> FP32;
            };
        }
    }

    @Inject
    LibTorchProviderConfig config;

    private volatile Precision activePrecision = Precision.FP32;
    private final AtomicBoolean initialized = new AtomicBoolean(false);

    /**
     * Determine the best precision for the current hardware.
     *
     * @return the recommended precision mode
     */
    public Precision detectBestPrecision() {
        if (!config.gpu().enabled()) {
            // CPU — FP32 is the safest option (BF16 can work on modern CPUs)
            log.info("CPU mode: using FP32 precision");
            return Precision.FP32;
        }

        // GPU — try BF16 first (better range than FP16), fallback to FP16
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            if (binding.hasSymbol(LibTorchBinding.CUDA_IS_BF16_SUPPORTED)) {
                MethodHandle bf16Check = binding.bind(
                        LibTorchBinding.CUDA_IS_BF16_SUPPORTED,
                        LibTorchBinding.CUDA_IS_BF16_SUPPORTED_DESC);
                boolean supported = (boolean) bf16Check.invoke();
                if (supported) {
                    log.info("GPU supports BF16 — using BF16 precision");
                    return Precision.BF16;
                }
            }
        } catch (Throwable e) {
            log.debug("BF16 check failed, trying FP16");
        }

        log.info("Using FP16 precision for GPU inference");
        return Precision.FP16;
    }

    /**
     * Initialize mixed precision based on configuration or auto-detection.
     */
    public void initialize() {
        String configured = System.getProperty("libtorch.provider.precision", "auto");

        if ("auto".equalsIgnoreCase(configured)) {
            this.activePrecision = detectBestPrecision();
        } else {
            this.activePrecision = Precision.fromString(configured);
        }

        log.infof("Mixed precision initialized: %s (dtype=%s)",
                activePrecision.name(), activePrecision.getTorchDtype());
        initialized.set(true);
    }

    public void initializeIfNeeded() {
        if (!initialized.get()) {
            synchronized (this) {
                if (!initialized.get()) {
                    initialize();
                }
            }
        }
    }

    /**
     * Cast input for hybrid attention path (prefers BF16 on supported GPUs,
     * otherwise FP16).
     */
    public Tensor castForHybridInput(Tensor input) {
        initializeIfNeeded();
        return castToActivePrecision(input);
    }

    /**
     * Cast a tensor to the active precision.
     * Used before feeding into the model when mixed precision is enabled.
     *
     * @param input tensor in FP32
     * @return tensor cast to the active precision (or same tensor if FP32)
     */
    public Tensor castToActivePrecision(Tensor input) {
        if (activePrecision == Precision.FP32) {
            return input; // No-op for FP32
        }

        try {
            int dtypeCode = activePrecision == Precision.FP16
                    ? ScalarType.HALF.code()
                    : ScalarType.BFLOAT16.code();
            return castTensorToDtype(input, dtypeCode);
        } catch (Throwable t) {
            log.warnf(t, "Failed to cast tensor to %s, using FP32 fallback", activePrecision);
        }

        return input; // Fallback — no cast
    }

    /**
     * Cast output tensor back to FP32 for response processing.
     *
     * @param output model output tensor (possibly FP16/BF16)
     * @return tensor in FP32
     */
    public Tensor castToFP32(Tensor output) {
        if (activePrecision == Precision.FP32) {
            return output;
        }

        try {
            return castTensorToDtype(output, ScalarType.FLOAT.code());
        } catch (Throwable t) {
            log.warnf(t, "Failed to cast tensor to FP32");
        }

        return output;
    }

    /**
     * Get the active precision mode.
     */
    public Precision getActivePrecision() {
        return activePrecision;
    }

    /**
     * Check if mixed precision is active (anything other than FP32).
     */
    public boolean isMixedPrecisionActive() {
        return activePrecision != Precision.FP32;
    }

    private Tensor castTensorToDtype(Tensor tensor, int dtypeCode) throws Throwable {
        LibTorchBinding binding = LibTorchBinding.getInstance();
        if (!binding.hasSymbol(LibTorchBinding.TENSOR_TO_DTYPE)) {
            return tensor;
        }
        MethodHandle castFn = binding.bind(
                LibTorchBinding.TENSOR_TO_DTYPE,
                LibTorchBinding.TENSOR_TO_DTYPE_DESC);
        var result = (java.lang.foreign.MemorySegment) castFn.invoke(
                tensor.nativeHandle(), dtypeCode);
        return new Tensor(result, java.lang.foreign.Arena.ofAuto());
    }
}
