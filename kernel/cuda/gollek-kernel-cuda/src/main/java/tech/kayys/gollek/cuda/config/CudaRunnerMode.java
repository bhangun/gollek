package tech.kayys.gollek.cuda.config;

/**
 * CUDA runner selection mode.
 *
 * <ul>
 * <li>AUTO: Detect CUDA availability and pick runner based on model size.</li>
 * <li>STANDARD: Use CudaRunner only.</li>
 * <li>OFFLOAD: Use CudaWeightOffloadingRunner only.</li>
 * <li>FORCE: Allow CUDA runners even when detection fails.</li>
 * <li>DISABLED: Remove CUDA runners from selection.</li>
 * </ul>
 */
public enum CudaRunnerMode {
    AUTO,
    STANDARD,
    OFFLOAD,
    FORCE,
    DISABLED;

    public static CudaRunnerMode from(String raw) {
        if (raw == null || raw.isBlank())
            return AUTO;
        return switch (raw.trim().toLowerCase()) {
            case "auto" -> AUTO;
            case "standard" -> STANDARD;
            case "offload", "weight-offload", "cuda-offload" -> OFFLOAD;
            case "force", "manual", "forced" -> FORCE;
            case "disabled", "off", "false" -> DISABLED;
            default -> AUTO;
        };
    }
}
