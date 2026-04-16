package tech.kayys.gollek.metal.config;

/**
 * Metal runner selection mode.
 *
 * <ul>
 * <li>AUTO: Detect Metal availability and pick runner based on model size.</li>
 * <li>STANDARD: Use MetalRunner only.</li>
 * <li>OFFLOAD: Use MetalWeightOffloadingRunner only.</li>
 * <li>FORCE: Allow Metal runners even when detection fails.</li>
 * <li>DISABLED: Remove Metal runners from selection.</li>
 * </ul>
 */
public enum MetalRunnerMode {
    AUTO,
    STANDARD,
    OFFLOAD,
    FORCE,
    DISABLED;

    public static MetalRunnerMode from(String raw) {
        if (raw == null || raw.isBlank())
            return AUTO;
        return switch (raw.trim().toLowerCase()) {
            case "auto" -> AUTO;
            case "standard" -> STANDARD;
            case "offload", "weight-offload", "metal-offload" -> OFFLOAD;
            case "force", "manual", "forced" -> FORCE;
            case "disabled", "off", "false" -> DISABLED;
            default -> AUTO;
        };
    }
}
