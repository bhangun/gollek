package tech.kayys.gollek.blackwell.config;

/**
 * Blackwell runner selection mode.
 *
 * <ul>
 * <li>AUTO: Detect Blackwell availability and pick runner based on model size.</li>
 * <li>STANDARD: Use BlackwellRunner only.</li>
 * <li>OFFLOAD: Use BlackwellWeightOffloadingRunner only.</li>
 * <li>FORCE: Allow Blackwell runners even when detection fails.</li>
 * <li>DISABLED: Remove Blackwell runners from selection.</li>
 * </ul>
 */
public enum BlackwellRunnerMode {
    AUTO,
    STANDARD,
    OFFLOAD,
    FORCE,
    DISABLED;

    public static BlackwellRunnerMode from(String raw) {
        if (raw == null || raw.isBlank())
            return AUTO;
        return switch (raw.trim().toLowerCase()) {
            case "auto" -> AUTO;
            case "standard" -> STANDARD;
            case "offload", "weight-offload", "blackwell-offload" -> OFFLOAD;
            case "force", "manual", "forced" -> FORCE;
            case "disabled", "off", "false" -> DISABLED;
            default -> AUTO;
        };
    }
}
