package tech.kayys.gollek.inference.libtorch;

import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.config.LibTorchProviderConfig;

import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;

/**
 * Resolves the effective advanced LibTorch execution mode after applying
 * configuration and GPU safety guards.
 */
final class LibTorchAdvancedModeResolver {

    private static final Logger log = Logger.getLogger(LibTorchAdvancedModeResolver.class);
    private static final Set<String> SUPPORTED_ATTENTION_MODES = Set.of("baseline", "hybrid_fp8_bf16");
    private static final Set<Integer> SAGE_ATTENTION2_EXPERIMENTAL_SM = Set.of(90);

    EffectiveAdvancedMode resolve(LibTorchProviderConfig config, LibTorchBinding binding) {
        if (config.advanced() == null || !config.advanced().enabled()) {
            return EffectiveAdvancedMode.baseline("advanced.disabled", parseAllowedSm(config));
        }
        if (!config.gpu().enabled()) {
            return EffectiveAdvancedMode.baseline("gpu.disabled", parseAllowedSm(config));
        }

        String attentionMode = normalizeAttentionMode(config.advanced().attentionMode());
        if (!SUPPORTED_ATTENTION_MODES.contains(attentionMode)) {
            log.warnf("Unknown advanced attention mode '%s'; forcing baseline", config.advanced().attentionMode());
            return EffectiveAdvancedMode.baseline("attention.invalid", parseAllowedSm(config));
        }

        boolean fp8RowwiseRequested = config.advanced().fp8RowwiseEnabled();
        boolean sageAttentionRequested = config.advanced().sageAttention2Enabled();
        boolean advancedRequested = !"baseline".equals(attentionMode) || fp8RowwiseRequested || sageAttentionRequested;
        if (!advancedRequested) {
            return EffectiveAdvancedMode.baseline("advanced.noop", parseAllowedSm(config));
        }

        Set<Integer> allowedSm = parseAllowedSm(config);
        Optional<Integer> detectedSm = detectGpuSm(binding, config.gpu().deviceIndex());
        if (detectedSm.isEmpty()) {
            return EffectiveAdvancedMode.baseline("gpu.sm.unknown", allowedSm);
        }
        if (!allowedSm.isEmpty() && !allowedSm.contains(detectedSm.get())) {
            return EffectiveAdvancedMode.baseline("gpu.sm.not-allowed", allowedSm, detectedSm);
        }

        // M4 scaffold: keep SageAttention2 opt-in visible in config, but force
        // rollback until kernels are implemented. Preserve any other advanced mode.
        boolean sageAttentionEffective = false;
        String sageAttentionRollbackReason = "none";
        if (sageAttentionRequested) {
            if (!SAGE_ATTENTION2_EXPERIMENTAL_SM.contains(detectedSm.get())) {
                log.warnf("SageAttention2 requested but SM %s is outside experimental allow-list %s; rolling back",
                        detectedSm.get(), SAGE_ATTENTION2_EXPERIMENTAL_SM);
                sageAttentionRollbackReason = "sageattention2.sm.not-allowed";
            } else {
                log.warn("SageAttention2 requested on allowed SM but kernels are not implemented yet; rolling back");
                sageAttentionRollbackReason = "sageattention2.not-implemented";
            }
        }

        boolean advancedEffective = !"baseline".equals(attentionMode) || fp8RowwiseRequested;
        if (!advancedEffective && sageAttentionRequested) {
            return EffectiveAdvancedMode.baselineWithSageRollback(
                    "sageattention2.rollback",
                    sageAttentionRollbackReason,
                    allowedSm,
                    detectedSm);
        }

        return new EffectiveAdvancedMode(
                true,
                attentionMode,
                fp8RowwiseRequested,
                sageAttentionRequested,
                sageAttentionEffective,
                sageAttentionRollbackReason,
                "advanced.enabled",
                detectedSm,
                allowedSm);
    }

    static Set<Integer> parseAllowedSm(LibTorchProviderConfig config) {
        Set<Integer> values = new LinkedHashSet<>();
        if (config.advanced() == null || config.advanced().allowedGpuSm() == null) {
            return values;
        }
        for (String token : config.advanced().allowedGpuSm().split(",")) {
            String trimmed = token.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            try {
                values.add(Integer.parseInt(trimmed));
            } catch (NumberFormatException ignored) {
                log.warnf("Ignoring invalid GPU SM token in libtorch.provider.advanced.allowed-gpu-sm: '%s'", trimmed);
            }
        }
        return values;
    }

    private String normalizeAttentionMode(String mode) {
        if (mode == null || mode.isBlank()) {
            return "baseline";
        }
        return mode.trim().toLowerCase(Locale.ROOT);
    }

    private Optional<Integer> detectGpuSm(LibTorchBinding binding, int deviceIndex) {
        Optional<Integer> override = parseOverrideSm();
        if (override.isPresent()) {
            return override;
        }
        if (binding == null || !binding.hasSymbol(LibTorchBinding.CUDA_DEVICE_SM)) {
            return Optional.empty();
        }
        try {
            int sm = (int) binding.bind(LibTorchBinding.CUDA_DEVICE_SM, LibTorchBinding.CUDA_DEVICE_SM_DESC)
                    .invoke(deviceIndex);
            return sm > 0 ? Optional.of(sm) : Optional.empty();
        } catch (Throwable t) {
            log.debugf(t, "Unable to resolve CUDA SM via native binding; forcing baseline fallback");
            return Optional.empty();
        }
    }

    private Optional<Integer> parseOverrideSm() {
        String raw = System.getProperty("gollek.libtorch.gpu.sm");
        if (raw == null || raw.isBlank()) {
            raw = System.getenv("GOLLEK_LIBTORCH_GPU_SM");
        }
        if (raw == null || raw.isBlank()) {
            return Optional.empty();
        }
        try {
            int parsed = Integer.parseInt(raw.trim());
            if (parsed > 0) {
                return Optional.of(parsed);
            }
        } catch (NumberFormatException ignored) {
            log.warnf("Ignoring invalid GPU SM override value: '%s'", raw);
        }
        return Optional.empty();
    }

    record EffectiveAdvancedMode(
            boolean advancedEnabled,
            String attentionMode,
            boolean fp8RowwiseEnabled,
            boolean sageAttention2Requested,
            boolean sageAttention2Enabled,
            String sageAttention2RollbackReason,
            String reason,
            Optional<Integer> detectedGpuSm,
            Set<Integer> allowedGpuSm) {

        static EffectiveAdvancedMode baseline(String reason, Set<Integer> allowedSm) {
            return baseline(reason, allowedSm, Optional.empty());
        }

        static EffectiveAdvancedMode baseline(String reason, Set<Integer> allowedSm, Optional<Integer> detectedSm) {
            return new EffectiveAdvancedMode(false, "baseline", false, false, false, "none", reason, detectedSm,
                    allowedSm);
        }

        static EffectiveAdvancedMode baselineWithSageRollback(
                String reason,
                String sageRollbackReason,
                Set<Integer> allowedSm,
                Optional<Integer> detectedSm) {
            return new EffectiveAdvancedMode(
                    false,
                    "baseline",
                    false,
                    true,
                    false,
                    sageRollbackReason == null || sageRollbackReason.isBlank() ? "unknown" : sageRollbackReason,
                    reason,
                    detectedSm,
                    allowedSm);
        }
    }
}
