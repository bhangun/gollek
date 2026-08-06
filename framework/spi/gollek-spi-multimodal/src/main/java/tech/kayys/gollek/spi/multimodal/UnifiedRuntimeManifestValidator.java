/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.spi.multimodal;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Shared contract checks for detachable unified multimodal runtime manifests.
 */
public final class UnifiedRuntimeManifestValidator {
    private static final String UNKNOWN_RUNTIME_ID = "unknown-unified-runtime";
    private static final Pattern RUNTIME_ID_PATTERN = Pattern.compile("[a-z][a-z0-9._-]*");
    private static final Pattern MODEL_FAMILY_ID_PATTERN = Pattern.compile("[a-z][a-z0-9_]*");
    private static final Pattern MODEL_TYPE_PATTERN = Pattern.compile("[a-z][a-z0-9_.-]*");

    private UnifiedRuntimeManifestValidator() {
    }

    public static List<UnifiedRuntimeManifestViolation> validate(UnifiedRuntimeManifest manifest) {
        if (manifest == null) {
            return List.of(new UnifiedRuntimeManifestViolation(
                    UNKNOWN_RUNTIME_ID,
                    "manifest_null",
                    "unified runtime manifest must not be null"));
        }

        List<UnifiedRuntimeManifestViolation> violations = new ArrayList<>();
        validateRuntimeId(manifest, violations);
        validateFamilies(manifest, violations);
        validateModelTypes(manifest, violations);
        validateModalities(manifest, violations);
        return List.copyOf(violations);
    }

    public static boolean valid(UnifiedRuntimeManifest manifest) {
        return validate(manifest).isEmpty();
    }

    private static void validateRuntimeId(
            UnifiedRuntimeManifest manifest,
            List<UnifiedRuntimeManifestViolation> violations) {
        String runtimeId = manifest.runtimeId();
        if (UNKNOWN_RUNTIME_ID.equals(runtimeId)) {
            violations.add(new UnifiedRuntimeManifestViolation(
                    runtimeId,
                    "runtime_id_unknown",
                    "runtime id must be explicitly set"));
        } else if (!RUNTIME_ID_PATTERN.matcher(runtimeId).matches()) {
            violations.add(new UnifiedRuntimeManifestViolation(
                    runtimeId,
                    "runtime_id_invalid",
                    "runtime id must match " + RUNTIME_ID_PATTERN.pattern()));
        }
    }

    private static void validateFamilies(
            UnifiedRuntimeManifest manifest,
            List<UnifiedRuntimeManifestViolation> violations) {
        if (manifest.modelFamilyIds().isEmpty()) {
            violations.add(new UnifiedRuntimeManifestViolation(
                    manifest.runtimeId(),
                    "missing_model_families",
                    "runtime manifest must claim at least one model family id"));
            return;
        }
        for (String familyId : manifest.modelFamilyIds()) {
            if (!MODEL_FAMILY_ID_PATTERN.matcher(familyId).matches()) {
                violations.add(new UnifiedRuntimeManifestViolation(
                        manifest.runtimeId(),
                        "invalid_model_family",
                        "model family id '" + familyId + "' must match " + MODEL_FAMILY_ID_PATTERN.pattern()));
            }
        }
    }

    private static void validateModelTypes(
            UnifiedRuntimeManifest manifest,
            List<UnifiedRuntimeManifestViolation> violations) {
        if (manifest.modelTypes().isEmpty()) {
            violations.add(new UnifiedRuntimeManifestViolation(
                    manifest.runtimeId(),
                    "missing_model_types",
                    "runtime manifest must claim at least one model_type"));
            return;
        }
        for (String modelType : manifest.modelTypes()) {
            if (!MODEL_TYPE_PATTERN.matcher(modelType).matches()) {
                violations.add(new UnifiedRuntimeManifestViolation(
                        manifest.runtimeId(),
                        "invalid_model_type",
                        "model_type '" + modelType + "' must match " + MODEL_TYPE_PATTERN.pattern()));
            }
        }
    }

    private static void validateModalities(
            UnifiedRuntimeManifest manifest,
            List<UnifiedRuntimeManifestViolation> violations) {
        if (manifest.inputModalities().isEmpty()) {
            violations.add(new UnifiedRuntimeManifestViolation(
                    manifest.runtimeId(),
                    "missing_input_modalities",
                    "runtime manifest must advertise at least one input modality"));
        }
        if (!manifest.productionReady()) {
            return;
        }
        boolean hasText = manifest.inputModalities().contains(UnifiedInputModality.TEXT);
        boolean hasNonText = manifest.inputModalities().stream()
                .anyMatch(modality -> modality != UnifiedInputModality.TEXT);
        if (!hasText) {
            violations.add(new UnifiedRuntimeManifestViolation(
                    manifest.runtimeId(),
                    "production_ready_without_text",
                    "READY unified runtimes must advertise TEXT input support"));
        }
        if (!hasNonText) {
            violations.add(new UnifiedRuntimeManifestViolation(
                    manifest.runtimeId(),
                    "production_ready_without_non_text_modality",
                    "READY unified runtimes must advertise at least one non-text modality"));
        }
    }
}
