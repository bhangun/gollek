/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.spi.multimodal;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class UnifiedRuntimeManifestTest {
    @Test
    void manifestNormalizesAndMatchesGemma4Unified() {
        UnifiedRuntimeManifest manifest = new UnifiedRuntimeManifest(
                " gemma4-unified-runtime ",
                " Gemma 4 Unified Runtime ",
                List.of(" gemma4 ", "gemma4"),
                List.of(" Gemma4_Unified "),
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE, UnifiedInputModality.AUDIO),
                UnifiedRuntimeReadiness.PENDING,
                "waiting for runtime plugin",
                List.of("processor_config.json", " image_processor_config.json "),
                List.of("tokenizer.json", " tokenizer.model "),
                Map.of(" checkpoint ", "google/gemma-4-12B-it"));

        assertEquals("gemma4-unified-runtime", manifest.runtimeId());
        assertEquals(List.of("gemma4"), manifest.modelFamilyIds());
        assertEquals(List.of("gemma4_unified"), manifest.modelTypes());
        assertTrue(manifest.supportsFamily("gemma4"));
        assertTrue(manifest.supportsModelType("GEMMA4_UNIFIED"));
        assertTrue(manifest.attached());
        assertFalse(manifest.productionReady());
        assertEquals("google/gemma-4-12B-it", manifest.metadata().get("checkpoint"));
    }

    @Test
    void defaultRuntimePlanFailsUntilImplemented() {
        UnifiedMultimodalRuntime runtime = () -> new UnifiedRuntimeManifest(
                "gemma4-unified-runtime",
                "Gemma 4 Unified Runtime",
                List.of("gemma4"),
                List.of("gemma4_unified"),
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE),
                UnifiedRuntimeReadiness.PENDING,
                "planning is not implemented",
                List.of(),
                List.of("tokenizer.json"),
                Map.of());
        UnifiedMultimodalRequest request = new UnifiedMultimodalRequest(
                "google/gemma-4-12B-it",
                "Gemma4_Unified",
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE),
                Map.of("prompt", "hello"),
                Map.of());

        UnsupportedOperationException error = assertThrows(
                UnsupportedOperationException.class,
                () -> runtime.plan(request));
        assertTrue(error.getMessage().contains("gemma4-unified-runtime"));
    }

    @Test
    void validatorRejectsIncompleteReadyManifest() {
        UnifiedRuntimeManifest manifest = new UnifiedRuntimeManifest(
                "",
                "",
                List.of(),
                List.of("gemma4_unified"),
                List.of(UnifiedInputModality.TEXT),
                UnifiedRuntimeReadiness.READY,
                "",
                List.of(),
                List.of("tokenizer.json"),
                Map.of());

        List<String> codes = UnifiedRuntimeManifestValidator.validate(manifest).stream()
                .map(UnifiedRuntimeManifestViolation::code)
                .toList();

        assertTrue(codes.contains("runtime_id_unknown"));
        assertTrue(codes.contains("missing_model_families"));
        assertTrue(codes.contains("production_ready_without_non_text_modality"));
    }

    @Test
    void validatorAcceptsReadyGemma4UnifiedManifest() {
        UnifiedRuntimeManifest manifest = new UnifiedRuntimeManifest(
                "gemma4-unified-runtime",
                "Gemma 4 Unified Runtime",
                List.of("gemma4"),
                List.of("gemma4_unified"),
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE, UnifiedInputModality.AUDIO),
                UnifiedRuntimeReadiness.READY,
                "ready",
                List.of("processor_config.json"),
                List.of("tokenizer.json"),
                Map.of());

        assertTrue(UnifiedRuntimeManifestValidator.validate(manifest).isEmpty());
    }

    @Test
    void registryFindsRuntimeByModelTypeAndCarriesValidationReport() {
        UnifiedMultimodalRuntime runtime = () -> new UnifiedRuntimeManifest(
                "gemma4-unified-runtime",
                "Gemma 4 Unified Runtime",
                List.of("gemma4"),
                List.of("gemma4_unified"),
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE),
                UnifiedRuntimeReadiness.READY,
                "ready",
                List.of("processor_config.json"),
                List.of("tokenizer.json"),
                Map.of());

        UnifiedRuntimeRegistry.UnifiedRuntimeReport report =
                UnifiedRuntimeRegistry.of(List.of(runtime))
                        .firstSupportingModelType("GEMMA4_UNIFIED")
                        .orElseThrow();

        assertEquals("gemma4-unified-runtime", report.runtimeId());
        assertTrue(report.manifestAvailable());
        assertTrue(report.valid());
        assertTrue(report.supportsModelType("gemma4_unified"));
    }

    @Test
    void registryReportsManifestFailuresWithoutThrowing() {
        UnifiedMultimodalRuntime runtime = () -> {
            throw new IllegalStateException("boom");
        };

        UnifiedRuntimeRegistry.UnifiedRuntimeReport report =
                UnifiedRuntimeRegistry.of(List.of(runtime)).reports().getFirst();

        assertFalse(report.manifestAvailable());
        assertFalse(report.valid());
        assertEquals("manifest_unavailable", report.violations().getFirst().code());
        assertTrue(report.diagnostics().contains("IllegalStateException"));
    }

    @Test
    void registryReportsDuplicateModelTypeClaims() {
        UnifiedMultimodalRuntime first = () -> registryManifest("gemma4-unified-a");
        UnifiedMultimodalRuntime second = () -> registryManifest("gemma4-unified-b");

        UnifiedRuntimeRegistry registry = UnifiedRuntimeRegistry.of(List.of(first, second));

        assertEquals(2, registry.reportsSupportingModelType("gemma4_unified").size());
        UnifiedRuntimeManifestViolation conflict = registry.modelTypeConflicts().getFirst();
        assertEquals("duplicate_model_type_claim", conflict.code());
        assertTrue(conflict.message().contains("gemma4_unified"));
        assertTrue(conflict.message().contains("gemma4-unified-a"));
        assertTrue(conflict.message().contains("gemma4-unified-b"));
    }

    private static UnifiedRuntimeManifest registryManifest(String runtimeId) {
        return new UnifiedRuntimeManifest(
                runtimeId,
                runtimeId,
                List.of("gemma4"),
                List.of("gemma4_unified"),
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE),
                UnifiedRuntimeReadiness.READY,
                "ready",
                List.of("processor_config.json"),
                List.of("tokenizer.json"),
                Map.of());
    }
}
