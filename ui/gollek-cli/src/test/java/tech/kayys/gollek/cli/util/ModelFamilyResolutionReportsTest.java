package tech.kayys.gollek.cli.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyResolution;
import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelFamilyResolutionReportsTest {
    @TempDir
    Path tempDir;

    @Test
    @SuppressWarnings("unchecked")
    void reportIncludesDirectArchitectureAndTokenizerDiagnostics() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "report-direct-family",
                        "Report Direct Family",
                        List.of("report_direct"),
                        List.of("ReportDirectForCausalLM"),
                        List.of(
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE,
                                ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("report-direct-bpe"));
            }
        };
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(plugin);
        ModelFamilyResolution resolution = new ModelFamilyResolution(
                ModelFamilyResolution.Status.RESOLVED,
                "report_direct",
                "ReportDirectForCausalLM",
                List.of("report-direct-family"),
                List.of(plugin.supportReport()),
                List.of(plugin.runtimeManifest()),
                plugin.tokenizerDescriptors(),
                List.of(),
                List.of(),
                "");

        Map<String, Object> report = ModelFamilyResolutionReports.report(
                resolution,
                Optional.of(tempDir),
                registry);
        Map<String, Object> schema = ModelFamilyResolutionReportContract.schema();
        Map<String, Object> schemaValidation =
                ModelFamilyResolutionReportContract.schemaValidationReport(schema);
        Map<String, Object> validation = ModelFamilyResolutionReportContract.validationReport(report);

        assertEquals(List.of(), ModelFamilyResolutionReportContract.validateSchema(schema));
        assertEquals(List.of(), ModelFamilyResolutionReportContract.validateReport(report));
        assertEquals(ModelFamilyResolutionReportFields.CONTRACT_ID,
                schema.get(ModelFamilyResolutionReportFields.Schema.CONTRACT_ID));
        assertEquals(ModelFamilyResolutionReportFields.schemaFingerprint(),
                schema.get(ModelFamilyResolutionReportFields.Schema.SCHEMA_FINGERPRINT));
        assertEquals(Boolean.TRUE, schemaValidation.get(ModelFamilyResolutionReportFields.Validation.PASSED));
        assertEquals(0, schemaValidation.get(ModelFamilyResolutionReportFields.Validation.PROBLEM_COUNT));
        assertEquals(Boolean.TRUE, validation.get(ModelFamilyResolutionReportFields.Validation.PASSED));
        assertEquals(0, validation.get(ModelFamilyResolutionReportFields.Validation.PROBLEM_COUNT));
        assertEquals(ModelFamilyResolutionReportFields.resolutionFields(), List.copyOf(report.keySet()));
        assertEquals("RESOLVED", report.get(ModelFamilyResolutionReportFields.Resolution.STATUS));
        assertEquals(Boolean.TRUE, report.get(ModelFamilyResolutionReportFields.Resolution.REQUIRES_ATTENTION));
        assertTrue(((List<String>) report.get(ModelFamilyResolutionReportFields.Resolution.PROBLEM_CODES))
                .contains("model_family_architecture_adapters_missing"));
        assertTrue(((List<String>) report.get(ModelFamilyResolutionReportFields.Resolution.PROBLEM_CODES))
                .contains("model_family_tokenizer_files_missing"));

        List<Map<String, Object>> supportReports =
                (List<Map<String, Object>>) report.get(ModelFamilyResolutionReportFields.Resolution.SUPPORT_REPORTS);
        assertEquals(ModelFamilyResolutionReportFields.supportReportFields(),
                List.copyOf(supportReports.getFirst().keySet()));

        List<Map<String, Object>> runtimeManifests =
                (List<Map<String, Object>>) report.get(ModelFamilyResolutionReportFields.Resolution.RUNTIME_MANIFESTS);
        assertEquals(ModelFamilyResolutionReportFields.runtimeManifestFields(),
                List.copyOf(runtimeManifests.getFirst().keySet()));

        Map<String, Object> directArchitecture =
                (Map<String, Object>) report.get(ModelFamilyResolutionReportFields.Resolution.DIRECT_ARCHITECTURE);
        assertEquals(ModelFamilyResolutionReportFields.directArchitectureFields(),
                List.copyOf(directArchitecture.keySet()));
        assertEquals(Boolean.TRUE,
                directArchitecture.get(ModelFamilyResolutionReportFields.DirectArchitecture.DIRECT_SUPPORT_EXPECTED));
        assertTrue(((List<String>) directArchitecture.get(
                ModelFamilyResolutionReportFields.DirectArchitecture.PROBLEM_CODES))
                .contains("model_family_architecture_adapters_missing"));

        Map<String, Object> runtimeCompatibility =
                (Map<String, Object>) report.get(ModelFamilyResolutionReportFields.Resolution.RUNTIME_COMPATIBILITY);
        assertEquals(ModelFamilyResolutionReportFields.runtimeCompatibilityFields(),
                List.copyOf(runtimeCompatibility.keySet()));
        Map<String, Object> directSafetensor =
                (Map<String, Object>) runtimeCompatibility.get(
                        ModelFamilyResolutionReportFields.RuntimeCompatibility.DIRECT_SAFETENSOR);
        assertEquals(ModelFamilyResolutionReportFields.directSafetensorCompatibilityFields(),
                List.copyOf(directSafetensor.keySet()));
        assertEquals("direct_safetensor",
                directSafetensor.get(ModelFamilyResolutionReportFields.DirectSafetensorCompatibility.RUNTIME_ID));
        assertFalse((Boolean) directSafetensor.get(
                ModelFamilyResolutionReportFields.DirectSafetensorCompatibility.COMPATIBLE));
        assertTrue(((List<String>) directSafetensor.get(
                ModelFamilyResolutionReportFields.DirectSafetensorCompatibility.PROBLEM_CODES))
                .contains("model_family_architecture_adapters_missing"));
        assertTrue(((List<String>) directSafetensor.get(
                ModelFamilyResolutionReportFields.DirectSafetensorCompatibility.PROBLEM_CODES))
                .contains("model_family_tokenizer_files_missing"));

        List<Map<String, Object>> tokenizers =
                (List<Map<String, Object>>) report.get(ModelFamilyResolutionReportFields.Resolution.TOKENIZERS);
        assertEquals(ModelFamilyResolutionReportFields.tokenizerFields(), List.copyOf(tokenizers.getFirst().keySet()));
        assertEquals("report-direct-bpe", tokenizers.getFirst().get(ModelFamilyResolutionReportFields.Tokenizer.ID));
        assertEquals(Boolean.TRUE,
                tokenizers.getFirst().get(ModelFamilyResolutionReportFields.Tokenizer.FILE_STATUS_AVAILABLE));
        assertEquals(Boolean.FALSE, tokenizers.getFirst().get(ModelFamilyResolutionReportFields.Tokenizer.USABLE));
        assertTrue(tokenizers.getFirst()
                .get(ModelFamilyResolutionReportFields.Tokenizer.MISSING_FILE_GROUPS)
                .toString()
                .contains("tokenizer.json"));

        Map<String, Object> driftedReport = new LinkedHashMap<>(report);
        driftedReport.remove(ModelFamilyResolutionReportFields.Resolution.TOKENIZERS);
        Map<String, Object> driftedValidation = ModelFamilyResolutionReportContract.validationReport(driftedReport);
        assertEquals(Boolean.FALSE,
                driftedValidation.get(ModelFamilyResolutionReportFields.Validation.PASSED));
        assertTrue(ModelFamilyResolutionReportContract.validateReport(driftedReport).stream()
                .anyMatch(problem -> problem.contains("resolution fields expected")));

        Map<String, Object> driftedSchema = new LinkedHashMap<>(schema);
        driftedSchema.put(ModelFamilyResolutionReportFields.Schema.SCHEMA_FINGERPRINT, "sha256:drifted");
        Map<String, Object> driftedSchemaValidation =
                ModelFamilyResolutionReportContract.schemaValidationReport(driftedSchema);
        assertEquals(Boolean.FALSE,
                driftedSchemaValidation.get(ModelFamilyResolutionReportFields.Validation.PASSED));
        assertEquals(1, driftedSchemaValidation.get(ModelFamilyResolutionReportFields.Validation.PROBLEM_COUNT));
        assertTrue(ModelFamilyResolutionReportContract.validateSchema(driftedSchema).stream()
                .anyMatch(problem -> problem.contains("schema.schemaFingerprint expected sha256:")));
    }
}
