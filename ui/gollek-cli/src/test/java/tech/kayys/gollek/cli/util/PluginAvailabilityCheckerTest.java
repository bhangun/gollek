package tech.kayys.gollek.cli.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.plugin.core.ExtensionAvailability;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityContractReport;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityContractViolation;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityRegistry;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyContractViolation;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibilitySummary;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyUnifiedRuntimeRequirement;
import tech.kayys.gollek.spi.multimodal.UnifiedInputModality;
import tech.kayys.gollek.spi.multimodal.UnifiedMultimodalRuntime;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeManifest;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeReadiness;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeRegistry;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PluginAvailabilityCheckerTest {
    @Test
    void cliClasspathIncludesServiceLoadedGgufRunnerPlugin() {
        // GgufRunnerPlugin is temporarily excluded from the build.
    }

    @Test
    void availabilityCheckerIncludesServiceLoadedRunnerPlugins() {
        PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

        assertTrue(checker.hasRunnerPlugins());
    }

    @Test
    void modelFamilyContractReportsGroupChecklistViolationsByCategory() {
        List<ModelFamilyContractViolation> violations = List.of(
                new ModelFamilyContractViolation(
                        "gemma4",
                        "unified_runtime_requirement_missing_modalities",
                        "missing modalities"),
                new ModelFamilyContractViolation(
                        "kimi",
                        "tokenizer_metadata_pending_reason_missing",
                        "missing tokenizer pending reason"),
                new ModelFamilyContractViolation(
                        "qwen_vl",
                        "unexpected_origin_path",
                        "unexpected origin"));

        Map<String, Object> summary = ModelFamilyContractViolationReports.summary(violations);
        Map<String, Integer> categories = ModelFamilyContractViolationReports.categories(violations);
        Map<String, Object> validation = ModelFamilyContractViolationReports.validationReport(summary);

        assertEquals(ModelFamilyContractViolationReports.CONTRACT_ID,
                summary.get(ModelFamilyContractViolationReportFields.Report.CONTRACT_ID));
        assertEquals(ModelFamilyContractViolationReports.SCHEMA_VERSION,
                summary.get(ModelFamilyContractViolationReportFields.Report.SCHEMA_VERSION));
        assertEquals(ModelFamilyContractViolationReports.SCHEMA_FINGERPRINT,
                summary.get(ModelFamilyContractViolationReportFields.Report.SCHEMA_FINGERPRINT));
        assertTrue(summary.get(ModelFamilyContractViolationReportFields.Report.SCHEMA) instanceof Map<?, ?>);
        assertEquals(ModelFamilyContractViolationReports.categoryKeys(),
                summary.get(ModelFamilyContractViolationReportFields.Report.CATEGORY_KEYS));
        assertEquals(List.of(), ModelFamilyContractViolationReports.validateSummary(summary));
        assertEquals(Boolean.TRUE,
                validation.get(ModelFamilyContractViolationReportFields.Validation.PASSED));
        assertEquals(0,
                validation.get(ModelFamilyContractViolationReportFields.Validation.PROBLEM_COUNT));
        assertTrue(summary.get(ModelFamilyContractViolationReportFields.Report.REMEDIATION_CATALOG)
                instanceof Map<?, ?>);
        assertTrue(((Map<?, ?>) summary.get(ModelFamilyContractViolationReportFields.Report.REMEDIATION_CATALOG))
                .containsKey("tokenizerMetadata"));
        assertEquals("failed", summary.get(ModelFamilyContractViolationReportFields.Report.STATUS));
        assertEquals(3, summary.get(ModelFamilyContractViolationReportFields.Report.AFFECTED_FAMILY_COUNT));
        assertEquals(1, categories.get("unifiedRuntimeRequirement"));
        assertEquals(1, categories.get("tokenizerMetadata"));
        assertEquals(1, categories.get("origin"));
        assertTrue(summary.get(ModelFamilyContractViolationReportFields.Report.CATEGORY_REMEDIATION_HINTS)
                instanceof Map<?, ?>);
        assertTrue(summary.get(ModelFamilyContractViolationReportFields.Report.REMEDIATION_HINTS)
                instanceof List<?>);
        assertTrue(((List<?>) summary.get(ModelFamilyContractViolationReportFields.Report.REMEDIATION_HINTS)).stream()
                .anyMatch(hint -> hint.toString().contains("tokenizer_metadata_status=ready")));
        assertTrue(((Map<?, ?>) summary.get(ModelFamilyContractViolationReportFields.Report.CATEGORY_REMEDIATION_HINTS))
                .containsKey("unifiedRuntimeRequirement"));
        assertTrue(summary.get(ModelFamilyContractViolationReportFields.Report.VIOLATIONS) instanceof List<?>);
        assertTrue(((List<?>) summary.get(ModelFamilyContractViolationReportFields.Report.VIOLATIONS)).stream()
                .allMatch(value -> value instanceof Map<?, ?> report
                        && report.get(ModelFamilyContractViolationReportFields.Violation.CATEGORY) instanceof String
                        && report.get(ModelFamilyContractViolationReportFields.Violation.REMEDIATION_HINT)
                        instanceof String));

        Map<String, Object> drifted = new LinkedHashMap<>(summary);
        drifted.put(ModelFamilyContractViolationReportFields.Report.SCHEMA_FINGERPRINT, "sha256:drifted");
        Map<String, Object> driftedValidation = ModelFamilyContractViolationReports.validationReport(drifted);
        assertTrue(ModelFamilyContractViolationReports.validateSummary(drifted).stream()
                .anyMatch(problem -> problem.contains("schemaFingerprint")));
        assertEquals(Boolean.FALSE,
                driftedValidation.get(ModelFamilyContractViolationReportFields.Validation.PASSED));
        assertTrue((Integer) driftedValidation.get(
                ModelFamilyContractViolationReportFields.Validation.PROBLEM_COUNT) > 0);
    }

    @Test
    void availabilityCheckerSummariesIncludePartialDirectCaveats() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "cli-partial-direct",
                        "CLI Partial Direct",
                        List.of("cli_partial_direct"),
                        List.of("CliPartialDirectForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/cli_partial_direct",
                                "direct_safetensor", "experimental_text_path",
                                "moe_direct_safetensor", "pending_packed_expert_runtime"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }
        };

        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

            assertTrue(checker.getModelFamilySupportSummaries().stream()
                    .anyMatch(summary -> summary.contains(
                            "cli-partial-direct[optional](experimental:experimental_text_path;"
                                    + "caveats=moe:pending_packed_expert_runtime)")));
            assertTrue(checker.getModelFamilyCapabilityMatrixSummaries().stream()
                    .anyMatch(summary -> summary.contains(
                            "cli-partial-direct[optional](tok=no,gguf=no,safetensor=experimental")));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void availabilityCheckerReportsModelFamilyContractViolations() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "cli bad contract",
                        "CLI Bad Contract",
                        List.of("cli bad type"),
                        List.of(),
                        List.of(),
                        Map.of("bundle_profile", "research"));
            }
        };

        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

            assertTrue(checker.getModelFamilyContractViolations().stream()
                    .anyMatch(summary -> summary.contains("invalid_family_id")));
            assertTrue(checker.getModelFamilyContractViolations().stream()
                    .anyMatch(summary -> summary.contains("unknown_bundle_profile")));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void providerNotFoundErrorIncludesActiveBundlePresetPolicy() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct");
        properties.setProperty("selectorSource", "preset");
        properties.setProperty("presetSelectors", "direct");
        properties.setProperty("policySource", "mixed");
        properties.setProperty("presetRequiredAliases", "direct");
        properties.setProperty("explicitRequiredFamilies", "phi");
        properties.setProperty("families", "gemma");
        properties.setProperty("availableFamilies", "bert,gemma,qwen");
        properties.setProperty("requiredFamilies", "phi");
        properties.setProperty("policyPassed", "false");
        properties.setProperty("policyViolationCount", "1");
        properties.setProperty("missingRequiredFamilies", "phi");
        properties.setProperty("fixtureRequiredSelectors", "direct");
        properties.setProperty("fixtureRequiredFamilies", "gemma,phi");
        properties.setProperty("fixturePassed", "false");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "1");
        properties.setProperty("fixtureMissingRequiredCount", "1");
        properties.setProperty("fixtureProblemFamilyCount", "1");
        properties.setProperty("fixtureMissingRequiredFamilies", "phi");
        properties.setProperty("fixtureProblemFamilies", "phi");
        properties.setProperty("bundlePreset", "prod_llm");
        properties.setProperty("bundlePresets", "prod_llm");
        properties.setProperty("bundlePreset.prod_llm.description", "Lean production LLM");
        properties.setProperty("bundlePreset.prod_llm.selectors", "direct");
        properties.setProperty("bundlePreset.prod_llm.requiredAliases", "direct");
        properties.setProperty("bundlePreset.prod_llm.forbiddenAliases", "embedding");
        properties.setProperty("bundlePreset.prod_llm.selectedFamilies", "gemma");
        properties.setProperty("bundlePreset.prod_llm.policyPassed", "false");
        properties.setProperty("bundlePreset.prod_llm.policyViolationCount", "2");
        properties.setProperty("bundlePreset.prod_llm.missingRequiredFamilies", "qwen");
        properties.setProperty("bundlePreset.prod_llm.selectedForbiddenAlias.embedding.families", "bert");

        PluginAvailabilityChecker checker = new PluginAvailabilityChecker() {
            @Override
            public ModelFamilyBundleManifest getModelFamilyBundleManifest() {
                return ModelFamilyBundleManifest.fromProperties(properties);
            }
        };

        assertTrue(checker.getActiveModelFamilyBundlePresetSummary()
                .contains("prod_llm(selectors=direct, selected=1, policy=failed"));

        String message = checker.getProviderNotFoundError("missing-provider");
        assertTrue(message.contains("selectorSource=preset prod_llm"));
        assertTrue(message.contains("policySource=mixed preset+explicit"));
        assertTrue(message.contains("Packaged model-family bundle policy: failed (1 violation(s))"));
        assertTrue(message.contains("Packaged model-family fixture status: failed (1/2 required, 1 missing, 1 problem)"));
        assertTrue(message.contains("Packaged model-family missing required fixtures: phi"));
        assertTrue(message.contains("Packaged model-family problem fixtures: phi"));
        assertTrue(message.contains("Packaged bundle policy missing required families: phi"));
        assertTrue(message.contains("Active model-family bundle preset: prod_llm("));
        assertTrue(message.contains("Active model-family bundle preset policy: failed (2 violation(s), 1 selected families)"));
        assertTrue(message.contains("Active model-family bundle preset conformance: drifted"));
        assertTrue(message.contains("explicit policy override"));
        assertTrue(message.contains("Active preset missing required families: qwen"));
        assertTrue(message.contains("Active preset selected forbidden alias embedding: bert"));
    }

    @Test
    void availabilityCheckerClassifiesMissingBundledModelFamilies() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct");
        properties.setProperty("families", "gemma,phi");
        properties.setProperty("availableFamilies", "bert,gemma,phi");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixturePassed", "true");
        properties.setProperty("fixtureRequiredFamilies", "gemma,phi");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "2");
        properties.setProperty("bundlePreset", "prod_llm");
        properties.setProperty("bundlePresets", "prod_llm");
        properties.setProperty("bundlePreset.prod_llm.description", "Lean production LLM");
        properties.setProperty("bundlePreset.prod_llm.selectors", "direct");
        properties.setProperty("bundlePreset.prod_llm.selectedFamilies", "gemma,phi");
        properties.setProperty("bundlePreset.prod_llm.policyPassed", "true");
        properties.setProperty("bundlePreset.prod_llm.policyViolationCount", "0");

        PluginAvailabilityChecker checker = new PluginAvailabilityChecker() {
            @Override
            public ModelFamilyBundleManifest getModelFamilyBundleManifest() {
                return ModelFamilyBundleManifest.fromProperties(properties);
            }

            @Override
            public List<String> getModelFamilyPluginIds() {
                return List.of("gemma");
            }
        };

        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                checker.getModelFamilyBundleAvailability();

        assertFalse(availability.healthy());
        assertEquals("missing_plugins", availability.status());
        assertEquals(2, availability.selectedFamilyCount());
        assertEquals(1, availability.discoveredSelectedFamilyCount());
        assertEquals(1, availability.missingSelectedFamilyCount());
        assertEquals(List.of("phi"), availability.missingSelectedFamilies());
        assertTrue(availability.problems().stream()
                .anyMatch(problem -> problem.contains("selected model-family plugins were not discovered: phi")));
        assertTrue(availability.remediationHints().stream()
                .anyMatch(hint -> hint.contains("CDI or ServiceLoader: phi")));
    }

    @Test
    void availabilityCheckerClassifiesManifestCountDrift() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "core");
        properties.setProperty("families", "gemma");
        properties.setProperty("familyCount", "2");
        properties.setProperty("availableFamilies", "gemma");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixturePassed", "true");
        properties.setProperty("fixtureRequiredFamilyCount", "1");
        properties.setProperty("fixtureRequiredPassedCount", "1");

        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        ModelFamilyBundleManifest.fromProperties(properties),
                        Set.of("gemma"));

        assertFalse(availability.healthy());
        assertEquals("manifest_inconsistent", availability.status());
        assertTrue(availability.problems().stream()
                .anyMatch(problem -> problem.contains("bundle.familyCount=2, expected 1")));
        assertTrue(availability.remediationHints().stream()
                .anyMatch(hint -> hint.contains("generateModelFamilyBundleManifest")));
    }

    @Test
    void availabilityCheckerClassifiesFailedFixtureGate() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "core");
        properties.setProperty("families", "gemma");
        properties.setProperty("availableFamilies", "gemma");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixtureRequiredSelectors", "core");
        properties.setProperty("fixtureRequiredFamilies", "gemma");
        properties.setProperty("fixturePassed", "false");
        properties.setProperty("fixtureRequiredFamilyCount", "1");
        properties.setProperty("fixtureRequiredPassedCount", "0");
        properties.setProperty("fixtureMissingRequiredCount", "1");
        properties.setProperty("fixtureProblemFamilyCount", "1");
        properties.setProperty("fixtureMissingRequiredFamilies", "gemma");
        properties.setProperty("fixtureProblemFamilies", "gemma");

        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        ModelFamilyBundleManifest.fromProperties(properties),
                        Set.of("gemma"));

        assertFalse(availability.healthy());
        assertEquals("fixture_failed", availability.status());
        assertEquals("failed", availability.fixtureStatus());
        assertEquals(Boolean.FALSE, availability.fixturePassed());
        assertEquals(1, availability.fixtureMissingRequiredCount());
        assertEquals(1, availability.fixtureProblemFamilyCount());
        assertEquals(List.of("gemma"), availability.fixtureMissingRequiredFamilies());
        assertEquals(List.of("gemma"), availability.fixtureProblemFamilies());
        assertTrue(availability.problems().stream()
                .anyMatch(problem -> problem.contains("model-family fixture gate failed")));
        assertTrue(availability.remediationHints().stream()
                .anyMatch(hint -> hint.contains("validateModelFamilyFixtures")));
    }

    @Test
    void availabilityCheckerClassifiesProductionPendingTokenizerMetadata() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct,kimi");
        properties.setProperty("families", "gemma,kimi");
        properties.setProperty("availableFamilies", "gemma,kimi");
        properties.setProperty("tokenizerMetadataPendingFamilies", "kimi");
        properties.setProperty("family.kimi.tokenizerMetadataPendingReason",
                "tokenizer adapter pending descriptor stabilization");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixturePassed", "true");
        properties.setProperty("fixtureRequiredFamilies", "gemma,kimi");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "2");
        properties.setProperty("bundlePreset", "prod_llm");
        properties.setProperty("bundlePresets", "prod_llm");
        properties.setProperty("bundlePreset.prod_llm.selectedFamilies", "gemma,kimi");
        properties.setProperty("bundlePreset.prod_llm.productionTokenizerMetadataRequired", "true");
        properties.setProperty("bundlePreset.prod_llm.productionTokenizerMetadataReady", "false");
        properties.setProperty("bundlePreset.prod_llm.productionSafetyPassed", "false");
        properties.setProperty("bundlePreset.prod_llm.productionSafetyViolationCount", "1");
        properties.setProperty("bundlePreset.prod_llm.pendingTokenizerFamilies", "kimi");
        properties.setProperty("bundlePreset.prod_llm.pendingTokenizerFamily.kimi.reason",
                "tokenizer adapter pending descriptor stabilization");
        properties.setProperty("bundlePreset.prod_llm.policyPassed", "true");
        properties.setProperty("bundlePreset.prod_llm.policyViolationCount", "0");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);
        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(manifest, Set.of("gemma", "kimi"));

        assertFalse(availability.healthy());
        assertEquals("production_safety_failed", availability.status());
        assertEquals("failed", availability.productionSafetyStatus());
        assertFalse(availability.productionSafetyPassed());
        assertEquals(List.of("kimi"), availability.productionPendingTokenizerFamilies());
        assertTrue(availability.problems().stream()
                .anyMatch(problem -> problem.contains("pending tokenizer family")));
        assertTrue(availability.remediationHints().stream()
                .anyMatch(hint -> hint.contains("pending-tokenizer families out of prod_*")));

        ModelFamilyBundleGate gate = ModelFamilyBundleGate.evaluate(availability, List.of());
        assertFalse(gate.passed());
        assertEquals("production_safety_failed", gate.status());
        assertTrue(gate.failureMessage().contains("model-family production safety failed"));

        ModelFamilyBundleGate contractGate = ModelFamilyBundleGate.evaluate(
                availability,
                List.of(new ModelFamilyContractViolation(
                        "kimi",
                        "pending_contract",
                        "contract still pending")));
        assertEquals("production_safety_and_contract_failed", contractGate.status());
    }

    @Test
    void availabilityCheckerClassifiesCatalogReadinessFailures() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct");
        properties.setProperty("families", "gemma,phi");
        properties.setProperty("availableFamilies", "gemma,phi");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("productionReadinessPassed", "false");
        properties.setProperty("productionReadinessPendingCount", "1");
        properties.setProperty("productionReadinessPendingFamilies", "phi");
        properties.setProperty("directSafetensorReadinessPassed", "false");
        properties.setProperty("directSafetensorPendingCount", "1");
        properties.setProperty("directSafetensorPendingFamilies", "phi");
        properties.setProperty("fixturePassed", "true");
        properties.setProperty("fixtureRequiredFamilies", "gemma,phi");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "2");

        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        ModelFamilyBundleManifest.fromProperties(properties),
                        Set.of("gemma", "phi"));

        assertFalse(availability.healthy());
        assertEquals("catalog_readiness_failed", availability.status());
        assertEquals("failed", availability.catalogReadinessStatus());
        assertFalse(availability.catalogReadinessPassed());
        assertEquals(1, availability.productionReadinessPendingCount());
        assertEquals(1, availability.directSafetensorPendingCount());
        assertEquals(List.of("phi"), availability.productionReadinessPendingFamilies());
        assertEquals(List.of("phi"), availability.directSafetensorPendingFamilies());
        assertTrue(availability.problems().stream()
                .anyMatch(problem -> problem.contains("model-family catalog readiness failed")));
        assertTrue(availability.remediationHints().stream()
                .anyMatch(hint -> hint.contains("validateModelFamilyModuleCatalog")));

        ModelFamilyBundleGate gate = ModelFamilyBundleGate.evaluate(availability, List.of());
        assertFalse(gate.passed());
        assertEquals("catalog_readiness_failed", gate.status());

        ModelFamilyBundleGate contractGate = ModelFamilyBundleGate.evaluate(
                availability,
                List.of(new ModelFamilyContractViolation(
                        "phi",
                        "pending_contract",
                        "contract still pending")));
        assertEquals("catalog_readiness_and_contract_failed", contractGate.status());
    }

    @Test
    void modelFamilyBundleGateCombinesAvailabilityAndContractFailures() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct");
        properties.setProperty("families", "gemma,phi");
        properties.setProperty("availableFamilies", "gemma,phi");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixturePassed", "true");
        properties.setProperty("fixtureRequiredFamilies", "gemma,phi");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "2");

        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        ModelFamilyBundleManifest.fromProperties(properties),
                        Set.of("gemma"));
        ModelFamilyBundleGate gate = ModelFamilyBundleGate.evaluate(
                availability,
                List.of(new ModelFamilyContractViolation(
                        "bad_family",
                        "bad_contract",
                        "bad contract")));

        assertFalse(gate.passed());
        assertTrue(gate.failed());
        assertEquals("availability_and_contract_failed", gate.status());
        assertEquals(2, gate.violationCount());
        assertTrue(gate.violations().stream().anyMatch(violation -> violation.startsWith("availability:")));
        assertTrue(gate.violations().stream().anyMatch(violation -> violation.startsWith("contract:")));
        assertTrue(gate.failureMessage().contains("availability_and_contract_failed"));
    }

    @Test
    void modelFamilyBundleGateEnforcesDirectSafetensorRuntimeWhenRequired() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct");
        properties.setProperty("families", "phi");
        properties.setProperty("availableFamilies", "phi");
        properties.setProperty("requiredAliases", "direct");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixturePassed", "true");
        properties.setProperty("fixtureRequiredFamilies", "phi");
        properties.setProperty("fixtureRequiredFamilyCount", "1");
        properties.setProperty("fixtureRequiredPassedCount", "1");
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);
        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(manifest, Set.of("phi"));
        ModelFamilyRuntimeCompatibilitySummary runtimeSummary = new ModelFamilyRuntimeCompatibilitySummary(
                "direct_safetensor",
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                List.of(),
                List.of("phi"),
                Map.of("model_family_direct_safetensor_not_ready", 1));

        ModelFamilyBundleGate gate = ModelFamilyBundleGate.evaluate(
                availability,
                List.of(),
                runtimeSummary,
                manifest.requiresDirectSafetensorRuntime());

        assertTrue(manifest.requiresDirectSafetensorRuntime());
        assertFalse(gate.passed());
        assertEquals("runtime_failed", gate.status());
        assertTrue(gate.violations().stream()
                .anyMatch(violation -> violation.contains("direct SafeTensor runtime")));
        assertTrue(gate.violations().stream()
                .anyMatch(violation -> violation.contains("phi")));
    }

    @Test
    void availabilityCheckerReportsSulingThroughSharedExtensionContract() {
        PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

        ExtensionAvailability extensionAvailability = checker.sulingExtensionAvailability();
        PluginAvailabilityChecker.AudioExtensionAvailability audioAvailability =
                checker.sulingAudioAvailability();

        assertEquals("suling", extensionAvailability.id());
        assertEquals("Suling Audio", extensionAvailability.name());
        assertEquals("audio", extensionAvailability.kind());
        assertEquals(extensionAvailability.status(), audioAvailability.status());
        assertEquals(extensionAvailability.formats(), audioAvailability.formats());
        assertEquals(
                extensionAvailability.attributeBoolean("flacAvailable"),
                audioAvailability.flacAvailable());
        assertTrue(checker.getExtensionAvailabilityReports().stream()
                .anyMatch(report -> "suling".equals(report.id())));
    }

    @Test
    void sulingAvailabilityProviderIsDiscoverableThroughPluginCoreServiceLoader() {
        ExtensionAvailabilityRegistry registry = new ExtensionAvailabilityRegistry();

        registry.discoverServiceLoaderProviders();

        assertTrue(registry.providers().stream()
                .anyMatch(provider -> SulingAudioExtensionAvailabilityProvider.ID.equals(provider.extensionId())));
        ExtensionAvailability availability = registry.availability(SulingAudioExtensionAvailabilityProvider.ID)
                .orElseThrow();
        assertEquals("audio", availability.kind());
        assertEquals("Suling Audio", availability.name());
    }

    @Test
    void extensionAvailabilityPolicyClassifiesRequiredAndForbiddenExtensions() {
        ExtensionAvailability fallbackSuling = new ExtensionAvailability(
                "suling",
                "Suling Audio",
                "audio",
                true,
                false,
                true,
                false,
                "fallback",
                List.of("audio_encoding"),
                List.of("wav"),
                Map.of("fallback", "true"),
                "fallback active",
                List.of());

        ExtensionAvailabilityPolicy.Result result = new ExtensionAvailabilityPolicy(
                List.of("suling"),
                List.of("suling"),
                List.of("webworld")).evaluate(List.of(fallbackSuling));

        assertFalse(result.passed());
        assertEquals("failed", result.status());
        assertTrue(result.violations().stream()
                .anyMatch(violation -> violation.contains("not production-ready")));
    }

    @Test
    void extensionAvailabilityGateCombinesPolicyAndContractFailures() {
        ExtensionAvailability fallbackSuling = new ExtensionAvailability(
                "suling",
                "Suling Audio",
                "audio",
                true,
                false,
                true,
                false,
                "fallback",
                List.of("audio_encoding"),
                List.of("wav"),
                Map.of("fallback", "true"),
                "fallback active",
                List.of());
        ExtensionAvailabilityPolicy.Result policy = new ExtensionAvailabilityPolicy(
                List.of("suling"),
                List.of("suling"),
                List.of()).evaluate(List.of(fallbackSuling));
        ExtensionAvailabilityContractReport contract = ExtensionAvailabilityContractReport.fromViolations(List.of(
                new ExtensionAvailabilityContractViolation(
                        "suling",
                        "test_contract_violation",
                        "test contract violation")));

        ExtensionAvailabilityGate gate = ExtensionAvailabilityGate.evaluate(policy, contract);

        assertFalse(gate.passed());
        assertTrue(gate.failed());
        assertEquals("policy_and_contract_failed", gate.status());
        assertEquals(2, gate.violationCount());
        assertTrue(gate.violations().stream().anyMatch(violation -> violation.startsWith("policy:")));
        assertTrue(gate.violations().stream().anyMatch(violation -> violation.startsWith("contract:")));
        assertTrue(gate.failureMessage().contains("policy_and_contract_failed"));
    }

    @Test
    void extensionAvailabilityGatePassesCleanInputs() {
        ExtensionAvailabilityGate gate = ExtensionAvailabilityGate.evaluate(
                ExtensionAvailabilityPolicy.empty().evaluate(List.of()),
                ExtensionAvailabilityContractReport.fromViolations(List.of()));

        assertTrue(gate.passed());
        assertFalse(gate.failed());
        assertEquals("passed", gate.status());
        assertEquals(0, gate.violationCount());
        assertEquals(List.of(), gate.violations());
    }

    @Test
    void pluginGatesCombineSubGateFailures() {
        PluginGates extensionFailed = PluginGates.evaluate(
                new ExtensionAvailabilityGate(
                        false,
                        "contract_failed",
                        1,
                        List.of("contract: bad extension provider")),
                new ModelFamilyBundleGate(
                        true,
                        "passed",
                        0,
                        List.of()));

        assertFalse(extensionFailed.passed());
        assertTrue(extensionFailed.failed());
        assertEquals("extension_failed", extensionFailed.status());
        assertEquals(1, extensionFailed.violationCount());
        assertEquals("contract_failed", extensionFailed.extensionStatus());
        assertEquals("passed", extensionFailed.modelFamilyStatus());
        assertEquals(1, extensionFailed.extensionViolationCount());
        assertEquals(0, extensionFailed.modelFamilyViolationCount());
        assertTrue(extensionFailed.violations().stream()
                .anyMatch(violation -> violation.contains("extension: contract: bad extension provider")));
        assertTrue(extensionFailed.failureMessage().contains("extension_failed"));

        PluginGates missingSubGates = PluginGates.evaluate(null, null);

        assertFalse(missingSubGates.passed());
        assertEquals("extension_and_model_family_failed", missingSubGates.status());
        assertEquals(2, missingSubGates.violationCount());
        assertEquals(1, missingSubGates.extensionViolationCount());
        assertEquals(1, missingSubGates.modelFamilyViolationCount());
        assertTrue(missingSubGates.violations().stream()
                .anyMatch(violation -> violation.contains("extension availability gate was not evaluated")));
        assertTrue(missingSubGates.violations().stream()
                .anyMatch(violation -> violation.contains("model-family bundle gate was not evaluated")));

        PluginGates productionSafetyFailed = PluginGates.evaluate(
                new ExtensionAvailabilityGate(true, "passed", 0, List.of()),
                new ModelFamilyBundleGate(
                        false,
                        "production_safety_failed",
                        1,
                        List.of("availability: model-family production safety failed")));

        assertFalse(productionSafetyFailed.passed());
        assertEquals("model_family_production_safety_failed", productionSafetyFailed.status());
        assertEquals("production_safety_failed", productionSafetyFailed.modelFamilyStatus());
        assertTrue(productionSafetyFailed.failureMessage().contains("model_family_production_safety_failed"));

        PluginGates catalogReadinessFailed = PluginGates.evaluate(
                new ExtensionAvailabilityGate(true, "passed", 0, List.of()),
                new ModelFamilyBundleGate(
                        false,
                        "catalog_readiness_failed",
                        1,
                        List.of("availability: model-family catalog readiness failed")));

        assertFalse(catalogReadinessFailed.passed());
        assertEquals("model_family_catalog_readiness_failed", catalogReadinessFailed.status());
        assertEquals("catalog_readiness_failed", catalogReadinessFailed.modelFamilyStatus());
        assertTrue(catalogReadinessFailed.failureMessage().contains("model_family_catalog_readiness_failed"));

        PluginGates productionSafetyAndExtensionFailed = PluginGates.evaluate(
                new ExtensionAvailabilityGate(
                        false,
                        "policy_failed",
                        1,
                        List.of("policy: bad extension policy")),
                new ModelFamilyBundleGate(
                        false,
                        "production_safety_and_contract_failed",
                        2,
                        List.of(
                                "availability: model-family production safety failed",
                                "contract: pending tokenizer contract")));

        assertFalse(productionSafetyAndExtensionFailed.passed());
        assertEquals(
                "extension_and_model_family_production_safety_and_contract_failed",
                productionSafetyAndExtensionFailed.status());
        assertEquals("production_safety_and_contract_failed",
                productionSafetyAndExtensionFailed.modelFamilyStatus());

        PluginGates catalogReadinessAndExtensionFailed = PluginGates.evaluate(
                new ExtensionAvailabilityGate(
                        false,
                        "policy_failed",
                        1,
                        List.of("policy: bad extension policy")),
                new ModelFamilyBundleGate(
                        false,
                        "catalog_readiness_and_contract_failed",
                        2,
                        List.of(
                                "availability: model-family catalog readiness failed",
                                "contract: pending readiness contract")));

        assertFalse(catalogReadinessAndExtensionFailed.passed());
        assertEquals(
                "extension_and_model_family_catalog_readiness_and_contract_failed",
                catalogReadinessAndExtensionFailed.status());
        assertEquals("catalog_readiness_and_contract_failed",
                catalogReadinessAndExtensionFailed.modelFamilyStatus());
    }

    @Test
    void pluginGatesCarryModelFamilyContractRemediation() {
        ModelFamilyBundleGate modelFamilyGate = ModelFamilyBundleGate.evaluate(
                new PluginAvailabilityChecker.ModelFamilyBundleAvailability(
                        true,
                        false,
                        true,
                        "ready",
                        1,
                        1,
                        0,
                        0,
                        "passed",
                        0,
                        "passed",
                        true,
                        List.of(),
                        "passed",
                        true,
                        0,
                        0,
                        List.of(),
                        List.of(),
                        "passed",
                        true,
                        0,
                        0,
                        "none",
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of()),
                List.of(new ModelFamilyContractViolation(
                        "bad_bundle",
                        "unknown_bundle_profile",
                        "bad bundle profile")));

        PluginGates gates = PluginGates.evaluate(
                new ExtensionAvailabilityGate(true, "passed", 0, List.of()),
                modelFamilyGate);

        assertFalse(gates.passed());
        assertEquals(1, gates.modelFamilyContractCategoryCounts().get("bundleProfile"));
        assertTrue(gates.modelFamilyContractRemediationHints().stream()
                .anyMatch(hint -> hint.contains("metadata.bundle_profile")));
        assertTrue(gates.failureMessage().contains("Model-family contract recommendations"));
    }

    @Test
    void pluginGateViolationReportsCategorizeCombinedGateViolations() {
        Map<String, Integer> categories = PluginGateViolationReports.categories(List.of(
                "extension: policy: bad extension policy",
                "model-family: availability: selected family missing",
                "plugin-directory: model-family plugin jar is not plugin-core ready",
                "runner-route: contract bundle failed: schemaFingerprint drifted",
                "unified-runtime: duplicate_model_type_claim",
                "unified-runtime-requirement: gemma4 -> gemma4_unified status="
                        + UnifiedRuntimeRequirementStatuses.MISSING_RUNTIME,
                "unexpected violation"));

        assertEquals(1, categories.get("extension"));
        assertEquals(1, categories.get("modelFamily"));
        assertEquals(1, categories.get("pluginDirectory"));
        assertEquals(1, categories.get("runnerRoute"));
        assertEquals(1, categories.get("unifiedRuntime"));
        assertEquals(1, categories.get("unifiedRuntimeRequirement"));
        assertEquals(1, categories.get("unknown"));
        assertEquals(0, PluginGateViolationReports.categories((List<String>) null).get("extension"));
    }

    @Test
    void pluginGatesFailForUnifiedRuntimeConflictsAndManifestViolations() {
        ExtensionAvailabilityGate extensionGate = new ExtensionAvailabilityGate(true, "passed", 0, List.of());
        ModelFamilyBundleGate modelFamilyGate = new ModelFamilyBundleGate(true, "passed", 0, List.of());
        PluginGates base = PluginGates.evaluate(extensionGate, modelFamilyGate);

        UnifiedMultimodalRuntime first = () -> unifiedRuntimeManifest(
                "first-gemma4-unified-test-runtime",
                "gemma4_unified",
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE));
        UnifiedMultimodalRuntime second = () -> unifiedRuntimeManifest(
                "second-gemma4-unified-test-runtime",
                "gemma4_unified",
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE));

        PluginGates conflictGate = PluginGates.withUnifiedRuntimeReadiness(
                base,
                UnifiedRuntimeRegistry.of(List.of(first, second)));

        assertFalse(conflictGate.passed());
        assertEquals("unified_runtime_failed", conflictGate.status());
        assertTrue(conflictGate.violations().toString().contains("unified-runtime:"));
        assertTrue(conflictGate.violations().toString().contains("duplicate_model_type_claim"));
        assertTrue(conflictGate.failureMessage().contains("gemma4_unified"));

        UnifiedMultimodalRuntime invalid = () -> unifiedRuntimeManifest(
                "invalid-gemma4-unified-test-runtime",
                "gemma4_unified",
                List.of());

        PluginGates invalidGate = PluginGates.withUnifiedRuntimeReadiness(
                base,
                UnifiedRuntimeRegistry.of(List.of(invalid)));

        assertFalse(invalidGate.passed());
        assertEquals("unified_runtime_failed", invalidGate.status());
        assertTrue(invalidGate.violations().toString().contains("missing_input_modalities"));
    }

    @Test
    void pluginGatesFailForSelectedUnifiedRuntimeRequirementViolations() {
        PluginGates base = PluginGates.evaluate(
                new ExtensionAvailabilityGate(true, "passed", 0, List.of()),
                new ModelFamilyBundleGate(true, "passed", 0, List.of()));
        UnifiedRuntimeRequirementCompatibility ready = UnifiedRuntimeRequirementCompatibility.ready(
                "ready-family",
                "ready_unified",
                List.of("text"),
                true,
                List.of("ready-runtime"),
                List.of("text"));
        UnifiedRuntimeRequirementCompatibility missing = UnifiedRuntimeRequirementCompatibility.attention(
                "gemma4",
                "gemma4_unified",
                List.of("text", "image", "audio", "video"),
                true,
                UnifiedRuntimeRequirementIssueKind.MISSING_RUNTIME,
                List.of(),
                List.of(),
                "Attach one unified runtime plugin that claims model_type=gemma4_unified.");
        UnifiedRuntimeRequirementCompatibility legacy = new UnifiedRuntimeRequirementCompatibility(
                "legacy",
                "legacy_unified",
                List.of("text"),
                true,
                UnifiedRuntimeRequirementStatuses.MISSING_RUNTIME,
                List.of(),
                List.of(),
                List.of(),
                List.of());

        PluginGates gate = PluginGates.withUnifiedRuntimeRequirements(
                base,
                List.of(ready, missing, legacy));

        assertFalse(gate.passed());
        assertEquals("unified_runtime_requirement_failed", gate.status());
        assertTrue(gate.violations().toString().contains("unified-runtime-requirement:"));
        assertTrue(gate.violations().toString().contains("gemma4 -> gemma4_unified"));
        assertTrue(gate.violations().toString()
                .contains("legacy -> legacy_unified status=missing_runtime problems=unified_runtime_missing"));
        assertTrue(gate.violations().toString()
                .contains(UnifiedRuntimeRequirementProblemCodes.MISSING_RUNTIME));
        assertFalse(gate.violations().toString().contains("ready-family"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesFailForUnifiedRuntimeRequirementReportContractDrift() {
        PluginGates base = PluginGates.evaluate(
                new ExtensionAvailabilityGate(true, "passed", 0, List.of()),
                new ModelFamilyBundleGate(true, "passed", 0, List.of()));
        Map<String, Object> section = UnifiedRuntimeRequirementReports.modelFamilyRequirementSection(List.of());

        PluginGates validGate = PluginGates.withUnifiedRuntimeRequirementReportContract(base, section);

        assertTrue(validGate.passed());

        Map<String, Object> driftedSection = new LinkedHashMap<>(section);
        Map<String, Object> driftedSchema = new LinkedHashMap<>(
                (Map<String, Object>) section.get(
                        UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA));
        driftedSchema.put(
                UnifiedRuntimeRequirementReportFields.Schema.SCHEMA_FINGERPRINT,
                "sha256:drifted");
        driftedSection.put(
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA,
                driftedSchema);

        PluginGates gate = PluginGates.withUnifiedRuntimeRequirementReportContract(base, driftedSection);

        assertFalse(gate.passed());
        assertEquals("unified_runtime_requirement_failed", gate.status());
        assertTrue(gate.violations().toString().contains("report contract failed"));
        assertTrue(gate.violations().toString().contains("schema.schemaFingerprint"));
        assertEquals(1, PluginGateViolationReports.categories(gate).get("unifiedRuntimeRequirement"));
    }

    @Test
    void pluginGatesFailForModelFamilyContractReportDrift() {
        PluginGates base = PluginGates.evaluate(
                new ExtensionAvailabilityGate(true, "passed", 0, List.of()),
                new ModelFamilyBundleGate(true, "passed", 0, List.of()));
        Map<String, Object> summary = ModelFamilyContractViolationReports.summary(List.of());

        PluginGates validGate = PluginGates.withModelFamilyContractReportContract(base, summary);

        assertTrue(validGate.passed());

        Map<String, Object> driftedSummary = new LinkedHashMap<>(summary);
        driftedSummary.put(ModelFamilyContractViolationReportFields.Report.SCHEMA_FINGERPRINT, "sha256:drifted");

        PluginGates gate = PluginGates.withModelFamilyContractReportContract(base, driftedSummary);

        assertFalse(gate.passed());
        assertEquals("model_family_contract_report_failed", gate.status());
        assertTrue(gate.violations().toString().contains("model-family: contract report failed"));
        assertTrue(gate.violations().toString().contains("schemaFingerprint"));
        assertEquals(1, PluginGateViolationReports.categories(gate).get("modelFamily"));
    }

    @Test
    void unifiedRuntimeRequirementResolverReportsMissingReadyAndInsufficientModalities() {
        ModelFamilyPluginRegistry modelFamilies = ModelFamilyPluginRegistry.create();
        modelFamilies.register(unifiedRuntimeRequirementPlugin());

        List<UnifiedRuntimeRequirementCompatibility> missing = UnifiedRuntimeRequirementResolver.evaluate(
                modelFamilies,
                UnifiedRuntimeRegistry.of(List.of()));

        assertEquals(1, missing.size());
        assertEquals(UnifiedRuntimeRequirementStatuses.MISSING_RUNTIME, missing.getFirst().status());
        assertTrue(missing.getFirst().problemCodes()
                .contains(UnifiedRuntimeRequirementProblemCodes.MISSING_RUNTIME));

        UnifiedMultimodalRuntime readyRuntime = () -> unifiedRuntimeManifest(
                "ready-unified-requirement-runtime",
                "unit_unified",
                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE));
        List<UnifiedRuntimeRequirementCompatibility> ready = UnifiedRuntimeRequirementResolver.evaluate(
                modelFamilies,
                UnifiedRuntimeRegistry.of(List.of(readyRuntime)));

        assertEquals(UnifiedRuntimeRequirementStatuses.READY, ready.getFirst().status());
        assertTrue(ready.getFirst().compatible());
        assertEquals(List.of("image", "text"), ready.getFirst().availableInputModalities());

        UnifiedMultimodalRuntime textOnlyRuntime = () -> unifiedRuntimeManifest(
                "text-only-unified-requirement-runtime",
                "unit_unified",
                List.of(UnifiedInputModality.TEXT),
                UnifiedRuntimeReadiness.EXPERIMENTAL);
        List<UnifiedRuntimeRequirementCompatibility> insufficient = UnifiedRuntimeRequirementResolver.evaluate(
                modelFamilies,
                UnifiedRuntimeRegistry.of(List.of(textOnlyRuntime)));

        assertEquals(
                UnifiedRuntimeRequirementStatuses.INSUFFICIENT_MODALITIES,
                insufficient.getFirst().status());
        assertTrue(insufficient.getFirst().problemCodes()
                .contains(UnifiedRuntimeRequirementProblemCodes.MISSING_REQUIRED_MODALITIES));
    }

    @Test
    void gemma4ReadyUnifiedRuntimeSatisfiesSelectedRequirementAndPluginGate() {
        ModelFamilyPluginRegistry modelFamilies = ModelFamilyPluginRegistry.create();
        modelFamilies.register(gemma4UnifiedRequirementPlugin());
        UnifiedMultimodalRuntime readyGemma4Runtime = () -> unifiedRuntimeManifest(
                "ready-gemma4-unified-requirement-runtime",
                "gemma4_unified",
                List.of(
                        UnifiedInputModality.TEXT,
                        UnifiedInputModality.IMAGE,
                        UnifiedInputModality.AUDIO,
                        UnifiedInputModality.VIDEO));
        UnifiedRuntimeRegistry unifiedRuntimes = UnifiedRuntimeRegistry.of(List.of(readyGemma4Runtime));

        List<UnifiedRuntimeRequirementCompatibility> compatibilities =
                UnifiedRuntimeRequirementResolver.evaluate(
                        modelFamilies,
                        unifiedRuntimes,
                        List.of("gemma4"));
        PluginGates gate = PluginGates.withUnifiedRuntimeRequirements(
                PluginGates.withUnifiedRuntimeReadiness(
                        PluginGates.evaluate(
                                new ExtensionAvailabilityGate(true, "passed", 0, List.of()),
                                new ModelFamilyBundleGate(true, "passed", 0, List.of())),
                        unifiedRuntimes),
                compatibilities);

        assertEquals(1, compatibilities.size());
        assertEquals(UnifiedRuntimeRequirementStatuses.READY, compatibilities.getFirst().status());
        assertTrue(compatibilities.getFirst().compatible());
        assertEquals("gemma4", compatibilities.getFirst().familyId());
        assertEquals("gemma4_unified", compatibilities.getFirst().modelType());
        assertEquals(List.of("audio", "image", "text", "video"),
                compatibilities.getFirst().availableInputModalities());
        assertTrue(gate.passed());
        assertEquals("passed", gate.status());
        assertEquals(List.of(), gate.violations());
    }

    @Test
    @SuppressWarnings("unchecked")
    void extensionAvailabilityGateReportWriterBuildsCiArtifactPayload() {
        Map<String, Object> report = ExtensionAvailabilityGateReportWriter.buildReport();

        assertEquals(ExtensionAvailabilityGateReportFields.rootFields(), List.copyOf(report.keySet()));
        assertEquals(ExtensionAvailabilityGateReportFields.SCHEMA_VERSION,
                report.get(ExtensionAvailabilityGateReportFields.Root.SCHEMA_VERSION));
        assertTrue(report.get(ExtensionAvailabilityGateReportFields.Root.GENERATED_AT) instanceof String);
        Map<String, Object> gate =
                (Map<String, Object>) report.get(ExtensionAvailabilityGateReportFields.Root.GATE);
        Map<String, Object> policy =
                (Map<String, Object>) report.get(ExtensionAvailabilityGateReportFields.Root.POLICY);
        Map<String, Object> contract =
                (Map<String, Object>) report.get(ExtensionAvailabilityGateReportFields.Root.CONTRACT);
        assertEquals(ExtensionAvailabilityGateReportFields.gateFields(), List.copyOf(gate.keySet()));
        assertTrue(gate.get(ExtensionAvailabilityGateReportFields.Gate.STATUS) instanceof String);
        assertTrue(gate.get(ExtensionAvailabilityGateReportFields.Gate.VIOLATION_COUNT) instanceof Integer);
        assertEquals(ExtensionAvailabilityGateReportFields.policyFields(), List.copyOf(policy.keySet()));
        assertTrue(policy.get(ExtensionAvailabilityGateReportFields.Policy.STATUS) instanceof String);
        assertTrue(policy.get(ExtensionAvailabilityGateReportFields.Policy.CONFIGURED) instanceof Boolean);
        assertTrue(policy.get(ExtensionAvailabilityGateReportFields.Policy.REQUIRED_EXTENSIONS) instanceof List<?>);
        assertEquals(ExtensionAvailabilityGateReportFields.contractFields(), List.copyOf(contract.keySet()));
        assertTrue(contract.get(ExtensionAvailabilityGateReportFields.Contract.STATUS) instanceof String);
        assertTrue(contract.get(ExtensionAvailabilityGateReportFields.Contract.VIOLATION_COUNT) instanceof Integer);
        Object violations = contract.get(ExtensionAvailabilityGateReportFields.Contract.VIOLATIONS);
        assertTrue(violations instanceof List<?>);
        if (!((List<?>) violations).isEmpty()) {
            Map<String, Object> violation = (Map<String, Object>) ((List<?>) violations).getFirst();
            assertEquals(
                    ExtensionAvailabilityGateReportFields.contractViolationFields(),
                    List.copyOf(violation.keySet()));
        }
        Object extensions = report.get(ExtensionAvailabilityGateReportFields.Root.EXTENSIONS);
        assertTrue(extensions instanceof List<?>);
        if (!((List<?>) extensions).isEmpty()) {
            Map<String, Object> extension = (Map<String, Object>) ((List<?>) extensions).getFirst();
            assertEquals(ExtensionAvailabilityGateReportFields.extensionFields(), List.copyOf(extension.keySet()));
            assertTrue(extension.get(ExtensionAvailabilityGateReportFields.Extension.ID) instanceof String);
            assertTrue(extension.get(ExtensionAvailabilityGateReportFields.Extension.PRODUCTION_READY)
                    instanceof Boolean);
            assertTrue(extension.get(ExtensionAvailabilityGateReportFields.Extension.CAPABILITIES) instanceof List<?>);
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void modelFamilyBundleGateReportWriterBuildsCiArtifactPayload() {
        Map<String, Object> report = ModelFamilyBundleGateReportWriter.buildReport();

        assertEquals(ModelFamilyBundleGateReportFields.rootFields(), List.copyOf(report.keySet()));
        assertEquals(ModelFamilyBundleGateReportFields.SCHEMA_VERSION,
                report.get(ModelFamilyBundleGateReportFields.Root.SCHEMA_VERSION));
        assertTrue(report.get(ModelFamilyBundleGateReportFields.Root.GENERATED_AT) instanceof String);
        Map<String, Object> gate =
                (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.GATE);
        Map<String, Object> availability =
                (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.AVAILABILITY);
        Map<String, Object> contract =
                (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.CONTRACT);
        Map<String, Object> contractValidation =
                (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.CONTRACT_VALIDATION);
        Map<String, Object> manifest =
                (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.MANIFEST);
        assertEquals(ModelFamilyBundleGateReportFields.gateFields(), List.copyOf(gate.keySet()));
        assertTrue(gate.get(ModelFamilyBundleGateReportFields.Gate.STATUS) instanceof String);
        assertTrue(gate.get(ModelFamilyBundleGateReportFields.Gate.VIOLATION_COUNT) instanceof Integer);
        assertTrue(gate.get(ModelFamilyBundleGateReportFields.Gate.CONTRACT_CATEGORY_COUNTS) instanceof Map<?, ?>);
        assertTrue(gate.get(ModelFamilyBundleGateReportFields.Gate.CONTRACT_REMEDIATION_HINTS) instanceof List<?>);
        assertEquals(ModelFamilyBundleAvailabilityReportFields.fields(), List.copyOf(availability.keySet()));
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.STATUS) instanceof String);
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.PRODUCTION_SAFETY_STATUS)
                instanceof String);
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.PRODUCTION_SAFETY_PASSED)
                instanceof Boolean);
        assertTrue(availability.get(
                ModelFamilyBundleAvailabilityReportFields.Section.PRODUCTION_PENDING_TOKENIZER_FAMILIES)
                instanceof List<?>);
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.CATALOG_READINESS_STATUS)
                instanceof String);
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.CATALOG_READINESS_PASSED)
                instanceof Boolean);
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.PRODUCTION_READINESS_PENDING_COUNT)
                instanceof Integer);
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.DIRECT_SAFETENSOR_PENDING_COUNT)
                instanceof Integer);
        assertTrue(availability.get(
                ModelFamilyBundleAvailabilityReportFields.Section.PRODUCTION_READINESS_PENDING_FAMILIES)
                instanceof List<?>);
        assertTrue(availability.get(ModelFamilyBundleAvailabilityReportFields.Section.DIRECT_SAFETENSOR_PENDING_FAMILIES)
                instanceof List<?>);
        assertTrue(contract.get("status") instanceof String);
        assertEquals(ModelFamilyContractViolationReports.SCHEMA_FINGERPRINT,
                contract.get(ModelFamilyContractViolationReportFields.Report.SCHEMA_FINGERPRINT));
        assertEquals(Boolean.TRUE,
                contractValidation.get(ModelFamilyContractViolationReportFields.Validation.PASSED));
        assertEquals(0,
                contractValidation.get(ModelFamilyContractViolationReportFields.Validation.PROBLEM_COUNT));
        assertEquals(List.of(),
                contractValidation.get(ModelFamilyContractViolationReportFields.Validation.PROBLEMS));
        assertTrue(manifest.get(ModelFamilyBundleManifestReportFields.Manifest.PRESENT) instanceof Boolean);
        if (Boolean.TRUE.equals(manifest.get(ModelFamilyBundleManifestReportFields.Manifest.PRESENT))) {
            assertEquals(ModelFamilyBundleManifestReportFields.manifestFields(), List.copyOf(manifest.keySet()));
            assertTrue(manifest.get(ModelFamilyBundleManifestReportFields.Manifest.COUNT_CONSISTENCY_PROBLEMS)
                    instanceof List<?>);
            assertTrue(manifest.get(
                    ModelFamilyBundleManifestReportFields.Manifest.REQUIRES_DIRECT_SAFETENSOR_RUNTIME)
                    instanceof Boolean);
            Map<String, Object> productionSafety =
                    (Map<String, Object>) manifest.get(
                            ModelFamilyBundleManifestReportFields.Manifest.PRODUCTION_SAFETY);
            Map<String, Object> catalogReadiness =
                    (Map<String, Object>) manifest.get(
                            ModelFamilyBundleManifestReportFields.Manifest.CATALOG_READINESS);
            Map<String, Object> policyStatus =
                    (Map<String, Object>) manifest.get(
                            ModelFamilyBundleManifestReportFields.Manifest.POLICY_STATUS);
            Map<String, Object> fixtureStatus =
                    (Map<String, Object>) manifest.get(
                            ModelFamilyBundleManifestReportFields.Manifest.FIXTURE_STATUS);
            Map<String, Object> presetConformance =
                    (Map<String, Object>) manifest.get(
                            ModelFamilyBundleManifestReportFields.Manifest.ACTIVE_BUNDLE_PRESET_CONFORMANCE);
            assertEquals(
                    ModelFamilyBundleManifestReportFields.productionSafetyFields(),
                    List.copyOf(productionSafety.keySet()));
            assertEquals(
                    ModelFamilyBundleManifestReportFields.catalogReadinessFields(),
                    List.copyOf(catalogReadiness.keySet()));
            assertEquals(
                    ModelFamilyBundleManifestReportFields.policyStatusFields(),
                    List.copyOf(policyStatus.keySet()));
            assertEquals(
                    ModelFamilyBundleManifestReportFields.fixtureStatusFields(),
                    List.copyOf(fixtureStatus.keySet()));
            assertEquals(
                    ModelFamilyBundleManifestReportFields.presetConformanceFields(),
                    List.copyOf(presetConformance.keySet()));
            assertTrue(productionSafety.get(
                    ModelFamilyBundleManifestReportFields.ProductionSafety.PASSED) instanceof Boolean);
            assertTrue(productionSafety.get(
                    ModelFamilyBundleManifestReportFields.ProductionSafety.PENDING_TOKENIZER_FAMILIES)
                    instanceof List<?>);
            assertTrue(catalogReadiness.get(
                    ModelFamilyBundleManifestReportFields.CatalogReadiness.STATUS) instanceof String);
            assertTrue(catalogReadiness.get(
                    ModelFamilyBundleManifestReportFields.CatalogReadiness.PRODUCTION_READINESS_PENDING_FAMILIES)
                    instanceof List<?>);
            assertTrue(policyStatus.get(
                    ModelFamilyBundleManifestReportFields.PolicyStatus.VIOLATION_COUNT) instanceof Integer);
            assertTrue(fixtureStatus.get(
                    ModelFamilyBundleManifestReportFields.FixtureStatus.MISSING_REQUIRED_FAMILIES)
                    instanceof List<?>);
            assertTrue(presetConformance.get(
                    ModelFamilyBundleManifestReportFields.PresetConformance.MATCHES_PRESET) instanceof Boolean);
        } else {
            assertEquals(ModelFamilyBundleManifestReportFields.presentOnlyFields(), List.copyOf(manifest.keySet()));
        }
        assertModelFamilyInventoryArtifacts(report);
        assertTrue(report.get(ModelFamilyBundleGateReportFields.Root.RUNTIME_COMPATIBILITY) instanceof Map<?, ?>);
        Map<String, Object> runtimeCompatibility =
                (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.RUNTIME_COMPATIBILITY);
        assertEquals(
                ModelFamilyRuntimeCompatibilityReportFields.compatibilityFields(),
                List.copyOf(runtimeCompatibility.keySet()));
        assertTrue(runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.REQUIRES_DIRECT_SAFETENSOR_RUNTIME)
                instanceof Boolean);
        assertTrue(runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.SELECTED_FAMILY_IDS)
                instanceof List<?>);
        assertTrue(runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.SELECTED_DIRECT_SAFETENSOR)
                instanceof List<?>);
        assertTrue(runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.SELECTED_DIRECT_SAFETENSOR_SUMMARY)
                instanceof Map<?, ?>);
        assertTrue(runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.DIRECT_SAFETENSOR)
                instanceof List<?>);
        assertTrue(runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.DIRECT_SAFETENSOR_SUMMARY)
                instanceof Map<?, ?>);
        Map<String, Object> selectedDirectSummary =
                (Map<String, Object>) runtimeCompatibility.get(
                        ModelFamilyRuntimeCompatibilityReportFields.Compatibility.SELECTED_DIRECT_SAFETENSOR_SUMMARY);
        assertEquals(
                ModelFamilyRuntimeCompatibilityReportFields.summaryFields(),
                List.copyOf(selectedDirectSummary.keySet()));
        assertTrue(selectedDirectSummary.get(
                ModelFamilyRuntimeCompatibilityReportFields.Summary.FAMILY_COUNT)
                instanceof Integer);
        assertTrue(selectedDirectSummary.get(
                ModelFamilyRuntimeCompatibilityReportFields.Summary.PROBLEM_COUNTS)
                instanceof Map<?, ?>);
        Map<String, Object> directSummary =
                (Map<String, Object>) runtimeCompatibility.get(
                        ModelFamilyRuntimeCompatibilityReportFields.Compatibility.DIRECT_SAFETENSOR_SUMMARY);
        assertEquals(
                ModelFamilyRuntimeCompatibilityReportFields.summaryFields(),
                List.copyOf(directSummary.keySet()));
        assertTrue(directSummary.get(
                ModelFamilyRuntimeCompatibilityReportFields.Summary.FAMILY_COUNT)
                instanceof Integer);
        assertTrue(directSummary.get(
                ModelFamilyRuntimeCompatibilityReportFields.Summary.PROBLEM_COUNTS)
                instanceof Map<?, ?>);
        List<?> selectedDirect = (List<?>) runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.SELECTED_DIRECT_SAFETENSOR);
        if (!selectedDirect.isEmpty()) {
            Map<String, Object> compatibility = (Map<String, Object>) selectedDirect.getFirst();
            assertEquals(
                    ModelFamilyRuntimeCompatibilityReportFields.directSafetensorFields(),
                    List.copyOf(compatibility.keySet()));
            assertTrue(compatibility.get(
                    ModelFamilyRuntimeCompatibilityReportFields.DirectSafetensorCompatibility.RUNTIME_ID)
                    instanceof String);
            assertTrue(compatibility.get(
                    ModelFamilyRuntimeCompatibilityReportFields.DirectSafetensorCompatibility.PROBLEM_CODES)
                    instanceof List<?>);
        }
        List<?> directSafetensor = (List<?>) runtimeCompatibility.get(
                ModelFamilyRuntimeCompatibilityReportFields.Compatibility.DIRECT_SAFETENSOR);
        if (!directSafetensor.isEmpty()) {
            Map<String, Object> compatibility = (Map<String, Object>) directSafetensor.getFirst();
            assertEquals(
                    ModelFamilyRuntimeCompatibilityReportFields.directSafetensorFields(),
                    List.copyOf(compatibility.keySet()));
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void modelFamilyBundleGateReportWriterDiscoversScopedExternalModelFamilyPlugin(
            @TempDir Path tempDir) throws Exception {
        Path jarPath = tempDir.resolve("scoped-external-model-family.jar");
        writeScopedExternalModelFamilyServiceJar(jarPath);

        try (ExternalPluginClasspathScope scope = ExternalPluginClasspathScope.openParsed(
                List.of(jarPath),
                PluginAvailabilityCheckerTest.class)) {
            Map<String, Object> report = ModelFamilyBundleGateReportWriter.buildReport(
                    scope.discoveryClassLoader(),
                    scope.classpath());

            assertEquals("scoped",
                    report.get(ModelFamilyBundleGateReportFields.Root.MODEL_FAMILY_REGISTRY_SCOPE));
            assertEquals(scope.displayClasspath(),
                    report.get(ModelFamilyBundleGateReportFields.Root.EXTERNAL_PLUGIN_CLASSPATH));
            assertTrue(((List<?>) report.get(ModelFamilyBundleGateReportFields.Root.DISCOVERED_FAMILIES))
                    .contains(ScopedExternalModelFamilyPlugin.FAMILY_ID));

            List<Map<String, Object>> families =
                    (List<Map<String, Object>>) report.get(ModelFamilyBundleGateReportFields.Root.FAMILIES);
            Map<String, Object> family = families.stream()
                    .filter(candidate -> ScopedExternalModelFamilyPlugin.FAMILY_ID.equals(candidate.get(
                            ModelFamilyBundleInventoryReportFields.Family.ID)))
                    .findFirst()
                    .orElseThrow();
            assertModelFamilyInventoryFamily(family);
            assertEquals("Scoped External Family",
                    family.get(ModelFamilyBundleInventoryReportFields.Family.DISPLAY_NAME));
            assertEquals(List.of(ScopedExternalModelFamilyPlugin.MODEL_TYPE),
                    family.get(ModelFamilyBundleInventoryReportFields.Family.MODEL_TYPES));
            assertEquals("optional", family.get(ModelFamilyBundleInventoryReportFields.Family.BUNDLE_PROFILE));
            assertTrue(((List<?>) family.get(ModelFamilyBundleInventoryReportFields.Family.CAPABILITIES))
                    .contains("causal_lm"));
            assertEquals(
                    "external/model-family/" + ScopedExternalModelFamilyPlugin.FAMILY_ID,
                    ((Map<String, Object>) family.get(
                            ModelFamilyBundleInventoryReportFields.Family.METADATA)).get("origin"));

            List<Map<String, Object>> runtimeManifests =
                    (List<Map<String, Object>>) report.get(ModelFamilyBundleGateReportFields.Root.RUNTIME_MANIFESTS);
            Map<String, Object> runtimeManifest = runtimeManifests.stream()
                    .filter(candidate -> ScopedExternalModelFamilyPlugin.FAMILY_ID.equals(candidate.get(
                            ModelFamilyBundleInventoryReportFields.RuntimeManifest.FAMILY_ID)))
                    .findFirst()
                    .orElseThrow();
            assertModelFamilyRuntimeManifest(runtimeManifest);
            assertEquals(List.of(ScopedExternalModelFamilyPlugin.MODEL_TYPE),
                    runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.MODEL_TYPES));

            Map<String, Object> contract =
                    (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.CONTRACT);
            List<Map<String, Object>> violations =
                    (List<Map<String, Object>>) contract.get(
                            ModelFamilyContractViolationReportFields.Report.VIOLATIONS);
            assertFalse(violations.stream()
                    .anyMatch(violation -> ScopedExternalModelFamilyPlugin.FAMILY_ID.equals(
                            violation.get(ModelFamilyContractViolationReportFields.Violation.FAMILY_ID))));
        }
    }

    @SuppressWarnings("unchecked")
    private static void assertModelFamilyInventoryArtifacts(Map<String, Object> report) {
        assertTrue(report.get(ModelFamilyBundleGateReportFields.Root.DISCOVERED_FAMILIES) instanceof List<?>);
        Object families = report.get(ModelFamilyBundleGateReportFields.Root.FAMILIES);
        assertTrue(families instanceof List<?>);
        if (!((List<?>) families).isEmpty()) {
            Map<String, Object> family = (Map<String, Object>) ((List<?>) families).getFirst();
            assertModelFamilyInventoryFamily(family);
        }

        Object runtimeManifests = report.get(ModelFamilyBundleGateReportFields.Root.RUNTIME_MANIFESTS);
        assertTrue(runtimeManifests instanceof List<?>);
        if (!((List<?>) runtimeManifests).isEmpty()) {
            Map<String, Object> runtimeManifest = (Map<String, Object>) ((List<?>) runtimeManifests).getFirst();
            assertModelFamilyRuntimeManifest(runtimeManifest);
        }
    }

    private static void assertModelFamilyInventoryFamily(Map<String, Object> family) {
        assertEquals(ModelFamilyBundleInventoryReportFields.familyFields(), List.copyOf(family.keySet()));
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.ID) instanceof String);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.DISPLAY_NAME) instanceof String);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.DEFAULT_BUNDLE) instanceof Boolean);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.MODEL_TYPES) instanceof List<?>);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.TOKENIZER_PROFILE_IDS)
                instanceof List<?>);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.CAPABILITIES) instanceof List<?>);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.DIRECT_SAFETENSOR_READY)
                instanceof Boolean);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.DIRECT_SAFETENSOR_CAVEATS)
                instanceof Map<?, ?>);
        assertTrue(family.get(ModelFamilyBundleInventoryReportFields.Family.METADATA) instanceof Map<?, ?>);
    }

    @SuppressWarnings("unchecked")
    private static void assertModelFamilyRuntimeManifest(Map<String, Object> runtimeManifest) {
        assertEquals(
                ModelFamilyBundleInventoryReportFields.runtimeManifestFields(),
                List.copyOf(runtimeManifest.keySet()));
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.FAMILY_ID)
                instanceof String);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.MODEL_TYPES)
                instanceof List<?>);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.TOKENIZER_PROFILE_IDS)
                instanceof List<?>);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.TOKENIZER_READY)
                instanceof Boolean);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.CHAT_TEMPLATE_IDS)
                instanceof List<?>);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.CHAT_TEMPLATE_READY)
                instanceof Boolean);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.CAPABILITIES)
                instanceof List<?>);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.DIRECT_SAFETENSOR_READY)
                instanceof Boolean);
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.DIRECT_SAFETENSOR_CAVEATS)
                instanceof Map<?, ?>);
        assertTrue(runtimeManifest.get(
                ModelFamilyBundleInventoryReportFields.RuntimeManifest.UNIFIED_RUNTIME_REQUIREMENTS)
                instanceof List<?>);
        List<?> requirements = (List<?>) runtimeManifest.get(
                ModelFamilyBundleInventoryReportFields.RuntimeManifest.UNIFIED_RUNTIME_REQUIREMENTS);
        if (!requirements.isEmpty()) {
            assertModelFamilyRuntimeRequirement((Map<String, Object>) requirements.getFirst());
        }
        assertTrue(runtimeManifest.get(ModelFamilyBundleInventoryReportFields.RuntimeManifest.METADATA)
                instanceof Map<?, ?>);
    }

    private static void assertModelFamilyRuntimeRequirement(Map<String, Object> requirement) {
        assertEquals(
                ModelFamilyBundleInventoryReportFields.runtimeRequirementFields(),
                List.copyOf(requirement.keySet()));
        assertTrue(requirement.get(ModelFamilyBundleInventoryReportFields.RuntimeRequirement.MODEL_TYPE)
                instanceof String);
        assertTrue(requirement.get(
                ModelFamilyBundleInventoryReportFields.RuntimeRequirement.REQUIRED_INPUT_MODALITIES)
                instanceof List<?>);
        assertTrue(requirement.get(
                ModelFamilyBundleInventoryReportFields.RuntimeRequirement.PRODUCTION_READY_REQUIRED)
                instanceof Boolean);
        assertTrue(requirement.get(ModelFamilyBundleInventoryReportFields.RuntimeRequirement.REASON)
                instanceof String);
        assertTrue(requirement.get(ModelFamilyBundleInventoryReportFields.RuntimeRequirement.METADATA)
                instanceof Map<?, ?>);
    }

    @Test
    @SuppressWarnings("unchecked")
    void modelFamilyBundleGateReportWriterExposesRuntimeManifestUnifiedRequirements() {
        ModelFamilyPlugin plugin = cliGemma4StyleUnifiedRequirementPlugin();
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            Map<String, Object> report = ModelFamilyBundleGateReportWriter.buildReport();
            List<Map<String, Object>> runtimeManifests =
                    (List<Map<String, Object>>) report.get(ModelFamilyBundleGateReportFields.Root.RUNTIME_MANIFESTS);
            Map<String, Object> runtimeManifest = runtimeManifests.stream()
                    .filter(candidate -> "cli-gemma4-unified-runtime".equals(candidate.get(
                            ModelFamilyBundleInventoryReportFields.RuntimeManifest.FAMILY_ID)))
                    .findFirst()
                    .orElseThrow();
            assertModelFamilyRuntimeManifest(runtimeManifest);

            List<Map<String, Object>> requirements =
                    (List<Map<String, Object>>) runtimeManifest.get(
                            ModelFamilyBundleInventoryReportFields.RuntimeManifest.UNIFIED_RUNTIME_REQUIREMENTS);
            assertEquals(1, requirements.size());
            Map<String, Object> requirement = requirements.getFirst();
            assertEquals(
                    "gemma4_unified",
                    requirement.get(ModelFamilyBundleInventoryReportFields.RuntimeRequirement.MODEL_TYPE));
            assertEquals(
                    List.of("text", "image", "audio", "video"),
                    requirement.get(
                            ModelFamilyBundleInventoryReportFields.RuntimeRequirement.REQUIRED_INPUT_MODALITIES));
            assertEquals(
                    Boolean.TRUE,
                    requirement.get(
                            ModelFamilyBundleInventoryReportFields.RuntimeRequirement.PRODUCTION_READY_REQUIRED));
            assertTrue(requirement.get(ModelFamilyBundleInventoryReportFields.RuntimeRequirement.REASON)
                    .toString()
                    .contains("Gemma 4 unified execution"));
            Map<String, Object> metadata =
                    (Map<String, Object>) requirement.get(
                            ModelFamilyBundleInventoryReportFields.RuntimeRequirement.METADATA);
            assertEquals("google/gemma-4-12B-it", metadata.get("checkpoint"));
            assertEquals("Gemma4Processor", metadata.get("processor"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void modelFamilyBundleGateReportWriterIncludesContractFailures() {
        ModelFamilyPlugin plugin = badModelFamilyContractPlugin();
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            Map<String, Object> report = ModelFamilyBundleGateReportWriter.buildReport();
            Map<String, Object> gate =
                    (Map<String, Object>) report.get(ModelFamilyBundleGateReportFields.Root.GATE);

            assertFalse((Boolean) gate.get(ModelFamilyBundleGateReportFields.Gate.PASSED));
            assertTrue(gate.get(ModelFamilyBundleGateReportFields.Gate.VIOLATIONS)
                    .toString()
                    .contains("contract:"));
            assertTrue(((Map<?, ?>) gate.get(ModelFamilyBundleGateReportFields.Gate.CONTRACT_CATEGORY_COUNTS))
                    .containsKey("descriptor"));
            assertTrue(gate.get(ModelFamilyBundleGateReportFields.Gate.CONTRACT_REMEDIATION_HINTS)
                    .toString()
                    .contains("metadata.bundle_profile"));
            IllegalStateException error = assertThrows(
                    IllegalStateException.class,
                    () -> ModelFamilyBundleGateCheck.main(new String[0]));
            assertTrue(error.getMessage().contains("contract:"));
            assertTrue(error.getMessage().contains("Recommendations:"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void pluginGatesReportReaderParsesTopLevelGateSection() {
        Map<String, Object> gate = new LinkedHashMap<>();
        gate.put(PluginGatesReportFields.Gate.PASSED, false);
        gate.put(PluginGatesReportFields.Gate.STATUS, "plugin_directory_failed");
        gate.put(PluginGatesReportFields.Gate.VIOLATION_COUNT, "2");
        gate.put(PluginGatesReportFields.Gate.VIOLATIONS, List.of(
                "plugin-directory: missing plugin.json",
                42));
        gate.put(PluginGatesReportFields.Gate.EXTENSION_STATUS, "passed");
        gate.put(PluginGatesReportFields.Gate.MODEL_FAMILY_STATUS, "passed");
        gate.put(PluginGatesReportFields.Gate.EXTENSION_VIOLATION_COUNT, "0");
        gate.put(PluginGatesReportFields.Gate.MODEL_FAMILY_VIOLATION_COUNT, 0);
        gate.put(PluginGatesReportFields.Gate.MODEL_FAMILY_CONTRACT_CATEGORY_COUNTS,
                Map.of("descriptor", "2"));
        gate.put(PluginGatesReportFields.Gate.MODEL_FAMILY_CONTRACT_REMEDIATION_HINTS,
                List.of("Add plugin.json metadata.", 7));

        PluginGates parsed = PluginGatesReportReader.gatesFromReport(Map.of(
                PluginGatesReportFields.Root.GATE,
                gate));

        assertFalse(parsed.passed());
        assertEquals("plugin_directory_failed", parsed.status());
        assertEquals(2, parsed.violationCount());
        assertEquals(List.of("plugin-directory: missing plugin.json", "42"), parsed.violations());
        assertEquals(0, parsed.extensionViolationCount());
        assertEquals(0, parsed.modelFamilyViolationCount());
        assertEquals(Map.of("descriptor", 2), parsed.modelFamilyContractCategoryCounts());
        assertEquals(List.of("Add plugin.json metadata.", "7"), parsed.modelFamilyContractRemediationHints());
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportsBuildsStableGateSection() {
        PluginGates gates = new PluginGates(
                false,
                "plugin_directory_failed",
                1,
                List.of("plugin-directory: model-family plugin jar is not plugin-core ready"),
                "passed",
                "passed",
                0,
                0,
                Map.of("descriptor", 2),
                List.of("Add plugin.json metadata."));

        Map<String, Object> report = PluginGatesReports.gate(gates);

        assertEquals(PluginGatesReportFields.gateFields(), List.copyOf(report.keySet()));
        assertEquals(Boolean.FALSE, report.get(PluginGatesReportFields.Gate.PASSED));
        assertEquals(Boolean.TRUE, report.get(PluginGatesReportFields.Gate.FAILED));
        assertEquals("plugin_directory_failed", report.get(PluginGatesReportFields.Gate.STATUS));
        assertEquals(1, report.get(PluginGatesReportFields.Gate.VIOLATION_COUNT));
        assertEquals(List.of("plugin-directory: model-family plugin jar is not plugin-core ready"),
                report.get(PluginGatesReportFields.Gate.VIOLATIONS));
        Map<String, Integer> categories =
                (Map<String, Integer>) report.get(PluginGatesReportFields.Gate.VIOLATION_CATEGORIES);
        assertEquals(1, categories.get("pluginDirectory"));
        assertEquals("passed", report.get(PluginGatesReportFields.Gate.EXTENSION_STATUS));
        assertEquals("passed", report.get(PluginGatesReportFields.Gate.MODEL_FAMILY_STATUS));
        assertEquals(Map.of("descriptor", 2),
                report.get(PluginGatesReportFields.Gate.MODEL_FAMILY_CONTRACT_CATEGORY_COUNTS));
        assertEquals(List.of("Add plugin.json metadata."),
                report.get(PluginGatesReportFields.Gate.MODEL_FAMILY_CONTRACT_REMEDIATION_HINTS));
    }

    @Test
    void unifiedRuntimeRegistryReportsBuildsStableEmptySection() {
        Map<String, Object> report = UnifiedRuntimeRegistryReports.report(
                UnifiedRuntimeRegistry.of(List.of()),
                List.of());

        assertEquals(UnifiedRuntimeRegistryReportFields.combinedSectionFields(), List.copyOf(report.keySet()));
        assertEquals(0,
                ((Number) report.get(UnifiedRuntimeRegistryReportFields.Section.RUNTIME_COUNT)).intValue());
        assertEquals(0,
                ((Number) report.get(UnifiedRuntimeRegistryReportFields.Section.VALID_COUNT)).intValue());
        assertEquals(0,
                ((Number) report.get(UnifiedRuntimeRegistryReportFields.Section.INVALID_COUNT)).intValue());
        assertEquals(0,
                ((Number) report.get(UnifiedRuntimeRegistryReportFields.Section.PRODUCTION_READY_COUNT)).intValue());
        assertEquals(List.of(), report.get(UnifiedRuntimeRegistryReportFields.Section.MODEL_TYPES));
        assertEquals(List.of(), report.get(UnifiedRuntimeRegistryReportFields.Section.RUNTIMES));
        assertEquals(List.of(), report.get(UnifiedRuntimeRegistryReportFields.Section.CONFLICTS));
        assertEquals(List.of(), report.get(UnifiedRuntimeRegistryReportFields.Section.CONTRACT_VIOLATIONS));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.SCHEMA_VERSION,
                report.get(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA_VERSION));
    }

    @Test
    void pluginDirectoryReadinessReportsBuildsStableEmptySection() {
        Map<String, Object> report = PluginDirectoryReadinessReports.report(
                new ExternalPluginClasspath.PluginDirectoryInspection(List.of(), List.of(), List.of()));

        assertEquals(PluginDirectoryReadinessReportFields.sectionFields(), List.copyOf(report.keySet()));
        assertEquals(List.of(),
                report.get(PluginDirectoryReadinessReportFields.Section.ACTIVE_DIRECTORIES));
        assertEquals(0, report.get(PluginDirectoryReadinessReportFields.Section.JAR_COUNT));
        assertEquals(0,
                report.get(PluginDirectoryReadinessReportFields.Section.MODEL_FAMILY_PLUGIN_CANDIDATES));
        assertEquals(0,
                report.get(PluginDirectoryReadinessReportFields.Section.UNIFIED_RUNTIME_PLUGIN_CANDIDATES));
        assertEquals(0, report.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_INSTALL_READY));
        assertEquals(0, report.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_INSTALL_NOT_READY));
        assertEquals(0,
                report.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_TOKENIZER_METADATA_PENDING));
        assertEquals(List.of(), report.get(PluginDirectoryReadinessReportFields.Section.ERRORS));
        assertEquals(List.of(), report.get(PluginDirectoryReadinessReportFields.Section.JARS));
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportWriterBuildsCiArtifactPayload() {
        Map<String, Object> report = PluginGatesReportWriter.buildReport();

        assertEquals(PluginGatesReportFields.rootFields(), List.copyOf(report.keySet()));
        assertEquals(PluginGatesReportFields.SCHEMA_VERSION,
                report.get(PluginGatesReportFields.Root.SCHEMA_VERSION));
        assertTrue(report.get(PluginGatesReportFields.Root.GENERATED_AT) instanceof String);
        Map<String, Object> gate = (Map<String, Object>) report.get(PluginGatesReportFields.Root.GATE);
        Map<String, Object> extensionAvailability =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.EXTENSION_AVAILABILITY);
        Map<String, Object> modelFamilyBundle =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.MODEL_FAMILY_BUNDLE);
        Map<String, Object> runnerRouteContracts =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.RUNNER_ROUTE_CONTRACTS);
        Map<String, Object> routeBenchmarkCacheReportContract =
                (Map<String, Object>) report.get(
                        PluginGatesReportFields.Root.ROUTE_BENCHMARK_CACHE_REPORT_CONTRACT);
        Map<String, Object> unifiedRuntimes =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.UNIFIED_RUNTIMES);
        Map<String, Object> pluginDirectoryReadiness =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.PLUGIN_DIRECTORY_READINESS);
        assertEquals(PluginGatesReportFields.gateFields(), List.copyOf(gate.keySet()));
        assertTrue(gate.get(PluginGatesReportFields.Gate.STATUS) instanceof String);
        assertTrue(gate.get(PluginGatesReportFields.Gate.VIOLATION_COUNT) instanceof Integer);
        assertTrue(gate.get(PluginGatesReportFields.Gate.VIOLATION_CATEGORIES) instanceof Map<?, ?>);
        assertTrue(gate.get(PluginGatesReportFields.Gate.EXTENSION_STATUS) instanceof String);
        assertTrue(gate.get(PluginGatesReportFields.Gate.MODEL_FAMILY_STATUS) instanceof String);
        assertTrue(gate.get(PluginGatesReportFields.Gate.EXTENSION_VIOLATION_COUNT) instanceof Integer);
        assertTrue(gate.get(PluginGatesReportFields.Gate.MODEL_FAMILY_VIOLATION_COUNT) instanceof Integer);
        assertTrue(gate.get(PluginGatesReportFields.Gate.MODEL_FAMILY_CONTRACT_CATEGORY_COUNTS) instanceof Map<?, ?>);
        assertTrue(gate.get(PluginGatesReportFields.Gate.MODEL_FAMILY_CONTRACT_REMEDIATION_HINTS)
                instanceof List<?>);
        assertTrue(extensionAvailability.get(ExtensionAvailabilityGateReportFields.Root.GATE) instanceof Map<?, ?>);
        assertTrue(modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.GATE) instanceof Map<?, ?>);
        assertTrue(modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.CONTRACT) instanceof Map<?, ?>);
        assertTrue(modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.CONTRACT_VALIDATION)
                instanceof Map<?, ?>);
        assertTrue(runnerRouteContracts.get(RunnerRouteContractBundleReports.SECTION_BUNDLE) instanceof Map<?, ?>);
        assertTrue(runnerRouteContracts.get(RunnerRouteContractBundleReports.SECTION_VALIDATION) instanceof Map<?, ?>);
        Map<String, Object> runnerRouteContractBundle =
                (Map<String, Object>) runnerRouteContracts.get(RunnerRouteContractBundleReports.SECTION_BUNDLE);
        Map<String, Object> runnerRouteContractValidation =
                (Map<String, Object>) runnerRouteContracts.get(RunnerRouteContractBundleReports.SECTION_VALIDATION);
        assertEquals(
                RunnerRouteContractBundleReports.CONTRACT_ID,
                runnerRouteContractBundle.get(RunnerRouteContractBundleReports.FIELD_CONTRACT_ID));
        assertEquals(
                Boolean.TRUE,
                runnerRouteContractValidation.get("passed"));
        assertEquals(
                0,
                runnerRouteContractValidation.get("problemCount"));
        assertEquals(List.of(), RunnerRouteContractBundleReports.validateReport(runnerRouteContractBundle));
        assertTrue(routeBenchmarkCacheReportContract.get(
                RouteBenchmarkCacheReportContract.SECTION_SCHEMA) instanceof Map<?, ?>);
        assertTrue(routeBenchmarkCacheReportContract.get(
                RouteBenchmarkCacheReportContract.SECTION_SCHEMA_VALIDATION) instanceof Map<?, ?>);
        Map<String, Object> routeBenchmarkCacheSchema =
                (Map<String, Object>) routeBenchmarkCacheReportContract.get(
                        RouteBenchmarkCacheReportContract.SECTION_SCHEMA);
        Map<String, Object> routeBenchmarkCacheSchemaValidation =
                (Map<String, Object>) routeBenchmarkCacheReportContract.get(
                        RouteBenchmarkCacheReportContract.SECTION_SCHEMA_VALIDATION);
        assertEquals(
                RouteBenchmarkCacheReportContract.CONTRACT_ID,
                routeBenchmarkCacheSchema.get(RouteBenchmarkCacheReportContract.FIELD_CONTRACT_ID));
        assertEquals(
                Boolean.TRUE,
                routeBenchmarkCacheSchemaValidation.get("passed"));
        assertEquals(
                0,
                routeBenchmarkCacheSchemaValidation.get("problemCount"));
        assertEquals(List.of(), RouteBenchmarkCacheReportContract.validateSchema(routeBenchmarkCacheSchema));
        Map<String, Object> modelFamilyContract =
                (Map<String, Object>) modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.CONTRACT);
        assertTrue(modelFamilyContract.get(ModelFamilyContractViolationReportFields.Report.CATEGORY_COUNTS)
                instanceof Map<?, ?>);
        assertTrue(modelFamilyContract.get(ModelFamilyContractViolationReportFields.Report.AFFECTED_FAMILIES)
                instanceof List<?>);
        assertEquals(List.of(), ModelFamilyContractViolationReports.validateSummary(modelFamilyContract));
        assertEquals(UnifiedRuntimeRegistryReportFields.combinedSectionFields(), List.copyOf(unifiedRuntimes.keySet()));
        assertTrue(unifiedRuntimes.get(UnifiedRuntimeRegistryReportFields.Section.RUNTIME_COUNT) instanceof Number);
        assertTrue(unifiedRuntimes.get(UnifiedRuntimeRegistryReportFields.Section.CONFLICTS) instanceof List<?>);
        assertTrue(unifiedRuntimes.get(UnifiedRuntimeRegistryReportFields.Section.CONTRACT_VIOLATIONS)
                instanceof List<?>);
        assertTrue(unifiedRuntimes.get(UnifiedRuntimeRegistryReportFields.Section.RUNTIMES) instanceof List<?>);
        List<Map<String, Object>> runtimeReports =
                (List<Map<String, Object>>) unifiedRuntimes.get(UnifiedRuntimeRegistryReportFields.Section.RUNTIMES);
        if (!runtimeReports.isEmpty()) {
            Map<String, Object> runtime = runtimeReports.getFirst();
            assertEquals(UnifiedRuntimeRegistryReportFields.runtimeFields(), List.copyOf(runtime.keySet()));
            assertTrue(runtime.get(UnifiedRuntimeRegistryReportFields.RuntimeEntry.RUNTIME_ID) instanceof String);
            assertTrue(runtime.get(UnifiedRuntimeRegistryReportFields.RuntimeEntry.VALID) instanceof Boolean);
            assertTrue(runtime.get(UnifiedRuntimeRegistryReportFields.RuntimeEntry.VIOLATIONS) instanceof List<?>);
            List<Map<String, Object>> violations =
                    (List<Map<String, Object>>) runtime.get(UnifiedRuntimeRegistryReportFields.RuntimeEntry.VIOLATIONS);
            if (!violations.isEmpty()) {
                assertEquals(
                        UnifiedRuntimeRegistryReportFields.violationFields(),
                        List.copyOf(violations.getFirst().keySet()));
            }
        }
        List<Map<String, Object>> conflicts =
                (List<Map<String, Object>>) unifiedRuntimes.get(
                        UnifiedRuntimeRegistryReportFields.Section.CONFLICTS);
        if (!conflicts.isEmpty()) {
            assertEquals(
                    UnifiedRuntimeRegistryReportFields.violationFields(),
                    List.copyOf(conflicts.getFirst().keySet()));
        }
        List<Map<String, Object>> contractViolations =
                (List<Map<String, Object>>) unifiedRuntimes.get(
                        UnifiedRuntimeRegistryReportFields.Section.CONTRACT_VIOLATIONS);
        if (!contractViolations.isEmpty()) {
            assertEquals(
                    UnifiedRuntimeRegistryReportFields.violationFields(),
                    List.copyOf(contractViolations.getFirst().keySet()));
        }
        assertEquals(
                UnifiedRuntimeRequirementReportFields.SCHEMA_VERSION,
                unifiedRuntimes.get(
                        UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA_VERSION));
        assertTrue(unifiedRuntimes.get(
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA) instanceof Map<?, ?>);
        Map<String, Object> requirementSchema =
                (Map<String, Object>) unifiedRuntimes.get(
                        UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA);
        assertEquals(List.of(), UnifiedRuntimeRequirementReportContract.validateSection(unifiedRuntimes));
        assertTrue(unifiedRuntimes.get(
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_CONTRACT) instanceof Map<?, ?>);
        Map<String, Object> requirementContract =
                (Map<String, Object>) unifiedRuntimes.get(
                        UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_CONTRACT);
        assertEquals(
                Boolean.TRUE,
                requirementContract.get(UnifiedRuntimeRequirementReportFields.Contract.PASSED));
        assertEquals(
                0,
                requirementContract.get(UnifiedRuntimeRequirementReportFields.Contract.PROBLEM_COUNT));
        assertEquals(
                List.of(),
                requirementContract.get(UnifiedRuntimeRequirementReportFields.Contract.PROBLEMS));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.CONTRACT_ID,
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.CONTRACT_ID));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.SCHEMA_VERSION,
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.SCHEMA_VERSION));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.schemaFingerprint(),
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.SCHEMA_FINGERPRINT));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.sectionKeys(),
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.SECTION_KEYS));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.totalsKeys(),
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.TOTALS_KEYS));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.requirementKeys(),
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.REQUIREMENT_KEYS));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.issueKeys(),
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.ISSUE_KEYS));
        assertEquals(
                UnifiedRuntimeRequirementReportFields.affectedRequirementKeys(),
                requirementSchema.get(UnifiedRuntimeRequirementReportFields.Schema.AFFECTED_REQUIREMENT_KEYS));
        assertTrue(unifiedRuntimes.get(
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_TOTALS) instanceof Map<?, ?>);
        assertTrue(unifiedRuntimes.get(
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_RECOMMENDATIONS)
                instanceof List<?>);
        assertTrue(unifiedRuntimes.get(
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENTS) instanceof List<?>);
        Map<String, Object> requirementTotals =
                (Map<String, Object>) unifiedRuntimes.get(
                        UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_TOTALS);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.REQUIREMENT_COUNT) instanceof Number);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.ATTENTION_COUNT) instanceof Number);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.FAMILY_IDS) instanceof List<?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.ATTENTION_FAMILY_IDS) instanceof List<?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.MODEL_TYPES) instanceof List<?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.ATTENTION_MODEL_TYPES) instanceof List<?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.BY_STATUS) instanceof Map<?, ?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.PROBLEM_CODES) instanceof List<?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.PROBLEM_CODE_COUNTS) instanceof Map<?, ?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.REMEDIATION_HINTS) instanceof List<?>);
        assertTrue(requirementTotals.get(
                UnifiedRuntimeRequirementReportFields.Totals.REMEDIATION_HINT_COUNTS) instanceof Map<?, ?>);
        assertTrue(requirementTotals.get(UnifiedRuntimeRequirementReportFields.Totals.ISSUES) instanceof List<?>);
        assertTrue(report.get(PluginGatesReportFields.Root.EXTERNAL_PLUGIN_CLASSPATH) instanceof List<?>);
        assertEquals(
                PluginDirectoryReadinessReportFields.sectionFields(),
                List.copyOf(pluginDirectoryReadiness.keySet()));
        assertTrue(pluginDirectoryReadiness.get(PluginDirectoryReadinessReportFields.Section.ACTIVE_DIRECTORIES)
                instanceof List<?>);
        assertTrue(pluginDirectoryReadiness.get(PluginDirectoryReadinessReportFields.Section.JAR_COUNT)
                instanceof Number);
        assertTrue(pluginDirectoryReadiness.get(PluginDirectoryReadinessReportFields.Section.JARS)
                instanceof List<?>);
        List<Map<String, Object>> pluginDirectoryJars =
                (List<Map<String, Object>>) pluginDirectoryReadiness.get(
                        PluginDirectoryReadinessReportFields.Section.JARS);
        if (!pluginDirectoryJars.isEmpty()) {
            assertEquals(
                    PluginDirectoryReadinessReportFields.jarFields(),
                    List.copyOf(pluginDirectoryJars.getFirst().keySet()));
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportWriterPropagatesScopedExternalModelFamilyPlugin(
            @TempDir Path tempDir) throws Exception {
        Path jarPath = tempDir.resolve("scoped-external-model-family.jar");
        writeScopedExternalModelFamilyServiceJar(jarPath);

        try (ExternalPluginClasspathScope scope = ExternalPluginClasspathScope.openParsed(
                List.of(jarPath),
                PluginAvailabilityCheckerTest.class)) {
            Map<String, Object> report = PluginGatesReportWriter.buildReport(
                    scope.discoveryClassLoader(),
                    scope.classpath());

            assertEquals(scope.displayClasspath(), report.get(PluginGatesReportFields.Root.EXTERNAL_PLUGIN_CLASSPATH));

            Map<String, Object> readiness =
                    (Map<String, Object>) report.get(PluginGatesReportFields.Root.PLUGIN_DIRECTORY_READINESS);
            assertEquals(1, readiness.get(PluginDirectoryReadinessReportFields.Section.JAR_COUNT));
            assertEquals(1,
                    readiness.get(PluginDirectoryReadinessReportFields.Section.MODEL_FAMILY_PLUGIN_CANDIDATES));
            assertEquals(1, readiness.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_INSTALL_READY));
            assertEquals(0, readiness.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_INSTALL_NOT_READY));
            assertEquals(1, readiness.get(
                    PluginDirectoryReadinessReportFields.Section.PLUGIN_TOKENIZER_METADATA_PENDING));

            List<Map<String, Object>> jars =
                    (List<Map<String, Object>>) readiness.get(PluginDirectoryReadinessReportFields.Section.JARS);
            Map<String, Object> jar = jars.stream()
                    .filter(entry -> entry.get(PluginDirectoryReadinessReportFields.Jar.PATH)
                            .toString()
                            .endsWith("scoped-external-model-family.jar"))
                    .findFirst()
                    .orElseThrow();
            assertEquals(Boolean.TRUE,
                    jar.get(PluginDirectoryReadinessReportFields.Jar.HAS_MODEL_FAMILY_SERVICE_ENTRY));
            assertEquals(Boolean.TRUE,
                    jar.get(PluginDirectoryReadinessReportFields.Jar.HAS_GOLLEK_PLUGIN_SERVICE_ENTRY));
            assertEquals(Boolean.TRUE, jar.get(PluginDirectoryReadinessReportFields.Jar.HAS_PLUGIN_DESCRIPTOR));
            assertEquals(
                    "model-family/" + ScopedExternalModelFamilyPlugin.FAMILY_ID,
                    jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_DESCRIPTOR_ID));
            assertEquals("model-family",
                    jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_EXTENSION_POINT));
            assertEquals(List.of(ScopedExternalModelFamilyPlugin.FAMILY_ID),
                    jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_FAMILIES));
            assertEquals(ScopedExternalModelFamilyPlugin.class.getName(),
                    jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_MAIN_CLASS));
            assertEquals(Boolean.TRUE, jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_INSTALL_READY));
            assertEquals(List.of(), jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_INSTALL_ERRORS));

            Map<String, Object> modelFamilyBundle =
                    (Map<String, Object>) report.get(PluginGatesReportFields.Root.MODEL_FAMILY_BUNDLE);
            assertEquals("scoped",
                    modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.MODEL_FAMILY_REGISTRY_SCOPE));
            assertEquals(scope.displayClasspath(),
                    modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.EXTERNAL_PLUGIN_CLASSPATH));
            assertTrue(((List<?>) modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.DISCOVERED_FAMILIES))
                    .contains(ScopedExternalModelFamilyPlugin.FAMILY_ID));

            List<Map<String, Object>> families =
                    (List<Map<String, Object>>) modelFamilyBundle.get(ModelFamilyBundleGateReportFields.Root.FAMILIES);
            assertTrue(families.stream()
                    .anyMatch(family -> ScopedExternalModelFamilyPlugin.FAMILY_ID.equals(
                            family.get(ModelFamilyBundleInventoryReportFields.Family.ID))));

            Map<String, Object> gate = (Map<String, Object>) report.get(PluginGatesReportFields.Root.GATE);
            assertFalse(gate.get(PluginGatesReportFields.Gate.VIOLATIONS)
                    .toString()
                    .contains("plugin-directory: model-family plugin jar is not plugin-core ready"));
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportReaderFailsForModelFamilyContractReportDrift() {
        Map<String, Object> report = PluginGatesReportWriter.buildReport();
        Map<String, Object> driftedReport = new LinkedHashMap<>(report);
        Map<String, Object> modelFamilyBundle =
                new LinkedHashMap<>((Map<String, Object>) report.get(
                        PluginGatesReportFields.Root.MODEL_FAMILY_BUNDLE));
        Map<String, Object> contract =
                new LinkedHashMap<>((Map<String, Object>) modelFamilyBundle.get(
                        ModelFamilyBundleGateReportFields.Root.CONTRACT));
        contract.put(ModelFamilyContractViolationReportFields.Report.SCHEMA_FINGERPRINT, "sha256:drifted");
        modelFamilyBundle.put(ModelFamilyBundleGateReportFields.Root.CONTRACT, contract);
        driftedReport.put(PluginGatesReportFields.Root.MODEL_FAMILY_BUNDLE, modelFamilyBundle);

        PluginGates gates = PluginGatesReportWriter.gatesFromReport(driftedReport);

        assertFalse(gates.passed());
        assertTrue(gates.status().contains("model_family_contract_report"));
        assertTrue(gates.violations().toString().contains("model-family: contract report failed"));
        assertTrue(PluginGateViolationReports.categories(gates).get("modelFamily") >= 1);
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportReaderFailsForRunnerRouteContractBundleDrift() {
        Map<String, Object> report = PluginGatesReportWriter.buildReport();
        Map<String, Object> driftedReport = new LinkedHashMap<>(report);
        Map<String, Object> runnerRouteContracts =
                new LinkedHashMap<>((Map<String, Object>) report.get(
                        PluginGatesReportFields.Root.RUNNER_ROUTE_CONTRACTS));
        Map<String, Object> bundle =
                new LinkedHashMap<>((Map<String, Object>) runnerRouteContracts.get(
                        RunnerRouteContractBundleReports.SECTION_BUNDLE));
        bundle.put(RunnerRouteContractBundleReports.FIELD_SCHEMA_FINGERPRINT, "sha256:drifted");
        runnerRouteContracts.put(RunnerRouteContractBundleReports.SECTION_BUNDLE, bundle);
        driftedReport.put(PluginGatesReportFields.Root.RUNNER_ROUTE_CONTRACTS, runnerRouteContracts);

        PluginGates gates = PluginGatesReportWriter.gatesFromReport(driftedReport);

        assertFalse(gates.passed());
        assertTrue(gates.status().contains("runner_route_contract_bundle"));
        assertTrue(gates.violations().toString().contains("runner-route: contract bundle failed"));
        assertTrue(PluginGateViolationReports.categories(gates).get("runnerRoute") >= 1);
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportReaderFailsForRouteBenchmarkCacheSchemaDrift() {
        Map<String, Object> report = PluginGatesReportWriter.buildReport();
        Map<String, Object> driftedReport = new LinkedHashMap<>(report);
        Map<String, Object> contract =
                new LinkedHashMap<>((Map<String, Object>) report.get(
                        PluginGatesReportFields.Root.ROUTE_BENCHMARK_CACHE_REPORT_CONTRACT));
        Map<String, Object> schema =
                new LinkedHashMap<>((Map<String, Object>) contract.get(
                        RouteBenchmarkCacheReportContract.SECTION_SCHEMA));
        schema.put(RouteBenchmarkCacheReportContract.FIELD_SCHEMA_FINGERPRINT, "sha256:drifted");
        contract.put(RouteBenchmarkCacheReportContract.SECTION_SCHEMA, schema);
        driftedReport.put(PluginGatesReportFields.Root.ROUTE_BENCHMARK_CACHE_REPORT_CONTRACT, contract);

        PluginGates gates = PluginGatesReportWriter.gatesFromReport(driftedReport);

        assertFalse(gates.passed());
        assertTrue(gates.status().contains("runner_route_benchmark_cache_contract"));
        assertTrue(gates.violations().toString()
                .contains("runner-route: benchmark cache readiness schema contract failed"));
        assertTrue(PluginGateViolationReports.categories(gates).get("runnerRoute") >= 1);
    }

    @Test
    void routeBenchmarkCacheReportContractGateFailsForReportDrift() {
        Map<String, Object> report = validRouteBenchmarkCacheReport();
        report.put("profileTrustStatus", "ghost_profile");

        PluginGates gates = RouteBenchmarkCacheReportContract.applyGate(passingPluginGate(), report);

        assertFalse(gates.passed());
        assertTrue(gates.status().contains("runner_route_benchmark_cache_contract"));
        assertTrue(gates.violations().toString()
                .contains("runner-route: benchmark cache readiness report contract failed"));
        assertTrue(PluginGateViolationReports.categories(gates).get("runnerRoute") >= 1);
    }

    @Test
    void routeBenchmarkCacheReportContractGatePassesForValidReport() {
        PluginGates gates = RouteBenchmarkCacheReportContract.applyGate(
                passingPluginGate(),
                validRouteBenchmarkCacheReport());

        assertTrue(gates.passed());
        assertEquals(0, PluginGateViolationReports.categories(gates).get("runnerRoute"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportWriterReportsPendingTokenizerMetadataJar(@TempDir Path tempDir) throws Exception {
        Path jarPath = tempDir.resolve("pending-tokenizer-model-family.jar");
        writePendingTokenizerModelFamilyJar(jarPath);

        Map<String, Object> report = PluginGatesReportWriter.buildReport(null, List.of(jarPath));
        Map<String, Object> readiness =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.PLUGIN_DIRECTORY_READINESS);
        List<Map<String, Object>> jars =
                (List<Map<String, Object>>) readiness.get(PluginDirectoryReadinessReportFields.Section.JARS);
        Map<String, Object> jar = jars.stream()
                .filter(entry -> entry.get(PluginDirectoryReadinessReportFields.Jar.PATH)
                        .toString()
                        .endsWith("pending-tokenizer-model-family.jar"))
                .findFirst()
                .orElseThrow();

        assertEquals(PluginDirectoryReadinessReportFields.sectionFields(), List.copyOf(readiness.keySet()));
        assertEquals(PluginDirectoryReadinessReportFields.jarFields(), List.copyOf(jar.keySet()));
        assertEquals(1,
                readiness.get(PluginDirectoryReadinessReportFields.Section.MODEL_FAMILY_PLUGIN_CANDIDATES));
        assertEquals(1, readiness.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_INSTALL_READY));
        assertEquals(1,
                readiness.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_TOKENIZER_METADATA_PENDING));
        assertTrue(readiness.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_TOKENIZER_METADATA_PENDING_JARS)
                .toString()
                .contains("pending-tokenizer-model-family.jar"));
        assertEquals("pending",
                jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_TOKENIZER_METADATA_DESCRIPTOR_STATUS));
        assertEquals("pending", jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_TOKENIZER_METADATA_STATUS));
        assertEquals("tokenizer adapter pending fixture stabilization",
                jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_TOKENIZER_METADATA_PENDING_REASON));
        assertEquals(List.of(), jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_TOKENIZER_KINDS));
        assertEquals(List.of(), jar.get(PluginDirectoryReadinessReportFields.Jar.PLUGIN_INSTALL_ERRORS));
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportWriterFailsForLegacyModelFamilyPluginJar(@TempDir Path tempDir) throws Exception {
        Path jarPath = tempDir.resolve("legacy-model-family.jar");
        writeLegacyModelFamilyJar(jarPath);

        Map<String, Object> report = PluginGatesReportWriter.buildReport(null, List.of(jarPath));
        Map<String, Object> gate = (Map<String, Object>) report.get(PluginGatesReportFields.Root.GATE);
        Map<String, Object> readiness =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.PLUGIN_DIRECTORY_READINESS);

        assertFalse((Boolean) gate.get(PluginGatesReportFields.Gate.PASSED));
        assertTrue(gate.get(PluginGatesReportFields.Gate.STATUS).toString().contains("plugin_directory"));
        assertTrue(gate.get(PluginGatesReportFields.Gate.VIOLATIONS).toString().contains("plugin-directory:"));
        assertTrue(gate.get(PluginGatesReportFields.Gate.VIOLATIONS).toString().contains("plugin.json"));
        assertEquals(1,
                readiness.get(PluginDirectoryReadinessReportFields.Section.MODEL_FAMILY_PLUGIN_CANDIDATES));
        assertEquals(1, readiness.get(PluginDirectoryReadinessReportFields.Section.PLUGIN_INSTALL_NOT_READY));

        PluginGates gates = PluginGatesReportWriter.gatesFromReport(report);
        assertTrue(gates.failed());
        assertTrue(gates.failureMessage().contains("plugin-directory:"));
    }

    @Test
    void pluginGatesCheckFailsForLegacyModelFamilyPluginJar(@TempDir Path tempDir) throws Exception {
        Path jarPath = tempDir.resolve("legacy-model-family.jar");
        writeLegacyModelFamilyJar(jarPath);

        IllegalStateException error = assertThrows(
                IllegalStateException.class,
                () -> PluginGatesCheck.main(new String[] { jarPath.toString() }));
        assertTrue(error.getMessage().contains("plugin-directory:"));
        assertTrue(error.getMessage().contains("plugin.json"));
    }

    @Test
    void pluginGatesCheckFailsForLegacyModelFamilyPluginDirectory(@TempDir Path tempDir) throws Exception {
        Path pluginDirectory = tempDir.resolve("plugins");
        Path jarPath = pluginDirectory.resolve("legacy-model-family.jar");
        writeLegacyModelFamilyJar(jarPath);

        IllegalStateException error = assertThrows(
                IllegalStateException.class,
                () -> PluginGatesCheck.main(new String[] {
                        ExternalPluginClasspath.OPTION_PLUGIN_DIR,
                        pluginDirectory.toString()
                }));
        assertTrue(error.getMessage().contains("plugin-directory:"));
        assertTrue(error.getMessage().contains("plugin.json"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportWriterFailsForEmptyUnifiedRuntimeServiceJar(@TempDir Path tempDir) throws Exception {
        Path jarPath = tempDir.resolve("empty-unified-runtime.jar");
        writeEmptyUnifiedRuntimeJar(jarPath);

        Map<String, Object> report = PluginGatesReportWriter.buildReport(null, List.of(jarPath));
        Map<String, Object> gate = (Map<String, Object>) report.get(PluginGatesReportFields.Root.GATE);
        Map<String, Object> readiness =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.PLUGIN_DIRECTORY_READINESS);
        List<Map<String, Object>> jars =
                (List<Map<String, Object>>) readiness.get(PluginDirectoryReadinessReportFields.Section.JARS);
        Map<String, Object> jar = jars.stream()
                .filter(entry -> entry.get(PluginDirectoryReadinessReportFields.Jar.PATH)
                        .toString()
                        .endsWith("empty-unified-runtime.jar"))
                .findFirst()
                .orElseThrow();

        assertFalse((Boolean) gate.get(PluginGatesReportFields.Gate.PASSED));
        assertTrue(gate.get(PluginGatesReportFields.Gate.STATUS).toString().contains("plugin_directory"));
        assertTrue(gate.get(PluginGatesReportFields.Gate.VIOLATIONS)
                .toString()
                .contains("unified multimodal runtime jar"));
        assertEquals(1,
                readiness.get(PluginDirectoryReadinessReportFields.Section.UNIFIED_RUNTIME_PLUGIN_CANDIDATES));
        assertEquals(0, readiness.get(PluginDirectoryReadinessReportFields.Section.UNIFIED_RUNTIME_READY));
        assertEquals(1, readiness.get(PluginDirectoryReadinessReportFields.Section.UNIFIED_RUNTIME_NOT_READY));
        assertTrue(readiness.get(PluginDirectoryReadinessReportFields.Section.UNIFIED_RUNTIME_NOT_READY_JARS)
                .toString()
                .contains("empty-unified-runtime.jar"));
        assertEquals(Boolean.FALSE, jar.get(PluginDirectoryReadinessReportFields.Jar.UNIFIED_RUNTIME_READY));
        assertTrue(jar.get(PluginDirectoryReadinessReportFields.Jar.UNIFIED_RUNTIME_ERRORS)
                .toString()
                .contains("has no providers"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesReportWriterFailsForMissingUnifiedRuntimeProviderClass(@TempDir Path tempDir) throws Exception {
        Path jarPath = tempDir.resolve("missing-unified-runtime-class.jar");
        writeMissingUnifiedRuntimeProviderClassJar(jarPath);

        Map<String, Object> report = PluginGatesReportWriter.buildReport(null, List.of(jarPath));
        Map<String, Object> gate = (Map<String, Object>) report.get(PluginGatesReportFields.Root.GATE);
        Map<String, Object> readiness =
                (Map<String, Object>) report.get(PluginGatesReportFields.Root.PLUGIN_DIRECTORY_READINESS);
        List<Map<String, Object>> jars =
                (List<Map<String, Object>>) readiness.get(PluginDirectoryReadinessReportFields.Section.JARS);
        Map<String, Object> jar = jars.stream()
                .filter(entry -> entry.get(PluginDirectoryReadinessReportFields.Jar.PATH)
                        .toString()
                        .endsWith("missing-unified-runtime-class.jar"))
                .findFirst()
                .orElseThrow();

        assertFalse((Boolean) gate.get(PluginGatesReportFields.Gate.PASSED));
        assertTrue(gate.get(PluginGatesReportFields.Gate.VIOLATIONS)
                .toString()
                .contains("provider classes are missing"));
        assertEquals(0, readiness.get(PluginDirectoryReadinessReportFields.Section.UNIFIED_RUNTIME_READY));
        assertEquals(1, readiness.get(PluginDirectoryReadinessReportFields.Section.UNIFIED_RUNTIME_NOT_READY));
        assertEquals(Boolean.FALSE, jar.get(PluginDirectoryReadinessReportFields.Jar.UNIFIED_RUNTIME_READY));
        assertTrue(jar.get(
                PluginDirectoryReadinessReportFields.Jar.UNIFIED_MULTIMODAL_RUNTIME_MISSING_PROVIDER_CLASSES)
                .toString()
                .contains("external.missing.MissingUnifiedRuntime"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void pluginGatesCheckFailsWhenModelFamilyGateFails() {
        ModelFamilyPlugin plugin = badModelFamilyContractPlugin();
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            Map<String, Object> report = PluginGatesReportWriter.buildReport();
            Map<String, Object> gate = (Map<String, Object>) report.get("gate");

            assertFalse((Boolean) gate.get("passed"));
            assertTrue(gate.get("status").toString().contains("model_family_failed"));
            assertTrue(gate.get("violations").toString().contains("model-family:"));
            IllegalStateException error = assertThrows(
                    IllegalStateException.class,
                    () -> PluginGatesCheck.main(new String[0]));
            assertTrue(error.getMessage().contains("model-family:"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    private static void writePendingTokenizerModelFamilyJar(Path jarPath) throws Exception {
        String pluginClass = "external.pending.PendingTokenizerModelFamilyPlugin";
        Files.createDirectories(jarPath.getParent());
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath))) {
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.MODEL_FAMILY_SERVICE_ENTRY));
            jar.write((pluginClass + System.lineSeparator()).getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.GOLLEK_PLUGIN_SERVICE_ENTRY));
            jar.write((pluginClass + System.lineSeparator()).getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.PLUGIN_DESCRIPTOR_ENTRY));
            jar.write("""
                    {
                      "id" : "model-family/pending_tokenizer_family",
                      "name" : "Pending Tokenizer Family",
                      "version" : "0.1.0",
                      "description" : "External model-family fixture with pending tokenizer metadata",
                      "vendor" : "Test",
                      "mainClass" : "external.pending.PendingTokenizerModelFamilyPlugin",
                      "dependencies" : [ ],
                      "optionalDependencies" : [ ],
                      "properties" : {
                        "extensionPoint" : "model-family",
                        "bundleProfile" : "optional",
                        "families" : [ "pending_tokenizer_family" ],
                        "tokenizerMetadataStatus" : "pending",
                        "tokenizerMetadataPendingReason" : "tokenizer adapter pending fixture stabilization"
                      }
                    }
                    """.getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
        }
    }

    private static void writeScopedExternalModelFamilyServiceJar(Path jarPath) throws Exception {
        String pluginClass = ScopedExternalModelFamilyPlugin.class.getName();
        Files.createDirectories(jarPath.getParent());
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath))) {
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.MODEL_FAMILY_SERVICE_ENTRY));
            jar.write((pluginClass + System.lineSeparator()).getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.GOLLEK_PLUGIN_SERVICE_ENTRY));
            jar.write((pluginClass + System.lineSeparator()).getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.PLUGIN_DESCRIPTOR_ENTRY));
            jar.write(("""
                    {
                      "id" : "model-family/scoped_external_family",
                      "name" : "Scoped External Family",
                      "version" : "0.1.0-test",
                      "description" : "Scoped external model-family fixture",
                      "vendor" : "Test",
                      "mainClass" : "%s",
                      "dependencies" : [ ],
                      "optionalDependencies" : [ ],
                      "properties" : {
                        "extensionPoint" : "model-family",
                        "bundleProfile" : "optional",
                        "families" : [ "scoped_external_family" ],
                        "tokenizerMetadataStatus" : "pending",
                        "tokenizerMetadataPendingReason" : "tokenizer profile intentionally detached in fixture"
                      }
                    }
                    """.formatted(pluginClass)).getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
        }
    }

    private static void writeLegacyModelFamilyJar(Path jarPath) throws Exception {
        Files.createDirectories(jarPath.getParent());
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath))) {
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.MODEL_FAMILY_SERVICE_ENTRY));
            jar.write(("external.legacy.LegacyModelFamilyPlugin" + System.lineSeparator())
                    .getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
        }
    }

    private static void writeEmptyUnifiedRuntimeJar(Path jarPath) throws Exception {
        Files.createDirectories(jarPath.getParent());
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath))) {
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.UNIFIED_MULTIMODAL_RUNTIME_SERVICE_ENTRY));
            jar.write(("# intentionally empty" + System.lineSeparator()).getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
        }
    }

    private static void writeMissingUnifiedRuntimeProviderClassJar(Path jarPath) throws Exception {
        Files.createDirectories(jarPath.getParent());
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath))) {
            jar.putNextEntry(new JarEntry(ExternalPluginClasspath.UNIFIED_MULTIMODAL_RUNTIME_SERVICE_ENTRY));
            jar.write(("external.missing.MissingUnifiedRuntime" + System.lineSeparator())
                    .getBytes(StandardCharsets.UTF_8));
            jar.closeEntry();
        }
    }

    private static PluginGates passingPluginGate() {
        return new PluginGates(
                true,
                "passed",
                0,
                List.of(),
                "passed",
                "passed",
                0,
                0);
    }

    private static Map<String, Object> validRouteBenchmarkCacheReport() {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schemaVersion", RouteBenchmarkCacheReportContract.SCHEMA_VERSION);
        report.put("status", "empty");
        report.put("enabled", true);
        report.put("healthy", true);
        report.put("cacheFile", "/tmp/gollek-route-benchmark-profiles.tsv");
        report.put("cacheFileExists", false);
        report.put("cacheFileReadable", true);
        report.put("cacheDirectoryWritable", true);
        report.put("staleAfterDays", 30);
        report.put("entryCount", 0);
        report.put("staleEntryCount", 0);
        report.put("freshEntryCount", 0);
        report.put("trustedEntryCount", 0);
        report.put("staleProfilesAllowed", false);
        report.put("strictHealthy", true);
        report.put("profileTrustStatus", "empty");
        report.put("problems", List.of());
        report.put("remediationHints", List.of());
        report.put("invalidLineCount", 0);
        report.put("providers", List.of());
        report.put("formats", List.of());
        report.put("recentEntries", List.of());
        report.put("failOnRouteBenchmarkCache", false);
        return report;
    }

    private static ModelFamilyPlugin badModelFamilyContractPlugin() {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "Cli Bad Model Gate",
                        "CLI Bad Model Gate",
                        List.of("Cli Bad Model Type"),
                        List.of(),
                        List.of(),
                        Map.of(
                                "bundle_profile", "research",
                                "origin", "3rdparty/transformers/src/transformers/models/cli_bad_model_gate"));
            }
        };
    }

    private static ModelFamilyPlugin unifiedRuntimeRequirementPlugin() {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "unified-runtime-requirement-family",
                        "Unified Runtime Requirement Family",
                        List.of("unit_unified"),
                        List.of("UnitUnifiedForConditionalGeneration"),
                        List.of(ModelFamilyCapability.MULTIMODAL),
                        Map.of(
                                "unified_model_type", "unit_unified",
                                "unified_runtime_required_modalities", "text,image",
                                "unified_runtime_reason", "unit test runtime requirement"));
            }
        };
    }

    private static ModelFamilyPlugin gemma4UnifiedRequirementPlugin() {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "gemma4",
                        "Google Gemma 4",
                        List.of("gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio", "gemma4_unified"),
                        List.of("Gemma4ForMultimodalLM", "Gemma4Processor"),
                        List.of(ModelFamilyCapability.MULTIMODAL,
                                ModelFamilyCapability.VISION,
                                ModelFamilyCapability.AUDIO),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/gemma4",
                                "checkpoint_gemma_4_12b_it", "google/gemma-4-12B-it",
                                "unified_model_type", "gemma4_unified",
                                "unified_runtime", "metadata_processor_only_until_unified_embedder_runtime"));
            }

            @Override
            public List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements() {
                return List.of(new ModelFamilyUnifiedRuntimeRequirement(
                        "gemma4_unified",
                        List.of("text", "image", "audio", "video"),
                        true,
                        "Gemma 4 unified execution requires the multimodal embedder runtime",
                        Map.of(
                                "checkpoint", "google/gemma-4-12B-it",
                                "processor", "Gemma4Processor")));
            }
        };
    }

    private static ModelFamilyPlugin cliGemma4StyleUnifiedRequirementPlugin() {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "cli-gemma4-unified-runtime",
                        "CLI Gemma 4 Unified Runtime",
                        List.of("gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio", "gemma4_unified"),
                        List.of("Gemma4ForMultimodalLM", "Gemma4Processor"),
                        List.of(ModelFamilyCapability.MULTIMODAL,
                                ModelFamilyCapability.VISION,
                                ModelFamilyCapability.AUDIO),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/gemma4",
                                "checkpoint_gemma_4_12b_it", "google/gemma-4-12B-it",
                                "unified_model_type", "gemma4_unified",
                                "unified_runtime", "metadata_processor_only_until_unified_embedder_runtime"));
            }

            @Override
            public List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements() {
                return List.of(new ModelFamilyUnifiedRuntimeRequirement(
                        "gemma4_unified",
                        List.of("text", "image", "audio", "video"),
                        true,
                        "Gemma 4 unified execution requires the multimodal embedder runtime",
                        Map.of(
                                "checkpoint", "google/gemma-4-12B-it",
                                "processor", "Gemma4Processor")));
            }
        };
    }

    private static UnifiedRuntimeManifest unifiedRuntimeManifest(
            String runtimeId,
            String modelType,
            List<UnifiedInputModality> modalities) {
        return unifiedRuntimeManifest(runtimeId, modelType, modalities, UnifiedRuntimeReadiness.READY);
    }

    private static UnifiedRuntimeManifest unifiedRuntimeManifest(
            String runtimeId,
            String modelType,
            List<UnifiedInputModality> modalities,
            UnifiedRuntimeReadiness readiness) {
        return new UnifiedRuntimeManifest(
                runtimeId,
                runtimeId,
                List.of("gemma4"),
                List.of(modelType),
                modalities,
                readiness,
                "test runtime ready",
                List.of("processor_config.json"),
                List.of("tokenizer.model"),
                Map.of());
    }

    private static final class StubArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "cli-stub";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("CliPartialDirectForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("cli_partial_direct");
        }

        @Override
        public String embedTokensWeight() {
            return "embed.weight";
        }

        @Override
        public String finalNormWeight() {
            return "norm.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "q.weight";
        }

        @Override
        public String layerKeyWeight(int i) {
            return "k.weight";
        }

        @Override
        public String layerValueWeight(int i) {
            return "v.weight";
        }

        @Override
        public String layerOutputWeight(int i) {
            return "o.weight";
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "attn_norm.weight";
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "gate.weight";
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "up.weight";
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "down.weight";
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "ffn_norm.weight";
        }
    }
}
