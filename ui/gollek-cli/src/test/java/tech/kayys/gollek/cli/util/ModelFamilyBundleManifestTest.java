package tech.kayys.gollek.cli.util;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyContractValidator;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyContractViolation;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyResolution;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelFamilyBundleManifestTest {

    @Test
    void parsesGeneratedBundleMetadata() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:abc123");
        properties.setProperty("selectors", "core,direct,gemma,all,custom_selector");
        properties.setProperty("families", "gemma,llama,phi");
        properties.setProperty("familyCount", "3");
        properties.setProperty("profiles", "core,optional");
        properties.setProperty("availableFamilies", "bert,flava,gemma,llama,phi");
        properties.setProperty("availableProfiles", "core,metadata_only,optional");
        properties.setProperty("availableSelectors", "core,direct,gemma,llama,optional");
        properties.setProperty("tokenizerMetadataPendingFamilies", "kimi");
        properties.setProperty("family.kimi.tokenizerMetadataPendingReason",
                "tokenizer adapter pending descriptor stabilization");
        properties.setProperty("selectorSource", "explicit");
        properties.setProperty("explicitSelectors", "direct,gemma");
        properties.setProperty("presetSelectors", "direct");
        properties.setProperty("defaultSelectors", "core");
        properties.setProperty("policySource", "mixed");
        properties.setProperty("presetRequiredAliases", "direct");
        properties.setProperty("presetForbiddenAliases", "embedding");
        properties.setProperty("explicitRequiredFamilies", "qwen");
        properties.setProperty("requiredFamilies", "qwen");
        properties.setProperty("requiredAliases", "direct");
        properties.setProperty("forbiddenAliases", "embedding");
        properties.setProperty("policyPassed", "false");
        properties.setProperty("policyViolationCount", "2");
        properties.setProperty("missingRequiredFamilies", "qwen");
        properties.setProperty("selectedForbiddenAlias.embedding.families", "bert");
        properties.setProperty("fixtureRequiredSelectors", "core,direct");
        properties.setProperty("fixtureRequiredFamilies", "gemma,phi");
        properties.setProperty("fixturePassed", "false");
        properties.setProperty("fixtureRequiredFingerprint", "sha256:required");
        properties.setProperty("fixtureInventoryFingerprint", "sha256:inventory");
        properties.setProperty("fixtureAvailableFamilyCount", "5");
        properties.setProperty("fixtureFamilyCount", "3");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "1");
        properties.setProperty("fixtureMissingRequiredCount", "1");
        properties.setProperty("fixtureProblemFamilyCount", "2");
        properties.setProperty("fixtureMissingRequiredFamilies", "phi");
        properties.setProperty("fixtureProblemFamilies", "bert,phi");
        properties.setProperty("bundlePreset", "prod_llm");
        properties.setProperty("productionReadinessPassed", "true");
        properties.setProperty("productionReadinessPendingCount", "0");
        properties.setProperty("productionReadinessPendingFamilies", "");
        properties.setProperty("directSafetensorReadinessPassed", "true");
        properties.setProperty("directSafetensorPendingCount", "0");
        properties.setProperty("directSafetensorPendingFamilies", "");
        properties.setProperty("availableBundlePresets", "prod_llm,research_all");
        properties.setProperty("bundlePresets", "prod_llm,research_all");
        properties.setProperty("bundlePreset.prod_llm.description", "Lean production LLM");
        properties.setProperty("bundlePreset.prod_llm.selectors", "direct");
        properties.setProperty("bundlePreset.prod_llm.requiredAliases", "direct");
        properties.setProperty("bundlePreset.prod_llm.forbiddenAliases", "embedding");
        properties.setProperty("bundlePreset.prod_llm.selectedFamilies", "gemma,llama,phi");
        properties.setProperty("bundlePreset.prod_llm.selectedCount", "3");
        properties.setProperty("bundlePreset.prod_llm.productionTokenizerMetadataRequired", "true");
        properties.setProperty("bundlePreset.prod_llm.productionTokenizerMetadataReady", "true");
        properties.setProperty("bundlePreset.prod_llm.productionSafetyPassed", "true");
        properties.setProperty("bundlePreset.prod_llm.productionSafetyViolationCount", "0");
        properties.setProperty("bundlePreset.prod_llm.pendingTokenizerFamilies", "");
        properties.setProperty("bundlePreset.prod_llm.policyPassed", "true");
        properties.setProperty("bundlePreset.prod_llm.policyViolationCount", "0");
        properties.setProperty("bundlePreset.prod_llm.missingRequiredCount", "0");
        properties.setProperty("bundlePreset.prod_llm.selectedForbiddenCount", "0");
        properties.setProperty("bundlePreset.prod_llm.missingRequiredAliasCount", "0");
        properties.setProperty("bundlePreset.prod_llm.selectedForbiddenAliasCount", "0");
        properties.setProperty("bundlePreset.research_all.description", "Everything");
        properties.setProperty("bundlePreset.research_all.selectors", "all");
        properties.setProperty("bundlePreset.research_all.selectedFamilies", "bert,flava,gemma,llama,phi");
        properties.setProperty("bundlePreset.research_all.selectedCount", "5");
        properties.setProperty("bundlePreset.research_all.productionTokenizerMetadataRequired", "false");
        properties.setProperty("bundlePreset.research_all.productionTokenizerMetadataReady", "false");
        properties.setProperty("bundlePreset.research_all.productionSafetyPassed", "true");
        properties.setProperty("bundlePreset.research_all.productionSafetyViolationCount", "0");
        properties.setProperty("bundlePreset.research_all.policyPassed", "false");
        properties.setProperty("bundlePreset.research_all.policyViolationCount", "1");
        properties.setProperty("bundlePreset.research_all.missingRequiredAliases", "direct");
        properties.setProperty("bundlePreset.research_all.missingRequiredCount", "0");
        properties.setProperty("bundlePreset.research_all.selectedForbiddenCount", "0");
        properties.setProperty("bundlePreset.research_all.missingRequiredAliasCount", "1");
        properties.setProperty("bundlePreset.research_all.selectedForbiddenAliasCount", "0");
        properties.setProperty("bundlePreset.research_all.missingRequiredAlias.direct.families", "qwen");
        properties.setProperty("bundleAliases", "direct,vlm");
        properties.setProperty("bundleAlias.direct.description", "Lean direct runtime");
        properties.setProperty("bundleAlias.direct.families", "gemma,llama,phi");
        properties.setProperty("bundleAlias.direct.familyCount", "3");
        properties.setProperty("bundleAlias.vlm.description", "Vision-language metadata");
        properties.setProperty("bundleAlias.vlm.families", "flava,gemma");
        properties.setProperty("bundleAlias.vlm.familyCount", "2");
        properties.setProperty("family.bert.selected", "false");
        properties.setProperty("family.bert.profile", "metadata_only");
        properties.setProperty("family.flava.selected", "false");
        properties.setProperty("family.flava.profile", "metadata_only");
        properties.setProperty("family.gemma.selected", "true");
        properties.setProperty("family.gemma.profile", "core");
        properties.setProperty("family.gemma.path", ":models:gollek-model-gemma");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);

        assertTrue(manifest.present());
        assertEquals(ModelFamilyBundleManifest.CURRENT_SCHEMA_VERSION, manifest.schemaVersion());
        assertEquals("sha256:abc123", manifest.displayFingerprint());
        assertFalse(manifest.detached());
        assertEquals(List.of("core", "direct", "gemma", "all", "custom_selector"), manifest.selectors());
        assertEquals(List.of("gemma", "llama", "phi"), manifest.families());
        assertEquals("core, direct, gemma, all, custom_selector", manifest.joinedSelectors());
        assertEquals(List.of("direct"), manifest.requestedAliases());
        assertEquals(List.of("core"), manifest.requestedProfiles());
        assertEquals(List.of("gemma"), manifest.requestedFamilies());
        assertEquals(List.of("all"), manifest.reservedSelectors());
        assertEquals(List.of("custom_selector"), manifest.unknownSelectors());
        assertEquals(List.of("core", "direct", "gemma", "llama", "optional"), manifest.availableSelectors());
        assertEquals("core, direct, gemma, llama, optional", manifest.joinedAvailableSelectors());
        assertEquals(List.of("kimi"), manifest.tokenizerMetadataPendingFamilies());
        assertEquals(Map.of("kimi", "tokenizer adapter pending descriptor stabilization"),
                manifest.tokenizerMetadataPendingReasons());
        assertEquals("explicit", manifest.selectorSource());
        assertEquals("explicit -Pgollek.modelFamilies", manifest.displaySelectorSource());
        assertEquals(List.of("direct", "gemma"), manifest.explicitSelectors());
        assertEquals(List.of("direct"), manifest.presetSelectors());
        assertEquals(List.of("core"), manifest.defaultSelectors());
        assertEquals("mixed", manifest.policySource());
        assertEquals("mixed preset+explicit", manifest.displayPolicySource());
        assertEquals(List.of("direct"), manifest.presetRequiredAliases());
        assertEquals(List.of("embedding"), manifest.presetForbiddenAliases());
        assertEquals(List.of("qwen"), manifest.explicitRequiredFamilies());
        assertEquals(List.of("qwen"), manifest.bundlePolicy().requiredFamilies());
        assertEquals(List.of("direct"), manifest.bundlePolicy().requiredAliases());
        assertEquals(List.of("embedding"), manifest.bundlePolicy().forbiddenAliases());
        assertEquals(Boolean.FALSE, manifest.bundlePolicy().passed());
        assertEquals("failed", manifest.bundlePolicy().statusLabel());
        assertEquals("failed (2 violation(s))", manifest.displayBundlePolicyStatus());
        assertEquals(List.of("qwen"), manifest.bundlePolicy().violations().missingRequiredFamilies());
        assertEquals(List.of("bert"),
                manifest.bundlePolicy().violations().selectedForbiddenAliases().get("embedding"));
        assertEquals(List.of("core", "direct"), manifest.fixtureStatus().requiredSelectors());
        assertEquals(List.of("gemma", "phi"), manifest.fixtureStatus().requiredFamilies());
        assertEquals(Boolean.FALSE, manifest.fixtureStatus().passed());
        assertEquals("failed", manifest.fixtureStatus().statusLabel());
        assertEquals("failed (1/2 required, 1 missing, 2 problem)", manifest.displayFixtureStatus());
        assertEquals("sha256:required", manifest.fixtureStatus().requiredFingerprint());
        assertEquals("sha256:inventory", manifest.fixtureStatus().inventoryFingerprint());
        assertEquals(5, manifest.fixtureStatus().availableFamilyCount());
        assertEquals(3, manifest.fixtureStatus().fixtureFamilyCount());
        assertEquals(2, manifest.fixtureStatus().requiredFamilyCount());
        assertEquals(1, manifest.fixtureStatus().requiredPassedCount());
        assertEquals(1, manifest.fixtureStatus().missingRequiredCount());
        assertEquals(2, manifest.fixtureStatus().problemFamilyCount());
        assertEquals(List.of("phi"), manifest.fixtureStatus().missingRequiredFamilies());
        assertEquals(List.of("bert", "phi"), manifest.fixtureStatus().problemFamilies());
        assertEquals("prod_llm", manifest.bundlePreset());
        assertTrue(manifest.requiresDirectSafetensorRuntime());
        assertEquals(Boolean.TRUE, manifest.productionReadinessPassed());
        assertTrue(manifest.productionReadinessStatusKnown());
        assertEquals("passed", manifest.productionReadinessStatusLabel());
        assertEquals(0, manifest.productionReadinessPendingCount());
        assertEquals(List.of(), manifest.productionReadinessPendingFamilies());
        assertEquals(Boolean.TRUE, manifest.directSafetensorReadinessPassed());
        assertTrue(manifest.directSafetensorReadinessStatusKnown());
        assertEquals("passed", manifest.directSafetensorReadinessStatusLabel());
        assertEquals(0, manifest.directSafetensorPendingCount());
        assertEquals(List.of(), manifest.directSafetensorPendingFamilies());
        assertTrue(manifest.catalogReadinessPassed());
        assertEquals("passed", manifest.catalogReadinessStatusLabel());
        assertEquals("passed (0 production pending, 0 direct SafeTensor pending)",
                manifest.displayCatalogReadinessStatus());
        assertTrue(manifest.productionTokenizerMetadataRequired());
        assertTrue(manifest.productionTokenizerMetadataReady());
        assertTrue(manifest.productionSafetyPassed());
        assertEquals("passed", manifest.productionSafetyStatusLabel());
        assertEquals("passed (tokenizer metadata ready)", manifest.displayProductionSafetyStatus());
        assertEquals(List.of(), manifest.selectedTokenizerMetadataPendingFamilies());
        assertTrue(manifest.hasBundlePreset());
        assertEquals("prod_llm - Lean production LLM", manifest.displayBundlePreset());
        assertTrue(manifest.activeBundlePreset().isPresent());
        assertEquals("prod_llm", manifest.activeBundlePreset().get().id());
        assertEquals("passed (0 violation(s), 3 selected families)", manifest.displayBundlePresetPolicyStatus());
        ModelFamilyBundleManifest.BundlePresetConformance conformance = manifest.activeBundlePresetConformance();
        assertEquals("prod_llm", conformance.presetId());
        assertTrue(conformance.presetMetadataPresent());
        assertFalse(conformance.selectorsMatch());
        assertFalse(conformance.policyInputsMatch());
        assertTrue(conformance.explicitSelectorOverride());
        assertTrue(conformance.explicitPolicyOverride());
        assertEquals(List.of("core", "gemma", "all", "custom_selector"), conformance.selectorAdditions());
        assertEquals(List.of("qwen"), conformance.requiredFamilyAdditions());
        assertEquals("drifted", conformance.statusLabel());
        assertTrue(manifest.displayActiveBundlePresetConformance().contains("drifted"));
        assertEquals(List.of("prod_llm", "research_all"), manifest.availableBundlePresets());
        assertEquals("prod_llm, research_all", manifest.joinedAvailableBundlePresets());
        assertEquals(2, manifest.bundlePresets().size());
        assertEquals("prod_llm", manifest.bundlePresets().get(0).id());
        assertEquals(List.of("direct"), manifest.bundlePresets().get(0).selectors());
        assertEquals(List.of("direct"), manifest.bundlePresets().get(0).requiredAliases());
        assertEquals(List.of("embedding"), manifest.bundlePresets().get(0).forbiddenAliases());
        assertEquals(List.of("gemma", "llama", "phi"), manifest.bundlePresets().get(0).selectedFamilies());
        assertEquals(3, manifest.bundlePresets().get(0).selectedCount());
        assertEquals(Boolean.TRUE, manifest.bundlePresets().get(0).policyPassed());
        assertTrue(manifest.bundlePresets().get(0).policyStatusKnown());
        assertEquals("passed", manifest.bundlePresets().get(0).policyStatusLabel());
        assertEquals(0, manifest.bundlePresets().get(0).policyViolationCount());
        assertEquals(Boolean.TRUE, manifest.bundlePresets().get(0).productionTokenizerMetadataRequired());
        assertEquals(Boolean.TRUE, manifest.bundlePresets().get(0).productionTokenizerMetadataReady());
        assertEquals(Boolean.TRUE, manifest.bundlePresets().get(0).productionSafetyPassed());
        assertEquals("passed", manifest.bundlePresets().get(0).productionSafetyStatusLabel());
        assertEquals(0, manifest.bundlePresets().get(0).productionSafetyViolationCount());
        assertEquals(List.of(), manifest.bundlePresets().get(0).pendingTokenizerFamilies());
        assertEquals(Boolean.FALSE, manifest.bundlePresets().get(1).policyPassed());
        assertEquals(Boolean.FALSE, manifest.bundlePresets().get(1).productionTokenizerMetadataRequired());
        assertEquals(Boolean.TRUE, manifest.bundlePresets().get(1).productionSafetyPassed());
        assertEquals(List.of("qwen"),
                manifest.bundlePresets().get(1).policyViolations().missingRequiredAliases().get("direct"));
        assertTrue(manifest.joinedBundlePresets().contains("prod_llm(selectors=direct, selected=3, policy=passed"));
        assertEquals(2, manifest.bundleAliases().size());
        assertEquals("direct", manifest.bundleAliases().get(0).id());
        assertEquals("Lean direct runtime", manifest.bundleAliases().get(0).description());
        assertEquals(List.of("gemma", "llama", "phi"), manifest.bundleAliases().get(0).families());
        assertEquals("direct(3 families), vlm(2 families)", manifest.joinedBundleAliases());
        assertEquals("direct(3/3 complete)", manifest.joinedCompleteBundleAliases());
        assertEquals("vlm(1/2 partial)", manifest.joinedPartialBundleAliases());
        assertTrue(manifest.isFamilySelected("model-family/gemma"));
        assertFalse(manifest.isFamilySelected("bert"));
        assertEquals(List.of("bert", "flava"), manifest.omittedFamilies());
        assertEquals(List.of("bert[metadata_only]", "flava[metadata_only]"), manifest.omittedFamiliesWithProfiles());
        assertEquals("bert[metadata_only], flava[metadata_only]", manifest.joinedOmittedFamiliesWithProfiles());
        assertEquals("core", manifest.familyProfiles().get("gemma"));
        assertEquals(":models:gollek-model-gemma", manifest.familyPaths().get("gemma"));
        assertEquals(List.of(), manifest.countConsistencyProblems());
    }

    @Test
    void nonProductionPresetCanExposePendingTokenizersWithoutProductionSafetyViolations() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("bundlePresets", "research_all");
        properties.setProperty("bundlePreset.research_all.description", "Research model farm");
        properties.setProperty("bundlePreset.research_all.selectors", "all");
        properties.setProperty("bundlePreset.research_all.selectedFamilies", "gemma,kimi");
        properties.setProperty("bundlePreset.research_all.selectedCount", "2");
        properties.setProperty("bundlePreset.research_all.productionTokenizerMetadataRequired", "false");
        properties.setProperty("bundlePreset.research_all.productionTokenizerMetadataReady", "false");
        properties.setProperty("bundlePreset.research_all.productionSafetyPassed", "true");
        properties.setProperty("bundlePreset.research_all.productionSafetyViolationCount", "0");
        properties.setProperty("bundlePreset.research_all.pendingTokenizerFamilies", "kimi");
        properties.setProperty("bundlePreset.research_all.policyPassed", "true");
        properties.setProperty("bundlePreset.research_all.policyViolationCount", "0");

        ModelFamilyBundleManifest.BundlePreset preset =
                ModelFamilyBundleManifest.fromProperties(properties).bundlePresets().getFirst();

        assertEquals(0, preset.productionSafetyViolationCount());
        assertEquals(List.of("kimi"), preset.pendingTokenizerFamilies());
        assertEquals(List.of(), preset.countConsistencyProblems());
        assertTrue(preset.productionSafetyCompactStatus().contains("0 production safety violation(s)"));
        assertTrue(preset.productionSafetyCompactStatus().contains("1 pending tokenizer family(s)"));
    }

    @Test
    void reportsGeneratedCountMetadataDrift() {
        Properties properties = new Properties();
        properties.setProperty("families", "gemma,llama");
        properties.setProperty("familyCount", "3");
        properties.setProperty("productionReadinessPendingFamilies", "phi,qwen");
        properties.setProperty("productionReadinessPendingCount", "1");
        properties.setProperty("directSafetensorPendingFamilies", "phi");
        properties.setProperty("directSafetensorPendingCount", "0");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("missingRequiredFamilies", "qwen");
        properties.setProperty("fixtureRequiredFamilies", "gemma,llama");
        properties.setProperty("fixtureRequiredFamilyCount", "1");
        properties.setProperty("fixtureMissingRequiredFamilies", "qwen");
        properties.setProperty("fixtureMissingRequiredCount", "0");
        properties.setProperty("fixtureProblemFamilies", "gemma,llama");
        properties.setProperty("fixtureProblemFamilyCount", "1");
        properties.setProperty("bundlePresets", "prod_llm");
        properties.setProperty("bundlePreset.prod_llm.selectedFamilies", "gemma,llama");
        properties.setProperty("bundlePreset.prod_llm.selectedCount", "1");
        properties.setProperty("bundlePreset.prod_llm.missingRequiredFamilies", "qwen");
        properties.setProperty("bundlePreset.prod_llm.missingRequiredCount", "0");
        properties.setProperty("bundlePreset.prod_llm.policyViolationCount", "0");
        properties.setProperty("bundleAliases", "direct");
        properties.setProperty("bundleAlias.direct.families", "gemma,llama");
        properties.setProperty("bundleAlias.direct.familyCount", "4");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);

        assertEquals(List.of(
                "bundle.familyCount=3, expected 2",
                "bundle.productionReadinessPendingCount=1, expected 2",
                "bundle.directSafetensorPendingCount=0, expected 1",
                "policy.violationCount=0, expected 1",
                "fixture.requiredFamilyCount=1, expected 2",
                "fixture.missingRequiredCount=0, expected 1",
                "fixture.problemFamilyCount=1, expected 2",
                "bundlePreset.prod_llm.selectedCount=1, expected 2",
                "bundlePreset.prod_llm.policyViolationCount=0, expected 1",
                "bundlePreset.prod_llm.missingRequiredCount=0, expected 1",
                "bundleAlias.direct.familyCount=4, expected 2"
        ), manifest.countConsistencyProblems());
    }

    @Test
    void toleratesLegacyManifestWithoutSchemaOrFingerprint() {
        Properties properties = new Properties();
        properties.setProperty("families", "gemma");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);

        assertEquals(0, manifest.schemaVersion());
        assertEquals("-", manifest.displayFingerprint());
    }

    @Test
    void reportsBuildSelectedFamiliesMissingFromDiscovery() {
        Properties properties = new Properties();
        properties.setProperty("families", "gemma,llama,phi");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);

        assertEquals(List.of("phi"), manifest.missingDiscovered(Set.of("model-family/gemma", "llama")));
    }

    @Test
    void detectsDetachedBuildSelector() {
        Properties properties = new Properties();
        properties.setProperty("selectors", "none");
        properties.setProperty("families", "");
        properties.setProperty("profiles", "");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);

        assertTrue(manifest.detached());
        assertEquals("(detached)", manifest.displayFamilies());
        assertEquals("(detached)", manifest.displayProfiles());
        assertEquals(List.of(), manifest.missingDiscovered(Set.of()));
    }

    @Test
    void infersSelectedStateForOlderManifests() {
        Properties properties = new Properties();
        properties.setProperty("families", "gemma");
        properties.setProperty("availableFamilies", "bert,gemma");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);

        assertFalse(manifest.isFamilySelected("bert"));
        assertTrue(manifest.isFamilySelected("gemma"));
        assertEquals(List.of("bert"), manifest.omittedFamilies());
    }

    @Test
    void respectsExplicitDetachedFlag() {
        Properties properties = new Properties();
        properties.setProperty("selectors", "custom");
        properties.setProperty("detached", "true");

        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.fromProperties(properties);

        assertTrue(manifest.detached());
    }

    @Test
    void packagedBundleDiscoversSelectedModelFamilyPlugins() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        DiscoveredModelFamilyPlugins discovered = serviceLoaderPluginsByFamilyId();
        Map<String, ModelFamilyPlugin> plugins = discovered.uniquePluginsByFamilyId();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), discovered.discoveryErrors(),
                "packaged model-family plugin discovery should not throw");
        assertEquals(Map.of(), discovered.duplicateSelectedFamilyIds(manifest),
                "selected model families should resolve to exactly one ServiceLoader plugin");
        assertEquals(List.of(), manifest.missingDiscovered(plugins.keySet()),
                () -> "selected model families missing from ServiceLoader discovery: "
                        + manifest.missingDiscovered(plugins.keySet()));

        List<ModelFamilyContractViolation> violations = manifest.families().stream()
                .map(plugins::get)
                .flatMap(plugin -> ModelFamilyContractValidator.validate(plugin).stream())
                .toList();
        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "selected model-family plugins should satisfy the shared plugin contract");

    }

    @Test
    void packagedBundleDiscoveryHasUniqueFamilyIds() {
        DiscoveredModelFamilyPlugins discovered = serviceLoaderPluginsByFamilyId();

        assertEquals(List.of(), discovered.discoveryErrors(),
                "packaged model-family plugin discovery should not throw");
        assertEquals(Map.of(), discovered.duplicateFamilyIds(),
                "packaged model-family plugins should not publish duplicate family ids");
    }

    @Test
    void packagedCliClasspathCarriesGemma4DirectAdapterOutsideDefaultBundleSelection() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        DiscoveredModelFamilyPlugins discovered = serviceLoaderPluginsByFamilyId();
        ModelFamilyPlugin gemma4 = discovered.uniquePluginsByFamilyId().get("gemma4");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertTrue(!manifest.families().contains("gemma4"),
                "default bundle selection should not require Gemma4 unified runtime gates");
        assertEquals(List.of(), discovered.discoveryErrors(),
                "packaged model-family plugin discovery should not throw");
        assertTrue(gemma4 != null, "Gemma4 plugin should stay on the CLI classpath for direct runtime resolution");
        assertTrue(gemma4.descriptor().capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE));
        assertEquals(List.of("gemma4"), gemma4.architectureAdapters().stream()
                .map(adapter -> adapter.id())
                .toList());

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        discovered.uniquePluginsByFamilyId().values().forEach(registry::register);
        ModelFamilyResolution resolution = registry.resolve(
                "gemma4_unified",
                "Gemma4UnifiedForConditionalGeneration");
        ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(resolution);
        assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
        assertEquals(List.of("gemma4"), resolution.familyIds());
        assertEquals("gemma4", compatibility.selectedArchitectureAdapterId());
    }

    @Test
    void packagedDefaultBundleStaysCleanCoreSelection() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertFalse(manifest.detached(), "default packaged bundle should attach core model-family plugins");
        assertEquals(List.of("core"), manifest.selectors());
        assertEquals(List.of("core"), manifest.defaultSelectors());
        assertEquals(List.of(), manifest.explicitSelectors());
        assertEquals(List.of(), manifest.presetSelectors());
        assertEquals("default", manifest.selectorSource());
        assertEquals("none", manifest.policySource());
        assertFalse(manifest.hasBundlePreset(), "default packaged bundle should not impersonate a preset build");
        assertEquals(List.of("core"), manifest.requestedProfiles());
        assertEquals(List.of(), manifest.requestedAliases());
        assertEquals(List.of(), manifest.unknownSelectors());
        assertEquals(List.of(), familyIdsWithUnexpectedProfiles(
                        manifest.families(),
                        manifest.familyProfiles(),
                        Set.of("core")),
                "default packaged bundle should select only core-profile families");
    }

    @Test
    void packagedDefaultBundleMatchesRequiredFixtureCoverage() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.FixtureStatus fixture = manifest.fixtureStatus();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of("core"), fixture.requiredSelectors());
        assertEquals(manifest.families(), fixture.requiredFamilies(),
                "default packaged families should match the required core fixture set");
        assertEquals(Boolean.TRUE, fixture.passed());
        assertEquals(fixture.requiredFamilyCount(), fixture.requiredPassedCount());
        assertEquals(manifest.families().size(), fixture.requiredFamilyCount());
        assertEquals(0, fixture.missingRequiredCount());
        assertEquals(0, fixture.problemFamilyCount());
        assertEquals(List.of(), fixture.missingRequiredFamilies());
        assertEquals(List.of(), fixture.problemFamilies());
    }

    @Test
    void packagedAvailableFamiliesCarryProfileAndPathMetadata() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), missingFamilyIds(manifest.families(), manifest.availableFamilies()),
                () -> "selected families missing from availableFamilies: "
                        + missingFamilyIds(manifest.families(), manifest.availableFamilies()));
        assertEquals(List.of(), missingMapKeys(manifest.availableFamilies(), manifest.familyProfiles()),
                () -> "available families missing profile metadata: "
                        + missingMapKeys(manifest.availableFamilies(), manifest.familyProfiles()));
        assertEquals(List.of(), missingMapKeys(manifest.availableFamilies(), manifest.familyPaths()),
                () -> "available families missing Gradle path metadata: "
                        + missingMapKeys(manifest.availableFamilies(), manifest.familyPaths()));
        assertEquals(List.of(), missingFamilyIds(
                        manifest.tokenizerMetadataPendingFamilies(),
                        manifest.availableFamilies()),
                () -> "tokenizer metadata pending families should be known catalog families: "
                        + missingFamilyIds(manifest.tokenizerMetadataPendingFamilies(), manifest.availableFamilies()));
        assertEquals(List.of(), missingMapKeys(
                        manifest.tokenizerMetadataPendingFamilies(),
                        manifest.tokenizerMetadataPendingReasons()),
                () -> "tokenizer metadata pending families should carry reasons: "
                        + missingMapKeys(
                                manifest.tokenizerMetadataPendingFamilies(),
                                manifest.tokenizerMetadataPendingReasons()));
    }

    @Test
    void packagedAvailableSelectorsResolveToKnownCatalogEntries() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), unknownAvailableSelectors(manifest),
                () -> "available selectors should resolve to known families, profiles, aliases, or reserved selectors: "
                        + unknownAvailableSelectors(manifest));
    }

    @Test
    void packagedAvailableProfilesMatchFamilyProfileCatalog() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(distinctSortedValues(manifest.familyProfiles()), manifest.availableProfiles(),
                "availableProfiles should match the distinct profiles advertised by family metadata");
        assertEquals(List.of(), missingFamilyIds(manifest.profiles(), manifest.availableProfiles()),
                () -> "selected profiles missing from availableProfiles: "
                        + missingFamilyIds(manifest.profiles(), manifest.availableProfiles()));
    }

    @Test
    void packagedBundlePresetCatalogMatchesGuardedPresetSet() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        List<String> guardedPresetIds = List.of(
                "metadata_inspection",
                "prod_embedding",
                "prod_llm",
                "prod_vlm_metadata",
                "research_all",
                "runtime_proxy");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(guardedPresetIds, manifest.availableBundlePresets(),
                "packaged bundle preset catalog should stay in sync with explicit preset guard tests");
        assertEquals(guardedPresetIds, manifest.bundlePresets().stream()
                .map(ModelFamilyBundleManifest.BundlePreset::id)
                .toList());
    }

    @Test
    void packagedBundlePresetSelectionsStayCountedAndAvailable() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), manifest.bundlePresets().stream()
                        .filter(preset -> preset.selectedCount() != preset.selectedFamilies().size())
                        .map(ModelFamilyBundleManifest.BundlePreset::id)
                        .toList(),
                "packaged bundle preset selectedCount should match selectedFamilies size");
        assertEquals(List.of(), manifest.bundlePresets().stream()
                        .flatMap(preset -> missingFamilyIds(preset.selectedFamilies(), manifest.availableFamilies())
                                .stream()
                                .map(familyId -> preset.id() + ":" + familyId))
                        .toList(),
                "packaged bundle presets should only select available families");
    }

    @Test
    void packagedBundlePresetInputsResolveToKnownCatalogEntries() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), manifest.bundlePresets().stream()
                        .flatMap(preset -> unknownPresetSelectors(preset, manifest).stream()
                                .map(selector -> preset.id() + ":" + selector))
                        .toList(),
                "packaged bundle preset selectors should resolve to known catalog entries");
        assertEquals(List.of(), manifest.bundlePresets().stream()
                        .flatMap(preset -> unknownPresetFamilyPolicyInputs(preset, manifest).stream()
                                .map(familyId -> preset.id() + ":" + familyId))
                        .toList(),
                "packaged bundle preset family policies should reference available families");
        assertEquals(List.of(), manifest.bundlePresets().stream()
                        .flatMap(preset -> unknownPresetAliasPolicyInputs(preset, manifest).stream()
                                .map(aliasId -> preset.id() + ":" + aliasId))
                        .toList(),
                "packaged bundle preset alias policies should reference available aliases");
    }

    @Test
    void packagedPassingBundlePresetsDoNotCarryPolicyViolationDetails() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), manifest.bundlePresets().stream()
                        .filter(preset -> Boolean.TRUE.equals(preset.policyPassed()))
                        .filter(preset -> preset.policyViolationCount() != 0
                                || hasPolicyViolationDetails(preset.policyViolations()))
                        .map(ModelFamilyBundleManifest.BundlePreset::id)
                        .toList(),
                "passing packaged bundle presets should not carry stale policy violation metadata");
    }

    @Test
    void packagedBundleAliasCatalogMatchesGuardedAliasSet() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        List<String> guardedAliasIds = List.of("audio", "direct", "embedding", "moe", "research", "vision", "vlm");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(guardedAliasIds, manifest.bundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAlias::id)
                .toList(),
                "packaged bundle alias catalog should stay in sync with explicit alias guard expectations");
        assertEquals(List.of(), manifest.bundleAliases().stream()
                .flatMap(alias -> missingFamilyIds(alias.families(), manifest.availableFamilies()).stream()
                        .map(familyId -> alias.id() + ":" + familyId))
                .toList(),
                "packaged bundle aliases should only reference available model families");
    }

    @Test
    void packagedResearchAliasTracksAllNonCoreFamilies() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundleAlias research = bundleAlias(manifest, "research");
        List<String> nonCoreFamilies = manifest.familyProfiles().entrySet().stream()
                .filter(entry -> !entry.getValue().equals("core"))
                .map(Map.Entry::getKey)
                .sorted()
                .toList();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(nonCoreFamilies, research.families(),
                "research alias should track every non-core model-family plugin");
    }

    @Test
    void packagedProductionAliasProfilesStayInTheirRuntimeBuckets() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundleAlias direct = bundleAlias(manifest, "direct");
        ModelFamilyBundleManifest.BundleAlias embedding = bundleAlias(manifest, "embedding");
        ModelFamilyBundleManifest.BundleAlias moe = bundleAlias(manifest, "moe");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), familyIdsWithUnexpectedProfiles(
                        direct.families(),
                        manifest.familyProfiles(),
                        Set.of("core", "optional")),
                "direct alias should only contain core or optional direct-runtime families");
        assertEquals(List.of(), familyIdsWithUnexpectedProfiles(
                        embedding.families(),
                        manifest.familyProfiles(),
                        Set.of("metadata_only")),
                "embedding alias should stay metadata-only until an embedding runtime adapter exists");
        assertEquals(List.of(), selectedFamilyIds(embedding.families(), direct.families()),
                () -> "embedding alias should not overlap direct alias families: "
                        + selectedFamilyIds(embedding.families(), direct.families()));
        assertEquals(List.of(), selectedFamilyIds(embedding.families(), moe.families()),
                () -> "embedding alias should not overlap moe alias families: "
                        + selectedFamilyIds(embedding.families(), moe.families()));
    }

    @Test
    void packagedVlmAliasStaysWithinVisionCatalogAndOutsideEmbedding() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundleAlias vlm = bundleAlias(manifest, "vlm");
        ModelFamilyBundleManifest.BundleAlias vision = bundleAlias(manifest, "vision");
        ModelFamilyBundleManifest.BundleAlias embedding = bundleAlias(manifest, "embedding");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), missingFamilyIds(vlm.families(), vision.families()),
                () -> "vlm alias families should also be present in the broader vision alias: "
                        + missingFamilyIds(vlm.families(), vision.families()));
        assertEquals(List.of(), selectedFamilyIds(vlm.families(), embedding.families()),
                () -> "vlm alias should not overlap embedding-only families: "
                        + selectedFamilyIds(vlm.families(), embedding.families()));
    }

    @Test
    void packagedSpecializedAliasesStayOutOfCoreRuntimeSelection() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundleAlias audio = bundleAlias(manifest, "audio");
        ModelFamilyBundleManifest.BundleAlias moe = bundleAlias(manifest, "moe");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), familyIdsWithUnexpectedProfiles(
                        audio.families(),
                        manifest.familyProfiles(),
                        Set.of("metadata_only", "optional")),
                "audio alias should remain opt-in and avoid core runtime families");
        assertEquals(List.of(), familyIdsWithUnexpectedProfiles(
                        moe.families(),
                        manifest.familyProfiles(),
                        Set.of("metadata_only", "optional", "experimental")),
                "moe alias should remain opt-in and avoid core runtime families");
    }

    @Test
    void packagedSelectedDirectAliasFamiliesExposeDirectAdapters() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundleAlias direct = bundleAlias(manifest, "direct");
        DiscoveredModelFamilyPlugins discovered = serviceLoaderPluginsByFamilyId();
        Map<String, ModelFamilyPlugin> plugins = discovered.uniquePluginsByFamilyId();
        List<String> selectedDirectFamilies = selectedFamilyIds(direct.families(), manifest.families());

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of(), discovered.discoveryErrors(),
                "packaged model-family plugin discovery should not throw");
        assertEquals(Map.of(), discovered.duplicateFamilyIds(selectedDirectFamilies),
                "selected direct alias families should resolve to exactly one ServiceLoader plugin");
        assertEquals(List.of(), missingFamilyIds(selectedDirectFamilies, plugins),
                () -> "selected direct alias families missing from ServiceLoader discovery: "
                        + missingFamilyIds(selectedDirectFamilies, plugins));
        assertEquals(List.of(), selectedDirectFamilies.stream()
                        .filter(familyId -> !plugins.get(familyId).descriptor().capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE))
                        .toList(),
                "selected direct alias families should advertise direct safetensor support");
        assertEquals(List.of(), selectedDirectFamilies.stream()
                        .filter(familyId -> plugins.get(familyId).architectureAdapters().isEmpty())
                        .toList(),
                "selected direct alias families should expose at least one architecture adapter");
    }

    @Test
    void packagedProdLlmPresetMetadataMatchesDirectAliasPolicy() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundlePreset prodLlm = bundlePreset(manifest, "prod_llm");
        ModelFamilyBundleManifest.BundleAlias direct = bundleAlias(manifest, "direct");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of("direct"), prodLlm.selectors());
        assertEquals(List.of("direct"), prodLlm.requiredAliases());
        assertEquals(List.of("embedding"), prodLlm.forbiddenAliases());
        assertEquals(Boolean.TRUE, prodLlm.policyPassed());
        assertEquals(0, prodLlm.policyViolationCount());
        assertEquals(direct.families(), prodLlm.selectedFamilies(),
                "prod_llm preset should package the complete direct alias family set");
        assertEquals(List.of(), missingFamilyIds(prodLlm.selectedFamilies(), manifest.availableFamilies()),
                () -> "prod_llm selected families missing from availableFamilies: "
                        + missingFamilyIds(prodLlm.selectedFamilies(), manifest.availableFamilies()));
    }

    @Test
    void packagedProdEmbeddingPresetMetadataMatchesEmbeddingAliasPolicy() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundlePreset prodEmbedding = bundlePreset(manifest, "prod_embedding");
        ModelFamilyBundleManifest.BundleAlias embedding = bundleAlias(manifest, "embedding");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of("embedding"), prodEmbedding.selectors());
        assertEquals(List.of("embedding"), prodEmbedding.requiredAliases());
        assertEquals(List.of("direct", "moe"), prodEmbedding.forbiddenAliases());
        assertEquals(Boolean.TRUE, prodEmbedding.policyPassed());
        assertEquals(0, prodEmbedding.policyViolationCount());
        assertEquals(embedding.families(), prodEmbedding.selectedFamilies(),
                "prod_embedding preset should package the complete embedding alias family set");
        assertEquals(List.of(), missingFamilyIds(prodEmbedding.selectedFamilies(), manifest.availableFamilies()),
                () -> "prod_embedding selected families missing from availableFamilies: "
                        + missingFamilyIds(prodEmbedding.selectedFamilies(), manifest.availableFamilies()));
    }

    @Test
    void packagedProdVlmMetadataPresetKeepsTextAnchorsAndVlmAlias() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundlePreset prodVlmMetadata = bundlePreset(manifest, "prod_vlm_metadata");
        ModelFamilyBundleManifest.BundleAlias vlm = bundleAlias(manifest, "vlm");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of("core", "vlm"), prodVlmMetadata.selectors());
        assertEquals(List.of("gemma", "llama", "mistral", "phi", "qwen"), prodVlmMetadata.requiredFamilies());
        assertEquals(List.of("vlm"), prodVlmMetadata.requiredAliases());
        assertEquals(List.of("embedding"), prodVlmMetadata.forbiddenAliases());
        assertEquals(Boolean.TRUE, prodVlmMetadata.policyPassed());
        assertEquals(0, prodVlmMetadata.policyViolationCount());
        assertEquals(List.of(), missingFamilyIds(vlm.families(), prodVlmMetadata.selectedFamilies()),
                () -> "prod_vlm_metadata selected families missing vlm alias members: "
                        + missingFamilyIds(vlm.families(), prodVlmMetadata.selectedFamilies()));
        assertEquals(List.of(), missingFamilyIds(prodVlmMetadata.requiredFamilies(), prodVlmMetadata.selectedFamilies()),
                () -> "prod_vlm_metadata selected families missing required text anchors: "
                        + missingFamilyIds(prodVlmMetadata.requiredFamilies(), prodVlmMetadata.selectedFamilies()));
        assertEquals(List.of(), missingFamilyIds(prodVlmMetadata.selectedFamilies(), manifest.availableFamilies()),
                () -> "prod_vlm_metadata selected families missing from availableFamilies: "
                        + missingFamilyIds(prodVlmMetadata.selectedFamilies(), manifest.availableFamilies()));
    }

    @Test
    void packagedMetadataInspectionPresetTracksMetadataOnlyFamiliesAndExcludesDirectAlias() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundlePreset metadataInspection = bundlePreset(manifest, "metadata_inspection");
        ModelFamilyBundleManifest.BundleAlias direct = bundleAlias(manifest, "direct");
        List<String> metadataOnlyFamilies = manifest.familyProfiles().entrySet().stream()
                .filter(entry -> entry.getValue().equals("metadata_only"))
                .map(Map.Entry::getKey)
                .sorted()
                .toList();

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of("metadata_only"), metadataInspection.selectors());
        assertEquals(List.of("direct"), metadataInspection.forbiddenAliases());
        assertEquals(Boolean.TRUE, metadataInspection.policyPassed());
        assertEquals(0, metadataInspection.policyViolationCount());
        assertEquals(metadataOnlyFamilies, metadataInspection.selectedFamilies(),
                "metadata_inspection preset should package exactly metadata_only profile families");
        assertEquals(List.of(), selectedFamilyIds(direct.families(), metadataInspection.selectedFamilies()),
                () -> "metadata_inspection selected forbidden direct alias members: "
                        + selectedFamilyIds(direct.families(), metadataInspection.selectedFamilies()));
        assertEquals(List.of(), missingFamilyIds(metadataInspection.selectedFamilies(), manifest.availableFamilies()),
                () -> "metadata_inspection selected families missing from availableFamilies: "
                        + missingFamilyIds(metadataInspection.selectedFamilies(), manifest.availableFamilies()));
    }

    @Test
    void packagedRuntimeProxyPresetDetachesModelFamilyPlugins() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundlePreset runtimeProxy = bundlePreset(manifest, "runtime_proxy");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of("none"), runtimeProxy.selectors());
        assertEquals(List.of(), runtimeProxy.requiredFamilies());
        assertEquals(List.of(), runtimeProxy.forbiddenFamilies());
        assertEquals(List.of(), runtimeProxy.requiredAliases());
        assertEquals(List.of(), runtimeProxy.forbiddenAliases());
        assertEquals(List.of(), runtimeProxy.selectedFamilies(),
                "runtime_proxy preset should detach all model-family plugins");
        assertEquals(Boolean.TRUE, runtimeProxy.policyPassed());
        assertEquals(0, runtimeProxy.policyViolationCount());
    }

    @Test
    void packagedResearchAllPresetTracksEveryAvailableFamilyWithoutPolicyRestrictions() {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyBundleManifest.BundlePreset researchAll = bundlePreset(manifest, "research_all");

        assertTrue(manifest.present(), "packaged model-family bundle manifest should be generated");
        assertEquals(List.of("all"), researchAll.selectors());
        assertEquals(List.of(), researchAll.requiredFamilies());
        assertEquals(List.of(), researchAll.forbiddenFamilies());
        assertEquals(List.of(), researchAll.requiredAliases());
        assertEquals(List.of(), researchAll.forbiddenAliases());
        assertEquals(manifest.availableFamilies(), researchAll.selectedFamilies(),
                "research_all preset should package every available model-family plugin");
        assertEquals(Boolean.TRUE, researchAll.policyPassed());
        assertEquals(0, researchAll.policyViolationCount());
    }

    private static DiscoveredModelFamilyPlugins serviceLoaderPluginsByFamilyId() {
        Map<String, List<ModelFamilyPlugin>> plugins = new LinkedHashMap<>();
        List<String> discoveryErrors = new ArrayList<>();
        var iterator = ServiceLoader.load(ModelFamilyPlugin.class).iterator();
        while (true) {
            ModelFamilyPlugin plugin;
            try {
                if (!iterator.hasNext()) {
                    break;
                }
                plugin = iterator.next();
            } catch (RuntimeException | ServiceConfigurationError error) {
                discoveryErrors.add("ServiceLoader: " + error.getClass().getSimpleName() + ": " + error.getMessage());
                continue;
            }

            try {
                ModelFamilyDescriptor descriptor = plugin.descriptor();
                plugins.computeIfAbsent(descriptor.id(), ignored -> new ArrayList<>()).add(plugin);
            } catch (RuntimeException error) {
                discoveryErrors.add(plugin.getClass().getName() + ".descriptor(): "
                        + error.getClass().getSimpleName() + ": " + error.getMessage());
            }
        }
        return new DiscoveredModelFamilyPlugins(plugins, discoveryErrors);
    }

    private static ModelFamilyBundleManifest.BundlePreset bundlePreset(
            ModelFamilyBundleManifest manifest,
            String id) {
        return manifest.bundlePresets().stream()
                .filter(preset -> preset.id().equals(id))
                .findFirst()
                .orElseThrow();
    }

    private static ModelFamilyBundleManifest.BundleAlias bundleAlias(
            ModelFamilyBundleManifest manifest,
            String id) {
        return manifest.bundleAliases().stream()
                .filter(alias -> alias.id().equals(id))
                .findFirst()
                .orElseThrow();
    }

    private static List<String> missingFamilyIds(List<String> familyIds, Map<String, ModelFamilyPlugin> plugins) {
        return familyIds.stream()
                .filter(familyId -> !plugins.containsKey(familyId))
                .toList();
    }

    private static List<String> missingFamilyIds(List<String> familyIds, List<String> availableFamilyIds) {
        Set<String> available = Set.copyOf(availableFamilyIds);
        return familyIds.stream()
                .filter(familyId -> !available.contains(familyId))
                .toList();
    }

    private static List<String> missingMapKeys(List<String> familyIds, Map<String, ?> metadataByFamilyId) {
        return familyIds.stream()
                .filter(familyId -> !metadataByFamilyId.containsKey(familyId))
                .toList();
    }

    private static List<String> distinctSortedValues(Map<String, String> valuesByKey) {
        return valuesByKey.values().stream()
                .distinct()
                .sorted()
                .toList();
    }

    private static List<String> unknownAvailableSelectors(ModelFamilyBundleManifest manifest) {
        Set<String> knownSelectors = new java.util.LinkedHashSet<>();
        knownSelectors.addAll(manifest.availableFamilies());
        knownSelectors.addAll(manifest.availableProfiles());
        manifest.bundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAlias::id)
                .forEach(knownSelectors::add);
        knownSelectors.add("all");
        knownSelectors.add("none");
        return manifest.availableSelectors().stream()
                .filter(selector -> !knownSelectors.contains(selector))
                .toList();
    }

    private static List<String> unknownPresetSelectors(
            ModelFamilyBundleManifest.BundlePreset preset,
            ModelFamilyBundleManifest manifest) {
        Set<String> knownSelectors = new java.util.LinkedHashSet<>(manifest.availableSelectors());
        knownSelectors.add("all");
        knownSelectors.add("none");
        return preset.selectors().stream()
                .filter(selector -> !knownSelectors.contains(selector))
                .toList();
    }

    private static List<String> unknownPresetFamilyPolicyInputs(
            ModelFamilyBundleManifest.BundlePreset preset,
            ModelFamilyBundleManifest manifest) {
        Set<String> availableFamilies = Set.copyOf(manifest.availableFamilies());
        return java.util.stream.Stream.concat(
                        preset.requiredFamilies().stream(),
                        preset.forbiddenFamilies().stream())
                .filter(familyId -> !availableFamilies.contains(familyId))
                .distinct()
                .toList();
    }

    private static List<String> unknownPresetAliasPolicyInputs(
            ModelFamilyBundleManifest.BundlePreset preset,
            ModelFamilyBundleManifest manifest) {
        Set<String> availableAliases = manifest.bundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAlias::id)
                .collect(java.util.stream.Collectors.toSet());
        return java.util.stream.Stream.concat(
                        preset.requiredAliases().stream(),
                        preset.forbiddenAliases().stream())
                .filter(aliasId -> !availableAliases.contains(aliasId))
                .distinct()
                .toList();
    }

    private static List<String> selectedFamilyIds(List<String> familyIds, List<String> selectedFamilyIds) {
        Set<String> selected = Set.copyOf(selectedFamilyIds);
        return familyIds.stream()
                .filter(selected::contains)
                .toList();
    }

    private static List<String> familyIdsWithUnexpectedProfiles(
            List<String> familyIds,
            Map<String, String> familyProfiles,
            Set<String> expectedProfiles) {
        return familyIds.stream()
                .filter(familyId -> !expectedProfiles.contains(familyProfiles.get(familyId)))
                .toList();
    }

    private static boolean hasPolicyViolationDetails(
            ModelFamilyBundleManifest.BundlePolicyViolations violations) {
        return !violations.missingRequiredFamilies().isEmpty()
                || !violations.selectedForbiddenFamilies().isEmpty()
                || !violations.missingRequiredAliases().isEmpty()
                || !violations.selectedForbiddenAliases().isEmpty();
    }

    private record DiscoveredModelFamilyPlugins(
            Map<String, List<ModelFamilyPlugin>> pluginsByFamilyId,
            List<String> discoveryErrors) {

        private DiscoveredModelFamilyPlugins {
            pluginsByFamilyId = Map.copyOf(pluginsByFamilyId);
            discoveryErrors = List.copyOf(discoveryErrors);
        }

        private Map<String, ModelFamilyPlugin> uniquePluginsByFamilyId() {
            Map<String, ModelFamilyPlugin> unique = new LinkedHashMap<>();
            pluginsByFamilyId.forEach((familyId, plugins) -> {
                if (plugins.size() == 1) {
                    unique.put(familyId, plugins.get(0));
                }
            });
            return Map.copyOf(unique);
        }

        private Map<String, List<String>> duplicateFamilyIds() {
            Map<String, List<String>> duplicates = new LinkedHashMap<>();
            pluginsByFamilyId.forEach((familyId, plugins) -> {
                if (plugins.size() > 1) {
                    duplicates.put(familyId, plugins.stream()
                            .map(plugin -> plugin.getClass().getName())
                            .toList());
                }
            });
            return Map.copyOf(duplicates);
        }

        private Map<String, List<String>> duplicateFamilyIds(List<String> familyIds) {
            Map<String, List<String>> selectedDuplicates = new LinkedHashMap<>();
            Map<String, List<String>> duplicates = duplicateFamilyIds();
            for (String familyId : familyIds) {
                List<String> pluginClasses = duplicates.get(familyId);
                if (pluginClasses != null) {
                    selectedDuplicates.put(familyId, pluginClasses);
                }
            }
            return Map.copyOf(selectedDuplicates);
        }

        private Map<String, List<String>> duplicateSelectedFamilyIds(ModelFamilyBundleManifest manifest) {
            return duplicateFamilyIds(manifest.families());
        }
    }
}
