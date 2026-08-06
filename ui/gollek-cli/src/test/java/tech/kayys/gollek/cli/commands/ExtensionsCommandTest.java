package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.cli.util.ExtensionAvailabilityPolicy;
import tech.kayys.gollek.cli.util.ExternalPluginClasspath;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields;
import tech.kayys.gollek.cli.util.RouteBenchmarkCacheReportContract;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields;
import tech.kayys.gollek.sdk.route.RouteReportPayloadFields;
import tech.kayys.gollek.sdk.route.RunnerRouteReportFields;
import tech.kayys.gollek.cli.util.UnifiedRuntimeRequirementReportFields;
import tech.kayys.gollek.cli.util.UnifiedRuntimeRequirementProblemCodes;
import tech.kayys.gollek.plugin.core.ExtensionAvailability;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityProvider;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;

import javax.tools.JavaCompiler;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExtensionsCommandTest {
    private static final String EXTERNAL_EXTENSION_ID = "modules-external-extension";
    private static final String EXTERNAL_MODEL_FAMILY_ID = "modules_external_family";
    private static final String EXTERNAL_MODEL_TYPE = "modules_external_model";

    @Test
    void pluginTokenizerKindAllowlistIncludesSpecializedModelFarmAdapters() {
        assertTrue(ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS.contains("custom-bpe"));
        assertTrue(ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS.contains("jieba"));
        assertTrue(ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS.contains("tekken"));
        assertTrue(ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS.contains("tiktoken"));
        assertTrue(ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS.contains("jamba"));
        assertTrue(ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS.contains("tokenizer-json"));
        assertTrue(ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS.contains("composite-tokenizer"));
    }

    @Test
    void jsonOutputIsParseableForAutomation() throws Exception {
        ExtensionsCommand command = new ExtensionsCommand();
        command.jsonOutput = true;

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            command.run();
        } finally {
            System.setOut(originalOut);
        }

        JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
        assertEquals(1, root.path("schemaVersion").asInt());
        assertTrue(root.path("externalPluginClasspath").isArray());
        assertTrue(root.path("externalPluginClasspath").isEmpty());
        assertTrue(root.path("pluginDirectories").path("commandDirectories").isArray());
        assertTrue(root.path("pluginDirectories").path("configuredDirectories").isArray());
        assertTrue(root.path("pluginDirectories").path("activeDirectories").isArray());
        assertTrue(root.path("pluginDirectories").path("defaultDirectory").isTextual());
        assertTrue(root.path("pluginDirectories").path("defaultDirectoryExists").isBoolean());
        assertTrue(root.path("pluginDirectories").path("defaultDirectoryAutoloadEnabled").isBoolean());
        assertTrue(root.path("pluginDirectories").path("defaultDirectoryActive").isBoolean());
        assertTrue(root.path("pluginDirectories").path("jarReadiness").path("jarCount").canConvertToInt());
        assertTrue(root.path("pluginDirectories").path("jarReadiness").path("jars").isArray());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginInstallNotReady")
                .canConvertToInt());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataReady")
                .canConvertToInt());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataPending")
                .canConvertToInt());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataMissing")
                .canConvertToInt());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataInvalid")
                .canConvertToInt());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataReadyJars")
                .isArray());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataPendingJars")
                .isArray());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataMissingJars")
                .isArray());
        assertTrue(root.path("pluginDirectories")
                .path("jarReadiness")
                .path("pluginTokenizerMetadataInvalidJars")
                .isArray());
        assertTrue(root.path("kernel").path("displayName").isTextual());
        assertTrue(root.path("runtimeModules").isArray());
        assertTrue(root.path("runnerPlugins").path("plugins").isArray());
        JsonNode routeBenchmarkCache = root.path("runnerPlugins").path("routeBenchmarkCache");
        assertEquals(1, routeBenchmarkCache.path("schemaVersion").asInt());
        assertTrue(routeBenchmarkCache.path("status").isTextual());
        assertTrue(routeBenchmarkCache.path("enabled").isBoolean());
        assertTrue(routeBenchmarkCache.path("healthy").isBoolean());
        assertTrue(routeBenchmarkCache.path("cacheFile").isTextual());
        assertTrue(routeBenchmarkCache.path("cacheFileExists").isBoolean());
        assertTrue(routeBenchmarkCache.path("cacheFileReadable").isBoolean());
        assertTrue(routeBenchmarkCache.path("cacheDirectoryWritable").isBoolean());
        assertTrue(routeBenchmarkCache.path("staleAfterDays").canConvertToInt());
        assertTrue(routeBenchmarkCache.path("entryCount").canConvertToInt());
        assertTrue(routeBenchmarkCache.path("staleEntryCount").canConvertToInt());
        assertTrue(routeBenchmarkCache.path("freshEntryCount").canConvertToInt());
        assertTrue(routeBenchmarkCache.path("trustedEntryCount").canConvertToInt());
        assertTrue(routeBenchmarkCache.path("staleProfilesAllowed").isBoolean());
        assertFalse(routeBenchmarkCache.path("failOnRouteBenchmarkCache").asBoolean());
        assertTrue(routeBenchmarkCache.path("strictHealthy").isBoolean());
        assertTrue(routeBenchmarkCache.path("profileTrustStatus").isTextual());
        assertTrue(routeBenchmarkCache.path("problems").isArray());
        assertTrue(routeBenchmarkCache.path("remediationHints").isArray());
        assertTrue(routeBenchmarkCache.path("invalidLineCount").canConvertToInt());
        assertTrue(routeBenchmarkCache.path("providers").isArray());
        assertTrue(routeBenchmarkCache.path("formats").isArray());
        assertTrue(routeBenchmarkCache.path("recentEntries").isArray());
        assertRouteBenchmarkCacheContract(root.path("runnerPlugins"));
        JsonNode runnerRouteReportSchema = root.path("runnerPlugins").path("routeReportSchema");
        assertEquals(
                RunnerRouteReportFields.CONTRACT_ID,
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.CONTRACT_ID).asText());
        assertEquals(
                RunnerRouteReportFields.SCHEMA_VERSION,
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.SCHEMA_VERSION).asInt());
        assertEquals(
                RunnerRouteReportFields.schemaFingerprint(),
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.SCHEMA_FINGERPRINT).asText());
        assertEquals(
                RunnerRouteReportFields.METADATA_ROOT,
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.METADATA_ROOT).asText());
        assertEquals(
                RunnerRouteReportFields.METADATA_PREFIX,
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.METADATA_PREFIX).asText());
        assertEquals(
                RunnerRouteReportFields.VALIDATION_METADATA_ROOT,
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.VALIDATION_METADATA_ROOT).asText());
        assertEquals(
                RunnerRouteReportFields.VALIDATION_METADATA_PREFIX,
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.VALIDATION_METADATA_PREFIX).asText());
        assertTrue(containsTextValue(
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.REPORT_FIELDS),
                RunnerRouteReportFields.Report.NORMALIZED_RUNNER));
        assertTrue(containsTextValue(
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.REPORT_FIELDS),
                RunnerRouteReportFields.Report.ROUTE_PROFILE_STATUS));
        assertTrue(containsTextValue(
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.REQUIRED_REPORT_FIELDS),
                RunnerRouteReportFields.Report.MODE));
        assertTrue(containsTextValue(
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.REQUIRED_REPORT_FIELDS),
                RunnerRouteReportFields.Report.ROUTE_PROFILE_SOURCE));
        assertTrue(containsTextValue(
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.SELECTION_SOURCES),
                RunnerRouteReportFields.SelectionSource.PROVIDER_CLI));
        assertTrue(containsTextValue(
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.ROUTE_PROFILE_STATUSES),
                RunnerRouteReportFields.RouteProfileStatus.REDIRECTED));
        assertTrue(containsTextValue(
                runnerRouteReportSchema.path(RunnerRouteReportFields.Schema.ROUTE_PROFILE_SOURCES),
                RunnerRouteReportFields.RouteProfileSource.BENCHMARK_CACHE));
        JsonNode runnerRouteReportSchemaValidation =
                root.path("runnerPlugins").path("routeReportSchemaValidation");
        assertEquals(
                RunnerRouteReportFields.schemaFingerprint(),
                runnerRouteReportSchemaValidation
                        .path(RunnerRouteReportFields.Validation.SCHEMA_FINGERPRINT)
                        .asText());
        assertTrue(runnerRouteReportSchemaValidation
                .path(RunnerRouteReportFields.Validation.PASSED)
                .asBoolean());
        assertEquals(
                0,
                runnerRouteReportSchemaValidation
                        .path(RunnerRouteReportFields.Validation.PROBLEM_COUNT)
                        .asInt());
        assertTrue(runnerRouteReportSchemaValidation
                .path(RunnerRouteReportFields.Validation.PROBLEMS)
                .isArray());
        JsonNode routeReportPayloadSchema = root.path("runnerPlugins").path("routeReportPayloadSchema");
        assertEquals(
                RouteReportPayloadFields.CONTRACT_ID,
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.CONTRACT_ID).asText());
        assertEquals(
                RouteReportPayloadFields.SCHEMA_VERSION,
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.SCHEMA_VERSION).asInt());
        assertEquals(
                RouteReportPayloadFields.schemaFingerprint(),
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.SCHEMA_FINGERPRINT).asText());
        assertEquals(
                RouteReportPayloadFields.VALIDATION_ROOT,
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.VALIDATION_ROOT).asText());
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.PREFLIGHT_PROBLEMS));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.PREFLIGHT_PROBLEM_CODES));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.NEXT_ACTION_COUNT));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.NEXT_ACTION_KINDS));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RunnerRouteReportFields.METADATA_ROOT));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.ROUTE_RUNNER));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.ROUTE_SELECTION_SOURCE));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.ROUTE_REDIRECTED));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.ROUTE_PROFILE_STATUS));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.ROUTE_PROFILE_SOURCE));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.REQUIRED_PAYLOAD_FIELDS),
                RoutePreflightDiagnosticFields.VALIDATION_ROOT));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.OPTIONAL_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.LOCAL_PATH));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.OPTIONAL_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.ROUTE_REDIRECT_REASON));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.OPTIONAL_PAYLOAD_FIELDS),
                RouteReportPayloadFields.Payload.ROUTE_PROFILE_ADVICE));
        assertTrue(containsTextValue(
                routeReportPayloadSchema.path(RouteReportPayloadFields.Schema.VALIDATION_FIELDS),
                RouteReportPayloadFields.Validation.PASSED));
        JsonNode routeReportPayloadSchemaValidation =
                root.path("runnerPlugins").path("routeReportPayloadSchemaValidation");
        assertEquals(
                RouteReportPayloadFields.schemaFingerprint(),
                routeReportPayloadSchemaValidation
                        .path(RouteReportPayloadFields.Validation.SCHEMA_FINGERPRINT)
                        .asText());
        assertTrue(routeReportPayloadSchemaValidation
                .path(RouteReportPayloadFields.Validation.PASSED)
                .asBoolean());
        assertEquals(
                0,
                routeReportPayloadSchemaValidation
                        .path(RouteReportPayloadFields.Validation.PROBLEM_COUNT)
                        .asInt());
        JsonNode routePreflightDiagnosticSchema =
                root.path("runnerPlugins").path("routePreflightDiagnosticSchema");
        assertEquals(
                RoutePreflightDiagnosticFields.CONTRACT_ID,
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.CONTRACT_ID).asText());
        assertEquals(
                RoutePreflightDiagnosticFields.SCHEMA_VERSION,
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.SCHEMA_VERSION).asInt());
        assertEquals(
                RoutePreflightDiagnosticFields.schemaFingerprint(),
                routePreflightDiagnosticSchema
                        .path(RoutePreflightDiagnosticFields.Schema.SCHEMA_FINGERPRINT)
                        .asText());
        assertEquals(
                RoutePreflightDiagnosticFields.VALIDATION_ROOT,
                routePreflightDiagnosticSchema
                        .path(RoutePreflightDiagnosticFields.Schema.VALIDATION_ROOT)
                        .asText());
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.PROBLEM_FIELDS),
                RoutePreflightDiagnosticFields.Problem.CODE));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.ACTION_FIELDS),
                RoutePreflightDiagnosticFields.Action.ARGV));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.PROBLEM_DETAIL_FIELDS),
                RoutePreflightDiagnosticFields.ProblemDetail.READY_RUNTIME_CAPABILITIES));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.PROBLEM_DETAIL_FIELDS),
                RoutePreflightDiagnosticFields.ProblemDetail.HEADER_INSPECTION));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.EXECUTION_PROFILE_FIELDS),
                RoutePreflightDiagnosticFields.ExecutionProfile.GUARDED_TEXT_DECODER_READY));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.HEADER_INSPECTION_FIELDS),
                RoutePreflightDiagnosticFields.HeaderInspection.PAYLOAD_BYTES_LOADED));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.TENSOR_INVENTORY_FIELDS),
                RoutePreflightDiagnosticFields.TensorInventory.PACKED_MOE_EXPERT_TENSORS));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.COMPONENT_READINESS_FIELDS),
                RoutePreflightDiagnosticFields.ComponentReadiness.TEXT_DECODER_READY));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.INPUT_MODES),
                RoutePreflightDiagnosticFields.InputMode.TEXT));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.RUNTIME_CAPABILITIES),
                RoutePreflightDiagnosticFields.RuntimeCapability.GEMMA4_GUARDED_TEXT_DECODER));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.VALIDATION_FIELDS),
                RoutePreflightDiagnosticFields.Validation.PASSED));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.PROBLEM_CODES),
                RoutePreflightDiagnosticFields.ProblemCode.MODEL_NOT_LOCAL));
        assertTrue(containsTextValue(
                routePreflightDiagnosticSchema.path(RoutePreflightDiagnosticFields.Schema.ACTION_KINDS),
                RoutePreflightDiagnosticFields.ActionKind.INSPECT_MODULES));
        JsonNode routePreflightDiagnosticSchemaValidation =
                root.path("runnerPlugins").path("routePreflightDiagnosticSchemaValidation");
        assertEquals(
                RoutePreflightDiagnosticFields.schemaFingerprint(),
                routePreflightDiagnosticSchemaValidation
                        .path(RoutePreflightDiagnosticFields.Validation.SCHEMA_FINGERPRINT)
                        .asText());
        assertTrue(routePreflightDiagnosticSchemaValidation
                .path(RoutePreflightDiagnosticFields.Validation.PASSED)
                .asBoolean());
        assertEquals(
                0,
                routePreflightDiagnosticSchemaValidation
                        .path(RoutePreflightDiagnosticFields.Validation.PROBLEM_COUNT)
                        .asInt());
        JsonNode routeContractBundle = root.path("runnerPlugins").path("routeContractBundle");
        assertEquals(
                RunnerRouteContractBundle.CONTRACT_ID,
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_CONTRACT_ID).asText());
        assertEquals(
                RunnerRouteContractBundle.SCHEMA_VERSION,
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_SCHEMA_VERSION).asInt());
        assertEquals(
                RunnerRouteContractBundle.schemaFingerprint(),
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_SCHEMA_FINGERPRINT).asText());
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_CONTRACT_IDS),
                RunnerRouteReportFields.CONTRACT_ID));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_CONTRACT_IDS),
                RouteReportPayloadFields.CONTRACT_ID));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_CONTRACT_IDS),
                RoutePreflightDiagnosticFields.CONTRACT_ID));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_CONTRACT_IDS),
                RunnerRoutePolicyContract.CONTRACT_ID));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_CONTRACT_IDS),
                RouteBenchmarkCacheReportContract.CONTRACT_ID));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_PAYLOAD_ROOTS),
                RouteReportPayloadFields.Payload.NEXT_ACTIONS));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_PAYLOAD_ROOTS),
                "routeBenchmarkCache"));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_VALIDATION_ROOTS),
                RouteReportPayloadFields.VALIDATION_ROOT));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_VALIDATION_ROOTS),
                RoutePreflightDiagnosticFields.VALIDATION_ROOT));
        assertTrue(containsTextValue(
                routeContractBundle.path(RunnerRouteContractBundle.FIELD_VALIDATION_ROOTS),
                "routeBenchmarkCacheReportValidation"));
        JsonNode routeContractBundleValidation =
                root.path("runnerPlugins").path("routeContractBundleValidation");
        assertEquals(
                RunnerRouteContractBundle.schemaFingerprint(),
                routeContractBundleValidation
                        .path(RunnerRouteContractBundle.FIELD_SCHEMA_FINGERPRINT)
                        .asText());
        assertTrue(routeContractBundleValidation.path("passed").asBoolean());
        assertEquals(0, routeContractBundleValidation.path("problemCount").asInt());
        JsonNode runnerSelectionPolicy = root.path("runnerPlugins").path("selectionPolicy");
        assertEquals(
                RunnerRoutePolicyContract.CONTRACT_ID,
                runnerSelectionPolicy.path("contractId").asText());
        assertEquals(
                RunnerRoutePolicyContract.SCHEMA_VERSION,
                runnerSelectionPolicy.path("schemaVersion").asInt());
        assertEquals(
                RunnerRoutePolicyContract.schemaFingerprint(),
                runnerSelectionPolicy.path("schemaFingerprint").asText());
        assertEquals(RunnerRoutePolicy.AUTO, runnerSelectionPolicy.path("defaultRunner").asText());
        assertTrue(containsTextValue(
                runnerSelectionPolicy.path("supportedRunners"),
                RunnerRoutePolicy.GGUF));
        assertTrue(containsTextValue(
                runnerSelectionPolicy.path("aliases").path(RunnerRoutePolicy.LITERT),
                "tflite"));
        assertTrue(containsRunnerEffect(
                runnerSelectionPolicy.path("effects"),
                RunnerRoutePolicy.GGUF,
                "gguf",
                "gguf",
                true));
        assertTrue(root.path("extensionAvailability").path("extensions").isArray());
        assertEquals("global", root.path("extensionAvailability").path("extensionRegistryScope").asText());
        JsonNode extensionTotals = root.path("extensionAvailability").path("totals");
        assertTrue(extensionTotals.path("extensions").canConvertToInt());
        assertTrue(extensionTotals.path("healthy").canConvertToInt());
        assertTrue(extensionTotals.path("byStatus").isObject());
        assertTrue(extensionTotals.path("byKind").path("audio").canConvertToInt());
        assertTrue(root.path("extensionAvailability").path("byKind").path("audio").isArray());
        assertTrue(root.path("extensionAvailability").path("policy").path("status").isTextual());
        assertTrue(root.path("extensionAvailability")
                .path("policy")
                .path("configuration")
                .path("requiredProductionExtensions")
                .path("source")
                .isTextual());
        assertTrue(root.path("extensionAvailability").path("contract").path("status").isTextual());
        assertTrue(root.path("extensionAvailability").path("contract").path("violationCount").canConvertToInt());
        assertTrue(root.path("extensionAvailability").path("contract").path("byExtensionId").isObject());
        assertTrue(root.path("extensionAvailability").path("contractViolations").isArray());
        assertTrue(root.path("extensionAvailability").path("gate").path("status").isTextual());
        assertTrue(root.path("extensionAvailability").path("gate").path("violationCount").canConvertToInt());
        assertFalse(root.path("extensionAvailability").path("gate").path("failOnExtensionGate").asBoolean());
        JsonNode sulingExtension = root.path("extensionAvailability").path("byId").path("suling");
        assertEquals("audio", sulingExtension.path("kind").asText());
        assertTrue(sulingExtension.path("status").isTextual());
        assertTrue(sulingExtension.path("capabilities").isArray());
        assertTrue(sulingExtension.path("attributes").isObject());
        JsonNode sulingAudio = root.path("audioExtensions").path("suling");
        assertEquals(sulingExtension.path("status").asText(), sulingAudio.path("status").asText());
        assertEquals(sulingExtension.path("formats"), sulingAudio.path("formats"));
        assertTrue(sulingAudio.path("flacAvailable").isBoolean());
        assertTrue(root.path("modelFamilyBundle").path("selection").isArray());
        assertEquals("global", root.path("modelFamilyBundle").path("modelFamilyRegistryScope").asText());
        assertTrue(root.path("modelFamilyBundle").has("bundlePreset"));
        assertTrue(root.path("modelFamilyBundle").path("selectorSource").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("explicitSelectors").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetSelectors").isArray());
        assertTrue(root.path("modelFamilyBundle").path("defaultSelectors").isArray());
        assertTrue(root.path("modelFamilyBundle").path("policySource").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("presetRequiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetForbiddenFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetRequiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetForbiddenAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitRequiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitForbiddenFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitRequiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitForbiddenAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("forbiddenFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("forbiddenAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("policyStatus").path("status").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("policyViolations").path("missingRequiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("fixtureStatus").path("status").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("fixtureStatus").path("requiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("fixtureStatus").path("missingRequiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("status").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("healthy").isBoolean());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("summary").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("gate").path("status").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("gate").path("violationCount").canConvertToInt());
        assertTrue(root.path("modelFamilyBundle").path("gate").path("violations").isArray());
        assertTrue(root.path("modelFamilyBundle").path("gate").path("contractCategoryCounts").isObject());
        assertTrue(root.path("modelFamilyBundle").path("gate").path("contractRemediationHints").isArray());
        assertFalse(root.path("modelFamilyBundle").path("gate").path("failOnModelFamilyGate").asBoolean());
        assertTrue(root.path("pluginGates").path("status").isTextual());
        assertTrue(root.path("pluginGates").path("violationCount").canConvertToInt());
        assertTrue(root.path("pluginGates").path("violations").isArray());
        assertTrue(root.path("pluginGates").path("violationCategories").path("extension").canConvertToInt());
        assertTrue(root.path("pluginGates").path("violationCategories").path("modelFamily").canConvertToInt());
        assertTrue(root.path("pluginGates").path("violationCategories").path("runnerRoute").canConvertToInt());
        assertTrue(root.path("pluginGates").path("violationCategories").path("unifiedRuntime").canConvertToInt());
        assertTrue(root.path("pluginGates")
                .path("violationCategories")
                .path("unifiedRuntimeRequirement")
                .canConvertToInt());
        assertTrue(root.path("pluginGates").path("extensionStatus").isTextual());
        assertTrue(root.path("pluginGates").path("modelFamilyStatus").isTextual());
        assertTrue(root.path("pluginGates").path("extensionViolationCount").canConvertToInt());
        assertTrue(root.path("pluginGates").path("modelFamilyViolationCount").canConvertToInt());
        assertTrue(root.path("pluginGates").path("modelFamilyContractCategoryCounts").isObject());
        assertTrue(root.path("pluginGates").path("modelFamilyContractRemediationHints").isArray());
        assertFalse(root.path("pluginGates").path("failOnPluginGates").asBoolean());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("fixtureStatus").isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("fixtureMissingRequiredFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("problems").isArray());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("remediationHints").isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("missingSelectedFamilies")
                .isArray());
        assertEquals("global", root.path("unifiedRuntimes").path("registryScope").asText());
        assertTrue(root.path("unifiedRuntimes").path("runtimeCount").canConvertToInt());
        assertTrue(root.path("unifiedRuntimes").path("validCount").canConvertToInt());
        assertTrue(root.path("unifiedRuntimes").path("invalidCount").canConvertToInt());
        assertTrue(root.path("unifiedRuntimes").path("productionReadyCount").canConvertToInt());
        assertTrue(root.path("unifiedRuntimes").path("modelTypes").isArray());
        assertTrue(root.path("unifiedRuntimes").path("runtimes").isArray());
        assertTrue(root.path("unifiedRuntimes").path("conflicts").isArray());
        assertTrue(root.path("unifiedRuntimes").path("contractViolations").isArray());
        String modelFamilyRequirementSchemaVersion =
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA_VERSION;
        assertEquals(
                UnifiedRuntimeRequirementReportFields.SCHEMA_VERSION,
                root.path("unifiedRuntimes")
                        .path(modelFamilyRequirementSchemaVersion)
                        .asInt());
        JsonNode unifiedRuntimeRequirementSchema = root.path("unifiedRuntimes")
                .path(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_SCHEMA);
        assertEquals(
                UnifiedRuntimeRequirementReportFields.CONTRACT_ID,
                unifiedRuntimeRequirementSchema.path(
                        UnifiedRuntimeRequirementReportFields.Schema.CONTRACT_ID).asText());
        assertEquals(
                UnifiedRuntimeRequirementReportFields.SCHEMA_VERSION,
                unifiedRuntimeRequirementSchema.path(
                        UnifiedRuntimeRequirementReportFields.Schema.SCHEMA_VERSION).asInt());
        assertEquals(
                UnifiedRuntimeRequirementReportFields.schemaFingerprint(),
                unifiedRuntimeRequirementSchema.path(
                        UnifiedRuntimeRequirementReportFields.Schema.SCHEMA_FINGERPRINT).asText());
        assertTrue(unifiedRuntimeRequirementSchema.path(
                UnifiedRuntimeRequirementReportFields.Schema.SECTION_KEYS).isArray());
        assertTrue(containsTextValue(
                unifiedRuntimeRequirementSchema.path(UnifiedRuntimeRequirementReportFields.Schema.TOTALS_KEYS),
                UnifiedRuntimeRequirementReportFields.Totals.ISSUES));
        assertTrue(containsTextValue(
                unifiedRuntimeRequirementSchema.path(UnifiedRuntimeRequirementReportFields.Schema.REQUIREMENT_KEYS),
                UnifiedRuntimeRequirementReportFields.Requirement.PROBLEM_CODES));
        assertTrue(containsTextValue(
                unifiedRuntimeRequirementSchema.path(UnifiedRuntimeRequirementReportFields.Schema.ISSUE_KEYS),
                UnifiedRuntimeRequirementReportFields.Issue.AFFECTED_REQUIREMENTS));
        assertTrue(containsTextValue(
                unifiedRuntimeRequirementSchema.path(
                        UnifiedRuntimeRequirementReportFields.Schema.AFFECTED_REQUIREMENT_KEYS),
                UnifiedRuntimeRequirementReportFields.Requirement.AVAILABLE_INPUT_MODALITIES));
        JsonNode unifiedRuntimeRequirementContract = root.path("unifiedRuntimes")
                .path(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_CONTRACT);
        assertTrue(unifiedRuntimeRequirementContract.path(
                UnifiedRuntimeRequirementReportFields.Contract.PASSED).asBoolean());
        assertEquals(
                0,
                unifiedRuntimeRequirementContract.path(
                        UnifiedRuntimeRequirementReportFields.Contract.PROBLEM_COUNT).asInt());
        assertTrue(unifiedRuntimeRequirementContract.path(
                UnifiedRuntimeRequirementReportFields.Contract.PROBLEMS).isArray());
        assertTrue(root.path("unifiedRuntimes")
                .path(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_TOTALS)
                .path(UnifiedRuntimeRequirementReportFields.Totals.REQUIREMENT_COUNT)
                .canConvertToInt());
        assertTrue(root.path("unifiedRuntimes")
                .path(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_TOTALS)
                .path(UnifiedRuntimeRequirementReportFields.Totals.ISSUES)
                .isArray());
        assertTrue(root.path("unifiedRuntimes")
                .path(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_RECOMMENDATIONS)
                .isArray());
        assertTrue(root.path("unifiedRuntimes")
                .path(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENTS)
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("runtimeCompatibility")
                .path("selectedDirectSafetensor")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("runtimeCompatibility")
                .path("selectedDirectSafetensorSummary")
                .path("familyCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyBundle")
                .path("runtimeCompatibility")
                .path("requiresDirectSafetensorRuntime")
                .isBoolean());
        assertTrue(root.path("modelFamilyBundle")
                .path("productionSafety")
                .path("passed")
                .isBoolean());
        assertTrue(root.path("modelFamilyBundle")
                .path("productionSafety")
                .path("status")
                .isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("productionSafety")
                .path("pendingTokenizerFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("catalogReadiness")
                .path("status")
                .isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("catalogReadiness")
                .path("passed")
                .isBoolean());
        assertTrue(root.path("modelFamilyBundle")
                .path("catalogReadiness")
                .path("productionReadinessPendingFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("catalogReadiness")
                .path("directSafetensorPendingFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("productionSafetyStatus")
                .isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("productionPendingTokenizerFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("catalogReadinessStatus")
                .isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("catalogReadinessPassed")
                .isBoolean());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("productionReadinessPendingFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("directSafetensorPendingFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("tokenizerCoverage")
                .path("familyCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyBundle")
                .path("tokenizerCoverage")
                .path("tokenizerMetadataReadyCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyBundle")
                .path("tokenizerCoverage")
                .path("tokenizerMetadataPendingCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyBundle")
                .path("tokenizerCoverage")
                .path("missingFamilyIds")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("tokenizerCoverage")
                .path("pendingFamilyIds")
                .isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("tokenizerCoverage")
                .path("pendingReasons")
                .isObject());
        assertTrue(root.path("modelFamilyBundle").path("availableBundlePresets").isArray());
        assertTrue(root.path("modelFamilyBundle").has("activeBundlePreset"));
        assertTrue(root.path("modelFamilyBundle").path("requiresDirectSafetensorRuntime").isBoolean());
        assertTrue(root.path("modelFamilyBundle")
                .path("activeBundlePresetConformance")
                .path("status")
                .isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("activeBundlePresetConformance")
                .path("selectorsMatch")
                .isBoolean());
        assertTrue(root.path("modelFamilyBundle")
                .path("activeBundlePresetConformance")
                .path("policyInputsMatch")
                .isBoolean());
        if (!root.path("modelFamilyBundle").path("activeBundlePreset").isNull()) {
            assertTrue(root.path("modelFamilyBundle")
                    .path("activeBundlePreset")
                    .path("policyStatus")
                    .path("status")
                    .isTextual());
        }
        assertTrue(root.path("modelFamilyBundle").path("bundlePresets").isArray());
        if (!root.path("modelFamilyBundle").path("bundlePresets").isEmpty()) {
            JsonNode preset = root.path("modelFamilyBundle").path("bundlePresets").get(0);
            assertTrue(preset.path("selectedFamilies").isArray());
            assertTrue(preset.path("selectedCount").canConvertToInt());
            assertTrue(preset.path("productionSafety").path("status").isTextual());
            assertTrue(preset.path("productionSafety").path("pendingTokenizerFamilies").isArray());
            assertTrue(preset.path("policyStatus").path("status").isTextual());
            assertTrue(preset.path("policyViolations").path("missingRequiredAliases").isArray());
        }
        assertTrue(root.path("modelFamilyBundle").path("requestedFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requestedProfiles").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requestedAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("bundleAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("completeAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("partialAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("tokenizerMetadataPendingFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("tokenizerMetadataPendingReasons").isObject());
        assertTrue(root.path("modelFamilyPlugins").path("plugins").isArray());
        assertEquals("global", root.path("modelFamilyPlugins").path("modelFamilyRegistryScope").asText());
        assertTrue(root.path("modelFamilyPlugins").path("capabilityMatrix").isArray());
        assertTrue(root.path("modelFamilyPlugins").path("capabilityTotals").path("families").canConvertToInt());
        assertTrue(root.path("modelFamilyPlugins")
                .path("tokenizerCoverage")
                .path("tokenizerMetadataMissingCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyPlugins")
                .path("tokenizerCoverage")
                .path("tokenizerMetadataPendingCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyPlugins")
                .path("tokenizerCoverage")
                .path("pendingFamilyIds")
                .isArray());
        assertTrue(root.path("modelFamilyPlugins")
                .path("tokenizerCoverage")
                .path("pendingReasons")
                .isObject());
        assertTrue(root.path("modelFamilyPlugins")
                .path("tokenizerCoverage")
                .path("tokenizerKindCounts")
                .isObject());
        assertTrue(root.path("modelFamilyPlugins").path("runtimeManifests").isArray());
        if (!root.path("modelFamilyPlugins").path("runtimeManifests").isEmpty()) {
            assertTrue(root.path("modelFamilyPlugins")
                    .path("runtimeManifests")
                    .get(0)
                    .path("unifiedRuntimeRequirements")
                    .isArray());
        }
        assertTrue(root.path("modelFamilyPlugins")
                .path("runtimeCompatibility")
                .path("directSafetensor")
                .isArray());
        assertTrue(root.path("modelFamilyPlugins")
                .path("runtimeCompatibility")
                .path("directSafetensorSummary")
                .path("familyCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyPlugins")
                .path("runtimeCompatibility")
                .path("directSafetensorSummary")
                .path("problemCounts")
                .isObject());
        JsonNode resolutionReportSchema = root.path("modelFamilyPlugins").path("resolutionReportSchema");
        assertEquals(
                ModelFamilyResolutionReportFields.CONTRACT_ID,
                resolutionReportSchema.path(ModelFamilyResolutionReportFields.Schema.CONTRACT_ID).asText());
        assertEquals(
                ModelFamilyResolutionReportFields.SCHEMA_VERSION,
                resolutionReportSchema.path(ModelFamilyResolutionReportFields.Schema.SCHEMA_VERSION).asInt());
        assertEquals(
                ModelFamilyResolutionReportFields.schemaFingerprint(),
                resolutionReportSchema.path(ModelFamilyResolutionReportFields.Schema.SCHEMA_FINGERPRINT).asText());
        assertTrue(containsTextValue(
                resolutionReportSchema.path(ModelFamilyResolutionReportFields.Schema.RESOLUTION_FIELDS),
                ModelFamilyResolutionReportFields.Resolution.RUNTIME_COMPATIBILITY));
        assertTrue(containsTextValue(
                resolutionReportSchema.path(ModelFamilyResolutionReportFields.Schema.DIRECT_ARCHITECTURE_FIELDS),
                ModelFamilyResolutionReportFields.DirectArchitecture.SELECTED_ADAPTER_ID));
        assertTrue(containsTextValue(
                resolutionReportSchema.path(ModelFamilyResolutionReportFields.Schema.TOKENIZER_FIELDS),
                ModelFamilyResolutionReportFields.Tokenizer.MISSING_FILE_GROUPS));
        JsonNode resolutionReportSchemaValidation =
                root.path("modelFamilyPlugins").path("resolutionReportSchemaValidation");
        assertEquals(
                ModelFamilyResolutionReportFields.schemaFingerprint(),
                resolutionReportSchemaValidation
                        .path(ModelFamilyResolutionReportFields.Validation.SCHEMA_FINGERPRINT)
                        .asText());
        assertTrue(resolutionReportSchemaValidation
                .path(ModelFamilyResolutionReportFields.Validation.PASSED)
                .asBoolean());
        assertEquals(
                0,
                resolutionReportSchemaValidation
                        .path(ModelFamilyResolutionReportFields.Validation.PROBLEM_COUNT)
                        .asInt());
        assertTrue(resolutionReportSchemaValidation
                .path(ModelFamilyResolutionReportFields.Validation.PROBLEMS)
                .isArray());
        assertTrue(root.path("modelFamilyPlugins").path("contract").isObject());
        assertEquals(
                "model-family-contract-violation-report",
                root.path("modelFamilyPlugins").path("contract").path("contractId").asText());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contract")
                .path("schemaVersion")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contract")
                .path("schemaFingerprint")
                .isTextual());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contract")
                .path("schema")
                .isObject());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contract")
                .path("categoryKeys")
                .isArray());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contract")
                .path("remediationCatalog")
                .isObject());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contract")
                .path("categoryCounts")
                .isObject());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contractValidation")
                .path("passed")
                .isBoolean());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contractValidation")
                .path("problemCount")
                .canConvertToInt());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contractValidation")
                .path("problems")
                .isArray());
        assertTrue(root.path("modelFamilyPlugins")
                .path("contractViolationCategories")
                .isObject());
        assertTrue(root.path("modelFamilyPlugins").path("contractViolations").isArray());
        assertEquals(1, root.path("pluginDoctor").path("routeBenchmarkCache").path("schemaVersion").asInt());
        assertTrue(root.path("pluginDoctor").path("routeBenchmarkCache").path("profileTrustStatus").isTextual());
        assertTrue(root.path("pluginDoctor").path("routeBenchmarkCache").path("trustedEntryCount").canConvertToInt());
        assertFalse(root.path("pluginDoctor").path("routeBenchmarkCache").path("failOnRouteBenchmarkCache").asBoolean());
        assertTrue(root.path("pluginDoctor").path("routeBenchmarkCache").path("problems").isArray());
        assertTrue(root.path("pluginDoctor").path("routeBenchmarkCache").path("remediationHints").isArray());
        assertRouteBenchmarkCacheContract(root.path("pluginDoctor"));
        assertEquals("global", root.path("pluginDoctor").path("unifiedRuntimes").path("registryScope").asText());
        assertTrue(root.path("pluginDoctor").path("unifiedRuntimes").path("runtimeCount").canConvertToInt());
        assertTrue(root.path("dynamicPlugins").path("plugins").isArray());
    }

    @Test
    void jsonOutputLoadsExternalPluginsFromPluginClasspath(@TempDir Path tempDir) throws Exception {
        Path classesDir = compileExternalModulesFixture(tempDir);
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.externalPluginClasspath = List.of(classesDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            assertEquals("scoped", root.path("extensionAvailability").path("extensionRegistryScope").asText());
            assertEquals("scoped", root.path("modelFamilyBundle").path("modelFamilyRegistryScope").asText());
            assertEquals("scoped", root.path("modelFamilyPlugins").path("modelFamilyRegistryScope").asText());
            assertTrue(root.path("externalPluginClasspath").get(0).asText()
                    .contains(classesDir.getFileName().toString()));
            assertTrue(containsTextField(
                    root.path("extensionAvailability").path("extensions"),
                    "id",
                    EXTERNAL_EXTENSION_ID));
            assertTrue(containsTextField(
                    root.path("modelFamilyPlugins").path("plugins"),
                    "id",
                    EXTERNAL_MODEL_FAMILY_ID));
            assertTrue(containsTextField(
                    root.path("modelFamilyPlugins").path("runtimeManifests"),
                    "familyId",
                    EXTERNAL_MODEL_FAMILY_ID));
            assertTrue(ExtensionAvailabilityRegistry.global().availability(EXTERNAL_EXTENSION_ID).isEmpty());
            assertEquals("NOT_FOUND", ModelFamilyPluginRegistry.global()
                    .resolveModelType(EXTERNAL_MODEL_TYPE)
                    .status()
                    .name());
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonOutputReportsExternalUnifiedRuntimeConflictsFromPluginClasspath(@TempDir Path tempDir)
            throws Exception {
        Path classesDir = compileExternalUnifiedRuntimeFixture(tempDir);

        ExtensionsCommand command = new ExtensionsCommand();
        command.jsonOutput = true;
        command.externalPluginClasspath = List.of(classesDir.toString());

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            command.run();
        } finally {
            System.setOut(originalOut);
        }

        JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
        JsonNode unifiedRuntimes = root.path("unifiedRuntimes");
        assertEquals("scoped", unifiedRuntimes.path("registryScope").asText());
        assertTrue(unifiedRuntimes.path("runtimeCount").asInt() >= 2);
        assertTrue(containsTextValue(unifiedRuntimes.path("modelTypes"), "gemma4_unified"));
        assertTrue(containsTextField(unifiedRuntimes.path("runtimes"),
                "runtimeId",
                "first-command-gemma4-runtime"));
        assertTrue(containsTextField(unifiedRuntimes.path("runtimes"),
                "runtimeId",
                "second-command-gemma4-runtime"));
        assertTrue(containsTextField(unifiedRuntimes.path("conflicts"),
                "code",
                "duplicate_model_type_claim"));
        assertTrue(unifiedRuntimes.path("conflicts").toString().contains("gemma4_unified"));
        assertTrue(unifiedRuntimes.path("conflicts").toString().contains("first-command-gemma4-runtime"));
        assertTrue(unifiedRuntimes.path("conflicts").toString().contains("second-command-gemma4-runtime"));
        assertTrue(root.path("pluginDoctor")
                .path("unifiedRuntimes")
                .path("conflicts")
                .toString()
                .contains("duplicate_model_type_claim"));
        assertTrue(root.path("pluginDoctor").path("recommendations").toString()
                .contains("Detach duplicate unified runtime plugins"));
        assertFalse(root.path("pluginGates").path("passed").asBoolean());
        assertTrue(root.path("pluginGates").path("status").asText().contains("unified_runtime"));
        assertTrue(root.path("pluginGates").path("violations").toString().contains("unified-runtime:"));
        assertTrue(root.path("pluginGates").path("violations").toString()
                .contains("duplicate_model_type_claim"));
    }

    @Test
    void failOnPluginGatesThrowsForExternalUnifiedRuntimeConflicts(@TempDir Path tempDir)
            throws Exception {
        Path classesDir = compileExternalUnifiedRuntimeFixture(tempDir);

        ExtensionsCommand command = new ExtensionsCommand();
        command.jsonOutput = true;
        command.failOnPluginGates = true;
        command.externalPluginClasspath = List.of(classesDir.toString());

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        IllegalStateException error;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            error = assertThrows(IllegalStateException.class, command::run);
        } finally {
            System.setOut(originalOut);
        }

        assertTrue(error.getMessage().contains("unified-runtime:"));
        assertTrue(error.getMessage().contains("duplicate_model_type_claim"));
        JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
        assertTrue(root.path("pluginGates").path("failOnPluginGates").asBoolean());
        assertFalse(root.path("pluginGates").path("passed").asBoolean());
        assertTrue(root.path("pluginGates").path("status").asText().contains("unified_runtime"));
    }

    @Test
    void jsonDoctorOutputIsFocusedAndLoadsExternalPluginsFromPluginClasspath(@TempDir Path tempDir) throws Exception {
        Path classesDir = compileExternalModulesFixture(tempDir);
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginClasspath = List.of(classesDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            assertEquals(1, root.path("schemaVersion").asInt());
            assertEquals("pluginDoctor", root.path("kind").asText());
            assertEquals("scoped", root.path("registryScope").asText());
            assertTrue(root.path("classpath").path("active").asBoolean());
            assertTrue(root.path("status").isTextual());
            assertTrue(root.path("recommendations").isArray());
            assertFalse(root.has("kernel"));
            assertTrue(root.path("externalPluginClasspath").get(0).asText()
                    .contains(classesDir.getFileName().toString()));
            assertTrue(containsTextValue(root.path("extensions").path("ids"), EXTERNAL_EXTENSION_ID));
            assertTrue(containsTextValue(root.path("modelFamilies").path("ids"), EXTERNAL_MODEL_FAMILY_ID));
            assertTrue(root.path("modelFamilies").path("productionSafety").path("status").isTextual());
            assertTrue(root.path("modelFamilies").path("productionSafety").path("pendingTokenizerFamilies").isArray());
            assertEquals(1, root.path("routeBenchmarkCache").path("schemaVersion").asInt());
            assertTrue(root.path("routeBenchmarkCache").path("profileTrustStatus").isTextual());
            assertTrue(root.path("routeBenchmarkCache").path("trustedEntryCount").canConvertToInt());
            assertFalse(root.path("routeBenchmarkCache").path("failOnRouteBenchmarkCache").asBoolean());
            assertTrue(root.path("routeBenchmarkCache").path("problems").isArray());
            assertTrue(root.path("routeBenchmarkCache").path("remediationHints").isArray());
            assertRouteBenchmarkCacheContract(root);
            assertTrue(ExtensionAvailabilityRegistry.global().availability(EXTERNAL_EXTENSION_ID).isEmpty());
            assertEquals("NOT_FOUND", ModelFamilyPluginRegistry.global()
                    .resolveModelType(EXTERNAL_MODEL_TYPE)
                    .status()
                    .name());
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void textDoctorOutputShowsScopedExternalPluginSummary(@TempDir Path tempDir) throws Exception {
        Path classesDir = compileExternalModulesFixture(tempDir);
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.pluginDoctor = true;
            command.externalPluginClasspath = List.of(classesDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            String text = output.toString(StandardCharsets.UTF_8);
            assertTrue(text.contains("=== Plugin Doctor ==="));
            assertTrue(text.contains("Scope: scoped"));
            assertTrue(text.contains("External plugin classpath:"));
            assertTrue(text.contains("Overall:"));
            assertTrue(text.contains("Gate categories:"));
            assertTrue(text.contains("Route benchmark cache:"));
            assertTrue(text.contains("Extension ids:"));
            assertTrue(text.contains(EXTERNAL_EXTENSION_ID));
            assertTrue(text.contains(EXTERNAL_MODEL_FAMILY_ID));
            assertTrue(text.contains("Bundle production safety:"));
            assertTrue(text.contains("Selected model-family requirements:"));
            assertTrue(text.contains("Requirement report contract:"));
            assertTrue(text.contains("problems=0"));
            assertTrue(text.contains("Recommendations"));
            assertTrue(ExtensionAvailabilityRegistry.global().availability(EXTERNAL_EXTENSION_ID).isEmpty());
            assertEquals("NOT_FOUND", ModelFamilyPluginRegistry.global()
                    .resolveModelType(EXTERNAL_MODEL_TYPE)
                    .status()
                    .name());
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void textOutputShowsUnifiedRuntimeRequirementProblemCodeSummary(@TempDir Path tempDir) throws Exception {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "cli_missing_unified_runtime_family",
                        "CLI Missing Unified Runtime Family",
                        List.of("cli_missing_unified"),
                        List.of("CliMissingUnifiedForConditionalGeneration"),
                        List.of(ModelFamilyCapability.MULTIMODAL),
                        Map.of(
                                "bundle_profile", "optional",
                                "unified_model_type", "cli_missing_unified",
                                "unified_runtime_required_modalities", "text,image",
                                "unified_runtime_reason", "test missing unified runtime"));
            }
        };
        ModelFamilyPluginRegistry.global().register(plugin);
        Path manifestRoot = tempDir.resolve("manifest-classpath");
        Path manifestDir = manifestRoot.resolve("META-INF");
        Files.createDirectories(manifestDir);
        Files.writeString(manifestDir.resolve("gollek-model-family-bundle.properties"), """
                schemaVersion=1
                bundleFingerprint=sha256:test
                selectors=optional
                families=cli_missing_unified_runtime_family
                familyCount=1
                availableFamilies=cli_missing_unified_runtime_family
                policyPassed=true
                policyViolationCount=0
                fixturePassed=true
                fixtureRequiredFamilies=cli_missing_unified_runtime_family
                fixtureRequiredFamilyCount=1
                fixtureRequiredPassedCount=1
                """, StandardCharsets.UTF_8);

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        ClassLoader originalClassLoader = Thread.currentThread().getContextClassLoader();
        try (URLClassLoader manifestClassLoader = new URLClassLoader(
                new java.net.URL[] { manifestRoot.toUri().toURL() },
                originalClassLoader) {
            @Override
            public java.net.URL getResource(String name) {
                if ("META-INF/gollek-model-family-bundle.properties".equals(name)) {
                    java.net.URL local = findResource(name);
                    if (local != null) {
                        return local;
                    }
                }
                return super.getResource(name);
            }
        }) {
            Thread.currentThread().setContextClassLoader(manifestClassLoader);
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            ExtensionsCommand command = new ExtensionsCommand();
            command.pluginDoctor = true;
            command.run();
        } finally {
            System.setOut(originalOut);
            Thread.currentThread().setContextClassLoader(originalClassLoader);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }

        String text = output.toString(StandardCharsets.UTF_8);
        assertTrue(text.contains("Selected model-family requirements:"));
        assertTrue(text.contains("Requirement report contract:"));
        assertTrue(text.contains("problems=0"));
        assertTrue(text.contains("Attention problem codes:"));
        assertTrue(text.contains("Attention issues:"));
        assertTrue(text.contains("Attention recommendations:"));
        assertTrue(text.contains(UnifiedRuntimeRequirementProblemCodes.MISSING_RUNTIME));
        assertTrue(text.contains("families=cli_missing_unified_runtime_family"));
        assertTrue(text.contains("model_types=cli_missing_unified"));
        assertTrue(text.contains("cli_missing_unified"));
        assertTrue(text.contains("Attach unified runtime plugins for missing selected model-family requirements"));
        assertFalse(text.contains("Attach production-ready unified runtimes for selected model-family requirements"));
    }

    @Test
    void jsonDoctorOutputReportsReadyPluginDirectoryModelFamilyJar(@TempDir Path tempDir) throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("ready"));
        addPluginCoreContract(classesDir);
        writeJar(classesDir, pluginDir.resolve("modules-external-ready.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            assertTrue(readiness.path("jarCount").asInt() >= 1);
            assertTrue(readiness.path("modelFamilyPluginCandidates").asInt() >= 1);
            assertTrue(readiness.path("pluginInstallReady").asInt() >= 1);
            assertTrue(readiness.path("pluginTokenizerMetadataReady").asInt() >= 1);
            assertTrue(readiness.path("pluginTokenizerMetadataPending").canConvertToInt());
            assertTrue(readiness.path("pluginTokenizerMetadataMissing").canConvertToInt());
            assertTrue(readiness.path("pluginTokenizerMetadataInvalid").canConvertToInt());
            assertTrue(containsPathFilename(
                    readiness.path("pluginTokenizerMetadataReadyJars"),
                    "modules-external-ready.jar"));
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-ready.jar");
            assertTrue(jar.path("pluginInstallCandidate").asBoolean());
            assertTrue(jar.path("pluginInstallReady").asBoolean());
            assertTrue(jar.path("pluginInstallErrors").isEmpty());
            assertEquals("external.modules.ModulesExternalModelFamilyPlugin",
                    jar.path("pluginMainClass").asText());
            assertEquals("model-family/modules_external_family",
                    jar.path("pluginDescriptorId").asText());
            assertEquals("model-family", jar.path("pluginExtensionPoint").asText());
            assertEquals(EXTERNAL_MODEL_FAMILY_ID, jar.path("pluginFamilies").get(0).asText());
            assertEquals("optional", jar.path("pluginBundleProfile").asText());
            assertEquals("hf-bpe", jar.path("pluginTokenizerKind").asText());
            assertEquals("hf-bpe", jar.path("pluginTokenizerKinds").get(0).asText());
            assertTrue(jar.path("pluginTokenizerMetadataDescriptorStatus").asText().isBlank());
            assertEquals("ready", jar.path("pluginTokenizerMetadataStatus").asText());
            assertTrue(jar.path("pluginTokenizerMetadataPendingReason").asText().isBlank());
            assertTrue(containsTextValue(root.path("modelFamilies").path("ids"), EXTERNAL_MODEL_FAMILY_ID));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void textDoctorOutputShowsReadyPluginDirectoryJarDetails(@TempDir Path tempDir) throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("ready-text"));
        addPluginCoreContract(classesDir);
        writeJar(classesDir, pluginDir.resolve("modules-external-ready.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            String text = output.toString(StandardCharsets.UTF_8);
            assertTrue(text.contains("Plugin jar readiness:"));
            assertTrue(text.contains("ready="));
            assertTrue(text.contains("tokenizer-metadata="));
            assertTrue(text.contains("modules-external-ready.jar"));
            assertTrue(text.contains("(ready): mainClass=external.modules.ModulesExternalModelFamilyPlugin"));
            assertTrue(text.contains("pluginId=model-family/modules_external_family"));
            assertTrue(text.contains("families=" + EXTERNAL_MODEL_FAMILY_ID));
            assertTrue(text.contains("tokenizerKind=hf-bpe"));
            assertTrue(text.contains("tokenizerKinds=hf-bpe"));
            assertTrue(text.contains("tokenizerMetadataStatus=ready"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonDoctorOutputReportsMissingPluginTokenizerMetadata(@TempDir Path tempDir)
            throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("missing-tokenizer-metadata"));
        addPluginCoreContract(classesDir, externalModelFamilyPluginDescriptorWithoutTokenizerMetadata());
        writeJar(classesDir, pluginDir.resolve("modules-external-missing-tokenizer.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            assertTrue(readiness.path("pluginInstallReady").asInt() >= 1);
            assertTrue(readiness.path("pluginTokenizerMetadataPending").canConvertToInt());
            assertTrue(readiness.path("pluginTokenizerMetadataMissing").asInt() >= 1);
            assertTrue(readiness.path("pluginTokenizerMetadataInvalid").canConvertToInt());
            assertTrue(containsPathFilename(
                    readiness.path("pluginTokenizerMetadataMissingJars"),
                    "modules-external-missing-tokenizer.jar"));
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-missing-tokenizer.jar");
            assertTrue(jar.path("pluginInstallReady").asBoolean());
            assertTrue(jar.path("pluginTokenizerKind").asText().isBlank());
            assertTrue(jar.path("pluginTokenizerKinds").isEmpty());
            assertEquals("missing", jar.path("pluginTokenizerMetadataStatus").asText());
            assertTrue(jar.path("pluginInstallErrors").isEmpty());
            assertFalse(root.path("pluginGates").path("violations").toString()
                    .contains("properties.tokenizerKind"));
            assertTrue(root.path("recommendations").toString()
                    .contains("properties.tokenizerKind or tokenizerKinds"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonDoctorOutputReportsPendingPluginTokenizerMetadata(@TempDir Path tempDir)
            throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("pending-tokenizer-metadata"));
        addPluginCoreContract(classesDir, externalModelFamilyPluginDescriptorWithPendingTokenizerMetadata());
        writeJar(classesDir, pluginDir.resolve("modules-external-pending-tokenizer.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            assertTrue(readiness.path("pluginInstallReady").asInt() >= 1);
            assertTrue(readiness.path("pluginTokenizerMetadataPending").asInt() >= 1);
            assertTrue(readiness.path("pluginTokenizerMetadataMissing").canConvertToInt());
            assertTrue(containsPathFilename(
                    readiness.path("pluginTokenizerMetadataPendingJars"),
                    "modules-external-pending-tokenizer.jar"));
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-pending-tokenizer.jar");
            assertTrue(jar.path("pluginInstallReady").asBoolean());
            assertTrue(jar.path("pluginInstallErrors").isEmpty());
            assertTrue(jar.path("pluginTokenizerKinds").isEmpty());
            assertEquals("pending", jar.path("pluginTokenizerMetadataDescriptorStatus").asText());
            assertEquals("pending", jar.path("pluginTokenizerMetadataStatus").asText());
            assertEquals("tokenizer adapter pending fixture stabilization",
                    jar.path("pluginTokenizerMetadataPendingReason").asText());
            assertFalse(root.path("pluginGates").path("violations").toString()
                    .contains("tokenizerMetadataStatus"));
            assertTrue(root.path("recommendations").toString()
                    .contains("pending-tokenizer model-family plugin jars"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonDoctorOutputReportsMultiplePluginTokenizerKinds(@TempDir Path tempDir)
            throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("multi-tokenizer-kinds"));
        addPluginCoreContract(classesDir, externalModelFamilyPluginDescriptorWithTokenizerKinds());
        writeJar(classesDir, pluginDir.resolve("modules-external-multi-tokenizer.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-multi-tokenizer.jar");
            assertTrue(jar.path("pluginInstallReady").asBoolean());
            assertEquals("sentencepiece", jar.path("pluginTokenizerKind").asText());
            assertEquals("sentencepiece", jar.path("pluginTokenizerKinds").get(0).asText());
            assertEquals("hf-bpe", jar.path("pluginTokenizerKinds").get(1).asText());
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonDoctorOutputAcceptsExtendedPluginTokenizerKind(@TempDir Path tempDir)
            throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("processor-tokenizer-kind"));
        addPluginCoreContract(classesDir,
                externalModelFamilyPluginDescriptorWithTokenizerKind("vision-processor"));
        writeJar(classesDir, pluginDir.resolve("modules-external-processor-tokenizer.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-processor-tokenizer.jar");
            assertTrue(jar.path("pluginInstallReady").asBoolean());
            assertTrue(jar.path("pluginInstallErrors").isEmpty());
            assertTrue(readiness.path("pluginTokenizerMetadataReady").asInt() >= 1);
            assertEquals("vision-processor", jar.path("pluginTokenizerKind").asText());
            assertEquals("vision-processor", jar.path("pluginTokenizerKinds").get(0).asText());
            assertEquals("ready", jar.path("pluginTokenizerMetadataStatus").asText());
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonDoctorOutputReportsMissingPluginFamiliesAsNotReady(@TempDir Path tempDir)
            throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("missing-families"));
        addPluginCoreContract(classesDir, externalModelFamilyPluginDescriptorWithoutFamilies());
        writeJar(classesDir, pluginDir.resolve("modules-external-missing-families.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-missing-families.jar");
            assertTrue(jar.path("pluginInstallCandidate").asBoolean());
            assertFalse(jar.path("pluginInstallReady").asBoolean());
            assertEquals("model-family/modules_external_family",
                    jar.path("pluginDescriptorId").asText());
            assertEquals("model-family", jar.path("pluginExtensionPoint").asText());
            assertTrue(jar.path("pluginFamilies").isEmpty());
            assertTrue(jar.path("pluginInstallErrors").toString().contains("properties.families"));
            assertTrue(root.path("pluginGates").path("violations").toString().contains("properties.families"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonDoctorOutputReportsInvalidPluginTokenizerKindAsNotReady(@TempDir Path tempDir)
            throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("invalid-tokenizer-kind"));
        addPluginCoreContract(classesDir, externalModelFamilyPluginDescriptorWithTokenizerKind("mystery-tokenizer"));
        writeJar(classesDir, pluginDir.resolve("modules-external-invalid-tokenizer.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-invalid-tokenizer.jar");
            assertTrue(jar.path("pluginInstallCandidate").asBoolean());
            assertFalse(jar.path("pluginInstallReady").asBoolean());
            assertTrue(readiness.path("pluginTokenizerMetadataInvalid").asInt() >= 1);
            assertTrue(containsPathFilename(
                    readiness.path("pluginTokenizerMetadataInvalidJars"),
                    "modules-external-invalid-tokenizer.jar"));
            assertEquals("mystery-tokenizer", jar.path("pluginTokenizerKind").asText());
            assertEquals("mystery-tokenizer", jar.path("pluginTokenizerKinds").get(0).asText());
            assertEquals("invalid", jar.path("pluginTokenizerMetadataStatus").asText());
            assertTrue(jar.path("pluginInstallErrors").toString().contains("properties.tokenizerKind"));
            assertTrue(root.path("pluginGates").path("violations").toString().contains("properties.tokenizerKind"));
            assertTrue(root.path("recommendations").toString()
                    .contains("vision-text-processor"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonDoctorOutputReportsLegacyPluginDirectoryModelFamilyJarAsNotReady(@TempDir Path tempDir)
            throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("legacy"));
        writeJar(classesDir, pluginDir.resolve("modules-external-legacy.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode readiness = root.path("pluginDirectories").path("jarReadiness");
            assertTrue(readiness.path("jarCount").asInt() >= 1);
            assertTrue(readiness.path("modelFamilyPluginCandidates").asInt() >= 1);
            assertTrue(readiness.path("pluginInstallNotReady").asInt() >= 1);
            JsonNode jar = pluginDirectoryJar(readiness.path("jars"), "modules-external-legacy.jar");
            assertTrue(jar.path("pluginInstallCandidate").asBoolean());
            assertFalse(jar.path("pluginInstallReady").asBoolean());
            assertTrue(jar.path("pluginInstallErrors").toString()
                    .contains("tech.kayys.gollek.spi.plugin.GollekPlugin"));
            assertTrue(jar.path("pluginInstallErrors").toString().contains("plugin.json"));
            assertTrue(root.path("pluginGates").path("status").asText().contains("plugin_directory"));
            assertTrue(root.path("pluginGates").path("violations").toString().contains("plugin-directory:"));
            assertTrue(root.path("recommendations").toString().contains("not-ready model-family plugin jars"));
            assertTrue(containsTextValue(root.path("modelFamilies").path("ids"), EXTERNAL_MODEL_FAMILY_ID));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void textDoctorOutputShowsPluginDirectoryJarReadiness(@TempDir Path tempDir) throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("legacy"));
        writeJar(classesDir, pluginDir.resolve("modules-external-legacy.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.pluginDoctor = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            String text = output.toString(StandardCharsets.UTF_8);
            assertTrue(text.contains("Plugin dirs active:"));
            assertTrue(text.contains(pluginDir.toAbsolutePath().normalize().toString()));
            assertTrue(text.contains("Plugin dir sources: command=1"));
            assertTrue(text.contains("Plugin jar readiness:"));
            assertTrue(text.contains("not-ready="));
            assertTrue(text.contains("tokenizer-metadata="));
            assertTrue(text.contains("modules-external-legacy.jar"));
            assertTrue(text.contains("plugin.json"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void jsonOutputIncludesExtensionAvailabilityContractViolations() throws Exception {
        ExtensionAvailabilityProvider provider = badExtensionAvailabilityProvider();
        ExtensionAvailabilityRegistry.global().register(provider);
        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode violations = new ObjectMapper()
                    .readTree(output.toString(StandardCharsets.UTF_8))
                    .path("extensionAvailability")
                    .path("contractViolations");
            assertTrue(containsViolationCode(violations, "provider_id_invalid"));
            assertTrue(containsViolationCode(violations, "availability_id_mismatch"));
            JsonNode contract = new ObjectMapper()
                    .readTree(output.toString(StandardCharsets.UTF_8))
                    .path("extensionAvailability")
                    .path("contract");
            assertEquals("failed", contract.path("status").asText());
            assertFalse(contract.path("passed").asBoolean());
            assertTrue(contract.path("violationCount").asInt() >= 2);
            assertTrue(contract.path("byExtensionId").path("Bad Extension").isArray());
            JsonNode gate = new ObjectMapper()
                    .readTree(output.toString(StandardCharsets.UTF_8))
                    .path("extensionAvailability")
                    .path("gate");
            assertEquals("contract_failed", gate.path("status").asText());
            assertFalse(gate.path("passed").asBoolean());
            assertTrue(gate.path("violations").get(0).asText().startsWith("contract:"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(provider.extensionId());
        }
    }

    @Test
    void jsonCapabilityMatrixIncludesArchitectureAdapterHealth() throws Exception {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "json_adapter_family",
                        "JSON Adapter Family",
                        List.of("json_adapter"),
                        List.of("JsonAdapterForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/json_adapter"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture(
                        "json-adapter",
                        "json_adapter",
                        "JsonAdapterForCausalLM"));
            }
        };
        ModelFamilyPluginRegistry.global().register(plugin);

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.run();
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }

        JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
        JsonNode row = matrixEntry(root, "json_adapter_family");
        assertEquals("json-adapter", row.path("architectureAdapterIds").get(0).asText());
        assertEquals(1, row.path("architectureAdapterCount").asInt());
        assertTrue(row.path("architectureAdapterPresent").asBoolean());
        assertTrue(root.path("modelFamilyPlugins")
                .path("capabilityTotals")
                .path("architectureAdapterCount")
                .asInt() >= 1);
    }

    @Test
    void jsonOutputIncludesConfiguredExtensionAvailabilityPolicy() throws Exception {
        withSystemProperty(ExtensionAvailabilityPolicy.REQUIRED_PRODUCTION_PROPERTY, "missing-audio", () -> {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                command.run();
            } finally {
                System.setOut(originalOut);
            }

            JsonNode policy = new ObjectMapper()
                    .readTree(output.toString(StandardCharsets.UTF_8))
                    .path("extensionAvailability")
                    .path("policy");
            assertTrue(policy.path("configured").asBoolean());
            assertFalse(policy.path("passed").asBoolean());
            assertEquals("failed", policy.path("status").asText());
            assertEquals("missing-audio", policy.path("requiredProductionExtensions").get(0).asText());
            assertTrue(policy.path("violations").get(0).asText().contains("production extension missing-audio"));
            JsonNode gate = new ObjectMapper()
                    .readTree(output.toString(StandardCharsets.UTF_8))
                    .path("extensionAvailability")
                    .path("gate");
            assertEquals("policy_failed", gate.path("status").asText());
            assertFalse(gate.path("passed").asBoolean());
            assertTrue(gate.path("violations").get(0).asText().startsWith("policy:"));
            assertEquals("system_property", policy
                    .path("configuration")
                    .path("requiredProductionExtensions")
                    .path("source")
                    .asText());
            assertEquals(
                    ExtensionAvailabilityPolicy.REQUIRED_PRODUCTION_PROPERTY,
                    policy.path("configuration")
                            .path("requiredProductionExtensions")
                            .path("propertyName")
                            .asText());
        });
    }

    @Test
    void textOutputShowsPartialDirectSafetensorCaveats() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "text_caveat_family",
                        "Text Caveat Family",
                        List.of("text_caveat"),
                        List.of("TextCaveatForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/text_caveat",
                                "direct_safetensor", "experimental_text_path",
                                "moe_direct_safetensor", "pending_packed_expert_runtime"));
            }
        };
        ModelFamilyPluginRegistry.global().register(plugin);

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            new ExtensionsCommand().run();
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }

        String text = output.toString(StandardCharsets.UTF_8);
        assertTrue(text.contains("Capability matrix summary:"));
        assertTrue(text.contains("Build gate:"));
        assertTrue(text.contains("Build direct runtime:"));
        assertTrue(text.contains("Partial direct SafeTensor caveats:"));
        assertTrue(text.contains("text_caveat_family: moe:pending_packed_expert_runtime"));
        assertTrue(text.contains("=== Unified Multimodal Runtimes ==="));
        assertTrue(text.contains("Selected model-family requirements:"));
        assertTrue(text.contains("=== Extension Availability ==="));
        assertTrue(text.contains("Contract:"));
        assertTrue(text.contains("Release gate:"));
        assertTrue(text.contains("=== Plugin Gates ==="));
        assertTrue(text.contains("Combined gate:"));
        assertTrue(text.contains("Gate categories:"));
        assertTrue(text.contains("suling"));
    }

    @Test
    void textOutputGroupsModelFamilyContractViolationsByCategory() {
        ModelFamilyPlugin plugin = badModelFamilyContractPlugin();
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                new ExtensionsCommand().run();
            } finally {
                System.setOut(originalOut);
            }

            String text = output.toString(StandardCharsets.UTF_8);
            assertTrue(text.contains("Model family contract violations:"));
            assertTrue(text.contains("Model family contract categories:"));
            assertTrue(text.contains("Model family contract recommendations:"));
            assertTrue(text.contains("descriptor="));
            assertTrue(text.contains("bundleProfile="));
            assertTrue(text.contains("[descriptor]"));
            assertTrue(text.contains("[bundleProfile]"));
            assertTrue(text.contains("invalid_family_id"));
            assertTrue(text.contains("unknown_bundle_profile"));
            assertTrue(text.contains("metadata.bundle_profile"));
            assertTrue(text.contains("Normalize descriptor"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void textOutputShowsConfiguredExtensionAvailabilityPolicy() throws Exception {
        withSystemProperty(ExtensionAvailabilityPolicy.REQUIRED_PRODUCTION_PROPERTY, "missing-audio", () -> {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                new ExtensionsCommand().run();
            } finally {
                System.setOut(originalOut);
            }

            String text = output.toString(StandardCharsets.UTF_8);
            assertTrue(text.contains("=== Extension Availability ==="));
            assertTrue(text.contains("Policy: "));
            assertTrue(text.contains("Release gate: "));
            assertTrue(text.contains("failed"));
            assertTrue(text.contains("production extension missing-audio"));
        });
    }

    @Test
    void failOnExtensionGateThrowsAfterPrintingReport() throws Exception {
        withSystemProperty(ExtensionAvailabilityPolicy.REQUIRED_PRODUCTION_PROPERTY, "missing-audio", () -> {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.failOnExtensionGate = true;

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            IllegalStateException error;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                error = assertThrows(IllegalStateException.class, command::run);
            } finally {
                System.setOut(originalOut);
            }

            assertTrue(error.getMessage().contains("policy_failed"));
            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            assertTrue(root.path("extensionAvailability").path("gate").path("failOnExtensionGate").asBoolean());
            assertEquals("policy_failed", root.path("extensionAvailability").path("gate").path("status").asText());
        });
    }

    @Test
    void failOnModelFamilyGateThrowsAfterPrintingReport() throws Exception {
        ModelFamilyPlugin plugin = badModelFamilyContractPlugin();
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.failOnModelFamilyGate = true;

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            IllegalStateException error;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                error = assertThrows(IllegalStateException.class, command::run);
            } finally {
                System.setOut(originalOut);
            }

            assertTrue(error.getMessage().contains("contract:"));
            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode gate = root.path("modelFamilyBundle").path("gate");
            assertTrue(gate.path("failOnModelFamilyGate").asBoolean());
            assertFalse(gate.path("passed").asBoolean());
            assertTrue(gate.path("violations").toString().contains("contract:"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void failOnPluginGatesThrowsAfterPrintingReport() throws Exception {
        ModelFamilyPlugin plugin = badModelFamilyContractPlugin();
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.failOnPluginGates = true;

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            IllegalStateException error;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                error = assertThrows(IllegalStateException.class, command::run);
            } finally {
                System.setOut(originalOut);
            }

            assertTrue(error.getMessage().contains("model-family:"));
            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode gate = root.path("pluginGates");
            assertTrue(gate.path("failOnPluginGates").asBoolean());
            assertFalse(gate.path("passed").asBoolean());
            assertTrue(gate.path("status").asText().contains("model_family_failed"));
            assertTrue(gate.path("violations").toString().contains("model-family:"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void failOnRouteBenchmarkCacheThrowsAfterPrintingReport() throws Exception {
        withSystemProperty(RunnerRouteBenchmarkCache.CACHE_ENABLED_PROPERTY, "false", () -> {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.failOnRouteBenchmarkCache = true;

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            IllegalStateException error;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                error = assertThrows(IllegalStateException.class, command::run);
            } finally {
                System.setOut(originalOut);
            }

            assertTrue(error.getMessage().contains("cache_disabled"));
            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode routeBenchmarkCache = root.path("runnerPlugins").path("routeBenchmarkCache");
            assertTrue(routeBenchmarkCache.path("failOnRouteBenchmarkCache").asBoolean());
            assertTrue(routeBenchmarkCache.path("problems").toString().contains("cache_disabled"));
        });
    }

    @Test
    void failOnPluginGatesThrowsForLegacyPluginDirectoryModelFamilyJar(@TempDir Path tempDir) throws Exception {
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Path classesDir = compileExternalModulesFixture(tempDir.resolve("legacy-gate"));
        writeJar(classesDir, pluginDir.resolve("modules-external-legacy.jar"));
        ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);

        try {
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.pluginDoctor = true;
            command.failOnPluginGates = true;
            command.externalPluginDirectories = List.of(pluginDir.toString());

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            IllegalStateException error;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                error = assertThrows(IllegalStateException.class, command::run);
            } finally {
                System.setOut(originalOut);
            }

            assertTrue(error.getMessage().contains("plugin-directory:"));
            assertTrue(error.getMessage().contains("plugin.json"));
            JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
            JsonNode gate = root.path("pluginGates");
            assertTrue(gate.path("failOnPluginGates").asBoolean());
            assertFalse(gate.path("passed").asBoolean());
            assertTrue(gate.path("status").asText().contains("plugin_directory"));
            assertTrue(gate.path("violations").toString().contains("plugin-directory:"));
            assertTrue(root.path("pluginDirectoryReadiness")
                    .path("pluginInstallNotReady")
                    .asInt() >= 1);
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(EXTERNAL_EXTENSION_ID);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_MODEL_FAMILY_ID);
        }
    }

    @Test
    void textOutputIncludesExtensionAvailabilityContractViolations() {
        ExtensionAvailabilityProvider provider = badExtensionAvailabilityProvider();
        ExtensionAvailabilityRegistry.global().register(provider);
        try {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            try {
                System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
                new ExtensionsCommand().run();
            } finally {
                System.setOut(originalOut);
            }

            String text = output.toString(StandardCharsets.UTF_8);
            assertTrue(text.contains("Contract violations:"));
            assertTrue(text.contains("provider_id_invalid"));
            assertTrue(text.contains("availability_id_mismatch"));
        } finally {
            ExtensionAvailabilityRegistry.global().unregister(provider.extensionId());
        }
    }

    private static Path compileExternalModulesFixture(Path tempDir) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertTrue(compiler != null, "Java compiler must be available for external plugin fixture compilation");

        Path sourceRoot = tempDir.resolve("src");
        Path classesDir = tempDir.resolve("classes");
        Path extensionSource = sourceRoot.resolve("external/modules/ModulesExternalExtensionProvider.java");
        Path modelFamilySource = sourceRoot.resolve("external/modules/ModulesExternalModelFamilyPlugin.java");
        Files.createDirectories(extensionSource.getParent());
        Files.createDirectories(classesDir);
        Files.writeString(extensionSource, externalExtensionSource(), StandardCharsets.UTF_8);
        Files.writeString(modelFamilySource, externalModelFamilySource(), StandardCharsets.UTF_8);

        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
            Iterable<? extends javax.tools.JavaFileObject> units =
                    fileManager.getJavaFileObjectsFromFiles(List.of(
                            extensionSource.toFile(),
                            modelFamilySource.toFile()));
            List<String> options = List.of(
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", classesDir.toString());
            Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
            assertTrue(Boolean.TRUE.equals(compiled), "external modules fixture should compile");
        }

        writeServiceDescriptor(
                classesDir,
                "tech.kayys.gollek.plugin.core.ExtensionAvailabilityProvider",
                "external.modules.ModulesExternalExtensionProvider");
        writeServiceDescriptor(
                classesDir,
                "tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin",
                "external.modules.ModulesExternalModelFamilyPlugin");
        return classesDir;
    }

    private static Path compileExternalUnifiedRuntimeFixture(Path tempDir) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertTrue(compiler != null, "Java compiler must be available for external runtime fixture compilation");

        Path sourceRoot = tempDir.resolve("src");
        Path classesDir = tempDir.resolve("classes");
        Path firstSource = sourceRoot.resolve("external/runtime/FirstCommandGemma4Runtime.java");
        Path secondSource = sourceRoot.resolve("external/runtime/SecondCommandGemma4Runtime.java");
        Files.createDirectories(firstSource.getParent());
        Files.createDirectories(classesDir);
        Files.writeString(
                firstSource,
                externalUnifiedRuntimeSource(
                        "FirstCommandGemma4Runtime",
                        "first-command-gemma4-runtime"),
                StandardCharsets.UTF_8);
        Files.writeString(
                secondSource,
                externalUnifiedRuntimeSource(
                        "SecondCommandGemma4Runtime",
                        "second-command-gemma4-runtime"),
                StandardCharsets.UTF_8);

        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
            Iterable<? extends javax.tools.JavaFileObject> units =
                    fileManager.getJavaFileObjectsFromFiles(List.of(
                            firstSource.toFile(),
                            secondSource.toFile()));
            List<String> options = List.of(
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", classesDir.toString());
            Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
            assertTrue(Boolean.TRUE.equals(compiled), "external unified runtime fixture should compile");
        }

        Path serviceFile = classesDir.resolve(
                "META-INF/services/tech.kayys.gollek.spi.multimodal.UnifiedMultimodalRuntime");
        Files.createDirectories(serviceFile.getParent());
        Files.writeString(
                serviceFile,
                """
                external.runtime.FirstCommandGemma4Runtime
                external.runtime.SecondCommandGemma4Runtime
                """,
                StandardCharsets.UTF_8);
        return classesDir;
    }

    private static void addPluginCoreContract(Path classesDir) throws Exception {
        addPluginCoreContract(classesDir, externalModelFamilyPluginDescriptor());
    }

    private static void addPluginCoreContract(Path classesDir, String pluginDescriptor) throws Exception {
        writeServiceDescriptor(
                classesDir,
                "tech.kayys.gollek.spi.plugin.GollekPlugin",
                "external.modules.ModulesExternalModelFamilyPlugin");
        Files.writeString(
                classesDir.resolve("plugin.json"),
                pluginDescriptor,
                StandardCharsets.UTF_8);
    }

    private static void writeJar(Path classesDir, Path jarPath) throws Exception {
        Files.createDirectories(jarPath.getParent());
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath));
                var files = Files.walk(classesDir)) {
            for (Path path : files.filter(Files::isRegularFile).sorted(Comparator.naturalOrder()).toList()) {
                String entryName = classesDir.relativize(path).toString().replace('\\', '/');
                jar.putNextEntry(new JarEntry(entryName));
                Files.copy(path, jar);
                jar.closeEntry();
            }
        }
    }

    private static void writeServiceDescriptor(
            Path classesDir,
            String serviceName,
            String implementationClass) throws Exception {
        Path serviceFile = classesDir.resolve("META-INF/services/" + serviceName);
        Files.createDirectories(serviceFile.getParent());
        Files.writeString(serviceFile, implementationClass + System.lineSeparator(), StandardCharsets.UTF_8);
    }

    private static String externalExtensionSource() {
        return """
                package external.modules;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.gollek.plugin.core.ExtensionAvailability;
                import tech.kayys.gollek.plugin.core.ExtensionAvailabilityProvider;

                public final class ModulesExternalExtensionProvider implements ExtensionAvailabilityProvider {
                    @Override
                    public String extensionId() {
                        return "modules-external-extension";
                    }

                    @Override
                    public String extensionName() {
                        return "Modules External Extension";
                    }

                    @Override
                    public String extensionKind() {
                        return "tokenizer";
                    }

                    @Override
                    public ExtensionAvailability availability() {
                        return new ExtensionAvailability(
                                extensionId(),
                                extensionName(),
                                extensionKind(),
                                true,
                                false,
                                true,
                                true,
                                "ready",
                                List.of("tokenizer"),
                                List.of("sentencepiece"),
                                Map.of("source", "extensions-command-fixture"),
                                "external modules fixture ready",
                                List.of());
                    }
                }
                """;
    }

    private static String externalUnifiedRuntimeSource(String className, String runtimeId) {
        return """
                package external.runtime;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.gollek.spi.multimodal.UnifiedInputModality;
                import tech.kayys.gollek.spi.multimodal.UnifiedMultimodalRuntime;
                import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeManifest;
                import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeReadiness;

                public final class %s implements UnifiedMultimodalRuntime {
                    @Override
                    public UnifiedRuntimeManifest manifest() {
                        return new UnifiedRuntimeManifest(
                                "%s",
                                "%s",
                                List.of("gemma4"),
                                List.of("gemma4_unified"),
                                List.of(UnifiedInputModality.TEXT, UnifiedInputModality.IMAGE),
                                UnifiedRuntimeReadiness.READY,
                                "fixture ready",
                                List.of("processor_config.json"),
                                List.of("tokenizer.model"),
                                Map.of("fixture", "extensions-command"));
                    }
                }
                """.formatted(className, runtimeId, runtimeId);
    }

    private static String externalModelFamilySource() {
        return """
                package external.modules;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
                import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

                public final class ModulesExternalModelFamilyPlugin implements ModelFamilyPlugin {
                    @Override
                    public ModelFamilyDescriptor descriptor() {
                        return new ModelFamilyDescriptor(
                                "modules_external_family",
                                "Modules External Family",
                                List.of("modules_external_model"),
                                List.of("ModulesExternalForCausalLM"),
                                List.of(ModelFamilyCapability.TOKENIZER),
                                Map.of(
                                        "bundle_profile", "optional",
                                        "origin", "external/modules-fixture"));
                    }

                    @Override
                    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("modules-external-tokenizer"));
                    }
                }
                """;
    }

    private static String externalModelFamilyPluginDescriptor() {
        return """
                {
                  "id" : "model-family/modules_external_family",
                  "name" : "Modules External Family",
                  "version" : "0.1.0",
                  "description" : "External model-family fixture for modules doctor",
                  "vendor" : "Test",
                  "mainClass" : "external.modules.ModulesExternalModelFamilyPlugin",
                  "dependencies" : [ ],
                  "optionalDependencies" : [ ],
                  "properties" : {
                    "extensionPoint" : "model-family",
                    "bundleProfile" : "optional",
                    "families" : [ "modules_external_family" ],
                    "tokenizerKind" : "hf-bpe"
                  }
                }
                """;
    }

    private static String externalModelFamilyPluginDescriptorWithTokenizerKind(String tokenizerKind) {
        return """
                {
                  "id" : "model-family/modules_external_family",
                  "name" : "Modules External Family",
                  "version" : "0.1.0",
                  "description" : "External model-family fixture with custom tokenizer kind",
                  "vendor" : "Test",
                  "mainClass" : "external.modules.ModulesExternalModelFamilyPlugin",
                  "dependencies" : [ ],
                  "optionalDependencies" : [ ],
                  "properties" : {
                    "extensionPoint" : "model-family",
                    "bundleProfile" : "optional",
                    "families" : [ "modules_external_family" ],
                    "tokenizerKind" : "%s"
                  }
                }
                """.formatted(tokenizerKind);
    }

    private static String externalModelFamilyPluginDescriptorWithoutTokenizerMetadata() {
        return """
                {
                  "id" : "model-family/modules_external_family",
                  "name" : "Modules External Family",
                  "version" : "0.1.0",
                  "description" : "External model-family fixture without tokenizer metadata",
                  "vendor" : "Test",
                  "mainClass" : "external.modules.ModulesExternalModelFamilyPlugin",
                  "dependencies" : [ ],
                  "optionalDependencies" : [ ],
                  "properties" : {
                    "extensionPoint" : "model-family",
                    "bundleProfile" : "optional",
                    "families" : [ "modules_external_family" ]
                  }
                }
                """;
    }

    private static String externalModelFamilyPluginDescriptorWithPendingTokenizerMetadata() {
        return """
                {
                  "id" : "model-family/modules_external_family",
                  "name" : "Modules External Family",
                  "version" : "0.1.0",
                  "description" : "External model-family fixture with pending tokenizer metadata",
                  "vendor" : "Test",
                  "mainClass" : "external.modules.ModulesExternalModelFamilyPlugin",
                  "dependencies" : [ ],
                  "optionalDependencies" : [ ],
                  "properties" : {
                    "extensionPoint" : "model-family",
                    "bundleProfile" : "optional",
                    "families" : [ "modules_external_family" ],
                    "tokenizerMetadataStatus" : "pending",
                    "tokenizerMetadataPendingReason" : "tokenizer adapter pending fixture stabilization"
                  }
                }
                """;
    }

    private static String externalModelFamilyPluginDescriptorWithTokenizerKinds() {
        return """
                {
                  "id" : "model-family/modules_external_family",
                  "name" : "Modules External Family",
                  "version" : "0.1.0",
                  "description" : "External model-family fixture with multiple tokenizer kinds",
                  "vendor" : "Test",
                  "mainClass" : "external.modules.ModulesExternalModelFamilyPlugin",
                  "dependencies" : [ ],
                  "optionalDependencies" : [ ],
                  "properties" : {
                    "extensionPoint" : "model-family",
                    "bundleProfile" : "optional",
                    "families" : [ "modules_external_family" ],
                    "tokenizerKind" : "sentencepiece",
                    "tokenizerKinds" : [ "sentencepiece", "hf-bpe" ]
                  }
                }
                """;
    }

    private static String externalModelFamilyPluginDescriptorWithoutFamilies() {
        return """
                {
                  "id" : "model-family/modules_external_family",
                  "name" : "Modules External Family",
                  "version" : "0.1.0",
                  "description" : "External model-family fixture missing families metadata",
                  "vendor" : "Test",
                  "mainClass" : "external.modules.ModulesExternalModelFamilyPlugin",
                  "dependencies" : [ ],
                  "optionalDependencies" : [ ],
                  "properties" : {
                    "extensionPoint" : "model-family",
                    "bundleProfile" : "optional",
                    "tokenizerKind" : "hf-bpe"
                  }
                }
                """;
    }

    private static JsonNode matrixEntry(JsonNode root, String familyId) {
        for (JsonNode row : root.path("modelFamilyPlugins").path("capabilityMatrix")) {
            if (familyId.equals(row.path("id").asText())) {
                return row;
            }
        }
        throw new AssertionError("matrix row not found: " + familyId);
    }

    private static JsonNode pluginDirectoryJar(JsonNode jars, String filename) {
        for (JsonNode jar : jars) {
            String path = jar.path("path").asText();
            if (!path.isBlank() && filename.equals(Path.of(path).getFileName().toString())) {
                return jar;
            }
        }
        throw new AssertionError("plugin directory jar not found: " + filename + " in " + jars);
    }

    private static boolean containsTextField(JsonNode values, String field, String expected) {
        for (JsonNode value : values) {
            if (expected.equals(value.path(field).asText())) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsTextValue(JsonNode values, String expected) {
        for (JsonNode value : values) {
            if (expected.equals(value.asText())) {
                return true;
            }
        }
        return false;
    }

    private static void assertRouteBenchmarkCacheContract(JsonNode root) {
        JsonNode schema = root.path("routeBenchmarkCacheReportSchema");
        assertEquals(
                RouteBenchmarkCacheReportContract.CONTRACT_ID,
                schema.path(RouteBenchmarkCacheReportContract.FIELD_CONTRACT_ID).asText());
        assertEquals(
                RouteBenchmarkCacheReportContract.SCHEMA_VERSION,
                schema.path(RouteBenchmarkCacheReportContract.FIELD_SCHEMA_VERSION).asInt());
        assertEquals(
                RouteBenchmarkCacheReportContract.schemaFingerprint(),
                schema.path(RouteBenchmarkCacheReportContract.FIELD_SCHEMA_FINGERPRINT).asText());
        assertTrue(containsTextValue(
                schema.path(RouteBenchmarkCacheReportContract.FIELD_REPORT_FIELDS),
                "profileTrustStatus"));
        assertTrue(containsTextValue(
                schema.path(RouteBenchmarkCacheReportContract.FIELD_REQUIRED_REPORT_FIELDS),
                "failOnRouteBenchmarkCache"));
        assertTrue(containsTextValue(
                schema.path(RouteBenchmarkCacheReportContract.FIELD_PROFILE_TRUST_STATUSES),
                "trusted_profiles_available"));
        assertTrue(containsTextValue(
                schema.path(RouteBenchmarkCacheReportContract.FIELD_PROBLEM_CODES),
                "cache_disabled"));

        JsonNode schemaValidation = root.path("routeBenchmarkCacheReportSchemaValidation");
        assertEquals(
                RouteBenchmarkCacheReportContract.schemaFingerprint(),
                schemaValidation.path(RouteBenchmarkCacheReportContract.FIELD_SCHEMA_FINGERPRINT).asText());
        assertTrue(schemaValidation.path("passed").asBoolean());
        assertEquals(0, schemaValidation.path("problemCount").asInt());

        JsonNode reportValidation = root.path("routeBenchmarkCacheReportValidation");
        assertEquals(
                RouteBenchmarkCacheReportContract.schemaFingerprint(),
                reportValidation.path(RouteBenchmarkCacheReportContract.FIELD_SCHEMA_FINGERPRINT).asText());
        assertTrue(reportValidation.path("passed").asBoolean());
        assertEquals(0, reportValidation.path("problemCount").asInt());
    }

    private static boolean containsRunnerEffect(
            JsonNode effects,
            String runner,
            String provider,
            String format,
            boolean forceGguf) {
        for (JsonNode effect : effects) {
            if (runner.equals(effect.path("runner").asText())
                    && provider.equals(effect.path("provider").asText())
                    && format.equals(effect.path("format").asText())
                    && forceGguf == effect.path("forceGguf").asBoolean()) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsPathFilename(JsonNode values, String filename) {
        for (JsonNode value : values) {
            String path = value.asText();
            if (!path.isBlank() && filename.equals(Path.of(path).getFileName().toString())) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsViolationCode(JsonNode violations, String code) {
        for (JsonNode violation : violations) {
            if (code.equals(violation.path("code").asText())) {
                return true;
            }
        }
        return false;
    }

    private static ExtensionAvailabilityProvider badExtensionAvailabilityProvider() {
        return new ExtensionAvailabilityProvider() {
            @Override
            public String extensionId() {
                return "Bad Extension";
            }

            @Override
            public String extensionName() {
                return "Bad Extension";
            }

            @Override
            public String extensionKind() {
                return "Audio";
            }

            @Override
            public ExtensionAvailability availability() {
                return new ExtensionAvailability(
                        "bad-extension",
                        "Bad Extension",
                        "audio",
                        true,
                        false,
                        true,
                        false,
                        "fallback",
                        List.of("audio_encoding"),
                        List.of("wav"),
                        Map.of(),
                        "bad provider metadata",
                        List.of());
            }
        };
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

    private static void withSystemProperty(String key, String value, ThrowingRunnable action) throws Exception {
        String previousValue = System.getProperty(key);
        try {
            System.setProperty(key, value);
            action.run();
        } finally {
            if (previousValue == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, previousValue);
            }
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    private record StubArchitecture(
            String id,
            String modelType,
            String architectureClassName) implements ModelArchitecture {

        @Override
        public List<String> supportedArchClassNames() {
            return List.of(architectureClassName);
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of(modelType);
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
