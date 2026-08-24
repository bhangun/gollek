package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;
import jakarta.inject.Inject;

import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportContract;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.context.RequestContext;

import javax.tools.JavaCompiler;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@QuarkusTest
public class ShowCommandTest {

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {
    };
    private static final String EXTERNAL_SHOW_FAMILY_ID = "show-external-family";
    private static final String EXTERNAL_SHOW_MODEL_TYPE = "show_external_model";

    @Inject
    ShowCommand showCommand;

    @InjectMock
    GollekSdk sdk;

    @TempDir
    Path tempDir;

    @Test
    public void testShowCommandModelFound() throws Exception {
        ModelInfo model = ModelInfo.builder()
                .modelId("test-model")
                .name("Test Model")
                .version("1.0")
                .format("GGUF")
                .metadata(Collections.emptyMap())
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();

        Mockito.when(sdk.getModelInfo(eq("test-model")))
                .thenReturn(Optional.of(model));

        showCommand.modelId = "test-model";
        showCommand.json = false;

        showCommand.run();

        Mockito.verify(sdk).getModelInfo(eq("test-model"));
    }

    @Test
    public void testShowCommandModelNotFound() throws Exception {
        Mockito.when(sdk.getModelInfo(any(String.class)))
                .thenReturn(Optional.empty());

        showCommand.modelId = "nonexistent";
        showCommand.json = false;

        showCommand.run();

        Mockito.verify(sdk).getModelInfo(eq("nonexistent"));
    }

    @Test
    public void testShowCommandJsonIncludesModelFamilyResolution() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_resolution",
                  "architectures": ["ShowResolutionForCausalLM"]
                }
                """);
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-resolution-family",
                        "Show Resolution Family",
                        List.of("show_resolution"),
                        List.of("ShowResolutionForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "metadata_only"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("show-resolution-bpe"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-model")
                .name("Show Model")
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode root = JSON.readTree(stdout.toString(StandardCharsets.UTF_8));
            assertValidModelFamilyReport(root);
            JsonNode modelFamily = root.path("modelFamily");
            assertEquals("RESOLVED", modelFamily.path("status").asText());
            assertTrue(!modelFamily.path("requiresAttention").asBoolean());
            assertTrue(modelFamily.path("problemCodes").isEmpty());
            assertEquals("show_resolution", modelFamily.path("modelType").asText());
            assertEquals("show-resolution-family", modelFamily.path("familyIds").get(0).asText());
            JsonNode runtimeManifest = modelFamily.path("runtimeManifests").get(0);
            assertEquals("show-resolution-family", runtimeManifest.path("familyId").asText());
            assertTrue(runtimeManifest.path("tokenizerReady").asBoolean());
            assertEquals("show-resolution-bpe", runtimeManifest.path("tokenizerProfileIds").get(0).asText());
            JsonNode directRuntime = modelFamily.path("runtimeCompatibility").path("directSafetensor");
            assertEquals("direct_safetensor", directRuntime.path("runtimeId").asText());
            assertTrue(!directRuntime.path("compatible").asBoolean());
            assertEquals("model_family_direct_safetensor_not_advertised",
                    directRuntime.path("problemCodes").get(0).asText());
            assertEquals("show-resolution-bpe", directRuntime.path("usableTokenizerIds").get(0).asText());
            assertEquals("show-resolution-bpe", modelFamily.path("tokenizers").get(0).path("id").asText());
            assertTrue(modelFamily.path("tokenizers").get(0).path("fileStatusAvailable").asBoolean());
            assertTrue(modelFamily.path("tokenizers").get(0).path("usable").asBoolean());
            assertEquals("tokenizer.json",
                    modelFamily.path("tokenizers").get(0).path("existingFileGroup").get(0).asText());
            assertTrue(modelFamily.path("summary").asText().contains("show-resolution-family"));
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonLoadsExternalModelFamilyFromPluginClasspath() throws Exception {
        Path modelDir = tempDir.resolve("model");
        Files.createDirectories(modelDir);
        Files.writeString(modelDir.resolve("config.json"), """
                {
                  "model_type": "show_external_model",
                  "architectures": ["ShowExternalForCausalLM"]
                }
                """);
        Files.writeString(modelDir.resolve("tokenizer.json"), "{}");
        Path classesDir = compileExternalShowModelFamilyFixture(tempDir);
        ModelInfo model = ModelInfo.builder()
                .modelId("show-external-model")
                .name("Show External Model")
                .format("SAFETENSORS")
                .metadata(Map.of("path", modelDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-external-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_SHOW_FAMILY_ID);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-external-model";
            showCommand.json = true;
            showCommand.externalPluginClasspath = List.of(classesDir.toString());

            showCommand.run();

            JsonNode root = JSON.readTree(stdout.toString(StandardCharsets.UTF_8));
            assertValidModelFamilyReport(root);
            JsonNode modelFamily = root.path("modelFamily");
            assertEquals("scoped", root.path("modelFamilyRegistryScope").asText());
            assertTrue(root.path("externalPluginClasspath").get(0).asText()
                    .contains(classesDir.getFileName().toString()));
            assertEquals("RESOLVED", modelFamily.path("status").asText());
            assertEquals(EXTERNAL_SHOW_MODEL_TYPE, modelFamily.path("modelType").asText());
            assertEquals(EXTERNAL_SHOW_FAMILY_ID, modelFamily.path("familyIds").get(0).asText());
            assertEquals(EXTERNAL_SHOW_FAMILY_ID,
                    modelFamily.path("runtimeManifests").get(0).path("familyId").asText());
            assertEquals("show-external-bpe", modelFamily.path("tokenizers").get(0).path("id").asText());
            assertTrue(ModelFamilyPluginRegistry.global()
                    .resolveModelType(EXTERNAL_SHOW_MODEL_TYPE)
                    .status()
                    .name()
                    .equals("NOT_FOUND"));
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(EXTERNAL_SHOW_FAMILY_ID);
            showCommand.json = false;
            showCommand.externalPluginClasspath = new java.util.ArrayList<>();
        }
    }

    @Test
    public void testShowCommandJsonIncludesModelFamilyRemediationWhenMissing() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_missing_resolution",
                  "architectures": ["ShowMissingResolutionForCausalLM"]
                }
                """);
        ModelInfo model = ModelInfo.builder()
                .modelId("show-missing-model")
                .name("Show Missing Model")
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-missing-model")))
                .thenReturn(Optional.of(model));
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-missing-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode root = JSON.readTree(stdout.toString(StandardCharsets.UTF_8));
            assertValidModelFamilyReport(root);
            JsonNode modelFamily = root.path("modelFamily");
            assertEquals("NOT_FOUND", modelFamily.path("status").asText());
            assertTrue(modelFamily.path("requiresAttention").asBoolean());
            assertEquals("model_family_not_found", modelFamily.path("problemCodes").get(0).asText());
            assertTrue(modelFamily.path("remediationHints").get(0).asText().contains("show_missing_resolution"));
        } finally {
            System.setOut(originalOut);
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonIncludesTokenizerFileDiagnosticsWhenMissing() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_tokenizer_missing",
                  "architectures": ["ShowTokenizerMissingForCausalLM"]
                }
                """);
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-tokenizer-missing-family",
                        "Show Tokenizer Missing Family",
                        List.of("show_tokenizer_missing"),
                        List.of("ShowTokenizerMissingForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "metadata_only"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.wordPiece("show-tokenizer-wordpiece"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-tokenizer-missing-model")
                .name("Show Tokenizer Missing Model")
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-tokenizer-missing-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-tokenizer-missing-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode root = JSON.readTree(stdout.toString(StandardCharsets.UTF_8));
            assertValidModelFamilyReport(root);
            JsonNode modelFamily = root.path("modelFamily");
            JsonNode tokenizer = modelFamily.path("tokenizers").get(0);
            assertTrue(modelFamily.path("requiresAttention").asBoolean());
            assertEquals("model_family_tokenizer_files_missing",
                    modelFamily.path("problemCodes").get(0).asText());
            assertEquals("show-tokenizer-wordpiece", tokenizer.path("id").asText());
            assertTrue(tokenizer.path("fileStatusAvailable").asBoolean());
            assertTrue(!tokenizer.path("usable").asBoolean());
            assertTrue(tokenizer.path("missingFileGroups").toString().contains("vocab.txt"));
            assertTrue(modelFamily.path("remediationHints").get(0).asText()
                    .contains("show-tokenizer-wordpiece"));
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonIncludesDirectArchitectureRouting() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_direct",
                  "architectures": ["ShowDirectForCausalLM"]
                }
                """);
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-direct-family",
                        "Show Direct Family",
                        List.of("show_direct"),
                        List.of("ShowDirectForCausalLM"),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new TestArchitecture(
                        "show-direct-adapter",
                        "show_direct",
                        "ShowDirectForCausalLM"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("show-direct-bpe"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-direct-model")
                .name("Show Direct Model")
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-direct-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-direct-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode root = JSON.readTree(stdout.toString(StandardCharsets.UTF_8));
            assertValidModelFamilyReport(root);
            JsonNode directArchitecture = root.path("modelFamily").path("directArchitecture");
            assertTrue(directArchitecture.path("directSupportExpected").asBoolean());
            assertEquals("show-direct-adapter", directArchitecture.path("adapterIds").get(0).asText());
            assertEquals("show-direct-adapter", directArchitecture.path("selectedAdapterId").asText());
            assertEquals("model_type", directArchitecture.path("selectedBy").asText());
            assertTrue(directArchitecture.path("problemCodes").isEmpty());
            JsonNode directRuntime = root.path("modelFamily")
                    .path("runtimeCompatibility")
                    .path("directSafetensor");
            assertTrue(directRuntime.path("compatible").asBoolean());
            assertEquals("show-direct-adapter",
                    directRuntime.path("selectedArchitectureAdapterId").asText());
            assertEquals("show-direct-bpe", directRuntime.path("usableTokenizerIds").get(0).asText());
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonFlagsDirectFamilyWithoutArchitectureAdapter() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_direct_missing_adapter",
                  "architectures": ["ShowDirectMissingAdapterForCausalLM"]
                }
                """);
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-direct-missing-adapter-family",
                        "Show Direct Missing Adapter Family",
                        List.of("show_direct_missing_adapter"),
                        List.of("ShowDirectMissingAdapterForCausalLM"),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-direct-missing-adapter-model")
                .name("Show Direct Missing Adapter Model")
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-direct-missing-adapter-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-direct-missing-adapter-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode root = JSON.readTree(stdout.toString(StandardCharsets.UTF_8));
            assertValidModelFamilyReport(root);
            JsonNode modelFamily = root.path("modelFamily");
            JsonNode directArchitecture = modelFamily.path("directArchitecture");
            assertTrue(modelFamily.path("requiresAttention").asBoolean());
            assertEquals("model_family_architecture_adapters_missing",
                    modelFamily.path("problemCodes").get(0).asText());
            assertEquals("model_family_architecture_adapters_missing",
                    directArchitecture.path("problemCodes").get(0).asText());
            assertTrue(modelFamily.path("remediationHints").get(0).asText().contains("architecture adapter"));
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    private static Path compileExternalShowModelFamilyFixture(Path tempDir) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertTrue(compiler != null, "Java compiler must be available for external show fixture compilation");

        Path sourceRoot = tempDir.resolve("external-src");
        Path classesDir = tempDir.resolve("external-classes");
        Path sourceFile = sourceRoot.resolve("external/show/ShowExternalModelFamilyPlugin.java");
        Files.createDirectories(sourceFile.getParent());
        Files.createDirectories(classesDir);
        Files.writeString(sourceFile, externalShowModelFamilySource(), StandardCharsets.UTF_8);

        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
            Iterable<? extends javax.tools.JavaFileObject> units =
                    fileManager.getJavaFileObjectsFromFiles(List.of(sourceFile.toFile()));
            List<String> options = List.of(
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", classesDir.toString());
            Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
            assertTrue(Boolean.TRUE.equals(compiled), "external show fixture should compile");
        }

        Path serviceFile = classesDir.resolve(
                "META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin");
        Files.createDirectories(serviceFile.getParent());
        Files.writeString(
                serviceFile,
                "external.show.ShowExternalModelFamilyPlugin" + System.lineSeparator(),
                StandardCharsets.UTF_8);
        return classesDir;
    }

    private static String externalShowModelFamilySource() {
        return """
                package external.show;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
                import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

                public final class ShowExternalModelFamilyPlugin implements ModelFamilyPlugin {
                    @Override
                    public ModelFamilyDescriptor descriptor() {
                        return new ModelFamilyDescriptor(
                                "show-external-family",
                                "Show External Family",
                                List.of("show_external_model"),
                                List.of("ShowExternalForCausalLM"),
                                List.of(ModelFamilyCapability.TOKENIZER),
                                Map.of(
                                        "bundle_profile", "optional",
                                        "origin", "external/show-fixture"));
                    }

                    @Override
                    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("show-external-bpe"));
                    }
                }
                """;
    }

    private static void assertValidModelFamilyReport(JsonNode root) {
        assertEquals(List.of(), ModelFamilyResolutionReportContract.validateReport(
                JSON.convertValue(root.path("modelFamily"), MAP_TYPE)));
        JsonNode validation = root.path("modelFamilyValidation");
        assertEquals(
                ModelFamilyResolutionReportFields.CONTRACT_ID,
                validation.path(ModelFamilyResolutionReportFields.Validation.CONTRACT_ID).asText());
        assertEquals(
                ModelFamilyResolutionReportFields.SCHEMA_VERSION,
                validation.path(ModelFamilyResolutionReportFields.Validation.SCHEMA_VERSION).asInt());
        assertEquals(
                ModelFamilyResolutionReportFields.schemaFingerprint(),
                validation.path(ModelFamilyResolutionReportFields.Validation.SCHEMA_FINGERPRINT).asText());
        assertTrue(validation.path(ModelFamilyResolutionReportFields.Validation.PASSED).asBoolean());
        assertEquals(0, validation.path(ModelFamilyResolutionReportFields.Validation.PROBLEM_COUNT).asInt());
        assertTrue(validation.path(ModelFamilyResolutionReportFields.Validation.PROBLEMS).isArray());
    }

    private record TestArchitecture(
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
