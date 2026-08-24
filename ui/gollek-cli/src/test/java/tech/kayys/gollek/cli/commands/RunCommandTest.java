package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;
import jakarta.inject.Inject;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import javax.tools.JavaCompiler;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class RunCommandTest {

        private static final String EXTERNAL_RUN_FAMILY_ID = "run-external-family";
        private static final String EXTERNAL_RUN_MODEL_TYPE = "run_external_model";

        @Inject
        RunCommand runCommand;

        @InjectMock
        GollekSdk sdk;

        @TempDir
        Path tempDir;

        @Test
        public void testRunCommand() throws Exception {
                ModelInfo modelInfo = ModelInfo.builder().modelId("test-model").build();

                // Mock response
                InferenceResponse mockResponse = InferenceResponse.builder()
                                .requestId("test-id")
                                .model("test-model")
                                .content("Blue like the ocean")
                                .build();

                Mockito.when(sdk.createCompletion(any(InferenceRequest.class)))
                                .thenReturn(mockResponse);

                Mockito.when(sdk.getModelInfo(eq("test-model")))
                                .thenReturn(Optional.of(modelInfo));

                Mockito.when(sdk.prepareModel(
                                eq("test-model"),
                                Mockito.<String>isNull(),
                                Mockito.<String>isNull(),
                                eq(false),
                                eq("Q4_K_M"),
                                eq(List.of()),
                                Mockito.<Consumer<PullProgress>>any()))
                                .thenReturn(new ModelResolution("test-model", null, modelInfo));

                // Set CLI options directly as they are injected by picocli,
                // but here we are testing the Runnable logic as a bean.
                // Since fields are package-private or private and injected by picocli,
                // we might need to rely on reflection or integration tests if we can't set them
                // easily.
                // However, looking at the previous implementation, the fields were
                // package-private.

                // Actually, Picocli fields are often private. Let's check the source again.
                // If they are package-private, I can set them. If they are private, I need
                // reflection.
                // I wrote them as package-private (default visibility) in the previous step.
                // "String modelId;" etc.

                runCommand.modelId = "test-model";
                runCommand.prompt = "Why is the sky blue?";
                runCommand.stream = false;

                // Execute
                runCommand.run();

                // Verify
                Mockito.verify(sdk).createCompletion(any(InferenceRequest.class));
        }

        @Test
        public void testRunCommandRequestsRuntimeProfile() throws Exception {
                ModelInfo modelInfo = ModelInfo.builder().modelId("profile-model").build();
                InferenceResponse mockResponse = InferenceResponse.builder()
                                .requestId("profile-id")
                                .model("profile-model")
                                .content("profiled")
                                .metadata("profile_mode", "onnx")
                                .build();
                List<InferenceRequest> capturedRequests = new ArrayList<>();

                Mockito.when(sdk.createCompletion(any(InferenceRequest.class)))
                                .thenAnswer(invocation -> {
                                        capturedRequests.add(invocation.getArgument(0));
                                        return mockResponse;
                                });
                Mockito.when(sdk.getModelInfo(eq("profile-model")))
                                .thenReturn(Optional.of(modelInfo));
                Mockito.when(sdk.prepareModel(
                                eq("profile-model"),
                                Mockito.<String>isNull(),
                                Mockito.<String>isNull(),
                                eq(false),
                                eq("Q4_K_M"),
                                eq(List.of()),
                                Mockito.<Consumer<PullProgress>>any()))
                                .thenReturn(new ModelResolution("profile-model", null, modelInfo));

                try {
                        runCommand.modelId = "profile-model";
                        runCommand.prompt = "Measure this.";
                        runCommand.stream = false;
                        runCommand.runtimeProfile = true;

                        runCommand.run();

                        assertEquals(1, capturedRequests.size());
                        InferenceRequest request = capturedRequests.get(0);
                        assertEquals(true, request.getParameters().get("profile"));
                        assertEquals(true, request.getParameters().get("onnx_profile"));
                        assertEquals(true, request.getMetadata().get("profile"));
                        assertEquals(true, request.getMetadata().get("onnx_profile"));
                } finally {
                        runCommand.modelId = null;
                        runCommand.prompt = null;
                        runCommand.stream = true;
                        runCommand.runtimeProfile = false;
                }
        }

        @Test
        public void testRunCommandAttachesExternalModelFamilyForSingleRun() throws Exception {
                Path classesDir = compileExternalRunModelFamilyFixture(tempDir);
                ModelInfo modelInfo = ModelInfo.builder()
                                .modelId("external-run-model")
                                .metadata(Map.of("model_type", EXTERNAL_RUN_MODEL_TYPE))
                                .build();

                InferenceResponse mockResponse = InferenceResponse.builder()
                                .requestId("external-run-id")
                                .model("external-run-model")
                                .content("external family active")
                                .build();

                Mockito.when(sdk.createCompletion(any(InferenceRequest.class)))
                                .thenAnswer(invocation -> {
                                        assertEquals("RESOLVED", ModelFamilyPluginRegistry.global()
                                                        .resolveModelType(EXTERNAL_RUN_MODEL_TYPE)
                                                        .status()
                                                        .name());
                                        return mockResponse;
                                });

                Mockito.when(sdk.getModelInfo(eq("external-run-model")))
                                .thenReturn(Optional.of(modelInfo));

                Mockito.when(sdk.prepareModel(
                                eq("external-run-model"),
                                Mockito.<String>isNull(),
                                Mockito.<String>isNull(),
                                eq(false),
                                eq("Q4_K_M"),
                                eq(List.of()),
                                Mockito.<Consumer<PullProgress>>any()))
                                .thenReturn(new ModelResolution("external-run-model", null, modelInfo));

                ModelFamilyPluginRegistry.global().unregister(EXTERNAL_RUN_FAMILY_ID);
                try {
                        runCommand.modelId = "external-run-model";
                        runCommand.prompt = "Use the external family.";
                        runCommand.stream = false;
                        runCommand.externalPluginClasspath = List.of(classesDir.toString());

                        runCommand.run();

                        Mockito.verify(sdk).createCompletion(any(InferenceRequest.class));
                        assertEquals("NOT_FOUND", ModelFamilyPluginRegistry.global()
                                        .resolveModelType(EXTERNAL_RUN_MODEL_TYPE)
                                        .status()
                                        .name());
                } finally {
                        ModelFamilyPluginRegistry.global().unregister(EXTERNAL_RUN_FAMILY_ID);
                        runCommand.modelId = null;
                        runCommand.prompt = null;
                        runCommand.stream = true;
                        runCommand.externalPluginClasspath = new ArrayList<>();
                }
        }

        private static Path compileExternalRunModelFamilyFixture(Path tempDir) throws Exception {
                JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
                assertTrue(compiler != null, "Java compiler must be available for external run fixture compilation");

                Path sourceRoot = tempDir.resolve("external-src");
                Path classesDir = tempDir.resolve("external-classes");
                Path sourceFile = sourceRoot.resolve("external/run/RunExternalModelFamilyPlugin.java");
                Files.createDirectories(sourceFile.getParent());
                Files.createDirectories(classesDir);
                Files.writeString(sourceFile, externalRunModelFamilySource(), StandardCharsets.UTF_8);

                try (StandardJavaFileManager fileManager =
                                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
                        Iterable<? extends javax.tools.JavaFileObject> units =
                                        fileManager.getJavaFileObjectsFromFiles(List.of(sourceFile.toFile()));
                        List<String> options = List.of(
                                        "-classpath", System.getProperty("java.class.path"),
                                        "-d", classesDir.toString());
                        Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
                        assertTrue(Boolean.TRUE.equals(compiled), "external run fixture should compile");
                }

                Path serviceFile = classesDir.resolve(
                                "META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin");
                Files.createDirectories(serviceFile.getParent());
                Files.writeString(
                                serviceFile,
                                "external.run.RunExternalModelFamilyPlugin" + System.lineSeparator(),
                                StandardCharsets.UTF_8);
                return classesDir;
        }

        private static String externalRunModelFamilySource() {
                return """
                                package external.run;

                                import java.util.List;
                                import java.util.Map;
                                import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
                                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
                                import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

                                public final class RunExternalModelFamilyPlugin implements ModelFamilyPlugin {
                                    @Override
                                    public ModelFamilyDescriptor descriptor() {
                                        return new ModelFamilyDescriptor(
                                                "run-external-family",
                                                "Run External Family",
                                                List.of("run_external_model"),
                                                List.of("RunExternalForCausalLM"),
                                                List.of(ModelFamilyCapability.TOKENIZER),
                                                Map.of(
                                                        "bundle_profile", "optional",
                                                        "origin", "external/run-fixture"));
                                    }

                                    @Override
                                    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                                        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("run-external-bpe"));
                                    }
                                }
                                """;
        }
}
