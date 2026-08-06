package tech.kayys.gollek.cli.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyResolution;

import javax.tools.JavaCompiler;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExternalModelFamilyPluginScopeTest {
    private static final String FAMILY_ID = "external_scope_family";
    private static final String MODEL_TYPE = "external_scope_model";
    private static final String SERVICE_DESCRIPTOR =
            "META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin";

    @TempDir
    Path tempDir;

    @Test
    void attachRegistersExternalFamilyOnlyUntilScopeCloses() throws Exception {
        Path classesDir = compileExternalModelFamily();
        writeServiceDescriptor(classesDir, "external.scope.ExternalScopeModelFamilyPlugin");
        ModelFamilyPluginRegistry.global().unregister(FAMILY_ID);

        try {
            try (ExternalModelFamilyPluginScope scope = ExternalModelFamilyPluginScope.attach(
                    List.of(classesDir.toString()),
                    ExternalModelFamilyPluginScopeTest.class,
                    null)) {
                assertTrue(scope.active());
                assertEquals(List.of(classesDir), scope.classpath());
                assertTrue(scope.displayClasspath().getFirst().contains(classesDir.getFileName().toString()));
                assertTrue(scope.detachablePluginIds().contains("model-family/" + FAMILY_ID));
                assertEquals(ModelFamilyResolution.Status.RESOLVED,
                        ModelFamilyPluginRegistry.global().resolveModelType(MODEL_TYPE).status());
            }

            assertEquals(ModelFamilyResolution.Status.NOT_FOUND,
                    ModelFamilyPluginRegistry.global().resolveModelType(MODEL_TYPE).status());
        } finally {
            ModelFamilyPluginRegistry.global().unregister(FAMILY_ID);
        }
    }

    @Test
    void attachCanResolveModelFamilyFromPluginDirectory() throws Exception {
        Path classesDir = compileExternalModelFamily();
        writeServiceDescriptor(classesDir, "external.scope.ExternalScopeModelFamilyPlugin");
        ModelFamilyPluginRegistry.global().unregister(FAMILY_ID);

        try {
            try (ExternalModelFamilyPluginScope scope = ExternalModelFamilyPluginScope.attach(
                    List.of(),
                    List.of(tempDir.toString()),
                    ExternalModelFamilyPluginScopeTest.class,
                    null)) {
                assertTrue(scope.active());
                assertTrue(scope.classpath().contains(classesDir.toAbsolutePath().normalize()));
                assertTrue(scope.detachablePluginIds().contains("model-family/" + FAMILY_ID));
                assertEquals(ModelFamilyResolution.Status.RESOLVED,
                        ModelFamilyPluginRegistry.global().resolveModelType(MODEL_TYPE).status());
            }

            assertEquals(ModelFamilyResolution.Status.NOT_FOUND,
                    ModelFamilyPluginRegistry.global().resolveModelType(MODEL_TYPE).status());
        } finally {
            ModelFamilyPluginRegistry.global().unregister(FAMILY_ID);
        }
    }

    @Test
    void emptyScopeUsesGlobalClasspathDefaults() {
        try (ExternalModelFamilyPluginScope scope = ExternalModelFamilyPluginScope.empty()) {
            assertFalse(scope.active());
            assertEquals(List.of(), scope.classpath());
            assertEquals(List.of(), scope.displayClasspath());
            assertEquals(List.of(), scope.detachablePluginIds());
        }
    }

    private Path compileExternalModelFamily() throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertTrue(compiler != null, "Java compiler must be available for external scope fixture compilation");

        Path sourceRoot = tempDir.resolve("src");
        Path classesDir = tempDir.resolve("classes");
        Path sourceFile = sourceRoot.resolve("external/scope/ExternalScopeModelFamilyPlugin.java");
        Files.createDirectories(sourceFile.getParent());
        Files.createDirectories(classesDir);
        Files.writeString(sourceFile, externalModelFamilySource(), StandardCharsets.UTF_8);

        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
            Iterable<? extends javax.tools.JavaFileObject> units =
                    fileManager.getJavaFileObjectsFromFiles(List.of(sourceFile.toFile()));
            List<String> options = List.of(
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", classesDir.toString());
            Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
            assertTrue(Boolean.TRUE.equals(compiled), "external scope fixture should compile");
        }
        return classesDir;
    }

    private void writeServiceDescriptor(Path classesDir, String implementationClass) throws Exception {
        Path serviceFile = classesDir.resolve(SERVICE_DESCRIPTOR);
        Files.createDirectories(serviceFile.getParent());
        Files.writeString(serviceFile, implementationClass + System.lineSeparator(), StandardCharsets.UTF_8);
    }

    private static String externalModelFamilySource() {
        return """
                package external.scope;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
                import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

                public final class ExternalScopeModelFamilyPlugin implements ModelFamilyPlugin {
                    @Override
                    public ModelFamilyDescriptor descriptor() {
                        return new ModelFamilyDescriptor(
                                "external_scope_family",
                                "External Scope Family",
                                List.of("external_scope_model"),
                                List.of("ExternalScopeForCausalLM"),
                                List.of(ModelFamilyCapability.TOKENIZER),
                                Map.of(
                                        "bundle_profile", "optional",
                                        "origin", "external/scope-fixture"));
                    }

                    @Override
                    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("external-scope-bpe"));
                    }
                }
                """;
    }
}
