package tech.kayys.gollek.cli.util;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyResolution;

import javax.tools.JavaCompiler;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelFamilyExternalPluginFixtureTest {
    private static final String CLEAN_FAMILY_ID = "external_tokenizer_fixture";
    private static final String BAD_FAMILY_ID = "bad external model";
    private static final String SERVICE_DESCRIPTOR =
            "META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin";

    @TempDir
    Path tempDir;

    @Test
    @SuppressWarnings("unchecked")
    void checkerLoadsExternalModelFamilyFromPluginClassLoader() throws Exception {
        Path classesDir = compileExternalModelFamily(
                "external.fixture.ExternalTokenizerModelFamilyPlugin",
                cleanModelFamilySource());
        writeServiceDescriptor(classesDir, "external.fixture.ExternalTokenizerModelFamilyPlugin");

        try (URLClassLoader pluginClassLoader = externalPluginClassLoader(classesDir)) {
            ModelFamilyPluginRegistry.global().unregister(CLEAN_FAMILY_ID);

            List<String> familyIds = PluginAvailabilityChecker.getGlobalModelFamilyPluginIds(pluginClassLoader);
            ModelFamilyPluginRegistry scopedRegistry =
                    new PluginAvailabilityChecker().getModelFamilyPluginRegistry(pluginClassLoader);

            assertTrue(familyIds.contains(CLEAN_FAMILY_ID));
            assertTrue(ModelFamilyPluginRegistry.global().contractViolations().isEmpty());
            assertEquals(ModelFamilyResolution.Status.NOT_FOUND,
                    ModelFamilyPluginRegistry.global().resolveModelType("external_tokenizer_fixture").status());

            ModelFamilyResolution resolution =
                    scopedRegistry.resolveModelType("external_tokenizer_fixture");
            assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
            assertEquals(List.of(CLEAN_FAMILY_ID), resolution.familyIds());
            assertEquals("external_tokenizer_fixture", resolution.tokenizerDescriptors().getFirst().id());

            ModelFamilyBundleGate gate =
                    PluginAvailabilityChecker.getGlobalModelFamilyBundleGate(pluginClassLoader);
            assertTrue(gate.passed());
            assertEquals("passed", gate.status());

            Map<String, Object> report = ModelFamilyBundleGateReportWriter.buildReport(pluginClassLoader);
            assertTrue(((List<String>) report.get("discoveredFamilies")).contains(CLEAN_FAMILY_ID));
            assertEquals("scoped", report.get("modelFamilyRegistryScope"));
            assertEquals("passed", ((Map<String, Object>) report.get("gate")).get("status"));

            Path reportFile = tempDir.resolve("external-model-family-gate.json");
            ModelFamilyBundleGateReportWriter.main(new String[] {
                    reportFile.toString(),
                    classesDir.toString()
            });
            JsonNode root = new ObjectMapper().readTree(reportFile.toFile());
            assertTrue(root.path("externalPluginClasspath").get(0).asText()
                    .contains(classesDir.getFileName().toString()));
            assertEquals("scoped", root.path("modelFamilyRegistryScope").asText());
            assertTrue(root.path("discoveredFamilies").toString().contains(CLEAN_FAMILY_ID));

            ModelFamilyBundleGateCheck.main(new String[] {
                    classesDir.toString()
            });
            assertEquals(ModelFamilyResolution.Status.NOT_FOUND,
                    ModelFamilyPluginRegistry.global().resolveModelType("external_tokenizer_fixture").status());
        } finally {
            ModelFamilyPluginRegistry.global().unregister(CLEAN_FAMILY_ID);
        }
    }

    @Test
    void checkerReportsExternalModelFamilyContractFailureThroughGate() throws Exception {
        Path classesDir = compileExternalModelFamily(
                "external.fixture.BadExternalModelFamilyPlugin",
                badModelFamilySource());
        writeServiceDescriptor(classesDir, "external.fixture.BadExternalModelFamilyPlugin");

        try (URLClassLoader pluginClassLoader = externalPluginClassLoader(classesDir)) {
            ModelFamilyPluginRegistry.global().unregister(BAD_FAMILY_ID);

            List<String> familyIds = PluginAvailabilityChecker.getGlobalModelFamilyPluginIds(pluginClassLoader);
            ModelFamilyBundleGate gate =
                    PluginAvailabilityChecker.getGlobalModelFamilyBundleGate(pluginClassLoader);

            assertTrue(familyIds.contains(BAD_FAMILY_ID));
            assertTrue(ModelFamilyPluginRegistry.global().contractViolations().stream()
                    .noneMatch(violation -> BAD_FAMILY_ID.equals(violation.familyId())));
            assertFalse(gate.passed());
            assertEquals("contract_failed", gate.status());
            assertTrue(gate.violations().stream()
                    .anyMatch(violation -> violation.contains("invalid_family_id")));

            IllegalStateException error = assertThrows(
                    IllegalStateException.class,
                    () -> ModelFamilyBundleGateCheck.main(new String[] {
                            classesDir.toString()
                    }));
            assertTrue(error.getMessage().contains("invalid_family_id"));
            assertTrue(ModelFamilyPluginRegistry.global().contractViolations().stream()
                    .noneMatch(violation -> BAD_FAMILY_ID.equals(violation.familyId())));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(BAD_FAMILY_ID);
        }
    }

    @Test
    void gateCheckRejectsMissingExternalModelFamilyClasspath() {
        Path missingPlugin = tempDir.resolve("missing-model-family.jar");

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> ModelFamilyBundleGateCheck.main(new String[] {
                        missingPlugin.toString()
                }));
        assertTrue(error.getMessage().contains("does not exist"));
    }

    private Path compileExternalModelFamily(String className, String source) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertTrue(compiler != null, "Java compiler must be available for external model-family fixture compilation");

        Path sourceRoot = tempDir.resolve("src");
        Path classesDir = tempDir.resolve("classes-" + className.toLowerCase(Locale.ROOT).replace('.', '-'));
        Path sourceFile = sourceRoot.resolve(className.replace('.', '/') + ".java");
        Files.createDirectories(sourceFile.getParent());
        Files.createDirectories(classesDir);
        Files.writeString(sourceFile, source, StandardCharsets.UTF_8);

        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
            Iterable<? extends javax.tools.JavaFileObject> units =
                    fileManager.getJavaFileObjectsFromFiles(List.of(sourceFile.toFile()));
            List<String> options = List.of(
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", classesDir.toString());
            Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
            assertTrue(Boolean.TRUE.equals(compiled), "external model-family fixture should compile: " + className);
        }
        return classesDir;
    }

    private void writeServiceDescriptor(Path classesDir, String implementationClass) throws Exception {
        Path serviceFile = classesDir.resolve(SERVICE_DESCRIPTOR);
        Files.createDirectories(serviceFile.getParent());
        Files.writeString(serviceFile, implementationClass + System.lineSeparator(), StandardCharsets.UTF_8);
    }

    private URLClassLoader externalPluginClassLoader(Path classesDir) throws Exception {
        URL[] urls = { classesDir.toUri().toURL() };
        return new URLClassLoader(urls, ModelFamilyExternalPluginFixtureTest.class.getClassLoader());
    }

    private static String cleanModelFamilySource() {
        return """
                package external.fixture;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
                import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

                public final class ExternalTokenizerModelFamilyPlugin implements ModelFamilyPlugin {
                    @Override
                    public ModelFamilyDescriptor descriptor() {
                        return new ModelFamilyDescriptor(
                                "external_tokenizer_fixture",
                                "External Tokenizer Fixture",
                                List.of("external_tokenizer_fixture"),
                                List.of("ExternalTokenizerFixtureForCausalLM"),
                                List.of(ModelFamilyCapability.TOKENIZER),
                                Map.of(
                                        "bundle_profile", "optional",
                                        "origin", "external/plugin-fixture/models/external_tokenizer_fixture"));
                    }

                    @Override
                    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("external_tokenizer_fixture"));
                    }
                }
                """;
    }

    private static String badModelFamilySource() {
        return """
                package external.fixture;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;

                public final class BadExternalModelFamilyPlugin implements ModelFamilyPlugin {
                    @Override
                    public ModelFamilyDescriptor descriptor() {
                        return new ModelFamilyDescriptor(
                                "Bad External Model",
                                "Bad External Model",
                                List.of("Bad External Model Type"),
                                List.of(),
                                List.of(ModelFamilyCapability.TOKENIZER),
                                Map.of(
                                        "bundle_profile", "research",
                                        "origin", "external/plugin-fixture/models/bad_external_model"));
                    }
                }
                """;
    }
}
