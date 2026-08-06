package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.tools.JavaCompiler;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeaturesCommandScaffoldTest {
    @Test
    void initScaffoldsModelFamilyPluginProject(@TempDir Path tempDir) throws Exception {
        Path root = tempDir.resolve("gollek-model-acme");
        FeaturesCommand.Init command = new FeaturesCommand.Init();
        command.targetDir = root.toString();
        command.kind = "model-family";
        command.modelFamilyId = "acme";
        command.modelTypes = List.of("acme");
        command.architectureClassNames = List.of("AcmeForCausalLM");
        command.modelFamilyCapabilities = List.of("causal_lm", "tokenizer", "gguf");
        command.tokenizerKind = "hf-bpe";

        assertEquals(0, command.call());

        Path javaFile = root.resolve(
                "lib/src/main/java/tech/kayys/gollek/extensions/modelacme/AcmeModelFamily.java");
        assertContains(javaFile, "implements ModelFamilyPlugin");
        assertContains(javaFile, "public static final String FAMILY_ID = \"acme\"");
        assertContains(javaFile, "ModelFamilyCapability.CAUSAL_LM");
        assertContains(javaFile, "ModelFamilyCapability.TOKENIZER");
        assertContains(javaFile, "ModelFamilyCapability.GGUF");
        assertContains(javaFile, "ModelTokenizerDescriptor.huggingFaceBpe");
        assertContains(javaFile, "\"tokenizer_metadata_status\", \"ready\"");

        Path serviceFile = root.resolve(
                "lib/src/main/resources/META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin");
        assertEquals(
                "tech.kayys.gollek.extensions.modelacme.AcmeModelFamily",
                Files.readString(serviceFile).trim());

        Path gollekPluginServiceFile = root.resolve(
                "lib/src/main/resources/META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin");
        assertEquals(
                "tech.kayys.gollek.extensions.modelacme.AcmeModelFamily",
                Files.readString(gollekPluginServiceFile).trim());

        JsonNode pluginDescriptor = new ObjectMapper().readTree(Files.readString(
                root.resolve("lib/src/main/resources/plugin.json")));
        assertEquals("model-family/acme", pluginDescriptor.path("id").asText());
        assertEquals(
                "tech.kayys.gollek.extensions.modelacme.AcmeModelFamily",
                pluginDescriptor.path("mainClass").asText());
        assertEquals("model-family", pluginDescriptor.path("properties").path("extensionPoint").asText());
        assertEquals("hf-bpe", pluginDescriptor.path("properties").path("tokenizerKind").asText());
        assertEquals("hf-bpe", pluginDescriptor.path("properties").path("tokenizerKinds").get(0).asText());

        JsonNode manifest = new ObjectMapper().readTree(Files.readString(
                root.resolve("lib/src/main/resources/META-INF/gollek-extension.json")));
        assertEquals(
                "tech.kayys.gollek.extensions.modelacme.AcmeModelFamily",
                manifest.path("services").path("model_family_plugins").get(0).asText());
        JsonNode family = manifest.path("model_families").get(0);
        assertEquals("acme", family.path("id").asText());
        assertEquals("acme", family.path("model_types").get(0).asText());
        assertEquals("AcmeForCausalLM", family.path("architecture_classes").get(0).asText());
        assertEquals("CAUSAL_LM", family.path("capabilities").get(0).asText());
        assertEquals("TOKENIZER", family.path("capabilities").get(1).asText());
        assertEquals("GGUF", family.path("capabilities").get(2).asText());
        assertEquals("ready", family.path("tokenizer_metadata_status").asText());
        assertEquals("hf-bpe", family.path("tokenizer_kind").asText());
        assertEquals("hf-bpe", family.path("tokenizer_kinds").get(0).asText());

        assertContains(root.resolve("lib/build.gradle.kts"), "gollek-spi-model");
        assertContains(root.resolve("README.md"),
                "gollek extensions inspect lib/build/libs/gollek-model-acme-0.1.0-SNAPSHOT.jar --strict");
        assertContains(root.resolve("README.md"),
                "gollek extensions install lib/build/libs/gollek-model-acme-0.1.0-SNAPSHOT.jar --plugin-dir ~/.gollek/plugins --force");
        assertContains(root.resolve("README.md"),
                "gollek extensions inspect lib/build/libs/gollek-model-acme-0.1.0-SNAPSHOT.jar --strict --json");
        assertContains(root.resolve("README.md"),
                "jar.artifact_tokenizer_metadata.tokenizer_metadata_status");
        assertContains(root.resolve("README.md"), "install_lock.tokenizer_metadata_status");
        assertContains(root.resolve("README.md"), "tokenizer_lock_status");
        assertContains(root.resolve("README.md"),
                "gollek extensions detach acme --plugin-dir ~/.gollek/plugins");
        assertContains(root.resolve("README.md"), "gollek modules --doctor --plugin-dir lib/build/libs");
        assertContains(root.resolve("README.md"), "tech.kayys.gollek.spi.plugin.GollekPlugin");
        assertContains(root.resolve("README.md"), "plugin.json");
    }

    @Test
    void initScaffoldsPendingTokenizerModelFamilyPluginProject(@TempDir Path tempDir) throws Exception {
        Path root = tempDir.resolve("gollek-model-moon");
        FeaturesCommand.Init command = new FeaturesCommand.Init();
        command.targetDir = root.toString();
        command.kind = "model-family";
        command.modelFamilyId = "moon";
        command.modelTypes = List.of("moon");
        command.architectureClassNames = List.of("MoonForCausalLM");
        command.modelFamilyCapabilities = List.of("causal_lm", "tokenizer");
        command.tokenizerMetadataStatus = "pending";
        command.tokenizerMetadataPendingReason = "tokenizer adapter pending fixture stabilization";

        assertEquals(0, command.call());

        Path javaFile = root.resolve(
                "lib/src/main/java/tech/kayys/gollek/extensions/modelmoon/MoonModelFamily.java");
        String javaSource = Files.readString(javaFile);
        assertContains(javaFile, "\"tokenizer_metadata_status\", \"pending\"");
        assertContains(javaFile, "\"tokenizer_metadata_pending_reason\", \"tokenizer adapter pending fixture stabilization\"");
        assertContains(javaFile, "public List<ModelTokenizerDescriptor> tokenizerDescriptors()");
        assertContains(javaFile, "return List.of();");
        assertFalse(javaSource.contains("ModelTokenizerDescriptor.huggingFaceBpe"));

        JsonNode pluginDescriptor = new ObjectMapper().readTree(Files.readString(
                root.resolve("lib/src/main/resources/plugin.json")));
        JsonNode properties = pluginDescriptor.path("properties");
        assertEquals("pending", properties.path("tokenizerMetadataStatus").asText());
        assertEquals("tokenizer adapter pending fixture stabilization",
                properties.path("tokenizerMetadataPendingReason").asText());
        assertTrue(properties.path("tokenizerKind").isMissingNode());
        assertTrue(properties.path("tokenizerKinds").isMissingNode());

        JsonNode manifest = new ObjectMapper().readTree(Files.readString(
                root.resolve("lib/src/main/resources/META-INF/gollek-extension.json")));
        JsonNode family = manifest.path("model_families").get(0);
        assertEquals("moon", family.path("id").asText());
        assertEquals("pending", family.path("tokenizer_metadata_status").asText());
        assertEquals("tokenizer adapter pending fixture stabilization",
                family.path("tokenizer_metadata_pending_reason").asText());
        assertTrue(family.path("tokenizer_kind").isMissingNode());
        assertTrue(family.path("tokenizer_kinds").isMissingNode());

        Path readme = root.resolve("README.md");
        assertContains(readme, "Tokenizer metadata: `pending (tokenizer adapter pending fixture stabilization)`");
        assertContains(readme, "Pending tokenizer metadata intentionally omits tokenizer descriptors");
        assertContains(readme, "pending-tokenizer plugins out of production inference presets");
    }

    @Test
    void initTreatsTokenizerKindNoneAsPendingModelFamilyMetadata(@TempDir Path tempDir) throws Exception {
        Path root = tempDir.resolve("gollek-model-void");
        FeaturesCommand.Init command = new FeaturesCommand.Init();
        command.targetDir = root.toString();
        command.kind = "model-family";
        command.modelFamilyId = "void";
        command.tokenizerKind = "none";

        assertEquals(0, command.call());

        JsonNode pluginDescriptor = new ObjectMapper().readTree(Files.readString(
                root.resolve("lib/src/main/resources/plugin.json")));
        JsonNode properties = pluginDescriptor.path("properties");
        assertEquals("pending", properties.path("tokenizerMetadataStatus").asText());
        assertEquals("tokenizer adapter pending scaffold stabilization",
                properties.path("tokenizerMetadataPendingReason").asText());
        assertTrue(properties.path("tokenizerKind").isMissingNode());
        assertTrue(properties.path("tokenizerKinds").isMissingNode());

        JsonNode manifest = new ObjectMapper().readTree(Files.readString(
                root.resolve("lib/src/main/resources/META-INF/gollek-extension.json")));
        JsonNode family = manifest.path("model_families").get(0);
        assertEquals("pending", family.path("tokenizer_metadata_status").asText());
        assertEquals("tokenizer adapter pending scaffold stabilization",
                family.path("tokenizer_metadata_pending_reason").asText());
        assertTrue(family.path("tokenizer_kind").isMissingNode());
        assertTrue(family.path("tokenizer_kinds").isMissingNode());
    }

    @Test
    void inspectAcceptsStrictModelFamilyPluginJar(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        FeaturesCommand.Inspect command = new FeaturesCommand.Inspect();
        command.jarPath = jarPath.toString();
        command.strict = true;
        command.json = true;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("ok", root.path("status").asText());
        assertTrue(root.path("warnings").isEmpty());
        assertTrue(root.path("errors").isEmpty());
        JsonNode jar = root.path("jar");
        assertTrue(jar.path("has_model_family_service_entry").asBoolean());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                jar.path("model_family_providers").get(0).asText());
        assertTrue(jar.path("has_gollek_plugin_service_entry").asBoolean());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                jar.path("gollek_plugin_providers").get(0).asText());
        assertTrue(jar.path("has_plugin_descriptor").asBoolean());
        assertEquals("model-family/inspect-family", jar.path("plugin_descriptor").path("id").asText());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                jar.path("plugin_descriptor").path("mainClass").asText());
        assertTrue(jar.path("providers").isEmpty());
        assertTrue(jar.path("adapter_providers").isEmpty());
        JsonNode family = jar.path("manifest").path("model_families").get(0);
        assertEquals("inspect-family", family.path("id").asText());
        assertEquals("InspectForCausalLM", family.path("architecture_classes").get(0).asText());
        assertEquals("hf-bpe", family.path("tokenizer_kind").asText());
        assertEquals("hf-bpe", family.path("tokenizer_kinds").get(0).asText());
    }

    @Test
    void inspectWarnsForModelFamilyPluginJarWithoutPluginCoreContract(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileLegacyModelFamilyPluginJar(tempDir);
        FeaturesCommand.Inspect command = new FeaturesCommand.Inspect();
        command.jarPath = jarPath.toString();
        command.json = true;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("warning", root.path("status").asText());
        assertTrue(root.path("valid").asBoolean());
        assertTrue(root.path("warnings").toString()
                .contains("tech.kayys.gollek.spi.plugin.GollekPlugin"));
        assertTrue(root.path("warnings").toString().contains("plugin.json"));
        JsonNode jar = root.path("jar");
        assertTrue(jar.path("has_model_family_service_entry").asBoolean());
        assertFalse(jar.path("has_gollek_plugin_service_entry").asBoolean());
        assertFalse(jar.path("has_plugin_descriptor").asBoolean());
    }

    @Test
    void strictInspectRejectsModelFamilyPluginJarWithoutPluginCoreContract(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileLegacyModelFamilyPluginJar(tempDir);
        FeaturesCommand.Inspect command = new FeaturesCommand.Inspect();
        command.jarPath = jarPath.toString();
        command.strict = true;
        command.json = true;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(1, captured.value());
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("error", root.path("status").asText());
        assertFalse(root.path("valid").asBoolean());
        assertTrue(root.path("errors").toString()
                .contains("tech.kayys.gollek.spi.plugin.GollekPlugin"));
        assertTrue(root.path("errors").toString().contains("plugin.json"));
    }

    @Test
    void installCanCopyModelFamilyPluginJarToPluginDirectory(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install command = new FeaturesCommand.Install();
        command.jarPath = jarPath.toString();
        command.pluginDir = pluginDir.toString();
        command.force = true;
        command.json = true;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        Path installedJar = pluginDir.resolve("inspect-model-family.jar");
        assertTrue(Files.isRegularFile(installedJar));
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("plugin", root.path("install_mode").asText());
        assertEquals(pluginDir.toAbsolutePath().normalize().toString(), root.path("plugin_dir").asText());
        assertEquals(installedJar.toAbsolutePath().normalize().toString(), root.path("target").asText());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                root.path("jar").path("model_family_providers").get(0).asText());
        assertEquals("model-family/inspect-family", root.path("lock_entry").path("plugin_id").asText());
        assertEquals("ready", root.path("lock_entry").path("plugin_tokenizer_metadata_status").asText());
        assertEquals("ready", root.path("lock_entry").path("tokenizer_metadata_status").asText());
        assertEquals("hf-bpe", root.path("lock_entry").path("tokenizer_kinds").get(0).asText());

        JsonNode lock = new ObjectMapper().readTree(Files.readString(
                pluginDir.resolve("gollek-extensions.lock.json")));
        JsonNode lockEntry = lock.path("extensions").path("inspect-model-family.jar");
        assertEquals("model-family/inspect-family", lockEntry.path("plugin_id").asText());
        assertEquals("Inspect Model Family", lockEntry.path("plugin_name").asText());
        assertEquals("0.1.0", lockEntry.path("plugin_version").asText());
        assertEquals("external.inspect.InspectModelFamilyPlugin", lockEntry.path("plugin_main_class").asText());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                lockEntry.path("gollek_plugin_providers").get(0).asText());
        assertEquals("inspect-family", lockEntry.path("model_family_ids").get(0).asText());
        assertEquals("hf-bpe", lockEntry.path("plugin_tokenizer_kind").asText());
        assertEquals("hf-bpe", lockEntry.path("plugin_tokenizer_kinds").get(0).asText());
        assertEquals("ready", lockEntry.path("plugin_tokenizer_metadata_status").asText());
        assertEquals("hf-bpe", lockEntry.path("model_family_tokenizer_kinds").get(0).asText());
        assertEquals("ready", lockEntry.path("model_family_tokenizer_metadata_statuses").get(0).asText());
        assertEquals("hf-bpe", lockEntry.path("tokenizer_kind").asText());
        assertEquals("hf-bpe", lockEntry.path("tokenizer_kinds").get(0).asText());
        assertEquals("ready", lockEntry.path("tokenizer_metadata_status").asText());
    }

    @Test
    void installDryRunReportsPlannedModelFamilyPluginLockEntry(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install command = new FeaturesCommand.Install();
        command.jarPath = jarPath.toString();
        command.pluginDir = pluginDir.toString();
        command.dryRun = true;
        command.json = true;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        Path installedJar = pluginDir.resolve("inspect-model-family.jar");
        assertFalse(Files.exists(installedJar));
        assertFalse(Files.exists(pluginDir.resolve("gollek-extensions.lock.json")));

        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertTrue(root.path("dry_run").asBoolean());
        assertFalse(root.path("locked").asBoolean());
        assertTrue(root.path("lock_entry").isMissingNode());
        JsonNode planned = root.path("would_lock_entry");
        assertEquals("inspect-model-family.jar", planned.path("filename").asText());
        assertEquals(installedJar.toAbsolutePath().normalize().toString(), planned.path("path").asText());
        assertEquals("model-family/inspect-family", planned.path("plugin_id").asText());
        assertEquals("ready", planned.path("plugin_tokenizer_metadata_status").asText());
        assertEquals("ready", planned.path("tokenizer_metadata_status").asText());
        assertEquals("hf-bpe", planned.path("tokenizer_kinds").get(0).asText());
        assertTrue(planned.path("sha256").asText().matches("[0-9a-f]{64}"));
        assertTrue(planned.path("size_bytes").asLong() > 0);
        assertTrue(planned.path("sha256_error").isMissingNode());
        assertTrue(planned.path("size_error").isMissingNode());
    }

    @Test
    void installRecordsPendingTokenizerMetadataInModelFamilyPluginLockEntry(@TempDir Path tempDir) throws Exception {
        Path jarPath = compilePendingTokenizerScaffoldJar(tempDir);

        FeaturesCommand.Inspect inspect = new FeaturesCommand.Inspect();
        inspect.jarPath = jarPath.toString();

        Captured<Integer> inspectCaptured = captureOut(inspect::call);

        assertEquals(0, inspectCaptured.value());
        assertTrue(inspectCaptured.output().contains(
                "tokenizer metadata: pending (tokenizer adapter pending fixture stabilization)"),
                inspectCaptured.output());

        FeaturesCommand.Inspect jsonInspect = new FeaturesCommand.Inspect();
        jsonInspect.jarPath = jarPath.toString();
        jsonInspect.json = true;

        Captured<Integer> jsonInspectCaptured = captureOut(jsonInspect::call);

        assertEquals(0, jsonInspectCaptured.value());
        JsonNode inspectedJar = new ObjectMapper()
                .readTree(jsonInspectCaptured.output())
                .path("jar");
        assertEquals("pending",
                inspectedJar.path("artifact_tokenizer_metadata").path("tokenizer_metadata_status").asText());
        assertEquals("tokenizer adapter pending fixture stabilization",
                inspectedJar.path("artifact_tokenizer_metadata")
                        .path("tokenizer_metadata_pending_reason")
                        .asText());

        FeaturesCommand.Doctor uninstalledDoctor = new FeaturesCommand.Doctor();
        uninstalledDoctor.featurePaths = List.of(jarPath.toString());

        Captured<Integer> uninstalledDoctorCaptured = captureOut(uninstalledDoctor::call);

        assertEquals(0, uninstalledDoctorCaptured.value());
        assertTrue(uninstalledDoctorCaptured.output().contains(
                "tokenizer metadata: pending (tokenizer adapter pending fixture stabilization)"),
                uninstalledDoctorCaptured.output());
        assertFalse(uninstalledDoctorCaptured.output().contains("tokenizer lock:"),
                uninstalledDoctorCaptured.output());

        FeaturesCommand.Doctor jsonUninstalledDoctor = new FeaturesCommand.Doctor();
        jsonUninstalledDoctor.featurePaths = List.of(jarPath.toString());
        jsonUninstalledDoctor.json = true;

        Captured<Integer> jsonUninstalledDoctorCaptured = captureOut(jsonUninstalledDoctor::call);

        assertEquals(0, jsonUninstalledDoctorCaptured.value());
        JsonNode uninstalledDoctorJar = findDoctorJar(
                new ObjectMapper()
                        .readTree(jsonUninstalledDoctorCaptured.output())
                        .path("doctor")
                        .path("jars"),
                "pending-tokenizer-model-family.jar");
        assertEquals("pending",
                uninstalledDoctorJar.path("artifact_tokenizer_metadata").path("tokenizer_metadata_status").asText());
        assertEquals("tokenizer adapter pending fixture stabilization",
                uninstalledDoctorJar.path("artifact_tokenizer_metadata")
                        .path("tokenizer_metadata_pending_reason")
                        .asText());
        assertTrue(uninstalledDoctorJar.path("tokenizer_lock_status").isMissingNode());

        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        install.json = true;

        Captured<Integer> captured = captureOut(install::call);

        assertEquals(0, captured.value());
        JsonNode root = new ObjectMapper().readTree(captured.output());
        JsonNode lockEntry = root.path("lock_entry");
        assertEquals("model-family/pending-tokenizer", lockEntry.path("plugin_id").asText());
        assertEquals("pending", lockEntry.path("plugin_tokenizer_metadata_status").asText());
        assertEquals("tokenizer adapter pending fixture stabilization",
                lockEntry.path("plugin_tokenizer_metadata_pending_reason").asText());
        assertEquals("pending", lockEntry.path("model_family_tokenizer_metadata_statuses").get(0).asText());
        assertEquals("tokenizer adapter pending fixture stabilization",
                lockEntry.path("model_family_tokenizer_metadata_pending_reasons")
                        .path("pending-tokenizer")
                        .asText());
        assertEquals("pending", lockEntry.path("tokenizer_metadata_status").asText());
        assertEquals("tokenizer adapter pending fixture stabilization",
                lockEntry.path("tokenizer_metadata_pending_reason").asText());
        assertTrue(lockEntry.path("tokenizer_kind").asText().isBlank());
        assertTrue(lockEntry.path("tokenizer_kinds").isEmpty());

        Path textPluginDir = tempDir.resolve("text-plugins");
        FeaturesCommand.Install textInstall = new FeaturesCommand.Install();
        textInstall.jarPath = jarPath.toString();
        textInstall.pluginDir = textPluginDir.toString();
        textInstall.force = true;

        Captured<Integer> textCaptured = captureOut(textInstall::call);

        assertEquals(0, textCaptured.value());
        assertTrue(textCaptured.output().contains(
                "tokenizer metadata: pending (tokenizer adapter pending fixture stabilization)"),
                textCaptured.output());

        FeaturesCommand.Doctor doctor = new FeaturesCommand.Doctor();
        doctor.featurePaths = List.of(pluginDir.toString());
        doctor.json = true;

        Captured<Integer> doctorCaptured = captureOut(doctor::call);

        assertEquals(0, doctorCaptured.value());
        JsonNode doctorRoot = new ObjectMapper().readTree(doctorCaptured.output());
        JsonNode summary = doctorRoot.path("doctor").path("summary");
        assertTrue(summary.path("tokenizer_lock_ok").asInt() >= 1);
        assertEquals(0, summary.path("tokenizer_lock_missing").asInt());
        assertEquals(0, summary.path("tokenizer_lock_stale").asInt());
        JsonNode jar = findDoctorJar(doctorRoot.path("doctor").path("jars"), "pending-tokenizer-model-family.jar");
        assertEquals("ok", jar.path("tokenizer_lock_status").asText());

        FeaturesCommand.Doctor textDoctor = new FeaturesCommand.Doctor();
        textDoctor.featurePaths = List.of(pluginDir.toString());

        Captured<Integer> textDoctorCaptured = captureOut(textDoctor::call);

        assertEquals(0, textDoctorCaptured.value());
        assertTrue(textDoctorCaptured.output().contains(
                "tokenizer metadata: pending (tokenizer adapter pending fixture stabilization)"),
                textDoctorCaptured.output());
        assertTrue(textDoctorCaptured.output().contains("tokenizer lock: ok"), textDoctorCaptured.output());
    }

    @Test
    void installRejectsModelFamilyPluginJarWithoutPluginCoreContract(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileLegacyModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install command = new FeaturesCommand.Install();
        command.jarPath = jarPath.toString();
        command.pluginDir = pluginDir.toString();
        command.force = true;
        command.json = true;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(1, captured.value());
        assertFalse(Files.exists(pluginDir.resolve("inspect-model-family.jar")));
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertTrue(root.path("error").asText().contains("Model-family plugin install requires"));
        assertTrue(root.path("error").asText().contains("tech.kayys.gollek.spi.plugin.GollekPlugin"));
        assertTrue(root.path("error").asText().contains("plugin.json"));
    }

    @Test
    void doctorReportsPluginReadinessForInstalledModelFamilyPlugin(@TempDir Path tempDir) throws Exception {
        Path fakeHome = tempDir.resolve("home");
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, captureOut(install::call).value());

        withSystemProperty("user.home", fakeHome.toString(), () -> {
            FeaturesCommand.Doctor doctor = new FeaturesCommand.Doctor();
            doctor.featurePaths = List.of(pluginDir.toString());
            doctor.json = true;

            Captured<Integer> captured = captureOut(doctor::call);

            assertEquals(0, captured.value());
            JsonNode root = new ObjectMapper().readTree(captured.output());
            JsonNode summary = root.path("doctor").path("summary");
            assertTrue(summary.path("plugin_install_candidates").asInt() >= 1);
            assertTrue(summary.path("plugin_install_ready").asInt() >= 1);
            assertEquals(0, summary.path("plugin_install_not_ready").asInt());
            assertTrue(summary.path("tokenizer_lock_ok").asInt() >= 1);
            assertEquals(0, summary.path("tokenizer_lock_missing").asInt());
            assertEquals(0, summary.path("tokenizer_lock_stale").asInt());

            JsonNode jar = findDoctorJar(root.path("doctor").path("jars"), "inspect-model-family.jar");
            assertTrue(jar.path("plugin_install_candidate").asBoolean());
            assertTrue(jar.path("plugin_install_ready").asBoolean());
            assertTrue(jar.path("plugin_install_errors").isEmpty());
            assertEquals("ok", jar.path("tokenizer_lock_status").asText());
            assertEquals("model-family/inspect-family", jar.path("plugin_descriptor").path("id").asText());
            assertEquals("external.inspect.InspectModelFamilyPlugin",
                    jar.path("gollek_plugin_providers").get(0).asText());
        });
    }

    @Test
    void doctorWarnsWhenModelFamilyTokenizerLockMetadataIsMissing(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, captureOut(install::call).value());

        ObjectMapper mapper = new ObjectMapper();
        Path lockPath = pluginDir.resolve("gollek-extensions.lock.json");
        ObjectNode lock = (ObjectNode) mapper.readTree(Files.readString(lockPath));
        removeTokenizerLockFields(lock.path("extensions").path("inspect-model-family.jar"));
        removeTokenizerLockFields(lock.path("features").path("inspect-model-family.jar"));
        Files.writeString(
                lockPath,
                mapper.writerWithDefaultPrettyPrinter().writeValueAsString(lock) + System.lineSeparator(),
                StandardCharsets.UTF_8);

        FeaturesCommand.Doctor doctor = new FeaturesCommand.Doctor();
        doctor.featurePaths = List.of(pluginDir.toString());
        doctor.json = true;

        Captured<Integer> captured = captureOut(doctor::call);

        assertEquals(0, captured.value());
        JsonNode root = mapper.readTree(captured.output());
        JsonNode summary = root.path("doctor").path("summary");
        assertEquals(0, summary.path("tokenizer_lock_ok").asInt());
        assertTrue(summary.path("tokenizer_lock_missing").asInt() >= 1);
        assertEquals(0, summary.path("tokenizer_lock_stale").asInt());
        JsonNode jar = findDoctorJar(root.path("doctor").path("jars"), "inspect-model-family.jar");
        assertEquals("warning", jar.path("validation").path("status").asText());
        assertEquals("missing", jar.path("tokenizer_lock_status").asText());
        assertTrue(jar.path("install_lock_warnings").toString().contains("missing tokenizer metadata"));
        assertTrue(jar.path("install_lock_warnings").toString().contains("doctor --repair"));
    }

    @Test
    void doctorReportsStaleModelFamilyTokenizerLockMetadata(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, captureOut(install::call).value());

        ObjectMapper mapper = new ObjectMapper();
        Path lockPath = pluginDir.resolve("gollek-extensions.lock.json");
        ObjectNode lock = (ObjectNode) mapper.readTree(Files.readString(lockPath));
        replaceTokenizerLockKind(lock.path("extensions").path("inspect-model-family.jar"), "sentencepiece");
        replaceTokenizerLockKind(lock.path("features").path("inspect-model-family.jar"), "sentencepiece");
        Files.writeString(
                lockPath,
                mapper.writerWithDefaultPrettyPrinter().writeValueAsString(lock) + System.lineSeparator(),
                StandardCharsets.UTF_8);

        FeaturesCommand.Doctor doctor = new FeaturesCommand.Doctor();
        doctor.featurePaths = List.of(pluginDir.toString());
        doctor.json = true;

        Captured<Integer> captured = captureOut(doctor::call);

        assertEquals(0, captured.value());
        JsonNode root = mapper.readTree(captured.output());
        JsonNode summary = root.path("doctor").path("summary");
        assertEquals(0, summary.path("tokenizer_lock_ok").asInt());
        assertEquals(0, summary.path("tokenizer_lock_missing").asInt());
        assertTrue(summary.path("tokenizer_lock_stale").asInt() >= 1);
        JsonNode jar = findDoctorJar(root.path("doctor").path("jars"), "inspect-model-family.jar");
        assertEquals("warning", jar.path("validation").path("status").asText());
        assertEquals("stale", jar.path("tokenizer_lock_status").asText());
        assertTrue(jar.path("install_lock_warnings").toString().contains("tokenizer metadata differs"));
        assertTrue(jar.path("install_lock_warnings").toString().contains("doctor --repair"));
    }

    @Test
    void doctorRepairDryRunPreviewsTokenizerLockMetadataBackfillWithoutWriting(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, captureOut(install::call).value());

        ObjectMapper mapper = new ObjectMapper();
        Path lockPath = pluginDir.resolve("gollek-extensions.lock.json");
        ObjectNode lock = (ObjectNode) mapper.readTree(Files.readString(lockPath));
        removeTokenizerLockFields(lock.path("extensions").path("inspect-model-family.jar"));
        removeTokenizerLockFields(lock.path("features").path("inspect-model-family.jar"));
        String oldLockText = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(lock)
                + System.lineSeparator();
        Files.writeString(lockPath, oldLockText, StandardCharsets.UTF_8);

        FeaturesCommand.Doctor doctor = new FeaturesCommand.Doctor();
        doctor.featurePaths = List.of(pluginDir.toString());
        doctor.repair = true;
        doctor.dryRun = true;
        doctor.json = true;

        Captured<Integer> captured = captureOut(doctor::call);

        assertEquals(0, captured.value());
        assertEquals(oldLockText, Files.readString(lockPath));
        JsonNode root = mapper.readTree(captured.output());
        assertTrue(root.path("dry_run").asBoolean());
        assertTrue(root.path("repair").path("dry_run").asBoolean());
        assertEquals(1, root.path("repair").path("summary").path("changed").asInt());
        JsonNode operation = root.path("repair").path("operations").get(0);
        assertEquals("lock_refresh_tokenizer_metadata", operation.path("action").asText());
        assertEquals("would_change", operation.path("status").asText());
        JsonNode summary = root.path("doctor").path("summary");
        assertEquals(0, summary.path("tokenizer_lock_ok").asInt());
        assertTrue(summary.path("tokenizer_lock_missing").asInt() >= 1);
        JsonNode jar = findDoctorJar(root.path("doctor").path("jars"), "inspect-model-family.jar");
        assertEquals("missing", jar.path("tokenizer_lock_status").asText());
    }

    @Test
    void doctorRepairBackfillsModelFamilyTokenizerLockMetadata(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, captureOut(install::call).value());

        ObjectMapper mapper = new ObjectMapper();
        Path lockPath = pluginDir.resolve("gollek-extensions.lock.json");
        ObjectNode lock = (ObjectNode) mapper.readTree(Files.readString(lockPath));
        removeTokenizerLockFields(lock.path("extensions").path("inspect-model-family.jar"));
        removeTokenizerLockFields(lock.path("features").path("inspect-model-family.jar"));
        Files.writeString(
                lockPath,
                mapper.writerWithDefaultPrettyPrinter().writeValueAsString(lock) + System.lineSeparator(),
                StandardCharsets.UTF_8);

        FeaturesCommand.Doctor doctor = new FeaturesCommand.Doctor();
        doctor.featurePaths = List.of(pluginDir.toString());
        doctor.repair = true;
        doctor.json = true;

        Captured<Integer> captured = captureOut(doctor::call);

        assertEquals(0, captured.value());
        JsonNode root = mapper.readTree(captured.output());
        JsonNode summary = root.path("doctor").path("summary");
        assertTrue(summary.path("tokenizer_lock_ok").asInt() >= 1);
        assertEquals(0, summary.path("tokenizer_lock_missing").asInt());
        assertEquals(0, summary.path("tokenizer_lock_stale").asInt());
        JsonNode jar = findDoctorJar(root.path("doctor").path("jars"), "inspect-model-family.jar");
        assertEquals("ok", jar.path("tokenizer_lock_status").asText());
        assertEquals(1, root.path("repair").path("summary").path("changed").asInt());
        JsonNode operation = root.path("repair").path("operations").get(0);
        assertEquals("lock_refresh_tokenizer_metadata", operation.path("action").asText());
        assertEquals("changed", operation.path("status").asText());

        JsonNode repaired = mapper.readTree(Files.readString(lockPath))
                .path("extensions")
                .path("inspect-model-family.jar");
        assertEquals("hf-bpe", repaired.path("plugin_tokenizer_kind").asText());
        assertEquals("hf-bpe", repaired.path("plugin_tokenizer_kinds").get(0).asText());
        assertEquals("hf-bpe", repaired.path("model_family_tokenizer_kinds").get(0).asText());
        assertEquals("hf-bpe", repaired.path("tokenizer_kind").asText());
        assertEquals("hf-bpe", repaired.path("tokenizer_kinds").get(0).asText());
    }

    @Test
    void doctorReportsLegacyModelFamilyPluginJarAsPluginNotReady(@TempDir Path tempDir) throws Exception {
        Path fakeHome = tempDir.resolve("home");
        Path jarPath = compileLegacyModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        Files.createDirectories(pluginDir);
        Files.copy(jarPath, pluginDir.resolve("inspect-model-family.jar"));

        withSystemProperty("user.home", fakeHome.toString(), () -> {
            FeaturesCommand.Doctor doctor = new FeaturesCommand.Doctor();
            doctor.featurePaths = List.of(pluginDir.toString());
            doctor.json = true;

            Captured<Integer> captured = captureOut(doctor::call);

            assertEquals(0, captured.value());
            JsonNode root = new ObjectMapper().readTree(captured.output());
            JsonNode summary = root.path("doctor").path("summary");
            assertTrue(summary.path("plugin_install_candidates").asInt() >= 1);
            assertTrue(summary.path("plugin_install_not_ready").asInt() >= 1);

            JsonNode jar = findDoctorJar(root.path("doctor").path("jars"), "inspect-model-family.jar");
            assertTrue(jar.path("plugin_install_candidate").asBoolean());
            assertFalse(jar.path("plugin_install_ready").asBoolean());
            assertTrue(jar.path("plugin_install_errors").toString()
                    .contains("tech.kayys.gollek.spi.plugin.GollekPlugin"));
            assertTrue(jar.path("plugin_install_errors").toString().contains("plugin.json"));
            assertEquals("warning", jar.path("validation").path("status").asText());
        });
    }

    @Test
    void detachCanRemoveModelFamilyPluginJarFromPluginDirectoryByFamilyId(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, install.call());

        Path installedJar = pluginDir.resolve("inspect-model-family.jar");
        assertTrue(Files.isRegularFile(installedJar));

        FeaturesCommand.Remove remove = new FeaturesCommand.Remove();
        remove.target = "inspect-family";
        remove.pluginDir = pluginDir.toString();
        remove.json = true;

        Captured<Integer> captured = captureOut(remove::call);

        assertEquals(0, captured.value());
        assertFalse(Files.exists(installedJar));
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("plugin", root.path("remove_mode").asText());
        assertEquals(pluginDir.toAbsolutePath().normalize().toString(), root.path("plugin_dir").asText());
        assertEquals("inspect-family",
                root.path("jar").path("manifest").path("model_families").get(0).path("id").asText());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                root.path("jar").path("model_family_providers").get(0).asText());
    }

    @Test
    void detachCanRemoveModelFamilyPluginJarFromPluginDirectoryByPluginId(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, install.call());

        Path installedJar = pluginDir.resolve("inspect-model-family.jar");
        assertTrue(Files.isRegularFile(installedJar));

        FeaturesCommand.Detach detach = new FeaturesCommand.Detach();
        detach.target = "model-family/inspect-family";
        detach.pluginDir = pluginDir.toString();
        detach.json = true;

        Captured<Integer> captured = captureOut(detach::call);

        assertEquals(0, captured.value());
        assertFalse(Files.exists(installedJar));
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("detach", root.path("action").asText());
        assertEquals("plugin", root.path("remove_mode").asText());
        assertEquals("model-family/inspect-family",
                root.path("jar").path("plugin_descriptor").path("id").asText());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                root.path("jar").path("gollek_plugin_providers").get(0).asText());
    }

    @Test
    void backupsCanListDetachedModelFamilyPluginBackupsByPluginId(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, install.call());

        FeaturesCommand.Detach detach = new FeaturesCommand.Detach();
        detach.target = "model-family/inspect-family";
        detach.pluginDir = pluginDir.toString();
        assertEquals(0, detach.call());

        FeaturesCommand.Backups backups = new FeaturesCommand.Backups();
        backups.target = "model-family/inspect-family";
        backups.pluginDir = pluginDir.toString();
        backups.json = true;

        Captured<Integer> captured = captureOut(backups::call);

        assertEquals(0, captured.value());
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("plugin", root.path("backup_mode").asText());
        assertEquals(pluginDir.toAbsolutePath().normalize().toString(), root.path("plugin_dir").asText());
        assertTrue(root.path("backups").isArray());
        assertTrue(root.path("backups").size() >= 1);
        assertEquals("model-family/inspect-family",
                root.path("backups").get(0).path("jar").path("plugin_descriptor").path("id").asText());
    }

    @Test
    void rollbackCanRestoreDetachedModelFamilyPluginByPluginId(@TempDir Path tempDir) throws Exception {
        Path jarPath = compileModelFamilyPluginJar(tempDir);
        Path pluginDir = tempDir.resolve("plugins");
        FeaturesCommand.Install install = new FeaturesCommand.Install();
        install.jarPath = jarPath.toString();
        install.pluginDir = pluginDir.toString();
        install.force = true;
        assertEquals(0, install.call());

        Path installedJar = pluginDir.resolve("inspect-model-family.jar");
        assertTrue(Files.isRegularFile(installedJar));

        FeaturesCommand.Detach detach = new FeaturesCommand.Detach();
        detach.target = "model-family/inspect-family";
        detach.pluginDir = pluginDir.toString();
        assertEquals(0, detach.call());
        assertFalse(Files.exists(installedJar));

        FeaturesCommand.Rollback rollback = new FeaturesCommand.Rollback();
        rollback.target = "model-family/inspect-family";
        rollback.pluginDir = pluginDir.toString();
        rollback.json = true;

        Captured<Integer> captured = captureOut(rollback::call);

        assertEquals(0, captured.value());
        assertTrue(Files.isRegularFile(installedJar));
        JsonNode root = new ObjectMapper().readTree(captured.output());
        assertEquals("plugin", root.path("rollback_mode").asText());
        assertEquals(pluginDir.toAbsolutePath().normalize().toString(), root.path("plugin_dir").asText());
        assertEquals(installedJar.toAbsolutePath().normalize().toString(), root.path("target_path").asText());
        assertEquals("model-family/inspect-family",
                root.path("jar").path("plugin_descriptor").path("id").asText());

        JsonNode lock = new ObjectMapper().readTree(Files.readString(
                pluginDir.resolve("gollek-extensions.lock.json")));
        JsonNode lockEntry = lock.path("extensions").path("inspect-model-family.jar");
        assertEquals("model-family/inspect-family", lockEntry.path("plugin_id").asText());
        assertEquals("external.inspect.InspectModelFamilyPlugin",
                lockEntry.path("gollek_plugin_providers").get(0).asText());
    }

    @Test
    void detachDefaultsToDefaultPluginDirectoryForModelFamilyPlugins(@TempDir Path tempDir) throws Exception {
        Path fakeHome = tempDir.resolve("home");
        Path jarPath = compileModelFamilyPluginJar(tempDir);

        withSystemProperty("user.home", fakeHome.toString(), () -> {
            FeaturesCommand.Install install = new FeaturesCommand.Install();
            install.jarPath = jarPath.toString();
            install.force = true;
            install.json = true;
            assertEquals(0, captureOut(install::call).value());

            Path installedJar = fakeHome.resolve(".gollek/plugins/inspect-model-family.jar");
            assertTrue(Files.isRegularFile(installedJar));

            FeaturesCommand.Detach detach = new FeaturesCommand.Detach();
            detach.target = "inspect-family";
            detach.json = true;

            Captured<Integer> captured = captureOut(detach::call);

            assertEquals(0, captured.value());
            assertFalse(Files.exists(installedJar));
            JsonNode root = new ObjectMapper().readTree(captured.output());
            assertEquals("detach", root.path("action").asText());
            assertEquals("plugin", root.path("remove_mode").asText());
            assertEquals(fakeHome.resolve(".gollek/plugins").toAbsolutePath().normalize().toString(),
                    root.path("plugin_dir").asText());
        });
    }

    private static void assertContains(Path path, String expected) throws Exception {
        assertTrue(Files.readString(path).contains(expected), path + " should contain " + expected);
    }

    private static JsonNode findDoctorJar(JsonNode jars, String filename) {
        for (JsonNode jar : jars) {
            String path = jar.path("path").asText();
            if (!path.isBlank() && filename.equals(Path.of(path).getFileName().toString())) {
                return jar;
            }
        }
        throw new AssertionError("doctor output should include " + filename + ": " + jars);
    }

    private static void removeTokenizerLockFields(JsonNode node) {
        assertTrue(node instanceof ObjectNode, "lock entry should be an object");
        ((ObjectNode) node).remove(List.of(
                "plugin_tokenizer_kind",
                "plugin_tokenizer_kinds",
                "plugin_tokenizer_metadata_status",
                "plugin_tokenizer_metadata_pending_reason",
                "model_family_tokenizer_kinds",
                "model_family_tokenizer_metadata_statuses",
                "model_family_tokenizer_metadata_pending_reasons",
                "tokenizer_kind",
                "tokenizer_kinds",
                "tokenizer_metadata_status",
                "tokenizer_metadata_pending_reason"));
    }

    private static void replaceTokenizerLockKind(JsonNode node, String tokenizerKind) {
        assertTrue(node instanceof ObjectNode, "lock entry should be an object");
        ((ObjectNode) node).put("tokenizer_kind", tokenizerKind);
    }

    private static Path compileModelFamilyPluginJar(Path tempDir) throws Exception {
        return compileModelFamilyPluginJar(tempDir, true);
    }

    private static Path compileLegacyModelFamilyPluginJar(Path tempDir) throws Exception {
        return compileModelFamilyPluginJar(tempDir, false);
    }

    private static Path compilePendingTokenizerScaffoldJar(Path tempDir) throws Exception {
        Path root = tempDir.resolve("gollek-model-pending-tokenizer");
        FeaturesCommand.Init command = new FeaturesCommand.Init();
        command.targetDir = root.toString();
        command.kind = "model-family";
        command.featureId = "pending-tokenizer-model-family";
        command.modelFamilyId = "pending-tokenizer";
        command.modelTypes = List.of("pending_tokenizer");
        command.architectureClassNames = List.of("PendingTokenizerForCausalLM");
        command.modelFamilyCapabilities = List.of("tokenizer");
        command.tokenizerMetadataStatus = "pending";
        command.tokenizerMetadataPendingReason = "tokenizer adapter pending fixture stabilization";
        assertEquals(0, command.call());

        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertTrue(compiler != null, "Java compiler must be available for model-family scaffold compilation");
        Path sourceRoot = root.resolve("lib/src/main/java");
        Path resourcesRoot = root.resolve("lib/src/main/resources");
        Path classesDir = tempDir.resolve("pending-tokenizer-classes");
        Files.createDirectories(classesDir);
        List<Path> sourceFiles;
        try (var files = Files.walk(sourceRoot)) {
            sourceFiles = files.filter(path -> path.toString().endsWith(".java"))
                    .sorted()
                    .toList();
        }
        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
            Iterable<? extends javax.tools.JavaFileObject> units =
                    fileManager.getJavaFileObjectsFromFiles(sourceFiles.stream().map(Path::toFile).toList());
            List<String> options = List.of(
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", classesDir.toString());
            Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
            assertTrue(Boolean.TRUE.equals(compiled), "pending model-family scaffold should compile");
        }
        try (var files = Files.walk(resourcesRoot)) {
            for (Path path : files.filter(Files::isRegularFile).sorted(Comparator.naturalOrder()).toList()) {
                Path target = classesDir.resolve(resourcesRoot.relativize(path).toString());
                Files.createDirectories(target.getParent());
                Files.copy(path, target);
            }
        }

        Path jarPath = tempDir.resolve("pending-tokenizer-model-family.jar");
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath));
                var files = Files.walk(classesDir)) {
            for (Path path : files.filter(Files::isRegularFile).sorted(Comparator.naturalOrder()).toList()) {
                String entryName = classesDir.relativize(path).toString().replace('\\', '/');
                jar.putNextEntry(new JarEntry(entryName));
                Files.copy(path, jar);
                jar.closeEntry();
            }
        }
        return jarPath;
    }

    private static Path compileModelFamilyPluginJar(Path tempDir, boolean includePluginCoreContract) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertTrue(compiler != null, "Java compiler must be available for model-family fixture compilation");
        Path sourceRoot = tempDir.resolve("src");
        Path classesDir = tempDir.resolve("classes");
        Path sourceFile = sourceRoot.resolve("external/inspect/InspectModelFamilyPlugin.java");
        Files.createDirectories(sourceFile.getParent());
        Files.createDirectories(classesDir);
        Files.writeString(sourceFile, inspectModelFamilySource(), StandardCharsets.UTF_8);
        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(null, Locale.ROOT, StandardCharsets.UTF_8)) {
            Iterable<? extends javax.tools.JavaFileObject> units =
                    fileManager.getJavaFileObjectsFromFiles(List.of(sourceFile.toFile()));
            List<String> options = List.of(
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", classesDir.toString());
            Boolean compiled = compiler.getTask(null, fileManager, null, options, null, units).call();
            assertTrue(Boolean.TRUE.equals(compiled), "model-family fixture should compile");
        }

        Path serviceFile = classesDir.resolve(
                "META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin");
        Files.createDirectories(serviceFile.getParent());
        Files.writeString(
                serviceFile,
                "external.inspect.InspectModelFamilyPlugin" + System.lineSeparator(),
                StandardCharsets.UTF_8);
        if (includePluginCoreContract) {
            Path gollekPluginServiceFile = classesDir.resolve(
                    "META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin");
            Files.writeString(
                    gollekPluginServiceFile,
                    "external.inspect.InspectModelFamilyPlugin" + System.lineSeparator(),
                    StandardCharsets.UTF_8);
            Files.writeString(
                    classesDir.resolve("plugin.json"),
                    inspectModelFamilyPluginDescriptor(),
                    StandardCharsets.UTF_8);
        }
        Files.writeString(
                classesDir.resolve("META-INF/gollek-extension.json"),
                inspectModelFamilyManifest(),
                StandardCharsets.UTF_8);

        Path jarPath = tempDir.resolve("inspect-model-family.jar");
        try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(jarPath));
                var files = Files.walk(classesDir)) {
            for (Path path : files.filter(Files::isRegularFile).sorted(Comparator.naturalOrder()).toList()) {
                String entryName = classesDir.relativize(path).toString().replace('\\', '/');
                jar.putNextEntry(new JarEntry(entryName));
                Files.copy(path, jar);
                jar.closeEntry();
            }
        }
        return jarPath;
    }

    private static String inspectModelFamilySource() {
        return """
                package external.inspect;

                import java.util.List;
                import java.util.Map;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
                import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

                public final class InspectModelFamilyPlugin implements ModelFamilyPlugin {
                    @Override
                    public ModelFamilyDescriptor descriptor() {
                        return new ModelFamilyDescriptor(
                                "inspect-family",
                                "Inspect Family",
                                List.of("inspect_model"),
                                List.of("InspectForCausalLM"),
                                List.of(ModelFamilyCapability.TOKENIZER),
                                Map.of("bundle_profile", "optional"));
                    }

                    @Override
                    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("inspect-family-tokenizer"));
                    }
                }
                """;
    }

    private static String inspectModelFamilyManifest() {
        return """
                {
                  "schema_version" : 1,
                  "id" : "inspect-model-family",
                  "name" : "Inspect Model Family",
                  "version" : "0.1.0",
                  "requires" : {
                    "gollek_spi" : "*"
                  },
                  "services" : {
                    "model_family_plugins" : [ "external.inspect.InspectModelFamilyPlugin" ]
                  },
                  "model_families" : [ {
                    "id" : "inspect-family",
                    "class" : "external.inspect.InspectModelFamilyPlugin",
                    "model_types" : [ "inspect_model" ],
                    "architecture_classes" : [ "InspectForCausalLM" ],
                    "capabilities" : [ "TOKENIZER" ],
                    "tokenizer_kind" : "hf-bpe",
                    "tokenizer_kinds" : [ "hf-bpe" ],
                    "bundle_profile" : "optional"
                  } ]
                }
                """;
    }

    private static String inspectModelFamilyPluginDescriptor() {
        return """
                {
                  "id" : "model-family/inspect-family",
                  "name" : "Inspect Model Family",
                  "version" : "0.1.0",
                  "description" : "Model-family fixture for CLI inspection",
                  "vendor" : "Test",
                  "mainClass" : "external.inspect.InspectModelFamilyPlugin",
                  "dependencies" : [ ],
                  "optionalDependencies" : [ ],
                  "properties" : {
                    "extensionPoint" : "model-family",
                    "bundleProfile" : "optional",
                    "families" : [ "inspect-family" ],
                    "tokenizerKind" : "hf-bpe",
                    "tokenizerKinds" : [ "hf-bpe" ]
                  }
                }
                """;
    }

    private static <T> Captured<T> captureOut(Callable<T> callable) throws Exception {
        PrintStream originalOut = System.out;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try (PrintStream capture = new PrintStream(output, true, StandardCharsets.UTF_8)) {
            System.setOut(capture);
            T value = callable.call();
            capture.flush();
            return new Captured<>(value, output.toString(StandardCharsets.UTF_8));
        } finally {
            System.setOut(originalOut);
        }
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

    private record Captured<T>(T value, String output) {
    }
}
