package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.cli.util.ExternalPluginClasspath;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.CodeSource;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarFile;
import java.util.regex.Pattern;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

@Dependent
@Unremovable
@Command(
        name = "extensions",
        aliases = { "features", "feature", "pipelines" },
        mixinStandardHelpOptions = true,
        description = "List and manage installable Gollek extensions and their exposed features",
        subcommands = {
                FeaturesCommand.Init.class,
                FeaturesCommand.Kinds.class,
                FeaturesCommand.Index.class,
                FeaturesCommand.Doctor.class,
                FeaturesCommand.Inspect.class,
                FeaturesCommand.Install.class,
                FeaturesCommand.Migrate.class,
                FeaturesCommand.FinalizeMigration.class,
                FeaturesCommand.PruneLegacy.class,
                FeaturesCommand.Remove.class,
                FeaturesCommand.Detach.class,
                FeaturesCommand.Rollback.class,
                FeaturesCommand.Backups.class
        })
public class FeaturesCommand implements Runnable {
    private static final String SERVICE_ENTRY = "META-INF/services/tech.kayys.gollek.spi.pipeline.ModelPipeline";
    private static final String ADAPTER_SERVICE_ENTRY = "META-INF/services/tech.kayys.gollek.spi.feature.FeatureAdapter";
    private static final String MODEL_FAMILY_KIND = "model-family";
    private static final String MODEL_FAMILY_SERVICE_ENTRY =
            "META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin";
    private static final String GOLLEK_PLUGIN_SERVICE_ENTRY =
            "META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin";
    private static final String MANIFEST_ENTRY = "META-INF/gollek-extension.json";
    private static final String LEGACY_MANIFEST_ENTRY = "META-INF/gollek-feature.json";
    private static final String PLUGIN_DESCRIPTOR_ENTRY = "plugin.json";
    private static final String PACKAGE_MANIFEST_ENTRY = "gollek-package.json";
    private static final String EXTENSION_PATH_PROPERTY = "gollek.extensions.path";
    private static final String EXTENSION_PATH_ENV = "GOLLEK_EXTENSION_PATH";
    private static final String LEGACY_FEATURE_PATH_PROPERTY = "gollek.features.path";
    private static final String LEGACY_FEATURE_PATH_ENV = "GOLLEK_FEATURE_PATH";
    private static final String DEFAULT_RUNTIME_VERSION = "dev";
    private static final String INSTALL_LOCKFILE = "gollek-extensions.lock.json";
    private static final String LEGACY_INSTALL_LOCKFILE = "gollek-features.lock.json";
    private static final String FEATURE_BACKUP_DIR = ".gollek-extension-backups";
    private static final String LEGACY_FEATURE_BACKUP_DIR = ".gollek-feature-backups";

    ModelPipelineRegistry pipelineRegistry = null;

    @Option(names = { "--details" }, description = "Show tags and descriptor metadata")
    boolean verbose;

    @Option(names = { "--paths" }, description = "Show implementation class and code-source paths")
    boolean paths;

    @Option(names = { "--json" }, description = "Print feature discovery as JSON")
    boolean json;

    @Option(names = { "--doctor" }, description = "Inspect runtime extension paths and service entries")
    boolean doctor;

    @Option(names = { "--dir", "--plugin-dir", "--path" }, split = ",",
            description = "Additional extension jar or directory to scan without loading it")
    List<String> scanPaths = new ArrayList<>();

    @Override
    public void run() {
        if (pipelineRegistry == null) {
            System.out.println("Extension pipeline registry is not available.");
            return;
        }
        List<FeatureRow> loadedFeatures = pipelineRegistry.all().stream()
                .map(FeaturesCommand::featureRow)
                .toList();
        FeatureDoctorReport scanReport = scanPaths.isEmpty() ? null : inspectConfiguredFeaturePaths(scanPaths);
        List<FeatureRow> scannedFeatures = scanReport == null ? List.of() : scannedFeatureRows(scanReport);
        FeatureDoctorReport doctorReport = doctor ? inspectRuntimeFeaturePaths(scanPaths) : null;
        List<FeatureRow> displayFeatures = new ArrayList<>();
        displayFeatures.addAll(loadedFeatures);
        displayFeatures.addAll(scannedFeatures);
        if (json) {
            printJson(displayFeatures, loadedFeatures.size(), scannedFeatures.size(), scanReport, doctorReport);
            return;
        }
        if (doctorReport != null) {
            printDoctor(doctorReport, false);
            System.out.println();
        } else if (scanReport != null) {
            printScanSummary(scanReport);
            System.out.println();
        }
        if (displayFeatures.isEmpty()) {
            System.out.println("No extension features discovered.");
            if (scanPaths.isEmpty()) {
                System.out.println("Install extension jars or packages into ~/.gollek/extensions or set GOLLEK_EXTENSION_PATH.");
            } else {
                System.out.println("No manifest-declared capabilities were found in the scanned extension paths.");
                System.out.println("Run gollek extensions doctor --dir <path> for validation details.");
            }
            return;
        }

        System.out.printf("%-18s %-22s %-13s %-12s %-15s %-15s %-8s %-14s%n",
                "ID", "NAME", "KIND", "FAMILY", "INPUTS", "OUTPUTS", "PRIOR", "SOURCE");
        System.out.println("-".repeat(126));
        for (FeatureRow pipeline : displayFeatures) {
            System.out.printf("%-18s %-22s %-13s %-12s %-15s %-15s %-8d %-14s%n",
                    truncate(pipeline.id(), 18),
                    truncate(pipeline.name(), 22),
                    truncate(pipeline.kind(), 13),
                    truncate(pipeline.family(), 12),
                    truncate(String.join(",", pipeline.inputs()), 15),
                    truncate(String.join(",", pipeline.outputs()), 15),
                    pipeline.priority(),
                    truncate(pipeline.source(), 14));
            if ((verbose || paths)) {
                if (verbose && !pipeline.tags().isEmpty()) {
                    System.out.printf("  tags: %s%n", String.join(", ", pipeline.tags()));
                }
                if (paths) {
                    System.out.printf("  class: %s%n", pipeline.className());
                    System.out.printf("  path:  %s%n", pipeline.codeSource());
                }
                if (verbose && !pipeline.metadata().isEmpty()) {
                    System.out.printf("  metadata: %s%n", pipeline.metadata());
                }
            }
        }
        System.out.printf("%n%d extension feature(s) discovered", displayFeatures.size());
        if (!scanPaths.isEmpty()) {
            System.out.printf(" (%d loaded, %d scanned)", loadedFeatures.size(), scannedFeatures.size());
        }
        System.out.println();
        if (!scanPaths.isEmpty()) {
            System.out.println("Scanned adapters are manifest entries only; install the jar or set GOLLEK_EXTENSION_PATH to load runtime services.");
        }
    }

    @Command(name = "kinds", aliases = { "templates", "scaffolds" }, mixinStandardHelpOptions = true,
            description = "List extension scaffold kinds and the service contract each kind generates")
    public static class Kinds implements Callable<Integer> {
        @Option(names = { "--json" }, description = "Print scaffold kinds as JSON")
        boolean json;

        @Override
        public Integer call() {
            List<ScaffoldKind> kinds = scaffoldKinds();
            if (json) {
                Map<String, Object> result = new LinkedHashMap<>();
                result.put("action", "kinds");
                result.put("count", kinds.size());
                result.put("kinds", kinds.stream().map(ScaffoldKind::toMap).toList());
                printObjectJson(result);
                return 0;
            }

            System.out.println("Extension Scaffold Kinds");
            System.out.println("------------------------");
            System.out.printf("%-14s %-15s %-18s %-18s %s%n",
                    "KIND", "SERVICE", "DEFAULT INPUTS", "DEFAULT OUTPUTS", "USE WHEN");
            System.out.println("-".repeat(112));
            for (ScaffoldKind kind : kinds) {
                System.out.printf("%-14s %-15s %-18s %-18s %s%n",
                        truncate(kind.kind(), 14),
                        truncate(kind.service(), 15),
                        truncate(String.join(",", kind.defaultInputs()), 18),
                        truncate(String.join(",", kind.defaultOutputs()), 18),
                        truncate(kind.summary(), 42));
            }
            System.out.println();
            System.out.println("Examples:");
            for (ScaffoldKind kind : kinds) {
                if (kind.exampleCommand() != null && !kind.exampleCommand().isBlank()) {
                    System.out.printf("  %-14s %s%n", kind.kind(), kind.exampleCommand());
                }
            }
            System.out.println();
            System.out.println("Tip: pipeline creates ModelPipeline, model-family creates ModelFamilyPlugin, and the other common kinds create FeatureAdapter.");
            return 0;
        }
    }

    @Command(name = "index", aliases = { "generate-index", "aot-index" }, mixinStandardHelpOptions = true,
            description = "Generate an extension inventory for launchers, CI, and GraalVM native-image builds")
    public static class Index implements Callable<Integer> {
        @Option(names = { "--dir", "--plugin-dir", "--path" }, split = ",",
                description = "Additional extension jar or directory to include in the index")
        List<String> featurePaths = new ArrayList<>();

        @Option(names = { "--strict" }, description = "Return non-zero for warnings as well as errors")
        boolean strict;

        @Option(names = { "--output", "-o" }, description = "Write the JSON index to a file; use '-' for stdout")
        String output;

        @Option(names = { "--json" }, description = "Print the JSON index to stdout")
        boolean json;

        @Override
        public Integer call() {
            try {
                FeatureDoctorReport report = inspectFeatureRoots(runtimeFeatureRoots(featurePaths));
                Map<String, Object> index = extensionIndex(report, strict);
                boolean stdoutOutput = output != null && "-".equals(output.trim());
                if (json || stdoutOutput) {
                    printObjectJson(index);
                }
                if (output != null && !output.isBlank() && !stdoutOutput) {
                    Path target = normalize(Path.of(output.trim()));
                    Path parent = target.getParent();
                    if (parent != null) {
                        Files.createDirectories(parent);
                    }
                    Files.writeString(
                            target,
                            objectJson(index) + System.lineSeparator(),
                            StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE,
                            StandardOpenOption.TRUNCATE_EXISTING);
                    if (!json) {
                        printIndexSummary(report, strict, target);
                    }
                } else if (!json && !stdoutOutput) {
                    printIndexSummary(report, strict, null);
                    System.out.println("Use --json to print the index or --output gollek-extensions.index.json to save it.");
                }
                return report.validationSummary(strict).exitCode();
            } catch (Exception e) {
                if (json) {
                    printErrorJson("index", e.getMessage());
                } else {
                    System.err.println("Failed to generate extension index: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "init", aliases = { "new" }, mixinStandardHelpOptions = true,
            description = "Scaffold an out-of-core extension project")
    public static class Init implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "<dir>", description = "Directory for the new extension project")
        String targetDir;

        @Option(names = { "--id" }, description = "Extension package id (default: directory name)")
        String featureId;

        @Option(names = { "--kind", "--adapter-kind", "--capability-kind" },
                description = "Capability kind: pipeline, model-family, training, optimization, quantization, backend, exporter, dataset, evaluator")
        String kind = FeatureAdapterKinds.PIPELINE;

        @Option(names = { "--pipeline-id" }, description = "Pipeline id selected by gollek run --pipeline")
        String pipelineId;

        @Option(names = { "--adapter-id", "--capability-id" }, description = "Adapter/capability id for non-pipeline extensions")
        String adapterId;

        @Option(names = { "--model-family-id", "--family-id" },
                description = "Model-family id for --kind model-family (default: --id or directory name)")
        String modelFamilyId;

        @Option(names = { "--model-type" }, split = ",",
                description = "Hugging Face config.model_type values claimed by --kind model-family")
        List<String> modelTypes = new ArrayList<>();

        @Option(names = { "--architecture", "--architecture-class" }, split = ",",
                description = "Hugging Face architecture class names claimed by --kind model-family")
        List<String> architectureClassNames = new ArrayList<>();

        @Option(names = { "--capability" }, split = ",",
                description = "Model-family capabilities such as causal_lm, tokenizer, gguf, onnx, vision")
        List<String> modelFamilyCapabilities = new ArrayList<>();

        @Option(names = { "--tokenizer-kind" },
                description = "Tokenizer descriptor for --kind model-family: hf-bpe, sentencepiece, wordpiece, custom, none")
        String tokenizerKind;

        @Option(names = { "--tokenizer-metadata-status" },
                description = "Tokenizer metadata status for --kind model-family: ready or pending")
        String tokenizerMetadataStatus;

        @Option(names = { "--tokenizer-pending-reason", "--tokenizer-metadata-pending-reason" },
                description = "Reason used when tokenizer metadata status is pending")
        String tokenizerMetadataPendingReason;

        @Option(names = { "--bundle-profile" },
                description = "Model-family bundle profile metadata: core, direct, vlm, embedding, research, optional")
        String bundleProfile;

        @Option(names = { "--name" }, description = "Human-readable feature name")
        String displayName;

        @Option(names = { "--package" }, description = "Java package for the pipeline class")
        String packageName;

        @Option(names = { "--class" }, description = "Java pipeline or adapter class name")
        String className;

        @Option(names = { "--family" }, description = "Capability family, such as ocr, tts, asr, vision, audio, eval, training")
        String family;

        @Option(names = { "--input" }, split = ",", description = "Pipeline modalities or adapter inputs")
        List<String> inputModalities = new ArrayList<>();

        @Option(names = { "--output" }, split = ",", description = "Pipeline modalities or adapter outputs")
        List<String> outputModalities = new ArrayList<>();

        @Option(names = { "--target" }, split = ",", description = "Adapter target list, such as llm, vision, audio, edge")
        List<String> targets = new ArrayList<>();

        @Option(names = { "--tag" }, split = ",", description = "Descriptor tags")
        List<String> tags = new ArrayList<>();

        @Option(names = { "--provider" }, description = "Optional provider hint for the generated descriptor, such as onnx or litert")
        String providerHint;

        @Option(names = { "--format" }, description = "Optional model format hint for the generated descriptor, such as onnx, litert, safetensors, or gguf")
        String formatHint;

        @Option(names = { "--gollek-path" }, description = "Optional local Gollek checkout path to include as a Gradle composite build")
        String gollekPath;

        @Option(names = { "--force" }, description = "Overwrite files that already exist")
        boolean force;

        @Option(names = { "--dry-run" }, description = "Show files that would be written without creating them")
        boolean dryRun;

        @Option(names = { "--json" }, description = "Print scaffold result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                Path root = normalize(Path.of(targetDir));
                String id = normalizeFeatureId(firstNonBlank(featureId, fileStem(root), "gollek-extension"));
                String normalizedKind = FeatureAdapterKinds.normalize(kind);
                boolean modelFamilyScaffold = MODEL_FAMILY_KIND.equals(normalizedKind);
                boolean pipelineScaffold = !modelFamilyScaffold && FeatureAdapterKinds.PIPELINE.equals(normalizedKind);
                String capabilityId = normalizeFeatureId(firstNonBlank(
                        modelFamilyScaffold ? modelFamilyId : null,
                        pipelineScaffold ? pipelineId : adapterId,
                        adapterId,
                        pipelineId,
                        id));
                String pipe = pipelineScaffold ? capabilityId : "";
                String name = firstNonBlank(displayName, titleFromId(id));
                String pkg = normalizePackageName(firstNonBlank(packageName, defaultPackageName(id)));
                String typeName = normalizeClassName(firstNonBlank(
                        className,
                        pascalCase(capabilityId)
                                + (modelFamilyScaffold ? "ModelFamily" : pipelineScaffold ? "Pipeline" : "Adapter")));
                String normalizedFamily = normalizeFeatureId(firstNonBlank(
                        family,
                        modelFamilyScaffold ? capabilityId : null,
                        pipelineScaffold ? "custom" : normalizedKind));
                List<ModalityType> inputs = pipelineScaffold ? parseModalities(inputModalities, "input") : List.of();
                List<ModalityType> outputs = pipelineScaffold ? parseModalities(outputModalities, "output") : List.of();
                List<String> adapterInputs = pipelineScaffold
                        ? List.of()
                        : normalizeScaffoldTokens(inputModalities, defaultAdapterInputs(normalizedKind));
                List<String> adapterOutputs = pipelineScaffold
                        ? List.of()
                        : normalizeScaffoldTokens(outputModalities, defaultAdapterOutputs(normalizedKind));
                List<String> normalizedFormats = normalizeScaffoldTokens(formatHint, List.of());
                List<String> normalizedTargets = normalizeScaffoldTokens(targets, List.of());
                List<String> normalizedTags = normalizeTags(tags, normalizedFamily, providerHint, formatHint, normalizedKind);
                List<String> normalizedModelTypes = normalizeScaffoldTokens(modelTypes, List.of(capabilityId));
                List<String> normalizedArchitectureClasses = normalizeCaseSensitiveScaffoldTokens(
                        architectureClassNames,
                        List.of(pascalCase(capabilityId) + "ForCausalLM"));
                List<String> normalizedModelFamilyCapabilities =
                        normalizeModelFamilyCapabilities(modelFamilyCapabilities, modelFamilyScaffold);
                boolean tokenizerKindNone = tokenizerKindNone(tokenizerKind);
                String normalizedTokenizerMetadataStatus = modelFamilyScaffold && tokenizerKindNone
                        && tokenizerMetadataStatus == null
                                ? "pending"
                                : normalizeTokenizerMetadataStatus(tokenizerMetadataStatus);
                if ("ready".equals(normalizedTokenizerMetadataStatus) && tokenizerKindNone) {
                    throw new IllegalArgumentException(
                            "Use --tokenizer-metadata-status pending instead of --tokenizer-kind none");
                }
                if ("pending".equals(normalizedTokenizerMetadataStatus)
                        && tokenizerKind != null
                        && !tokenizerKind.isBlank()
                        && !tokenizerKindNone) {
                    throw new IllegalArgumentException(
                            "--tokenizer-kind cannot be combined with --tokenizer-metadata-status pending");
                }
                if ("ready".equals(normalizedTokenizerMetadataStatus)
                        && tokenizerMetadataPendingReason != null
                        && !tokenizerMetadataPendingReason.isBlank()) {
                    throw new IllegalArgumentException(
                            "--tokenizer-pending-reason requires --tokenizer-metadata-status pending");
                }
                String normalizedTokenizerKind = "pending".equals(normalizedTokenizerMetadataStatus)
                        ? ""
                        : normalizeTokenizerKind(tokenizerKind);
                String normalizedTokenizerMetadataPendingReason = "pending".equals(normalizedTokenizerMetadataStatus)
                        ? firstNonBlank(tokenizerMetadataPendingReason, "tokenizer adapter pending scaffold stabilization")
                        : "";
                String normalizedBundleProfile = normalizeFeatureId(firstNonBlank(bundleProfile, "optional"));
                Path localGollek = gollekPath == null || gollekPath.isBlank()
                        ? null
                        : normalize(Path.of(gollekPath.trim()));

                FeatureScaffold scaffold = new FeatureScaffold(
                        root,
                        id,
                        normalizedKind,
                        capabilityId,
                        pipe,
                        name,
                        pkg,
                        typeName,
                        normalizedFamily,
                        inputs,
                        outputs,
                        adapterInputs,
                        adapterOutputs,
                        normalizedTags,
                        normalizeHint(providerHint),
                        normalizedFormats,
                        normalizedTargets,
                        normalizedModelTypes,
                        normalizedArchitectureClasses,
                        normalizedModelFamilyCapabilities,
                        normalizedTokenizerKind,
                        normalizedTokenizerMetadataStatus,
                        normalizedTokenizerMetadataPendingReason,
                        normalizedBundleProfile,
                        localGollek);
                List<Map<String, Object>> files = writeScaffold(scaffold, force, dryRun);

                Map<String, Object> result = new LinkedHashMap<>();
                result.put("action", "init");
                result.put("dry_run", dryRun);
                result.put("root", root.toString());
                result.put("feature_id", id);
                result.put("kind", normalizedKind);
                result.put("capability_id", capabilityId);
                if (modelFamilyScaffold) {
                    result.put("model_family_id", capabilityId);
                    result.put("model_types", normalizedModelTypes);
                    result.put("architecture_classes", normalizedArchitectureClasses);
                    result.put("model_family_capabilities", normalizedModelFamilyCapabilities);
                    result.put("tokenizer_metadata_status", normalizedTokenizerMetadataStatus);
                    if ("ready".equals(normalizedTokenizerMetadataStatus)) {
                        result.put("tokenizer_kind", normalizedTokenizerKind);
                        result.put("tokenizer_kinds", List.of(normalizedTokenizerKind));
                    } else {
                        result.put("tokenizer_metadata_pending_reason", normalizedTokenizerMetadataPendingReason);
                    }
                }
                if (pipelineScaffold) {
                    result.put("pipeline_id", pipe);
                }
                result.put("class", pkg + "." + typeName);
                result.put("files", files);
                if (json) {
                    printObjectJson(result);
                } else {
                    System.out.printf("%s extension project: %s%n", dryRun ? "Would scaffold" : "Scaffolded", root);
                    System.out.printf("extension: %s (%s)%n", id, name);
                    System.out.printf("kind:      %s%n", normalizedKind);
                    System.out.printf("capability: %s%n", capabilityId);
                    if (modelFamilyScaffold) {
                        System.out.printf("model family: %s%n", capabilityId);
                        System.out.printf("model types:  %s%n", String.join(", ", normalizedModelTypes));
                        System.out.printf("tokenizer:    %s%n", "ready".equals(normalizedTokenizerMetadataStatus)
                                ? normalizedTokenizerKind
                                : normalizedTokenizerMetadataStatus + " (" + normalizedTokenizerMetadataPendingReason + ")");
                    }
                    if (pipelineScaffold) {
                        System.out.printf("pipeline:  %s%n", pipe);
                    }
                    System.out.printf("class:    %s.%s%n", pkg, typeName);
                    for (Map<String, Object> file : files) {
                        System.out.printf("  %s %s%n", file.get("status"), file.get("path"));
                    }
                    System.out.println();
                    System.out.println("Next:");
                    System.out.println("  " + scaffoldBuildCommand(scaffold));
                    if (modelFamilyScaffold) {
                        System.out.printf("  gollek extensions inspect lib/build/libs/%s-0.1.0-SNAPSHOT.jar --strict%n", id);
                        System.out.printf("  gollek extensions install lib/build/libs/%s-0.1.0-SNAPSHOT.jar --plugin-dir ~/.gollek/plugins --force%n", id);
                        System.out.println("  GOLLEK_PLUGIN_DIRS=~/.gollek/plugins gollek modules --doctor");
                    } else {
                        System.out.printf("  gollek extensions inspect lib/build/libs/%s-0.1.0-SNAPSHOT.jar --strict%n", id);
                        System.out.printf("  gollek extensions install lib/build/libs/%s-0.1.0-SNAPSHOT.jar --force%n", id);
                        System.out.println("  gollek extensions doctor --strict");
                    }
                    if (pipelineScaffold) {
                        System.out.printf("  gollek run --model <model> --pipeline %s --prompt \"hello\"%n", pipe);
                    } else if (modelFamilyScaffold) {
                        System.out.printf("  gollek show <model-or-path> --plugin-dir ~/.gollek/plugins --json%n");
                    } else {
                        System.out.println("  gollek extensions --details --paths");
                    }
                }
                return 0;
            } catch (Exception e) {
                if (json) {
                    printErrorJson("init", e.getMessage());
                } else {
                    System.err.println("Failed to scaffold feature project: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "inspect", aliases = { "validate" }, mixinStandardHelpOptions = true,
            description = "Inspect and validate an extension jar or package before installing it")
    public static class Inspect implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "<jar-or-package>", description = "Extension jar or package zip to inspect")
        String jarPath;

        @Option(names = { "--strict" }, description = "Return non-zero for warnings as well as errors")
        boolean strict;

        @Option(names = { "--json" }, description = "Print inspection result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                Path source = normalize(Path.of(jarPath));
                if (!Files.isRegularFile(source)) {
                    throw new IllegalArgumentException("Extension artifact not found: " + source);
                }
                try (FeatureInstallSource input = resolveExtensionInstallSource(source)) {
                    FeatureJarReport inspected = inspectJar(input.jarPath()).withPath(input.displayJarPath());
                    FeatureJarReport report = input.packageReport() == null
                            ? applyInstallLock(source.getParent(), inspected, readInstallLock(source.getParent()))
                            : inspected;
                    FeatureJarValidation validation = validateArtifact(report, input.packageReport(), strict);
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("action", "inspect");
                    result.put("path", source.toString());
                    result.put("strict", strict);
                    result.put("status", validation.status());
                    result.put("valid", validation.valid());
                    result.put("errors", validation.errors());
                    result.put("warnings", validation.warnings());
                    result.put("runtime", runtimeCompatibilityContext().toMap());
                    if (input.packageReport() != null) {
                        result.put("package", input.packageReport().toMap());
                    }
                    result.put("jar", report.toMap());

                    if (json) {
                        printObjectJson(result);
                    } else {
                        if (input.packageReport() != null) {
                            printPackageInspect(input.packageReport());
                            System.out.println();
                        }
                        printInspect(report, validation);
                    }
                    return validation.valid() ? 0 : 1;
                }
            } catch (Exception e) {
                if (json) {
                    printErrorJson("inspect", e.getMessage());
                } else {
                    System.err.println("Failed to inspect extension artifact: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "doctor", mixinStandardHelpOptions = true,
            description = "Validate runtime extension paths and return a CI-friendly status code")
    public static class Doctor implements Callable<Integer> {
        @Option(names = { "--path", "--dir", "--plugin-dir" }, split = ",",
                description = "Additional extension jar or directory path to inspect")
        List<String> featurePaths = new ArrayList<>();

        @Option(names = { "--strict" }, description = "Return non-zero for warnings as well as errors")
        boolean strict;

        @Option(names = { "--json" }, description = "Print doctor report as JSON")
        boolean json;

        @Option(names = { "--repair" }, description = "Refresh feature install locks and remove stale lock entries")
        boolean repair;

        @Option(names = { "--dry-run" }, description = "Preview --repair changes without writing lockfiles")
        boolean dryRun;

        @Override
        public Integer call() {
            try {
                if (dryRun && !repair) {
                    throw new IllegalArgumentException("--dry-run is only valid with --repair");
                }
                List<Path> roots = runtimeFeatureRoots(featurePaths);
                FeatureDoctorReport report = inspectFeatureRoots(roots);
                FeatureLegacyPruneReport legacyMigration = inspectLegacyMigrationStatus();
                FeatureRepairReport repairReport = null;
                if (repair) {
                    repairReport = repairInstallLocks(roots, dryRun);
                    if (!dryRun) {
                        report = inspectFeatureRoots(roots);
                    }
                }
                FeatureDoctorSummary summary = report.validationSummary(strict);
                if (json) {
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("action", "doctor");
                    result.put("strict", strict);
                    result.put("repair_requested", repair);
                    result.put("dry_run", dryRun);
                    if (repairReport != null) {
                        result.put("repair", repairReport.toMap());
                    }
                    result.put("doctor", report.toMap(strict));
                    result.put("legacy_migration", legacyMigration.toStatusMap());
                    printObjectJson(result);
                } else {
                    if (repairReport != null) {
                        printRepair(repairReport);
                        System.out.println();
                    }
                    printDoctor(report, strict);
                    printLegacyMigrationStatus(legacyMigration);
                    System.out.printf("%nSummary: %d root(s), %d jar(s), %d ok, %d warning, %d error, %d collision(s), %d ignored legacy path setting(s), %d plugin-ready, %d plugin-not-ready, tokenizer-lock %d ok/%d missing/%d stale%n",
                            summary.roots(),
                            summary.jars(),
                            summary.ok(),
                            summary.warning(),
                            summary.error(),
                            summary.collisions(),
                            summary.ignoredLegacyPaths(),
                            summary.pluginInstallReady(),
                            summary.pluginInstallNotReady(),
                            summary.tokenizerLockOk(),
                            summary.tokenizerLockMissing(),
                            summary.tokenizerLockStale());
                }
                if (repairReport != null && repairReport.hasErrors()) {
                    return 1;
                }
                return summary.exitCode();
            } catch (Exception e) {
                if (json) {
                    printErrorJson("doctor", e.getMessage());
                } else {
                    System.err.println("Failed to inspect extension paths: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "install", mixinStandardHelpOptions = true,
            description = "Install an extension jar or package into Gollek's runtime extension directory")
    public static class Install implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "<jar-or-package>", description = "Extension jar or package zip to install")
        String jarPath;

        @Option(names = { "--dir" }, description = "Extension directory to install into (default: ~/.gollek/extensions)")
        String outputDir;

        @Option(names = { "--plugin-dir" },
                description = "Plugin directory to install model-family plugins into, such as ~/.gollek/plugins")
        String pluginDir;

        @Option(names = { "--as" }, description = "Installed jar filename (default: source filename)")
        String installedName;

        @Option(names = { "--force" }, description = "Replace an existing installed jar")
        boolean force;

        @Option(names = { "--dry-run" }, description = "Validate and print the destination without copying")
        boolean dryRun;

        @Option(names = { "--allow-no-service" }, description = "Allow jars without a ModelPipeline service entry")
        boolean allowNoService;

        @Option(names = { "--no-backup" }, description = "Do not save a rollback backup when replacing an existing jar")
        boolean noBackup;

        @Option(names = { "--json" }, description = "Print install result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                Path source = normalize(Path.of(jarPath));
                if (!Files.isRegularFile(source)) {
                    throw new IllegalArgumentException("Extension artifact not found: " + source);
                }
                if (pluginDir != null && !pluginDir.isBlank() && outputDir != null && !outputDir.isBlank()) {
                    throw new IllegalArgumentException("Use either --dir for extension installs or --plugin-dir for plugin installs, not both");
                }
                try (FeatureInstallSource input = resolveExtensionInstallSource(source)) {
                    FeatureJarReport report = inspectJar(input.jarPath()).withPath(input.displayJarPath());
                    if (report.error() != null) {
                        throw new IllegalArgumentException("Extension jar is not readable: " + report.error());
                    }
                    boolean pluginInstall = pluginInstallRequested(pluginDir, outputDir, report);
                    FeatureJarValidation validation = validateArtifact(report, input.packageReport(), false);
                    List<String> installErrors = new ArrayList<>(validation.errors());
                    if (input.packageReport() != null) {
                        installErrors.addAll(packageJarConsistencyWarnings(input.packageReport(), report));
                    }
                    if (allowNoService) {
                        installErrors.removeIf(error -> error.contains(SERVICE_ENTRY)
                                || error.contains(ADAPTER_SERVICE_ENTRY)
                                || error.contains(MODEL_FAMILY_SERVICE_ENTRY)
                                || error.contains(GOLLEK_PLUGIN_SERVICE_ENTRY));
                    }
                    if (pluginInstall && modelFamilyOnly(report)) {
                        installErrors.addAll(modelFamilyPluginInstallReadinessErrors(report));
                    }
                    if (!installErrors.isEmpty()) {
                        throw new IllegalArgumentException(String.join("; ", installErrors));
                    }

                    Path targetDir = installTargetDirectory(outputDir, pluginDir, report);
                    String filename = installedName == null || installedName.isBlank()
                            ? input.defaultInstalledName()
                            : installedName.trim();
                    if (!Path.of(filename).getFileName().toString().equals(filename)) {
                        throw new IllegalArgumentException("--as expects a filename, not a path: " + filename);
                    }
                    if (!filename.endsWith(".jar")) {
                        filename = filename + ".jar";
                    }
                    Path target = normalize(targetDir.resolve(filename));
                    ensureInside(targetDir, target);

                    boolean existed = Files.exists(target);
                    if (existed && !force) {
                        throw new IllegalArgumentException("Extension jar already exists: " + target
                                + " (pass --force to replace)");
                    }

                    FeatureBackupRecord backup = null;
                    boolean wouldBackup = existed && force && !noBackup;
                    Map<String, Object> lockEntry = null;
                    Map<String, Object> plannedLockEntry = null;
                    if (dryRun) {
                        plannedLockEntry = installLockEntry(target, input.jarPath(), report, input.packageReport());
                    }
                    if (!dryRun) {
                        Files.createDirectories(targetDir);
                        if (wouldBackup) {
                            backup = createFeatureBackup(targetDir, target, "install-replace");
                        }
                        if (force) {
                            Files.copy(input.jarPath(), target, StandardCopyOption.REPLACE_EXISTING);
                        } else {
                            Files.copy(input.jarPath(), target);
                        }
                        lockEntry = writeInstallLockEntry(targetDir, target, inspectJar(target), input.packageReport());
                    }

                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("action", "install");
                    result.put("install_mode", pluginInstall ? "plugin" : "extension");
                    result.put("source", source.toString());
                    result.put("target", target.toString());
                    if (pluginInstall) {
                        result.put("plugin_dir", targetDir.toString());
                    }
                    result.put("lockfile", installLockPath(targetDir).toString());
                    if (input.packageReport() != null) {
                        result.put("package", input.packageReport().toMap());
                    }
                    result.put("locked", !dryRun);
                    result.put("dry_run", dryRun);
                    result.put("replaced", existed && !dryRun);
                    result.put("would_replace", existed && dryRun && force);
                    result.put("backup_created", backup != null);
                    result.put("would_backup", dryRun && wouldBackup);
                    if (backup != null) {
                        result.put("backup", backup.toMap());
                    }
                    if (lockEntry != null) {
                        result.put("lock_entry", lockEntry);
                    }
                    if (plannedLockEntry != null) {
                        result.put("would_lock_entry", plannedLockEntry);
                    }
                    result.put("jar", report.toMap());
                    if (json) {
                        printObjectJson(result);
                    } else {
                        if (input.packageReport() != null) {
                            printPackageInspect(input.packageReport());
                            System.out.println();
                        }
                        System.out.printf("%s %s jar: %s%n",
                                dryRun ? "Would install" : "Installed",
                                pluginInstall ? "plugin" : "extension",
                                target);
                        if (backup != null) {
                            System.out.printf("backup: %s%n", backup.path());
                        } else if (dryRun && wouldBackup) {
                            System.out.println("backup: would create rollback backup before replacing existing jar");
                        }
                        System.out.printf("%s install lock: %s%n", dryRun ? "Would update" : "Updated", installLockPath(targetDir));
                        Map<String, Object> effectiveLockEntry = lockEntry == null ? plannedLockEntry : lockEntry;
                        List<String> tokenizerKinds = effectiveLockEntry == null
                                ? List.of()
                                : manifestStringList(effectiveLockEntry.get("tokenizer_kinds"));
                        String tokenizerMetadata = tokenizerMetadataLockSummary(effectiveLockEntry);
                        if (!tokenizerMetadata.isBlank()) {
                            System.out.printf("tokenizer metadata: %s%n", tokenizerMetadata);
                        }
                        if (!tokenizerKinds.isEmpty()) {
                            System.out.printf("tokenizer kinds: %s%n", String.join(", ", tokenizerKinds));
                        }
                        if (report.hasManifest()) {
                            System.out.printf("manifest: %s%n", manifestSummary(report.manifest()));
                        }
                        for (String provider : report.providers()) {
                            System.out.printf("provider: %s%n", provider);
                        }
                        for (String provider : report.adapterProviders()) {
                            System.out.printf("adapter provider: %s%n", provider);
                        }
                        for (String provider : report.modelFamilyProviders()) {
                            System.out.printf("model-family provider: %s%n", provider);
                        }
                        if (!report.hasServiceEntry() && !modelFamilyOnly(report)) {
                            System.out.println("warning: no ModelPipeline service entry found");
                        }
                        if (pluginInstall) {
                            System.out.printf("next: GOLLEK_PLUGIN_DIRS=%s gollek modules --doctor%n", targetDir);
                            System.out.printf("next: gollek show <model-or-path> --plugin-dir %s --json%n", targetDir);
                        } else if (modelFamilyOnly(report)) {
                            System.out.println("warning: model-family-only jars are loaded from plugin directories; pass --plugin-dir ~/.gollek/plugins for runtime use");
                        }
                    }
                    return 0;
                }
            } catch (Exception e) {
                if (json) {
                    printErrorJson("install", e.getMessage());
                } else {
                    System.err.println("Failed to install extension artifact: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "migrate", aliases = { "migration" }, mixinStandardHelpOptions = true,
            description = "Copy legacy ~/.gollek/features jars into the runtime extension directory")
    public static class Migrate implements Callable<Integer> {
        @Option(names = { "--from" }, description = "Legacy extension directory to migrate from (default: ~/.gollek/features)")
        String sourceDir;

        @Option(names = { "--to", "--dir" }, description = "Runtime extension directory to copy into (default: ~/.gollek/extensions)")
        String targetDir;

        @Option(names = { "--force" }, description = "Replace conflicting target jars after creating rollback backups")
        boolean force;

        @Option(names = { "--dry-run" }, description = "Preview migration without copying jars or updating locks")
        boolean dryRun;

        @Option(names = { "--no-backup" }, description = "Do not save rollback backups when --force replaces an existing jar")
        boolean noBackup;

        @Option(names = { "--json" }, description = "Print migration result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                Path source = legacyInstallDirectory(sourceDir);
                Path target = installDirectory(targetDir);
                FeatureMigrationReport report = migrateLegacyExtensions(source, target, force, dryRun, noBackup);
                if (json) {
                    printObjectJson(report.toMap());
                } else {
                    printMigration(report);
                }
                return report.hasErrors() ? 1 : 0;
            } catch (Exception e) {
                if (json) {
                    printErrorJson("migrate", e.getMessage());
                } else {
                    System.err.println("Failed to migrate extension jars: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "prune-legacy", aliases = { "cleanup-legacy", "prune-migrated" }, mixinStandardHelpOptions = true,
            description = "Archive legacy ~/.gollek/features jars that are already represented by runtime extensions")
    public static class PruneLegacy implements Callable<Integer> {
        @Parameters(index = "0", arity = "0..1", paramLabel = "<target>",
                description = "Optional jar filename, extension id, feature id, pipeline id, or provider class to prune")
        String target;

        @Option(names = { "--from" }, description = "Legacy extension directory to prune from (default: ~/.gollek/features)")
        String sourceDir;

        @Option(names = { "--to", "--dir" }, description = "Runtime extension directory to compare against (default: ~/.gollek/extensions)")
        String targetDir;

        @Option(names = { "--dry-run" }, description = "Preview legacy archive/delete operations without changing files")
        boolean dryRun;

        @Option(names = { "--json" }, description = "Print prune result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                Path source = legacyInstallDirectory(sourceDir);
                Path preferred = installDirectory(targetDir);
                FeatureLegacyPruneReport report = pruneLegacyExtensions(source, preferred, target, dryRun);
                if (json) {
                    printObjectJson(report.toMap());
                } else {
                    printLegacyPrune(report);
                }
                return report.hasErrors() ? 1 : 0;
            } catch (Exception e) {
                if (json) {
                    printErrorJson("prune-legacy", e.getMessage());
                } else {
                    System.err.println("Failed to prune legacy extension jars: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "finalize-migration", aliases = { "finish-migration", "finalise-migration" }, mixinStandardHelpOptions = true,
            description = "Finalize extension migration by archiving legacy jars already represented in ~/.gollek/extensions")
    public static class FinalizeMigration implements Callable<Integer> {
        @Parameters(index = "0", arity = "0..1", paramLabel = "<target>",
                description = "Optional jar filename, extension id, feature id, pipeline id, or provider class to finalize")
        String target;

        @Option(names = { "--from" }, description = "Legacy extension directory to finalize from (default: ~/.gollek/features)")
        String sourceDir;

        @Option(names = { "--to", "--dir" }, description = "Runtime extension directory to compare against (default: ~/.gollek/extensions)")
        String targetDir;

        @Option(names = { "--dry-run" }, description = "Preview legacy archive/delete operations without changing files")
        boolean dryRun;

        @Option(names = { "--json" }, description = "Print finalize result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                Path source = legacyInstallDirectory(sourceDir);
                Path preferred = installDirectory(targetDir);
                FeatureLegacyPruneReport report = pruneLegacyExtensions(
                        source,
                        preferred,
                        target,
                        dryRun,
                        "finalize-migration");
                if (json) {
                    printObjectJson(report.toMap("finalize-migration"));
                } else {
                    printFinalizeMigration(report);
                }
                return report.hasErrors() ? 1 : 0;
            } catch (Exception e) {
                if (json) {
                    printErrorJson("finalize-migration", e.getMessage());
                } else {
                    System.err.println("Failed to finalize extension migration: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "remove", aliases = { "uninstall" }, mixinStandardHelpOptions = true,
            description = "Remove an installed runtime extension or detachable plugin jar")
    public static class Remove implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "<target>",
                description = "Jar filename, extension/model-family id, feature id, pipeline id, or provider class")
        String target;

        @Option(names = { "--dir" }, description = "Extension directory to remove from (default: ~/.gollek/extensions)")
        String featureDir;

        @Option(names = { "--plugin-dir" },
                description = "Plugin directory to remove model-family plugins from, such as ~/.gollek/plugins")
        String pluginDir;

        @Option(names = { "--dry-run" }, description = "Print matched jar without deleting it")
        boolean dryRun;

        @Option(names = { "--no-backup" }, description = "Do not save a rollback backup before deleting the jar")
        boolean noBackup;

        @Option(names = { "--json" }, description = "Print remove result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                boolean explicitPluginDir = pluginDir != null && !pluginDir.isBlank();
                boolean explicitFeatureDir = featureDir != null && !featureDir.isBlank();
                if (explicitPluginDir && explicitFeatureDir) {
                    throw new IllegalArgumentException("Use either --dir for extension removals or --plugin-dir for plugin removals, not both");
                }
                boolean pluginRemove = explicitPluginDir || (!explicitFeatureDir && defaultPluginRemove());
                Path root = pluginRemove ? pluginDirectory(pluginDir) : installDirectory(featureDir);
                if (!Files.isDirectory(root)) {
                    throw new IllegalArgumentException((pluginRemove ? "Plugin" : "Extension")
                            + " directory does not exist: " + root);
                }

                List<FeatureJarReport> matches = installedJars(root).stream()
                        .map(FeaturesCommand::inspectJar)
                        .filter(report -> matchesTarget(root, report, target))
                        .toList();

                if (matches.isEmpty()) {
                    throw new IllegalArgumentException("No installed "
                            + (pluginRemove ? "plugin" : "extension")
                            + " jar matched: " + target);
                }
                if (matches.size() > 1) {
                    String candidates = matches.stream()
                            .map(FeatureJarReport::path)
                            .reduce((left, right) -> left + ", " + right)
                            .orElse("");
                    throw new IllegalArgumentException("Multiple "
                            + (pluginRemove ? "plugin" : "extension")
                            + " jars matched; use an exact filename: " + candidates);
                }

                FeatureJarReport match = matches.get(0);
                Path jar = normalize(Path.of(match.path()));
                ensureInside(root, jar);
                FeatureBackupRecord backup = null;
                if (!dryRun) {
                    if (!noBackup) {
                        backup = createFeatureBackup(root, jar, "remove");
                    }
                    Files.delete(jar);
                    removeInstallLockEntry(root, jar);
                }

                Map<String, Object> result = new LinkedHashMap<>();
                result.put("action", actionName());
                result.put("remove_mode", pluginRemove ? "plugin" : "extension");
                result.put("target", target);
                result.put("removed", !dryRun);
                result.put("dry_run", dryRun);
                if (pluginRemove) {
                    result.put("plugin_dir", root.toString());
                }
                result.put("lockfile", installLockPath(root).toString());
                result.put("backup_created", backup != null);
                result.put("would_backup", dryRun && !noBackup);
                if (backup != null) {
                    result.put("backup", backup.toMap());
                }
                result.put("jar", match.toMap());
                if (json) {
                    printObjectJson(result);
                } else {
                    System.out.printf("%s %s jar: %s%n",
                            dryRun ? "Would remove" : "Removed",
                            pluginRemove ? "plugin" : "extension",
                            jar);
                    if (backup != null) {
                        System.out.printf("backup: %s%n", backup.path());
                    } else if (dryRun && !noBackup) {
                        System.out.println("backup: would create rollback backup before deleting jar");
                    }
                    if (match.hasManifest()) {
                        System.out.printf("manifest: %s%n", manifestSummary(match.manifest()));
                    }
                }
                return 0;
            } catch (Exception e) {
                if (json) {
                    printErrorJson(actionName(), e.getMessage());
                } else {
                    System.err.println("Failed to remove extension/plugin jar: " + e.getMessage());
                }
                return 1;
            }
        }

        protected boolean defaultPluginRemove() {
            return false;
        }

        protected String actionName() {
            return "remove";
        }
    }

    @Command(name = "detach", aliases = { "unplug" }, mixinStandardHelpOptions = true,
            description = "Detach an installed model-family plugin jar from Gollek's plugin directory")
    public static class Detach extends Remove {
        @Override
        protected boolean defaultPluginRemove() {
            return true;
        }

        @Override
        protected String actionName() {
            return "detach";
        }
    }

    @Command(name = "rollback", aliases = { "restore" }, mixinStandardHelpOptions = true,
            description = "Restore an installed extension or model-family plugin jar from its latest rollback backup")
    public static class Rollback implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "<target>",
                description = "Jar filename, extension id, feature id, pipeline id, provider class, or removed jar filename")
        String target;

        @Option(names = { "--dir" }, description = "Extension directory to restore into (default: ~/.gollek/extensions)")
        String featureDir;

        @Option(names = { "--plugin-dir" },
                description = "Plugin directory to restore model-family plugin jars into, such as ~/.gollek/plugins")
        String pluginDir;

        @Option(names = { "--backup" }, description = "Specific backup jar filename or path to restore (default: latest)")
        String backup;

        @Option(names = { "--dry-run" }, description = "Print the rollback without restoring the jar")
        boolean dryRun;

        @Option(names = { "--json" }, description = "Print rollback result as JSON")
        boolean json;

        @Override
        public Integer call() {
            try {
                boolean explicitPluginDir = pluginDir != null && !pluginDir.isBlank();
                boolean explicitFeatureDir = featureDir != null && !featureDir.isBlank();
                if (explicitPluginDir && explicitFeatureDir) {
                    throw new IllegalArgumentException("Use either --dir for extension rollback or --plugin-dir for plugin rollback, not both");
                }
                boolean pluginRollback = explicitPluginDir;
                Path root = pluginRollback ? pluginDirectory(pluginDir) : installDirectory(featureDir);
                if (!Files.isDirectory(root)) {
                    throw new IllegalArgumentException((pluginRollback ? "Plugin" : "Extension")
                            + " directory does not exist: " + root);
                }

                Path targetPath;
                String filename;
                Path backupPath;
                try {
                    targetPath = resolveRollbackTarget(root, target);
                    filename = targetPath.getFileName() == null ? targetPath.toString() : targetPath.getFileName().toString();
                    try {
                        backupPath = selectFeatureBackup(root, filename, backup);
                    } catch (IllegalArgumentException e) {
                        FeatureBackupSelection selection = selectFeatureBackupByMetadata(root, target, backup);
                        if (selection == null) {
                            throw e;
                        }
                        targetPath = selection.targetPath();
                        filename = targetPath.getFileName() == null ? targetPath.toString() : targetPath.getFileName().toString();
                        backupPath = selection.backupPath();
                    }
                } catch (IllegalArgumentException e) {
                    FeatureBackupSelection selection = selectFeatureBackupByMetadata(root, target, backup);
                    if (selection == null) {
                        throw e;
                    }
                    targetPath = selection.targetPath();
                    filename = targetPath.getFileName() == null ? targetPath.toString() : targetPath.getFileName().toString();
                    backupPath = selection.backupPath();
                }
                FeatureJarReport backupReport = inspectJar(backupPath);
                if (!isRollbackFilenameTarget(target, targetPath) && !matchesTarget(root, backupReport, target)) {
                    FeatureBackupSelection selection = selectFeatureBackupByMetadata(root, target, backup);
                    if (selection == null) {
                        throw new IllegalArgumentException("No rollback backup matched feature target: " + target);
                    }
                    targetPath = selection.targetPath();
                    filename = targetPath.getFileName() == null ? targetPath.toString() : targetPath.getFileName().toString();
                    backupPath = selection.backupPath();
                    backupReport = inspectJar(backupPath);
                }
                if (backupReport.error() != null && !backupReport.error().isBlank()) {
                    throw new IllegalArgumentException("Backup extension jar is not readable: " + backupReport.error());
                }

                FeatureBackupRecord currentBackup = null;
                if (!dryRun) {
                    Files.createDirectories(root);
                    if (Files.exists(targetPath)) {
                        currentBackup = createFeatureBackup(root, targetPath, "rollback-current");
                    }
                    Files.copy(backupPath, targetPath, StandardCopyOption.REPLACE_EXISTING);
                    writeInstallLockEntry(root, targetPath, inspectJar(targetPath));
                }

                Map<String, Object> result = new LinkedHashMap<>();
                result.put("action", "rollback");
                result.put("rollback_mode", pluginRollback ? "plugin" : "extension");
                result.put("target", target);
                result.put("target_path", targetPath.toString());
                result.put("source_backup", backupPath.toString());
                if (pluginRollback) {
                    result.put("plugin_dir", root.toString());
                }
                result.put("lockfile", installLockPath(root).toString());
                result.put("restored", !dryRun);
                result.put("dry_run", dryRun);
                result.put("current_backup_created", currentBackup != null);
                if (currentBackup != null) {
                    result.put("current_backup", currentBackup.toMap());
                }
                result.put("jar", backupReport.toMap());
                if (json) {
                    printObjectJson(result);
                } else {
                    System.out.printf("%s %s jar rollback: %s%n",
                            dryRun ? "Would restore" : "Restored",
                            pluginRollback ? "plugin" : "extension",
                            targetPath);
                    System.out.printf("source backup: %s%n", backupPath);
                    if (currentBackup != null) {
                        System.out.printf("current backup: %s%n", currentBackup.path());
                    } else if (dryRun && Files.exists(targetPath)) {
                        System.out.println("current backup: would save current jar before restore");
                    }
                    if (backupReport.hasManifest()) {
                        System.out.printf("manifest: %s%n", manifestSummary(backupReport.manifest()));
                    }
                }
                return 0;
            } catch (Exception e) {
                if (json) {
                    printErrorJson("rollback", e.getMessage());
                } else {
                    System.err.println("Failed to rollback extension/plugin jar: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    @Command(name = "backups", aliases = { "backup", "history" }, mixinStandardHelpOptions = true,
            description = "List and prune extension or model-family plugin rollback backups")
    public static class Backups implements Callable<Integer> {
        @Parameters(index = "0", arity = "0..1", paramLabel = "<target>",
                description = "Optional jar filename, extension id, feature id, pipeline id, or provider class to filter backups")
        String target;

        @Option(names = { "--dir" }, description = "Extension directory to inspect (default: ~/.gollek/extensions)")
        String featureDir;

        @Option(names = { "--plugin-dir" },
                description = "Plugin directory to inspect for model-family plugin backups, such as ~/.gollek/plugins")
        String pluginDir;

        @Option(names = { "--json" }, description = "Print backup report as JSON")
        boolean json;

        @Option(names = { "--prune" }, description = "Delete older backups after keeping the newest N per jar")
        boolean prune;

        @Option(names = { "--keep" }, description = "Number of newest backups to keep per jar when pruning (default: ${DEFAULT-VALUE})")
        int keep = 5;

        @Option(names = { "--dry-run" }, description = "Preview --prune deletions without deleting backup files")
        boolean dryRun;

        @Override
        public Integer call() {
            try {
                if (dryRun && !prune) {
                    throw new IllegalArgumentException("--dry-run is only valid with --prune");
                }
                if (keep < 0) {
                    throw new IllegalArgumentException("--keep must be >= 0");
                }
                boolean explicitPluginDir = pluginDir != null && !pluginDir.isBlank();
                boolean explicitFeatureDir = featureDir != null && !featureDir.isBlank();
                if (explicitPluginDir && explicitFeatureDir) {
                    throw new IllegalArgumentException("Use either --dir for extension backups or --plugin-dir for plugin backups, not both");
                }
                boolean pluginBackups = explicitPluginDir;
                Path root = pluginBackups ? pluginDirectory(pluginDir) : installDirectory(featureDir);
                List<FeatureBackupReport> backups = inspectFeatureBackups(root, target);
                FeatureBackupPruneReport pruneReport = null;
                if (prune) {
                    pruneReport = pruneFeatureBackups(backups, keep, dryRun);
                    if (!dryRun) {
                        backups = inspectFeatureBackups(root, target);
                    }
                }

                FeatureBackupSummary summary = FeatureBackupSummary.from(backups);
                if (json) {
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("action", "backups");
                    result.put("backup_mode", pluginBackups ? "plugin" : "extension");
                    result.put("root", root.toString());
                    if (pluginBackups) {
                        result.put("plugin_dir", root.toString());
                    }
                    result.put("target", target == null ? "" : target);
                    result.put("prune_requested", prune);
                    result.put("dry_run", dryRun);
                    result.put("summary", summary.toMap());
                    if (pruneReport != null) {
                        result.put("prune", pruneReport.toMap());
                    }
                    result.put("backups", backups.stream().map(FeatureBackupReport::toMap).toList());
                    printObjectJson(result);
                } else {
                    if (pruneReport != null) {
                        printBackupPrune(pruneReport);
                        System.out.println();
                    }
                    printBackups(root, target, backups, summary);
                }
                return pruneReport != null && pruneReport.hasErrors() ? 1 : 0;
            } catch (Exception e) {
                if (json) {
                    printErrorJson("backups", e.getMessage());
                } else {
                    System.err.println("Failed to inspect extension backups: " + e.getMessage());
                }
                return 1;
            }
        }
    }

    private static List<Map<String, Object>> writeScaffold(
            FeatureScaffold scaffold,
            boolean force,
            boolean dryRun) throws java.io.IOException {
        List<ScaffoldFile> files = new ArrayList<>();
        files.add(new ScaffoldFile("settings.gradle.kts", settingsGradle(scaffold)));
        files.add(new ScaffoldFile("gradle.properties", gradleProperties()));
        files.add(new ScaffoldFile("lib/build.gradle.kts", featureBuildGradle(scaffold)));
        files.add(new ScaffoldFile("lib/src/main/java/" + scaffold.packageName().replace('.', '/')
                + "/" + scaffold.className() + ".java",
                scaffold.modelFamily()
                        ? modelFamilyPluginJava(scaffold)
                        : scaffold.pipeline() ? featurePipelineJava(scaffold) : featureAdapterJava(scaffold)));
        files.add(new ScaffoldFile("lib/src/main/resources/"
                + serviceEntry(scaffold),
                scaffold.fqcn() + System.lineSeparator()));
        if (scaffold.modelFamily()) {
            files.add(new ScaffoldFile("lib/src/main/resources/"
                    + GOLLEK_PLUGIN_SERVICE_ENTRY,
                    scaffold.fqcn() + System.lineSeparator()));
            files.add(new ScaffoldFile("lib/src/main/resources/"
                    + PLUGIN_DESCRIPTOR_ENTRY,
                    modelFamilyPluginDescriptor(scaffold)));
        }
        files.add(new ScaffoldFile("lib/src/main/resources/" + MANIFEST_ENTRY, featureManifest(scaffold)));
        files.add(new ScaffoldFile("README.md", scaffoldReadme(scaffold)));

        List<Map<String, Object>> result = new ArrayList<>();
        for (ScaffoldFile file : files) {
            Path target = normalize(scaffold.root().resolve(file.relativePath()));
            ensureInside(scaffold.root(), target);
            boolean exists = Files.exists(target);
            String status = dryRun
                    ? exists ? "would-overwrite" : "would-create"
                    : exists ? "overwritten" : "created";
            if (!dryRun) {
                Files.createDirectories(target.getParent());
                if (force) {
                    Files.writeString(target, file.content(), StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
                } else {
                    Files.writeString(target, file.content(), StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE_NEW);
                }
            }
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("path", target.toString());
            item.put("status", status);
            result.add(item);
        }
        return result;
    }

    private static String settingsGradle(FeatureScaffold scaffold) {
        StringBuilder text = new StringBuilder();
        text.append("rootProject.name = \"").append(escapeGradle(scaffold.featureId())).append("\"\n");
        text.append("include(\":lib\")\n");
        if (scaffold.gollekPath() != null) {
            String relative = relativePath(scaffold.root(), scaffold.gollekPath());
            text.append("\n");
            text.append("val gollekPath = file(\"").append(escapeGradle(relative)).append("\")\n");
            text.append("if (gollekPath.exists()) {\n");
            text.append("    includeBuild(gollekPath)\n");
            text.append("}\n");
        } else {
            text.append("\n");
            text.append("// Pass --gollek-path when scaffolding to consume a local Gollek checkout as a composite build.\n");
            text.append("// Otherwise publish Gollek SPI to mavenLocal before building this feature.\n");
        }
        return text.toString();
    }

    private static String gradleProperties() {
        return """
                org.gradle.jvmargs=-Xmx2g -Dfile.encoding=UTF-8
                org.gradle.parallel=true
                """;
    }

    private static String featureBuildGradle(FeatureScaffold scaffold) {
        return """
                plugins {
                    `java-library`
                }

                group = "tech.kayys.gollek.extensions"
                version = "0.1.0-SNAPSHOT"

                repositories {
                    mavenCentral()
                    mavenLocal()
                }

                dependencies {
                    api("tech.kayys.gollek:gollek-spi:0.1.0-SNAPSHOT")
                    api("tech.kayys.gollek:gollek-spi-provider:0.1.0-SNAPSHOT")
                    api("tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT")
                    api("tech.kayys.gollek:gollek-spi-multimodal:0.1.0-SNAPSHOT")
                    api("tech.kayys.gollek:gollek-spi-model:0.1.0-SNAPSHOT")
                    implementation("io.smallrye.reactive:mutiny:2.5.5")
                    compileOnly("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")

                    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
                    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
                }

                java {
                    toolchain {
                        languageVersion = JavaLanguageVersion.of(25)
                    }
                }

                tasks.withType<JavaCompile>().configureEach {
                    options.encoding = "UTF-8"
                    options.compilerArgs.add("--enable-preview")
                    options.compilerArgs.add("--add-modules=jdk.incubator.vector")
                }

                tasks.withType<Test>().configureEach {
                    useJUnitPlatform()
                    jvmArgs("--enable-preview", "--add-modules=jdk.incubator.vector")
                }

                tasks.jar {
                    archiveBaseName = "%s"
                }
                """.formatted(scaffold.featureId());
    }

    private static String serviceEntry(FeatureScaffold scaffold) {
        if (scaffold.modelFamily()) {
            return MODEL_FAMILY_SERVICE_ENTRY;
        }
        return scaffold.pipeline() ? SERVICE_ENTRY : ADAPTER_SERVICE_ENTRY;
    }

    private static String modelFamilyPluginDescriptor(FeatureScaffold scaffold) {
        Map<String, Object> descriptor = new LinkedHashMap<>();
        descriptor.put("id", "model-family/" + scaffold.capabilityId());
        descriptor.put("name", scaffold.displayName());
        descriptor.put("version", "0.1.0-SNAPSHOT");
        descriptor.put("description", "Out-of-core Gollek model-family plugin for " + scaffold.capabilityId());
        descriptor.put("vendor", "External");
        descriptor.put("mainClass", scaffold.fqcn());
        descriptor.put("dependencies", List.of());
        descriptor.put("optionalDependencies", List.of());
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("extensionPoint", MODEL_FAMILY_KIND);
        properties.put("bundleProfile", scaffold.bundleProfile());
        properties.put("families", scaffold.modelTypes());
        if (scaffold.tokenizerMetadataReady()) {
            properties.put("tokenizerKind", scaffold.tokenizerKind());
            properties.put("tokenizerKinds", List.of(scaffold.tokenizerKind()));
        } else if (scaffold.tokenizerMetadataPending()) {
            properties.put("tokenizerMetadataStatus", "pending");
            properties.put("tokenizerMetadataPendingReason", scaffold.tokenizerMetadataPendingReason());
        }
        descriptor.put("properties", properties);
        try {
            return new ObjectMapper()
                    .writerWithDefaultPrettyPrinter()
                    .writeValueAsString(descriptor) + System.lineSeparator();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to write plugin descriptor JSON", e);
        }
    }

    private static String featurePipelineJava(FeatureScaffold scaffold) {
        String inputs = modalityJavaList(scaffold.pipelineInputs());
        String outputs = modalityJavaList(scaffold.pipelineOutputs());
        String tags = stringJavaList(scaffold.tags());
        String provider = scaffold.providerHint().isBlank() ? "" : scaffold.providerHint();
        String format = scaffold.formatHint().isBlank() ? "" : scaffold.formatHint();
        return """
                package %s;

                import io.smallrye.mutiny.Uni;
                import jakarta.enterprise.context.ApplicationScoped;
                import tech.kayys.gollek.spi.inference.InferenceResponse;
                                                import tech.kayys.gollek.spi.pipeline.ModelPipelineRequest;
                
                import java.time.Duration;
                import java.util.List;
                import java.util.Locale;
                import java.util.Map;

                @ApplicationScoped
                public class %s implements ModelPipeline {
                    public static final String PIPELINE_ID = "%s";

                    @Override
                    public String id() {
                        return PIPELINE_ID;
                    }

                    @Override
                    public PipelineDescriptor descriptor() {
                        return new PipelineDescriptor(
                                PIPELINE_ID,
                                "%s",
                                "0.1.0",
                                "%s",
                                %s,
                                %s,
                                %s,
                                Map.of(
                                        "source", "external-extension",
                                        "kind", "feature-pipeline",
                                        "provider", "%s",
                                        "format", "%s"));
                    }

                    @Override
                    public int priority() {
                        return 50;
                    }

                    @Override
                    public boolean supports(ModelPipelineRequest request) {
                        if (request == null) {
                            return false;
                        }
                        if (matches(request.parameterText("pipeline").orElse(null))
                                || matches(request.parameterText("feature").orElse(null))
                                || matches(request.factText("pipeline").orElse(null))
                                || matches(request.factText("pipeline_id").orElse(null))) {
                            return true;
                        }
                        return request.parameterBoolean("feature_pipeline") && providerMatches(request) && formatMatches(request);
                    }

                    @Override
                    public Uni<InferenceResponse> infer(ModelPipelineRequest request) {
                        return Uni.createFrom().item(() -> {
                            long started = System.nanoTime();
                            Map<String, Object> metadata = Map.of(
                                    "pipeline", PIPELINE_ID,
                                    "feature", "%s",
                                    "provider", request.providerId() == null ? "" : request.providerId(),
                                    "format", request.factText("format").orElse(""));
                            return InferenceResponse.builder()
                                    .requestId(request.request().getRequestId())
                                    .model(request.request().getModel())
                                    .content("Extension pipeline '%s' is scaffolded. Replace this response with real orchestration.")
                                    .durationMs(Duration.ofNanos(System.nanoTime() - started).toMillis())
                                    .metadata(metadata)
                                    .build();
                        });
                    }

                    private boolean providerMatches(ModelPipelineRequest request) {
                        String expected = "%s";
                        return expected.isBlank()
                                || expected.equalsIgnoreCase(String.valueOf(request.providerId()));
                    }

                    private boolean formatMatches(ModelPipelineRequest request) {
                        String expected = "%s";
                        return expected.isBlank()
                                || request.factText("format")
                                        .map(value -> expected.equalsIgnoreCase(value))
                                        .orElse(false);
                    }

                    private boolean matches(String value) {
                        return value != null
                                && PIPELINE_ID.equals(value.trim().toLowerCase(Locale.ROOT));
                    }
                }
                """.formatted(
                scaffold.packageName(),
                scaffold.className(),
                scaffold.pipelineId(),
                escapeJava(scaffold.displayName()),
                scaffold.family(),
                inputs,
                outputs,
                tags,
                escapeJava(provider),
                escapeJava(format),
                scaffold.featureId(),
                scaffold.pipelineId(),
                escapeJava(provider),
                escapeJava(format));
    }

    private static String featureAdapterJava(FeatureScaffold scaffold) {
        String inputs = stringJavaList(scaffold.inputNames());
        String outputs = stringJavaList(scaffold.outputNames());
        String formats = stringJavaList(scaffold.formatHints());
        String targets = stringJavaList(scaffold.targets());
        String tags = stringJavaList(scaffold.tags());
        String provider = scaffold.providerHint().isBlank() ? "" : scaffold.providerHint();
        String format = scaffold.formatHint().isBlank() ? "" : scaffold.formatHint();
        return """
                package %s;

                import jakarta.enterprise.context.ApplicationScoped;
                import tech.kayys.gollek.spi.feature.FeatureAdapter;
                import tech.kayys.gollek.spi.feature.FeatureAdapterDescriptor;

                import java.util.List;
                import java.util.Map;

                @ApplicationScoped
                public class %s implements FeatureAdapter {
                    public static final String ADAPTER_ID = "%s";

                    @Override
                    public String id() {
                        return ADAPTER_ID;
                    }

                    @Override
                    public FeatureAdapterDescriptor descriptor() {
                        return new FeatureAdapterDescriptor(
                                ADAPTER_ID,
                                "%s",
                                "%s",
                                "0.1.0",
                                "%s",
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                Map.of(
                                        "source", "external-extension",
                                        "provider", "%s",
                                        "format", "%s"));
                    }

                    @Override
                    public int priority() {
                        return 50;
                    }
                }
                """.formatted(
                scaffold.packageName(),
                scaffold.className(),
                scaffold.capabilityId(),
                scaffold.kind(),
                escapeJava(scaffold.displayName()),
                scaffold.family(),
                inputs,
                outputs,
                formats,
                targets,
                tags,
                escapeJava(provider),
                escapeJava(format));
    }

    private static String modelFamilyPluginJava(FeatureScaffold scaffold) {
        String capabilities = modelFamilyCapabilityJavaList(scaffold.modelFamilyCapabilities());
        String modelTypes = stringJavaList(scaffold.modelTypes());
        String architectureClasses = stringJavaList(scaffold.architectureClassNames());
        String tokenizerDescriptors = modelFamilyTokenizerDescriptorJava(scaffold);
        String metadata = modelFamilyMetadataJava(scaffold);
        return """
                package %s;

                                import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
                import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
                import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

                import java.util.List;
                import java.util.Map;

                public final class %s implements ModelFamilyPlugin {
                    public static final String FAMILY_ID = "%s";

                    @Override
                    public ModelFamilyDescriptor descriptor() {
                        return new ModelFamilyDescriptor(
                                FAMILY_ID,
                                "%s",
                                %s,
                                %s,
                                %s,
                                %s);
                    }

                    @Override
                    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                        return %s;
                    }
                }
                """.formatted(
                scaffold.packageName(),
                scaffold.className(),
                scaffold.capabilityId(),
                escapeJava(scaffold.displayName()),
                modelTypes,
                architectureClasses,
                capabilities,
                metadata,
                tokenizerDescriptors);
    }

    private static String featureManifest(FeatureScaffold scaffold) {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("schema_version", 1);
        manifest.put("id", scaffold.featureId());
        manifest.put("name", scaffold.displayName());
        manifest.put("version", "0.1.0-SNAPSHOT");
        manifest.put("description", scaffold.pipeline()
                ? "Out-of-core Gollek extension pipeline"
                : "Out-of-core Gollek extension adapter");
        manifest.put("source", "external-extension");
        manifest.put("requires", Map.of("gollek_spi", runtimeGollekVersion()));
        manifest.put("loading", Map.of(
                "jvm", "service-loader",
                "native_image", "ahead-of-time",
                "native_image_note", "include this extension jar on the native-image build classpath"));
        if (scaffold.modelFamily()) {
            manifest.put("services", Map.of("model_family_plugins", List.of(scaffold.fqcn())));
            Map<String, Object> family = new LinkedHashMap<>();
            family.put("id", scaffold.capabilityId());
            family.put("class", scaffold.fqcn());
            family.put("model_types", scaffold.modelTypes());
            family.put("architecture_classes", scaffold.architectureClassNames());
            family.put("capabilities", scaffold.modelFamilyCapabilities());
            family.put("tokenizer_metadata_status", scaffold.tokenizerMetadataStatus());
            if (scaffold.tokenizerMetadataReady()) {
                family.put("tokenizer_kind", scaffold.tokenizerKind());
                family.put("tokenizer_kinds", List.of(scaffold.tokenizerKind()));
            } else if (scaffold.tokenizerMetadataPending()) {
                family.put("tokenizer_metadata_pending_reason", scaffold.tokenizerMetadataPendingReason());
            }
            family.put("bundle_profile", scaffold.bundleProfile());
            manifest.put("model_families", List.of(family));
        } else if (scaffold.pipeline()) {
            manifest.put("services", Map.of("model_pipelines", List.of(scaffold.fqcn())));
            Map<String, Object> pipeline = new LinkedHashMap<>();
            pipeline.put("id", scaffold.pipelineId());
            pipeline.put("class", scaffold.fqcn());
            pipeline.put("family", scaffold.family());
            pipeline.put("inputs", scaffold.inputNames());
            pipeline.put("outputs", scaffold.outputNames());
            if (!scaffold.providerHint().isBlank()) {
                pipeline.put("provider", scaffold.providerHint());
            }
            if (!scaffold.formatHint().isBlank()) {
                pipeline.put("format", scaffold.formatHint());
            }
            manifest.put("pipelines", List.of(pipeline));
        } else {
            manifest.put("services", Map.of("feature_adapters", List.of(scaffold.fqcn())));
            Map<String, Object> capability = new LinkedHashMap<>();
            capability.put("id", scaffold.capabilityId());
            capability.put("kind", scaffold.kind());
            capability.put("class", scaffold.fqcn());
            capability.put("family", scaffold.family());
            capability.put("inputs", scaffold.inputNames());
            capability.put("outputs", scaffold.outputNames());
            if (!scaffold.formatHints().isEmpty()) {
                capability.put("formats", scaffold.formatHints());
            }
            if (!scaffold.targets().isEmpty()) {
                capability.put("targets", scaffold.targets());
            }
            if (!scaffold.providerHint().isBlank()) {
                capability.put("provider", scaffold.providerHint());
            }
            manifest.put("capabilities", List.of(capability));
        }
        try {
            return new ObjectMapper()
                    .writerWithDefaultPrettyPrinter()
                    .writeValueAsString(manifest) + System.lineSeparator();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to write scaffold manifest JSON", e);
        }
    }

    private static String scaffoldReadme(FeatureScaffold scaffold) {
        if (scaffold.modelFamily()) {
            return """
                    # %s

                    Out-of-core Gollek model-family plugin.

                    This plugin implements `%s`, publishes reusable model metadata through `%s`,
                    and is visible to the core plugin manager through `%s` plus root `%s`:

                    ```text
                    %s
                    %s
                    %s
                    ```

                    ## Build

                    ```bash
                    %s
                    ```

                    ## Attach During Development

                    ```bash
                    gollek modules --doctor --plugin-dir lib/build/libs
                    gollek show <model-or-path> --plugin-dir lib/build/libs --json
                    ```

                    ## Install For Local Reuse

                    ```bash
                    gollek extensions inspect lib/build/libs/%s-0.1.0-SNAPSHOT.jar --strict
                    mkdir -p ~/.gollek/plugins
                    gollek extensions install lib/build/libs/%s-0.1.0-SNAPSHOT.jar --plugin-dir ~/.gollek/plugins --force
                    GOLLEK_PLUGIN_DIRS=~/.gollek/plugins gollek modules --doctor
                    ```

                    ## Tokenizer Metadata Checks

                    ```bash
                    gollek extensions inspect lib/build/libs/%s-0.1.0-SNAPSHOT.jar --strict --json
                    gollek extensions doctor --plugin-dir lib/build/libs --json
                    ```

                    Before install, JSON `jar.artifact_tokenizer_metadata.tokenizer_metadata_status`
                    reports `ready` or `pending`. After install, `install_lock.tokenizer_metadata_status`
                    and doctor `tokenizer_lock_status` show what production tooling will enforce.

                    Pending tokenizer metadata is valid for development and research scaffolds, but keep
                    pending-tokenizer plugins out of production inference presets until the tokenizer adapter
                    is ready.

                    ## Detach

                    ```bash
                    gollek extensions detach %s --plugin-dir ~/.gollek/plugins
                    ```

                    ## Family Contract

                    Model-family id: `%s`

                    Model types: `%s`

                    Architecture classes: `%s`

                    Capabilities: `%s`

                    Tokenizer metadata: `%s`

                    %s

                    Start with tokenizer/config metadata only. Add direct SafeTensor architecture adapters later
                    when the runtime implementation is ready and validated by `gollek modules --doctor`.
                    """.formatted(
                    scaffold.displayName(),
                    scaffold.fqcn(),
                    MODEL_FAMILY_SERVICE_ENTRY,
                    GOLLEK_PLUGIN_SERVICE_ENTRY,
                    PLUGIN_DESCRIPTOR_ENTRY,
                    MODEL_FAMILY_SERVICE_ENTRY,
                    GOLLEK_PLUGIN_SERVICE_ENTRY,
                    PLUGIN_DESCRIPTOR_ENTRY,
                    scaffoldBuildCommand(scaffold),
                    scaffold.featureId(),
                    scaffold.featureId(),
                    scaffold.featureId(),
                    scaffold.capabilityId(),
                    scaffold.capabilityId(),
                    String.join(", ", scaffold.modelTypes()),
                    String.join(", ", scaffold.architectureClassNames()),
                    String.join(", ", scaffold.modelFamilyCapabilities()),
                    modelFamilyTokenizerMetadataSummary(scaffold),
                    modelFamilyTokenizerMetadataReadmeNote(scaffold));
        }
        if (!scaffold.pipeline()) {
            return """
                    # %s

                    Out-of-core Gollek %s extension adapter.

                    ## Build

                    ```bash
                    %s
                    ```

                    ## Validate And Install

                    ```bash
                    gollek extensions inspect lib/build/libs/%s-0.1.0-SNAPSHOT.jar --strict
                    gollek extensions install lib/build/libs/%s-0.1.0-SNAPSHOT.jar --force
                    gollek extensions doctor --strict
                    gollek extensions --details --paths
                    ```

                    ## Adapter Contract

                    This project publishes `%s` through:

                    ```text
                    %s
                    ```

                    The manifest declares capability `%s` with kind `%s`. Domain registries for
                    training, optimization, quantization, backend, exporter, dataset, evaluator, or
                    other adapter families can discover this service and route it into their own APIs.

                    The generated manifest includes `requires.gollek_spi`, service metadata, and an
                    ahead-of-time loading note. `gollek extensions doctor --strict` checks compatibility
                    metadata before the jar is used at runtime, and `gollek extensions index` emits an
                    inventory for JVM launchers or native-image build inputs.
                    """.formatted(
                    scaffold.displayName(),
                    scaffold.kind(),
                    scaffoldBuildCommand(scaffold),
                    scaffold.featureId(),
                    scaffold.featureId(),
                    scaffold.fqcn(),
                    ADAPTER_SERVICE_ENTRY,
                    scaffold.capabilityId(),
                    scaffold.kind());
        }
        return """
                # %s

                Out-of-core Gollek extension pipeline.

                ## Build

                ```bash
                %s
                ```

                ## Validate And Install

                ```bash
                gollek extensions inspect lib/build/libs/%s-0.1.0-SNAPSHOT.jar --strict
                gollek extensions install lib/build/libs/%s-0.1.0-SNAPSHOT.jar --force
                gollek extensions doctor --strict
                ```

                ## Run During Development

                ```bash
                gollek run --model <model> --pipeline %s --prompt "hello"
                ```

                The generated pipeline is intentionally conservative. It only handles a request when `--pipeline %s`
                selects it, or after you add model-specific detection in `supports(...)`.

                The generated manifest includes `requires.gollek_spi`, service metadata, and an
                ahead-of-time loading note. `gollek extensions doctor --strict` checks compatibility
                metadata before the jar is used at runtime, and `gollek extensions index` emits an
                inventory for JVM launchers or native-image build inputs.
                """.formatted(
                scaffold.displayName(),
                scaffoldBuildCommand(scaffold),
                scaffold.featureId(),
                scaffold.featureId(),
                scaffold.pipelineId(),
                scaffold.pipelineId());
    }

    private static String scaffoldBuildCommand(FeatureScaffold scaffold) {
        if (scaffold.gollekPath() == null) {
            return "gradle :lib:build";
        }
        String relative = relativePath(scaffold.root(), scaffold.gollekPath()).replace('\\', '/');
        return relative + "/gradlew -p . :lib:build";
    }

    private static List<ModalityType> parseModalities(List<String> values, String label) {
        List<String> source = values == null || values.isEmpty() ? List.of("text") : values;
        List<ModalityType> result = new ArrayList<>();
        for (String raw : source) {
            String normalized = raw == null ? "" : raw.trim().replace('-', '_').toUpperCase(Locale.ROOT);
            if (normalized.isBlank()) {
                continue;
            }
            try {
                result.add(ModalityType.valueOf(normalized));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("Unknown " + label + " modality: " + raw);
            }
        }
        return result.isEmpty() ? List.of(ModalityType.TEXT) : List.copyOf(result);
    }

    private static List<String> normalizeScaffoldTokens(String configured, List<String> defaults) {
        return normalizeScaffoldTokens(
                configured == null || configured.isBlank() ? List.of() : List.of(configured),
                defaults);
    }

    private static List<String> normalizeScaffoldTokens(List<String> configured, List<String> defaults) {
        Set<String> values = new LinkedHashSet<>();
        if (configured != null) {
            for (String raw : configured) {
                if (raw == null) {
                    continue;
                }
                for (String part : raw.split(",")) {
                    String normalized = part.trim().toLowerCase(Locale.ROOT)
                            .replaceAll("[^a-z0-9._-]+", "-")
                            .replaceAll("[-_.]{2,}", "-")
                            .replaceAll("^[-_.]+|[-_.]+$", "");
                    if (!normalized.isBlank()) {
                        values.add(normalized);
                    }
                }
            }
        }
        if (values.isEmpty() && defaults != null) {
            for (String fallback : defaults) {
                String normalized = normalizeHint(fallback);
                if (!normalized.isBlank()) {
                    values.add(normalized);
                }
            }
        }
        return List.copyOf(values);
    }

    private static List<String> normalizeCaseSensitiveScaffoldTokens(List<String> configured, List<String> defaults) {
        Set<String> values = new LinkedHashSet<>();
        if (configured != null) {
            for (String raw : configured) {
                if (raw == null) {
                    continue;
                }
                for (String part : raw.split(",")) {
                    String normalized = part.trim();
                    if (!normalized.isBlank()) {
                        values.add(normalized);
                    }
                }
            }
        }
        if (values.isEmpty() && defaults != null) {
            for (String fallback : defaults) {
                String normalized = fallback == null ? "" : fallback.trim();
                if (!normalized.isBlank()) {
                    values.add(normalized);
                }
            }
        }
        return List.copyOf(values);
    }

    private static List<String> defaultAdapterInputs(String kind) {
        return switch (FeatureAdapterKinds.normalize(kind)) {
            case FeatureAdapterKinds.TRAINING -> List.of("dataset", "model");
            case FeatureAdapterKinds.BACKEND -> List.of("tensor");
            case FeatureAdapterKinds.DATASET -> List.of("source");
            case FeatureAdapterKinds.EVALUATOR -> List.of("model", "dataset");
            case FeatureAdapterKinds.RUNNER -> List.of("model", "request");
            default -> List.of("model");
        };
    }

    private static List<String> defaultAdapterOutputs(String kind) {
        return switch (FeatureAdapterKinds.normalize(kind)) {
            case FeatureAdapterKinds.TRAINING -> List.of("model", "metrics");
            case FeatureAdapterKinds.BACKEND -> List.of("tensor");
            case FeatureAdapterKinds.CONVERTER, FeatureAdapterKinds.EXPORTER -> List.of("artifact");
            case FeatureAdapterKinds.DATASET -> List.of("dataset");
            case FeatureAdapterKinds.EVALUATOR -> List.of("metrics");
            case FeatureAdapterKinds.RUNNER -> List.of("response");
            case FeatureAdapterKinds.TOOLING, FeatureAdapterKinds.ADAPTER -> List.of("artifact");
            default -> List.of("model");
        };
    }

    private static List<String> normalizeModelFamilyCapabilities(List<String> values, boolean modelFamilyScaffold) {
        if (!modelFamilyScaffold) {
            return List.of();
        }
        List<String> source = values == null || values.isEmpty() ? List.of("tokenizer") : values;
        List<String> capabilities = new ArrayList<>();
        for (String value : source) {
            String normalized = value == null
                    ? ""
                    : value.trim().replace('-', '_').replace('.', '_').toUpperCase(Locale.ROOT);
            if (normalized.isBlank()) {
                continue;
            }
            try {
                ModelFamilyCapability.valueOf(normalized);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("Unknown model-family capability: " + value, e);
            }
            if (!capabilities.contains(normalized)) {
                capabilities.add(normalized);
            }
        }
        return capabilities.isEmpty() ? List.of("TOKENIZER") : List.copyOf(capabilities);
    }

    private static String normalizeTokenizerKind(String tokenizerKind) {
        String normalized = normalizeFeatureId(firstNonBlank(tokenizerKind, "hf-bpe"));
        return switch (normalized) {
            case "hf", "huggingface", "huggingface-bpe", "bpe", "hf-bpe" -> "hf-bpe";
            case "spm", "sentence-piece", "sentencepiece", "sentencepiece-bpe" -> "sentencepiece";
            case "bert", "word-piece", "wordpiece" -> "wordpiece";
            case "custom", "none" -> normalized;
            default -> throw new IllegalArgumentException("Unknown tokenizer kind: " + tokenizerKind);
        };
    }

    private static String normalizeTokenizerMetadataStatus(String status) {
        String normalized = normalizeFeatureId(firstNonBlank(status, "ready"));
        return switch (normalized) {
            case "ready", "complete" -> "ready";
            case "pending", "todo", "not-ready", "not_ready" -> "pending";
            default -> throw new IllegalArgumentException("Unknown tokenizer metadata status: " + status);
        };
    }

    private static boolean tokenizerKindNone(String tokenizerKind) {
        return tokenizerKind != null
                && !tokenizerKind.isBlank()
                && "none".equals(normalizeFeatureId(tokenizerKind));
    }

    private static String modelFamilyCapabilityJavaList(List<String> capabilities) {
        List<String> normalized = capabilities == null || capabilities.isEmpty() ? List.of("TOKENIZER") : capabilities;
        return "List.of(" + normalized.stream()
                .map(capability -> "ModelFamilyCapability." + capability)
                .collect(Collectors.joining(", ")) + ")";
    }

    private static String modelFamilyMetadataJava(FeatureScaffold scaffold) {
        List<String> entries = new ArrayList<>();
        entries.add("\"bundle_profile\", \"" + escapeJava(scaffold.bundleProfile()) + "\"");
        entries.add("\"origin\", \"external/model-family\"");
        if (scaffold.tokenizerMetadataPending()) {
            entries.add("\"tokenizer_metadata_status\", \"pending\"");
            entries.add("\"tokenizer_metadata_pending_reason\", \""
                    + escapeJava(scaffold.tokenizerMetadataPendingReason()) + "\"");
        } else if (scaffold.tokenizerMetadataReady()) {
            entries.add("\"tokenizer_metadata_status\", \"ready\"");
        }
        entries.add("\"version\", \"0.1.0-SNAPSHOT\"");
        return "Map.of(" + String.join(", ", entries) + ")";
    }

    private static String modelFamilyTokenizerMetadataSummary(FeatureScaffold scaffold) {
        if (scaffold.tokenizerMetadataPending()) {
            return "pending (" + scaffold.tokenizerMetadataPendingReason() + ")";
        }
        return "ready (" + scaffold.tokenizerKind() + ")";
    }

    private static String modelFamilyTokenizerMetadataReadmeNote(FeatureScaffold scaffold) {
        if (scaffold.tokenizerMetadataPending()) {
            return "Pending tokenizer metadata intentionally omits tokenizer descriptors until the adapter contract is implemented.";
        }
        return "Ready tokenizer metadata publishes reusable tokenizer descriptors for runtime discovery.";
    }

    private static String modelFamilyTokenizerDescriptorJava(FeatureScaffold scaffold) {
        if (scaffold.tokenizerMetadataPending()) {
            return "List.of()";
        }
        return switch (scaffold.tokenizerKind()) {
            case "hf-bpe" -> "List.of(ModelTokenizerDescriptor.huggingFaceBpe(FAMILY_ID + \"-tokenizer\"))";
            case "sentencepiece" -> "List.of(ModelTokenizerDescriptor.sentencePieceBpe(FAMILY_ID + \"-tokenizer\"))";
            case "wordpiece" -> "List.of(ModelTokenizerDescriptor.wordPiece(FAMILY_ID + \"-tokenizer\"))";
            case "custom" -> "List.of(new ModelTokenizerDescriptor("
                    + "FAMILY_ID + \"-tokenizer\", "
                    + "tech.kayys.alkhawarizm.spi.model.ModelTokenizerKind.CUSTOM, "
                    + "List.of(), "
                    + "Map.of()))";
            case "none" -> "List.of()";
            default -> throw new IllegalArgumentException("Unknown tokenizer kind: " + scaffold.tokenizerKind());
        };
    }

    private static List<ScaffoldKind> scaffoldKinds() {
        return List.of(
                scaffoldKind(
                        FeatureAdapterKinds.PIPELINE,
                        "ModelPipeline",
                        List.of("text"),
                        List.of("text"),
                        List.of(),
                        "multi-step inference orchestration",
                        "gollek extensions init features/gollek-ocr-lab --kind pipeline --pipeline-id ocr-lab --input image,text --output text,document --provider onnx --format onnx"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.TRAINING,
                        List.of("llm", "vision", "audio"),
                        "training loops, LoRA, fine-tuning, experiments",
                        "gollek extensions init features/gollek-lora-lab --kind training --adapter-id lora-trainer --input dataset,model --output model,metrics --target llm"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.OPTIMIZATION,
                        List.of("llm", "edge"),
                        "graph rewrites, pruning, tuning, compile passes",
                        "gollek extensions init features/gollek-opt-lab --kind optimization --adapter-id graph-optimizer --input model --output model --target edge"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.QUANTIZATION,
                        List.of("llm"),
                        "AWQ, GPTQ, INT8, FP8, or format-aware quantizers",
                        "gollek extensions init features/gollek-awq-lab --kind quantization --adapter-id awq-quantizer --input model --output model --target llm --format safetensors,gguf"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.BACKEND,
                        List.of("cpu", "metal", "cuda", "rocm"),
                        "backend bridges, kernels, delegates, device runtimes",
                        "gollek extensions init features/gollek-metal-lab --kind backend --adapter-id metal-backend --input tensor --output tensor --target metal"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.EXPORTER,
                        List.of("onnx", "gguf", "safetensors"),
                        "model or artifact export workflows",
                        "gollek extensions init features/gollek-export-lab --kind exporter --adapter-id onnx-exporter --input model --output artifact --format onnx"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.CONVERTER,
                        List.of("onnx", "gguf", "safetensors"),
                        "model format conversion or migration tools",
                        "gollek extensions init features/gollek-convert-lab --kind converter --adapter-id model-converter --input model --output artifact"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.DATASET,
                        List.of("llm", "vision", "audio"),
                        "dataset loaders, preprocessors, corpus tools",
                        "gollek extensions init features/gollek-data-lab --kind dataset --adapter-id corpus-loader --input source --output dataset --target llm"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.EVALUATOR,
                        List.of("llm", "vision", "audio"),
                        "benchmarks, eval harnesses, scoring tools",
                        "gollek extensions init features/gollek-eval-lab --kind evaluator --adapter-id llm-evaluator --input model,dataset --output metrics --target llm"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.RUNNER,
                        List.of("llm", "vision", "audio"),
                        "runner adapters outside the core provider path",
                        "gollek extensions init features/gollek-runner-lab --kind runner --adapter-id custom-runner --input model,request --output response"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.TOOLING,
                        List.of("research", "ops"),
                        "research, diagnostics, packaging, or ops tooling",
                        "gollek extensions init features/gollek-tools-lab --kind tooling --adapter-id report-tool --input model --output artifact"),
                adapterScaffoldKind(
                        FeatureAdapterKinds.ADAPTER,
                        List.of("custom"),
                        "generic adapter when no common kind fits yet",
                        "gollek extensions init features/gollek-custom-lab --kind adapter --adapter-id custom-adapter --input model --output artifact"),
                scaffoldKind(
                        MODEL_FAMILY_KIND,
                        "ModelFamilyPlugin",
                        List.of("config.json", "tokenizer.json"),
                        List.of("ModelFamilyDescriptor", "ModelTokenizerDescriptor"),
                        List.of("llm", "production", "optional"),
                        "detachable model-family metadata, tokenizer descriptors, and optional architecture adapters",
                        "gollek extensions init features/gollek-model-acme --kind model-family --family-id acme --model-type acme --architecture AcmeForCausalLM --capability causal_lm,tokenizer --tokenizer-metadata-status pending --tokenizer-pending-reason scaffold"));
    }

    private static ScaffoldKind adapterScaffoldKind(
            String kind,
            List<String> defaultTargets,
            String summary,
            String exampleCommand) {
        return scaffoldKind(
                FeatureAdapterKinds.normalize(kind),
                "FeatureAdapter",
                defaultAdapterInputs(kind),
                defaultAdapterOutputs(kind),
                defaultTargets,
                summary,
                exampleCommand);
    }

    private static ScaffoldKind scaffoldKind(
            String kind,
            String service,
            List<String> defaultInputs,
            List<String> defaultOutputs,
            List<String> defaultTargets,
            String summary,
            String exampleCommand) {
        return new ScaffoldKind(
                kind,
                service,
                defaultInputs == null ? List.of() : List.copyOf(defaultInputs),
                defaultOutputs == null ? List.of() : List.copyOf(defaultOutputs),
                defaultTargets == null ? List.of() : List.copyOf(defaultTargets),
                summary == null ? "" : summary,
                exampleCommand == null ? "" : exampleCommand);
    }

    private static List<String> normalizeTags(
            List<String> configured,
            String family,
            String providerHint,
            String formatHint,
            String... extraHints) {
        Set<String> values = new LinkedHashSet<>();
        addTag(values, family);
        addTag(values, providerHint);
        addTag(values, formatHint);
        if (extraHints != null) {
            for (String extraHint : extraHints) {
                addTag(values, extraHint);
            }
        }
        if (configured != null) {
            configured.forEach(value -> addTag(values, value));
        }
        values.add("external-extension");
        return List.copyOf(values);
    }

    private static void addTag(Set<String> tags, String value) {
        if (value == null) {
            return;
        }
        for (String part : value.split(",")) {
            String tag = normalizeHint(part);
            if (!tag.isBlank()) {
                tags.add(tag);
            }
        }
    }

    private static String normalizeHint(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return "";
    }

    private static Object firstNonNull(Object... values) {
        if (values == null) {
            return null;
        }
        for (Object value : values) {
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private static String fileStem(Path path) {
        Path name = path.getFileName();
        return name == null ? "" : name.toString();
    }

    private static String normalizeFeatureId(String value) {
        String normalized = value == null ? "" : value.trim().toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9._-]+", "-")
                .replaceAll("[-_.]{2,}", "-")
                .replaceAll("^[-_.]+|[-_.]+$", "");
        return normalized.isBlank() ? "gollek-extension" : normalized;
    }

    private static String normalizePackageName(String value) {
        String[] parts = value == null ? new String[0] : value.trim().split("\\.");
        List<String> normalized = new ArrayList<>();
        for (String part : parts) {
            String identifier = javaIdentifier(part, false);
            if (!identifier.isBlank()) {
                normalized.add(identifier);
            }
        }
        if (normalized.isEmpty()) {
            return "tech.kayys.gollek.extensions.custom";
        }
        return String.join(".", normalized);
    }

    private static String normalizeClassName(String value) {
        String candidate = javaIdentifier(value, true);
        return candidate.isBlank() ? "CustomFeaturePipeline" : candidate;
    }

    private static String javaIdentifier(String value, boolean upperFirst) {
        String raw = value == null ? "" : value.trim();
        StringBuilder out = new StringBuilder();
        boolean capitalize = upperFirst;
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '_') {
                char next = capitalize ? Character.toUpperCase(c) : c;
                if (out.isEmpty() && !Character.isJavaIdentifierStart(next)) {
                    out.append('_');
                }
                out.append(Character.isJavaIdentifierPart(next) ? next : '_');
                capitalize = false;
            } else {
                capitalize = upperFirst;
            }
        }
        if (out.isEmpty()) {
            return "";
        }
        if (!Character.isJavaIdentifierStart(out.charAt(0))) {
            out.insert(0, '_');
        }
        return out.toString();
    }

    private static String defaultPackageName(String featureId) {
        String suffix = featureId.replaceFirst("^gollek[-_.]+", "").replaceAll("[^a-zA-Z0-9]+", "");
        if (suffix.isBlank()) {
            suffix = "custom";
        }
        return "tech.kayys.gollek.extensions." + suffix.toLowerCase(Locale.ROOT);
    }

    private static String pascalCase(String value) {
        StringBuilder out = new StringBuilder();
        boolean capitalize = true;
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                out.append(capitalize ? Character.toUpperCase(c) : c);
                capitalize = false;
            } else {
                capitalize = true;
            }
        }
        return out.isEmpty() ? "CustomFeature" : out.toString();
    }

    private static String titleFromId(String id) {
        StringBuilder out = new StringBuilder();
        for (String part : id.split("[-_.]+")) {
            if (part.isBlank()) {
                continue;
            }
            if (!out.isEmpty()) {
                out.append(' ');
            }
            out.append(Character.toUpperCase(part.charAt(0)));
            if (part.length() > 1) {
                out.append(part.substring(1));
            }
        }
        return out.isEmpty() ? "Gollek Feature" : out.toString();
    }

    private static String modalityJavaList(List<ModalityType> modalities) {
        return "List.of(" + modalities.stream()
                .map(type -> "ModalityType." + type.name())
                .reduce((left, right) -> left + ", " + right)
                .orElse("ModalityType.TEXT") + ")";
    }

    private static String stringJavaList(List<String> values) {
        return "List.of(" + values.stream()
                .map(value -> "\"" + escapeJava(value) + "\"")
                .reduce((left, right) -> left + ", " + right)
                .orElse("") + ")";
    }

    private static String relativePath(Path root, Path target) {
        try {
            return root.relativize(target).toString();
        } catch (Exception ignored) {
            return target.toString();
        }
    }

    private static String escapeJava(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String escapeGradle(String value) {
        return escapeJava(value).replace("$", "\\$");
    }

    private record FeatureScaffold(
            Path root,
            String featureId,
            String kind,
            String capabilityId,
            String pipelineId,
            String displayName,
            String packageName,
            String className,
            String family,
            List<ModalityType> pipelineInputs,
            List<ModalityType> pipelineOutputs,
            List<String> adapterInputs,
            List<String> adapterOutputs,
            List<String> tags,
            String providerHint,
            List<String> formatHints,
            List<String> targets,
            List<String> modelTypes,
            List<String> architectureClassNames,
            List<String> modelFamilyCapabilities,
            String tokenizerKind,
            String tokenizerMetadataStatus,
            String tokenizerMetadataPendingReason,
            String bundleProfile,
            Path gollekPath) {
        boolean modelFamily() {
            return MODEL_FAMILY_KIND.equals(kind);
        }

        boolean pipeline() {
            return !modelFamily() && FeatureAdapterKinds.PIPELINE.equals(kind);
        }

        boolean tokenizerMetadataReady() {
            return "ready".equals(tokenizerMetadataStatus);
        }

        boolean tokenizerMetadataPending() {
            return "pending".equals(tokenizerMetadataStatus);
        }

        String fqcn() {
            return packageName + "." + className;
        }

        String formatHint() {
            return formatHints == null || formatHints.isEmpty() ? "" : formatHints.get(0);
        }

        List<String> inputNames() {
            return pipeline() ? modalityNames(pipelineInputs) : adapterInputs;
        }

        List<String> outputNames() {
            return pipeline() ? modalityNames(pipelineOutputs) : adapterOutputs;
        }
    }

    private record ScaffoldKind(
            String kind,
            String service,
            List<String> defaultInputs,
            List<String> defaultOutputs,
            List<String> defaultTargets,
            String summary,
            String exampleCommand) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("kind", kind);
            value.put("service", service);
            value.put("default_inputs", defaultInputs);
            value.put("default_outputs", defaultOutputs);
            value.put("default_targets", defaultTargets);
            value.put("summary", summary);
            value.put("example", exampleCommand);
            return value;
        }
    }

    private record ScaffoldFile(
            String relativePath,
            String content) {
    }

    private static FeatureRow featureRow(ModelPipeline pipeline) {
        PipelineDescriptor descriptor = safeDescriptor(pipeline);
        Map<String, Object> metadata = descriptor == null ? Map.of() : descriptor.metadata();
        String source = metadataText(metadata, "source", "kind", "origin");
        if (source.isBlank()) {
            source = "classpath";
        }
        return new FeatureRow(
                pipeline.id(),
                descriptor == null ? pipeline.id() : descriptor.name(),
                descriptor == null ? "" : descriptor.version(),
                FeatureAdapterKinds.PIPELINE,
                descriptor == null ? "unknown" : descriptor.family(),
                descriptor == null ? List.of() : modalityNames(descriptor.inputModalities()),
                descriptor == null ? List.of() : modalityNames(descriptor.outputModalities()),
                descriptor == null ? List.of() : descriptor.tags(),
                metadata,
                pipeline.priority(),
                source,
                pipeline.getClass().getName(),
                codeSource(pipeline.getClass()));
    }

    private static PipelineDescriptor safeDescriptor(ModelPipeline pipeline) {
        try {
            return pipeline.descriptor();
        } catch (Exception ignored) {
            return null;
        }
    }

    private static List<String> modalityNames(List<ModalityType> modalities) {
        if (modalities == null || modalities.isEmpty()) {
            return List.of();
        }
        return modalities.stream()
                .map(type -> type.name().toLowerCase())
                .toList();
    }

    private void printJson(
            List<FeatureRow> features,
            int loadedCount,
            int scannedCount,
            FeatureDoctorReport scanReport,
            FeatureDoctorReport doctorReport) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("type", "gollek_extension_features");
        root.put("count", features.size());
        root.put("loaded_count", loadedCount);
        root.put("scanned_count", scannedCount);
        root.put("features", features.stream().map(FeatureRow::toMap).toList());
        root.put("pipelines", features.stream()
                .filter(feature -> FeatureAdapterKinds.PIPELINE.equals(feature.kind()))
                .map(FeatureRow::toMap)
                .toList());
        if (scanReport != null) {
            root.put("scan", scanReport.toMap());
        }
        if (doctorReport != null) {
            root.put("doctor", doctorReport.toMap());
        }
        try {
            System.out.println(new ObjectMapper()
                    .writerWithDefaultPrettyPrinter()
                    .writeValueAsString(root));
        } catch (Exception e) {
            System.err.println("Failed to write extension JSON: " + e.getMessage());
        }
    }

    private static void printScanSummary(FeatureDoctorReport report) {
        FeatureDoctorSummary summary = report.validationSummary(false);
        System.out.println("Extension Path Scan");
        System.out.println("-------------------");
        System.out.printf("Scanned: %d root(s), %d jar(s), %d ok, %d warning, %d error, %d collision(s), %d ignored legacy path setting(s)%n",
                summary.roots(),
                summary.jars(),
                summary.ok(),
                summary.warning(),
                summary.error(),
                summary.collisions(),
                summary.ignoredLegacyPaths());
    }

    private static void printIndexSummary(FeatureDoctorReport report, boolean strict, Path outputPath) {
        FeatureDoctorSummary summary = report.validationSummary(strict);
        int pipelineProviders = report.jars().stream().mapToInt(jar -> jar.providers().size()).sum();
        int adapterProviders = report.jars().stream().mapToInt(jar -> jar.adapterProviders().size()).sum();
        int modelFamilyProviders = report.jars().stream().mapToInt(jar -> jar.modelFamilyProviders().size()).sum();
        int capabilities = report.jars().stream()
                .map(FeatureJarReport::manifest)
                .mapToInt(manifest -> manifestCapabilities(manifest).size())
                .sum();
        System.out.println("Extension Index");
        System.out.println("---------------");
        System.out.printf("Indexed: %d root(s), %d jar(s), %d ok, %d warning, %d error%n",
                summary.roots(), summary.jars(), summary.ok(), summary.warning(), summary.error());
        System.out.printf("Services: %d ModelPipeline provider(s), %d FeatureAdapter provider(s), %d ModelFamilyPlugin provider(s), %d manifest capability entry(ies)%n",
                pipelineProviders, adapterProviders, modelFamilyProviders, capabilities);
        if (outputPath != null) {
            System.out.printf("Wrote index: %s%n", outputPath);
        }
        System.out.println("Native image: include indexed extension jars on the native-image build classpath; JVM runtime can still load them dynamically.");
    }

    private static FeatureDoctorReport inspectRuntimeFeaturePaths() {
        return inspectRuntimeFeaturePaths(List.of());
    }

    private static FeatureDoctorReport inspectRuntimeFeaturePaths(List<String> extraRoots) {
        return inspectFeatureRoots(runtimeFeatureRoots(extraRoots));
    }

    private static FeatureDoctorReport inspectConfiguredFeaturePaths(List<String> configuredRoots) {
        return inspectFeatureRoots(configuredFeatureRoots(configuredRoots));
    }

    private static FeatureDoctorReport inspectFeatureRoots(List<Path> roots) {
        List<FeatureRootReport> rootReports = new ArrayList<>();
        List<FeatureJarReport> jarReports = new ArrayList<>();
        for (Path root : roots) {
            boolean exists = Files.exists(root);
            boolean directory = Files.isDirectory(root);
            boolean regularFile = Files.isRegularFile(root);
            rootReports.add(new FeatureRootReport(root.toString(), exists, directory, regularFile));
            if (regularFile && isJar(root)) {
                FeatureInstallLockState lockState = readInstallLock(root.getParent());
                jarReports.add(applyInstallLock(root.getParent(), inspectJar(root), lockState));
            } else if (directory) {
                FeatureInstallLockState lockState = readInstallLock(root);
                Set<String> seenJars = new LinkedHashSet<>();
                try (var stream = Files.list(root)) {
                    stream.filter(path -> Files.isRegularFile(path) && isJar(path))
                            .map(FeaturesCommand::normalize)
                            .sorted()
                            .map(FeaturesCommand::inspectJar)
                            .map(report -> {
                                Path jarPath = normalize(Path.of(report.path()));
                                Path name = jarPath.getFileName();
                                if (name != null) {
                                    seenJars.add(name.toString());
                                }
                                return applyInstallLock(root, report, lockState);
                            })
                            .forEach(jarReports::add);
                    missingLockedJars(root, seenJars, lockState).forEach(jarReports::add);
                } catch (Exception e) {
                    jarReports.add(new FeatureJarReport(root.toString(), false, List.of(), false, List.of(),
                            false, List.of(), false, List.of(), Map.of(), "", null,
                            Map.of(), "", null, Map.of(), Map.of(), List.of(),
                            "Failed to inspect directory: " + e.getMessage()));
                }
            }
        }
        return new FeatureDoctorReport(rootReports, jarReports);
    }

    private static Map<String, Object> extensionIndex(FeatureDoctorReport report, boolean strict) {
        Map<String, Object> index = new LinkedHashMap<>();
        index.put("schema_version", 1);
        index.put("type", "gollek_extension_index");
        index.put("created_at", Instant.now().toString());
        index.put("strict", strict);
        index.put("runtime", runtimeCompatibilityContext().toMap());
        index.put("native_image", nativeImageIndexHints());
        Map<String, Object> serviceContracts = new LinkedHashMap<>();
        serviceContracts.put("model_pipeline", SERVICE_ENTRY);
        serviceContracts.put("feature_adapter", ADAPTER_SERVICE_ENTRY);
        serviceContracts.put("model_family_plugin", MODEL_FAMILY_SERVICE_ENTRY);
        serviceContracts.put("gollek_plugin", GOLLEK_PLUGIN_SERVICE_ENTRY);
        serviceContracts.put("plugin_descriptor", PLUGIN_DESCRIPTOR_ENTRY);
        index.put("service_contracts", serviceContracts);
        index.put("roots", report.roots().stream().map(FeatureRootReport::toMap).toList());
        index.put("summary", report.validationSummary(strict).toMap());
        index.put("collisions", report.collisionReport().toMap());
        index.put("jars", report.jars().stream()
                .map(jar -> extensionIndexJar(jar, strict))
                .toList());
        return index;
    }

    private static Map<String, Object> nativeImageIndexHints() {
        Map<String, Object> hints = new LinkedHashMap<>();
        hints.put("mode", "ahead-of-time");
        hints.put("dynamic_runtime_loading", false);
        hints.put("jvm_dynamic_loading", true);
        hints.put("build_time_requirement", "include extension jars on the native-image build classpath");
        hints.put("service_loader_entries", List.of(
                SERVICE_ENTRY,
                ADAPTER_SERVICE_ENTRY,
                MODEL_FAMILY_SERVICE_ENTRY,
                GOLLEK_PLUGIN_SERVICE_ENTRY));
        hints.put("plugin_descriptor_entry", PLUGIN_DESCRIPTOR_ENTRY);
        hints.put("note", "The index is manifest/service metadata only; extension-specific reflection or native libraries remain the extension author's responsibility.");
        return hints;
    }

    private static Map<String, Object> extensionIndexJar(FeatureJarReport jar, boolean strict) {
        FeatureJarValidation validation = validateJar(jar, strict);
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("path", jar.path());
        value.put("status", validation.status());
        value.put("valid", validation.valid());
        putFileDigest(value, jar.path());
        Map<String, Object> serviceEntries = new LinkedHashMap<>();
        serviceEntries.put(SERVICE_ENTRY, jar.providers());
        serviceEntries.put(ADAPTER_SERVICE_ENTRY, jar.adapterProviders());
        serviceEntries.put(MODEL_FAMILY_SERVICE_ENTRY, jar.modelFamilyProviders());
        serviceEntries.put(GOLLEK_PLUGIN_SERVICE_ENTRY, jar.gollekPluginProviders());
        value.put("service_entries", serviceEntries);
        value.put("providers", jar.providers());
        value.put("adapter_providers", jar.adapterProviders());
        value.put("model_family_providers", jar.modelFamilyProviders());
        value.put("gollek_plugin_providers", jar.gollekPluginProviders());
        if (validation.warnings() != null && !validation.warnings().isEmpty()) {
            value.put("warnings", validation.warnings());
        }
        if (validation.errors() != null && !validation.errors().isEmpty()) {
            value.put("errors", validation.errors());
        }
        Map<String, Object> manifest = jar.manifest();
        if (jar.hasManifest()) {
            value.put("extension", extensionIdentity(manifest));
            value.put("requires", manifest.getOrDefault("requires", Map.of()));
            value.put("manifest_entry", jar.manifestEntryName());
            if (manifest.containsKey("services")) {
                value.put("manifest_services", manifest.get("services"));
            }
            List<Map<String, Object>> capabilities = manifestCapabilities(manifest).stream()
                    .map(FeaturesCommand::extensionIndexCapability)
                    .toList();
            value.put("pipelines", capabilities.stream()
                    .filter(capability -> FeatureAdapterKinds.PIPELINE.equals(manifestValue(capability, "kind")))
                    .toList());
            value.put("model_families", capabilities.stream()
                    .filter(capability -> MODEL_FAMILY_KIND.equals(manifestValue(capability, "kind")))
                    .toList());
            value.put("capabilities", capabilities.stream()
                    .filter(capability -> !FeatureAdapterKinds.PIPELINE.equals(manifestValue(capability, "kind")))
                    .filter(capability -> !MODEL_FAMILY_KIND.equals(manifestValue(capability, "kind")))
                    .toList());
        } else if (jar.manifestError() != null && !jar.manifestError().isBlank()) {
            value.put("manifest_error", jar.manifestError());
        }
        if (jar.hasPluginDescriptor()) {
            value.put("plugin_descriptor_entry", jar.pluginDescriptorEntryName());
            value.put("plugin_descriptor", jar.pluginDescriptor());
        } else if (jar.pluginDescriptorError() != null && !jar.pluginDescriptorError().isBlank()) {
            value.put("plugin_descriptor_error", jar.pluginDescriptorError());
        }
        if (jar.hasInstallLock()) {
            value.put("install_lock", jar.installLock());
        }
        if (jar.installLockWarnings() != null && !jar.installLockWarnings().isEmpty()) {
            value.put("install_lock_warnings", jar.installLockWarnings());
        }
        if (jar.error() != null && !jar.error().isBlank()) {
            value.put("error", jar.error());
        }
        return value;
    }

    private static Map<String, Object> extensionIdentity(Map<String, Object> manifest) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("id", manifestValue(manifest, "id"));
        value.put("name", manifestValue(manifest, "name"));
        value.put("version", manifestValue(manifest, "version"));
        value.put("source", manifestValue(manifest, "source"));
        return value;
    }

    private static Map<String, Object> extensionIndexCapability(Map<String, Object> capability) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("source", firstNonBlank(manifestValue(capability, "_manifest_source"), "capabilities"));
        value.put("id", manifestValue(capability, "id"));
        value.put("kind", manifestCapabilityKind(capability));
        value.put("class", manifestValue(capability, "class"));
        value.put("family", manifestValue(capability, "family"));
        value.put("inputs", manifestStringList(capability.get("inputs")));
        value.put("outputs", manifestStringList(capability.get("outputs")));
        List<String> formats = manifestStringList(capability.get("formats"));
        String format = manifestValue(capability, "format");
        if (!format.isBlank()) {
            List<String> mergedFormats = new ArrayList<>(formats);
            mergedFormats.add(format);
            formats = normalizeScaffoldTokens(mergedFormats, List.of());
        }
        value.put("formats", formats);
        value.put("targets", manifestStringList(capability.get("targets")));
        String provider = manifestValue(capability, "provider");
        if (!provider.isBlank()) {
            value.put("provider", provider);
        }
        if (MODEL_FAMILY_KIND.equals(manifestCapabilityKind(capability))) {
            value.put("model_types", manifestStringList(capability.get("model_types")));
            value.put("architecture_classes", manifestStringList(capability.get("architecture_classes")));
            value.put("model_family_capabilities", manifestStringList(capability.get("capabilities")));
            String tokenizerKind = manifestValue(capability, "tokenizer_kind");
            if (!tokenizerKind.isBlank()) {
                value.put("tokenizer_kind", tokenizerKind);
            }
            List<String> tokenizerKinds = manifestStringList(capability.get("tokenizer_kinds"));
            if (tokenizerKinds.isEmpty() && !tokenizerKind.isBlank()) {
                tokenizerKinds = List.of(tokenizerKind);
            }
            if (!tokenizerKinds.isEmpty()) {
                value.put("tokenizer_kinds", tokenizerKinds);
            }
            String bundleProfile = manifestValue(capability, "bundle_profile");
            if (!bundleProfile.isBlank()) {
                value.put("bundle_profile", bundleProfile);
            }
        }
        return value;
    }

    private static void putFileDigest(Map<String, Object> value, String path) {
        try {
            Path jar = normalize(Path.of(path));
            if (Files.isRegularFile(jar)) {
                value.put("size_bytes", Files.size(jar));
                value.put("sha256", sha256(jar));
            }
        } catch (Exception e) {
            value.put("digest_error", e.getMessage());
        }
    }

    private static FeatureCollisionReport detectFeatureCollisions(List<FeatureJarReport> jars) {
        Map<String, List<FeatureCollisionClaim>> extensionIds = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> featureIds = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> pipelineIds = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> adapterIds = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> modelFamilyIds = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> providerClasses = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> adapterProviderClasses = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> modelFamilyProviderClasses = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> implementationClasses = new LinkedHashMap<>();
        Map<String, List<FeatureCollisionClaim>> providerIds = new LinkedHashMap<>();

        for (FeatureJarReport jar : jars == null ? List.<FeatureJarReport>of() : jars) {
            if (jar == null || jar.error() != null && !jar.error().isBlank()) {
                continue;
            }
            String path = jar.path();
            Map<String, Object> manifest = jar.manifest();
            addCollisionClaim(extensionIds, manifestValue(manifest, "id"), path, "manifest.id");
            for (String provider : jar.providers()) {
                addCollisionClaim(providerClasses, provider, path, SERVICE_ENTRY);
            }
            for (String provider : jar.adapterProviders()) {
                addCollisionClaim(adapterProviderClasses, provider, path, ADAPTER_SERVICE_ENTRY);
            }
            for (String provider : jar.modelFamilyProviders()) {
                addCollisionClaim(modelFamilyProviderClasses, provider, path, MODEL_FAMILY_SERVICE_ENTRY);
            }
            int index = 0;
            for (Map<String, Object> capability : manifestCapabilities(manifest)) {
                String source = firstNonBlank(manifestValue(capability, "_manifest_source"), "capabilities");
                String kind = manifestCapabilityKind(capability);
                String id = manifestValue(capability, "id");
                String claimSource = source + "[" + index + "]" + (kind.isBlank() ? "" : ":" + kind);
                addCollisionClaim(featureIds, id, path, claimSource);
                if (FeatureAdapterKinds.PIPELINE.equals(kind)) {
                    addCollisionClaim(pipelineIds, id, path, claimSource);
                } else if (FeatureAdapterKinds.ADAPTER.equals(kind)) {
                    addCollisionClaim(adapterIds, id, path, claimSource);
                } else if (MODEL_FAMILY_KIND.equals(kind)) {
                    addCollisionClaim(modelFamilyIds, id, path, claimSource);
                }
                addCollisionClaim(implementationClasses, manifestValue(capability, "class"), path, claimSource + ".class");
                addCollisionClaim(providerIds, manifestValue(capability, "provider"), path, claimSource + ".provider");
                index++;
            }
        }

        List<FeatureCollisionItem> items = new ArrayList<>();
        items.addAll(collisionItems("extension_id", extensionIds));
        items.addAll(collisionItems("feature_id", featureIds));
        items.addAll(collisionItems("pipeline_id", pipelineIds));
        items.addAll(collisionItems("adapter_id", adapterIds));
        items.addAll(collisionItems("model_family_id", modelFamilyIds));
        items.addAll(collisionItems("provider_class", providerClasses));
        items.addAll(collisionItems("adapter_provider_class", adapterProviderClasses));
        items.addAll(collisionItems("model_family_provider_class", modelFamilyProviderClasses));
        items.addAll(collisionItems("implementation_class", implementationClasses));
        items.addAll(collisionItems("provider_id", providerIds));
        items.sort((left, right) -> {
            int kind = left.kind().compareTo(right.kind());
            return kind != 0 ? kind : left.value().compareTo(right.value());
        });
        return new FeatureCollisionReport(List.copyOf(items));
    }

    private static void addCollisionClaim(
            Map<String, List<FeatureCollisionClaim>> target,
            String value,
            String path,
            String source) {
        if (value == null || value.isBlank()) {
            return;
        }
        String key = value.trim();
        FeatureCollisionClaim claim = new FeatureCollisionClaim(
                path == null ? "" : path,
                source == null || source.isBlank() ? "unknown" : source);
        List<FeatureCollisionClaim> claims = target.computeIfAbsent(key, ignored -> new ArrayList<>());
        boolean exists = false;
        for (FeatureCollisionClaim existing : claims) {
            if (existing.path().equals(claim.path()) && existing.source().equals(claim.source())) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            claims.add(claim);
        }
    }

    private static List<FeatureCollisionItem> collisionItems(
            String kind,
            Map<String, List<FeatureCollisionClaim>> claimsByValue) {
        List<FeatureCollisionItem> items = new ArrayList<>();
        for (Map.Entry<String, List<FeatureCollisionClaim>> entry : claimsByValue.entrySet()) {
            Set<String> paths = new LinkedHashSet<>();
            for (FeatureCollisionClaim claim : entry.getValue()) {
                paths.add(claim.path());
            }
            if (paths.size() > 1) {
                items.add(new FeatureCollisionItem(kind, entry.getKey(), List.copyOf(entry.getValue())));
            }
        }
        return items;
    }

    private static FeatureRepairReport repairInstallLocks(List<Path> roots, boolean dryRun) {
        Map<Path, FeatureRepairPlan> plans = new LinkedHashMap<>();
        List<FeatureRepairOperation> operations = new ArrayList<>();
        for (Path root : roots) {
            Path normalizedRoot = normalize(root);
            if (Files.isRegularFile(normalizedRoot) && isJar(normalizedRoot)) {
                Path parent = normalizedRoot.getParent();
                Path featureDir = parent == null ? normalize(Path.of(".")) : normalize(parent);
                plans.computeIfAbsent(featureDir, FeatureRepairPlan::new).jars.add(normalizedRoot);
            } else if (Files.isDirectory(normalizedRoot)) {
                FeatureRepairPlan plan = plans.computeIfAbsent(normalizedRoot, FeatureRepairPlan::new);
                plan.fullDirectory = true;
                try (var stream = Files.list(normalizedRoot)) {
                    stream.filter(path -> Files.isRegularFile(path) && isJar(path))
                            .map(FeaturesCommand::normalize)
                            .sorted()
                            .forEach(plan.jars::add);
                } catch (Exception e) {
                    operations.add(FeatureRepairOperation.error(normalizedRoot, normalizedRoot,
                            "scan_failed", "Failed to scan extension directory: " + e.getMessage()));
                }
            } else {
                operations.add(new FeatureRepairOperation(
                        normalizedRoot.toString(),
                        normalizedRoot.toString(),
                        "skip_root",
                        "skipped",
                        "Extension root is not a readable jar or directory"));
            }
        }

        for (FeatureRepairPlan plan : plans.values()) {
            operations.addAll(repairInstallLockPlan(plan, dryRun));
        }
        return new FeatureRepairReport(dryRun, operations);
    }

    private static List<FeatureRepairOperation> repairInstallLockPlan(FeatureRepairPlan plan, boolean dryRun) {
        List<FeatureRepairOperation> operations = new ArrayList<>();
        FeatureInstallLockState state = readInstallLock(plan.root);
        Map<String, Map<String, Object>> repaired = new LinkedHashMap<>(state.features());
        Set<String> seenJars = new LinkedHashSet<>();
        boolean changed = false;
        if (state.error() != null && !state.error().isBlank()) {
            changed = true;
            operations.add(new FeatureRepairOperation(
                    plan.root.toString(),
                    installLockPath(plan.root).toString(),
                    "lock_recreate",
                    dryRun ? "would_change" : "changed",
                    state.error()));
        }

        for (Path jar : plan.jars.stream().sorted().toList()) {
            String filename = jar.getFileName() == null ? jar.toString() : jar.getFileName().toString();
            seenJars.add(filename);
            FeatureJarReport report = inspectJar(jar);
            if (report.error() != null && !report.error().isBlank()) {
                operations.add(new FeatureRepairOperation(
                        plan.root.toString(),
                        jar.toString(),
                        "skip_unreadable",
                        "skipped",
                        "Extension jar is not readable: " + report.error()));
                continue;
            }
            Map<String, Object> current = installLockEntry(jar, report);
            Map<String, Object> previous = repaired.get(filename);
            if (previous == null || previous.isEmpty()) {
                repaired.put(filename, current);
                changed = true;
                operations.add(new FeatureRepairOperation(
                        plan.root.toString(),
                        jar.toString(),
                        "lock_add",
                        dryRun ? "would_change" : "changed",
                        "Extension jar was not tracked in install lock"));
            } else if (!sameInstallLockEntry(previous, current)) {
                Map<String, Object> next = repairedInstallLockEntry(jar, previous, current);
                repaired.put(filename, next);
                changed = true;
                boolean tokenizerOnly = sameInstallLockEntryExceptTokenizerMetadata(previous, current);
                operations.add(new FeatureRepairOperation(
                        plan.root.toString(),
                        jar.toString(),
                        tokenizerOnly ? "lock_refresh_tokenizer_metadata" : "lock_refresh",
                        dryRun ? "would_change" : "changed",
                        tokenizerOnly
                                ? "Extension jar tokenizer metadata was missing or stale in install lock"
                                : "Extension jar metadata differs from install lock"));
            } else {
                Map<String, Object> next = repairedInstallLockEntry(jar, previous, previous);
                if (!previous.equals(next)) {
                    repaired.put(filename, next);
                    changed = true;
                    operations.add(new FeatureRepairOperation(
                            plan.root.toString(),
                            jar.toString(),
                            "lock_enrich_package",
                            dryRun ? "would_change" : "changed",
                            "Extension install lock source package provenance was enriched"));
                } else {
                    operations.add(new FeatureRepairOperation(
                            plan.root.toString(),
                            jar.toString(),
                            "lock_keep",
                            "unchanged",
                            "Extension jar already matches install lock"));
                }
            }
        }

        if (plan.fullDirectory) {
            for (String filename : new ArrayList<>(repaired.keySet())) {
                if (!seenJars.contains(filename)) {
                    repaired.remove(filename);
                    changed = true;
                    operations.add(new FeatureRepairOperation(
                            plan.root.toString(),
                            normalize(plan.root.resolve(filename)).toString(),
                            "lock_remove_stale",
                            dryRun ? "would_change" : "changed",
                            "Install lock referenced a jar that is no longer present"));
                }
            }
        }

        if (changed && !dryRun) {
            try {
                writeInstallLock(plan.root, repaired);
            } catch (Exception e) {
                operations.add(FeatureRepairOperation.error(plan.root, installLockPath(plan.root),
                        "write_failed", "Failed to write install lock: " + e.getMessage()));
            }
        }
        return operations;
    }

    private static Map<String, Object> repairedInstallLockEntry(
            Path jarPath,
            Map<String, Object> previous,
            Map<String, Object> base) {
        Map<String, Object> next = new LinkedHashMap<>(base == null ? Map.of() : base);
        if (previous == null || !(previous.get("source_package") instanceof Map<?, ?> rawPackage)) {
            return next;
        }
        Map<String, Object> sourcePackage = enrichSourcePackageLock(rawPackage);
        if (sourcePackage.isEmpty()) {
            return next;
        }
        String currentJarHash = stringValue(next.get("sha256"));
        String previousJarHash = stringValue(previous.get("sha256"));
        String packageArtifactHash = stringValue(sourcePackage.get("artifact_sha256"));
        boolean sameJar = !currentJarHash.isBlank() && currentJarHash.equalsIgnoreCase(previousJarHash);
        boolean packageMatchesJar = !currentJarHash.isBlank()
                && !packageArtifactHash.isBlank()
                && currentJarHash.equalsIgnoreCase(packageArtifactHash);
        if (sameJar || packageMatchesJar) {
            next.put("source_package", sourcePackage);
        }
        return next;
    }

    private static Map<String, Object> enrichSourcePackageLock(Map<?, ?> rawPackage) {
        Map<String, Object> sourcePackage = objectMap(rawPackage);
        String packagePathText = stringValue(sourcePackage.get("path"));
        if (packagePathText.isBlank()) {
            return sourcePackage;
        }
        Path packagePath;
        try {
            packagePath = normalize(Path.of(packagePathText));
        } catch (InvalidPathException e) {
            return sourcePackage;
        }
        if (!Files.isRegularFile(packagePath)) {
            return sourcePackage;
        }
        putIfMissing(sourcePackage, "size_bytes", () -> Files.size(packagePath));
        putIfMissing(sourcePackage, "sha256", () -> sha256(packagePath));

        try (ZipFile zip = new ZipFile(packagePath.toFile())) {
            ZipEntry manifestEntry = sourcePackageManifestEntry(zip, sourcePackage);
            if (manifestEntry == null) {
                return sourcePackage;
            }
            putIfMissing(sourcePackage, "manifest_entry", manifestEntry::getName);
            Map<String, Object> packageManifest = readZipJsonObject(zip, manifestEntry);
            putManifestIfMissing(sourcePackage, packageManifest, "id");
            putManifestIfMissing(sourcePackage, packageManifest, "name");
            putManifestIfMissing(sourcePackage, packageManifest, "version");
            putManifestIfMissing(sourcePackage, packageManifest, "kind");
            if (!sourcePackage.containsKey("requires") && packageManifest.get("requires") instanceof Map<?, ?> rawRequires) {
                sourcePackage.put("requires", objectMap(rawRequires));
            }
            ZipEntry artifactEntry = sourcePackageArtifactEntry(zip, manifestEntry, packageManifest, sourcePackage);
            if (artifactEntry != null) {
                putIfMissing(sourcePackage, "artifact_entry", artifactEntry::getName);
                putIfMissing(sourcePackage, "artifact_size_bytes", artifactEntry::getSize);
                putIfMissing(sourcePackage, "artifact_sha256", () -> sha256(zip, artifactEntry));
            }
        } catch (Exception ignored) {
            return sourcePackage;
        }
        return sourcePackage;
    }

    private static ZipEntry sourcePackageManifestEntry(ZipFile zip, Map<String, Object> sourcePackage) {
        String entryName = stringValue(sourcePackage.get("manifest_entry"));
        if (!entryName.isBlank()) {
            ZipEntry entry = zip.getEntry(entryName);
            if (entry != null && !entry.isDirectory()) {
                return entry;
            }
        }
        try {
            return findPackageManifestEntry(zip);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static ZipEntry sourcePackageArtifactEntry(
            ZipFile zip,
            ZipEntry manifestEntry,
            Map<String, Object> packageManifest,
            Map<String, Object> sourcePackage) {
        String entryName = stringValue(sourcePackage.get("artifact_entry"));
        if (!entryName.isBlank()) {
            ZipEntry entry = zip.getEntry(entryName);
            if (entry != null && !entry.isDirectory()) {
                return entry;
            }
        }
        try {
            ZipEntry entry = zip.getEntry(safeZipEntryName(
                    zipEntryRoot(manifestEntry.getName()),
                    packageJarArtifactPath(packageManifest)));
            return entry == null || entry.isDirectory() ? null : entry;
        } catch (Exception ignored) {
            return null;
        }
    }

    private static void putManifestIfMissing(
            Map<String, Object> target,
            Map<String, Object> manifest,
            String key) {
        String value = manifestValue(manifest, key);
        if (!value.isBlank() && !target.containsKey(key)) {
            target.put(key, value);
        }
    }

    private static void putIfMissing(
            Map<String, Object> target,
            String key,
            CheckedSupplier<?> supplier) {
        if (target.containsKey(key)) {
            return;
        }
        try {
            Object value = supplier.get();
            if (value != null) {
                target.put(key, value);
            }
        } catch (Exception ignored) {
            // Repair should not fail just because optional provenance could not be enriched.
        }
    }

    @FunctionalInterface
    private interface CheckedSupplier<T> {
        T get() throws Exception;
    }

    private static boolean sameInstallLockEntry(Map<String, Object> previous, Map<String, Object> current) {
        if (previous == null || current == null) {
            return false;
        }
        return stringValue(previous.get("sha256")).equalsIgnoreCase(stringValue(current.get("sha256")))
                && longValue(previous.get("size_bytes"), -1L) == longValue(current.get("size_bytes"), -2L)
                && stringValue(previous.get("path")).equals(stringValue(current.get("path")))
                && manifestStringList(previous.get("providers")).equals(manifestStringList(current.get("providers")))
                && manifestStringList(previous.get("adapter_providers")).equals(manifestStringList(current.get("adapter_providers")))
                && manifestStringList(previous.get("model_family_providers")).equals(manifestStringList(current.get("model_family_providers")))
                && manifestStringList(previous.get("gollek_plugin_providers")).equals(manifestStringList(current.get("gollek_plugin_providers")))
                && stringValue(previous.get("plugin_id")).equals(stringValue(current.get("plugin_id")))
                && stringValue(previous.get("plugin_name")).equals(stringValue(current.get("plugin_name")))
                && stringValue(previous.get("plugin_version")).equals(stringValue(current.get("plugin_version")))
                && stringValue(previous.get("plugin_main_class")).equals(stringValue(current.get("plugin_main_class")))
                && stringValue(previous.get("plugin_tokenizer_kind")).equals(stringValue(current.get("plugin_tokenizer_kind")))
                && manifestStringList(previous.get("plugin_tokenizer_kinds")).equals(manifestStringList(current.get("plugin_tokenizer_kinds")))
                && stringValue(previous.get("plugin_tokenizer_metadata_status")).equals(stringValue(current.get("plugin_tokenizer_metadata_status")))
                && stringValue(previous.get("plugin_tokenizer_metadata_pending_reason")).equals(stringValue(current.get("plugin_tokenizer_metadata_pending_reason")))
                && stringValue(previous.get("feature_id")).equals(stringValue(current.get("feature_id")))
                && stringValue(previous.get("feature_name")).equals(stringValue(current.get("feature_name")))
                && stringValue(previous.get("feature_version")).equals(stringValue(current.get("feature_version")))
                && manifestStringList(previous.get("pipelines")).equals(manifestStringList(current.get("pipelines")))
                && manifestStringList(previous.get("model_family_ids")).equals(manifestStringList(current.get("model_family_ids")))
                && manifestStringList(previous.get("model_family_tokenizer_kinds")).equals(manifestStringList(current.get("model_family_tokenizer_kinds")))
                && manifestStringList(previous.get("model_family_tokenizer_metadata_statuses")).equals(manifestStringList(current.get("model_family_tokenizer_metadata_statuses")))
                && objectMap(previous.get("model_family_tokenizer_metadata_pending_reasons")).equals(objectMap(current.get("model_family_tokenizer_metadata_pending_reasons")))
                && stringValue(previous.get("tokenizer_kind")).equals(stringValue(current.get("tokenizer_kind")))
                && manifestStringList(previous.get("tokenizer_kinds")).equals(manifestStringList(current.get("tokenizer_kinds")))
                && stringValue(previous.get("tokenizer_metadata_status")).equals(stringValue(current.get("tokenizer_metadata_status")))
                && stringValue(previous.get("tokenizer_metadata_pending_reason")).equals(stringValue(current.get("tokenizer_metadata_pending_reason")))
                && manifestStringList(previous.get("capabilities")).equals(manifestStringList(current.get("capabilities")))
                && manifestStringList(previous.get("capability_kinds")).equals(manifestStringList(current.get("capability_kinds")));
    }

    private static boolean sameInstallLockEntryExceptTokenizerMetadata(
            Map<String, Object> previous,
            Map<String, Object> current) {
        if (previous == null || current == null) {
            return false;
        }
        return stringValue(previous.get("sha256")).equalsIgnoreCase(stringValue(current.get("sha256")))
                && longValue(previous.get("size_bytes"), -1L) == longValue(current.get("size_bytes"), -2L)
                && stringValue(previous.get("path")).equals(stringValue(current.get("path")))
                && manifestStringList(previous.get("providers")).equals(manifestStringList(current.get("providers")))
                && manifestStringList(previous.get("adapter_providers")).equals(manifestStringList(current.get("adapter_providers")))
                && manifestStringList(previous.get("model_family_providers")).equals(manifestStringList(current.get("model_family_providers")))
                && manifestStringList(previous.get("gollek_plugin_providers")).equals(manifestStringList(current.get("gollek_plugin_providers")))
                && stringValue(previous.get("plugin_id")).equals(stringValue(current.get("plugin_id")))
                && stringValue(previous.get("plugin_name")).equals(stringValue(current.get("plugin_name")))
                && stringValue(previous.get("plugin_version")).equals(stringValue(current.get("plugin_version")))
                && stringValue(previous.get("plugin_main_class")).equals(stringValue(current.get("plugin_main_class")))
                && stringValue(previous.get("feature_id")).equals(stringValue(current.get("feature_id")))
                && stringValue(previous.get("feature_name")).equals(stringValue(current.get("feature_name")))
                && stringValue(previous.get("feature_version")).equals(stringValue(current.get("feature_version")))
                && manifestStringList(previous.get("pipelines")).equals(manifestStringList(current.get("pipelines")))
                && manifestStringList(previous.get("model_family_ids")).equals(manifestStringList(current.get("model_family_ids")))
                && manifestStringList(previous.get("capabilities")).equals(manifestStringList(current.get("capabilities")))
                && manifestStringList(previous.get("capability_kinds")).equals(manifestStringList(current.get("capability_kinds")));
    }

    private static List<FeatureRow> scannedFeatureRows(FeatureDoctorReport report) {
        if (report == null || report.jars().isEmpty()) {
            return List.of();
        }
        List<FeatureRow> rows = new ArrayList<>();
        for (FeatureJarReport jar : report.jars()) {
            if (!jar.hasManifest()) {
                continue;
            }
            FeatureJarValidation validation = validateJar(jar, false);
            Map<String, Object> manifest = jar.manifest();
            String featureId = manifestValue(manifest, "id");
            String featureName = manifestValue(manifest, "name");
            String version = manifestValue(manifest, "version");
            for (Map<String, Object> capability : manifestCapabilities(manifest)) {
                String kind = manifestCapabilityKind(capability);
                String id = manifestValue(capability, "id");
                if (id.isBlank()) {
                    id = featureId;
                }
                if (id.isBlank()) {
                    continue;
                }
                String name = firstNonBlank(manifestValue(capability, "name"), featureName, id);
                String family = firstNonBlank(manifestValue(capability, "family"), kind);
                Map<String, Object> metadata = new LinkedHashMap<>();
                metadata.put("source", "scan");
                metadata.put("validation", validation.status());
                metadata.put("feature_id", featureId);
                metadata.put("feature_name", featureName);
                metadata.put("feature_version", version);
                metadata.put("kind", kind);
                metadata.put("manifest_source", manifestValue(capability, "_manifest_source"));
                metadata.put("jar", jar.path());
                putManifestValue(metadata, capability, "provider");
                putManifestValue(metadata, capability, "format");
                putManifestValue(metadata, capability, "formats");
                putManifestValue(metadata, capability, "targets");
                rows.add(new FeatureRow(
                        id,
                        name,
                        version,
                        kind,
                        family,
                        manifestStringList(capability.get("inputs")),
                        manifestStringList(capability.get("outputs")),
                        manifestStringList(capability.get("tags")),
                        metadata,
                        0,
                        "scan-" + validation.status(),
                        manifestValue(capability, "class"),
                        jar.path()));
            }
        }
        return List.copyOf(rows);
    }

    private static List<Path> installedJars(Path root) throws java.io.IOException {
        try (var stream = Files.list(root)) {
            return stream.filter(path -> Files.isRegularFile(path) && isJar(path))
                    .map(FeaturesCommand::normalize)
                    .sorted()
                    .toList();
        }
    }

    private static FeatureInstallSource resolveExtensionInstallSource(Path source) throws java.io.IOException {
        Path normalized = normalize(source);
        if (isJar(normalized)) {
            return new FeatureInstallSource(normalized, normalized, null, null);
        }
        if (isZip(normalized)) {
            return resolveExtensionPackageSource(normalized);
        }
        throw new IllegalArgumentException("Extension artifact must be a .jar or .zip package: " + source);
    }

    private static FeatureInstallSource resolveExtensionPackageSource(Path packagePath) throws java.io.IOException {
        Path tempDir = null;
        try (ZipFile zip = new ZipFile(packagePath.toFile())) {
            ZipEntry packageEntry = findPackageManifestEntry(zip);
            Map<String, Object> packageManifest = readZipJsonObject(zip, packageEntry);
            String packageRoot = zipEntryRoot(packageEntry.getName());
            String artifactPath = packageJarArtifactPath(packageManifest);
            String artifactEntryName = safeZipEntryName(packageRoot, artifactPath);
            ZipEntry artifactEntry = zip.getEntry(artifactEntryName);
            if (artifactEntry == null || artifactEntry.isDirectory()) {
                throw new IllegalArgumentException("Extension package jar artifact not found: " + artifactEntryName);
            }
            tempDir = Files.createTempDirectory("gollek-extension-package.");
            String filename = Path.of(artifactPath.replace('\\', '/')).getFileName().toString();
            Path tempJar = normalize(tempDir.resolve(filename));
            try (var input = zip.getInputStream(artifactEntry)) {
                Files.copy(input, tempJar, StandardCopyOption.REPLACE_EXISTING);
            }
            FeaturePackageReport packageReport = new FeaturePackageReport(
                    packagePath.toString(),
                    packageManifest,
                    packageEntry.getName(),
                    artifactEntryName,
                    filename,
                    Files.size(packagePath),
                    sha256(packagePath),
                    Files.size(tempJar),
                    sha256(tempJar));
            return new FeatureInstallSource(packagePath, tempJar, packageReport, tempDir);
        } catch (Exception e) {
            if (tempDir != null) {
                deleteRecursively(tempDir);
            }
            if (e instanceof java.io.IOException io) {
                throw io;
            }
            if (e instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    private static ZipEntry findPackageManifestEntry(ZipFile zip) {
        List<? extends ZipEntry> entries = zip.stream()
                .filter(entry -> entry != null && !entry.isDirectory())
                .filter(entry -> {
                    String name = entry.getName();
                    return PACKAGE_MANIFEST_ENTRY.equals(name) || name.endsWith("/" + PACKAGE_MANIFEST_ENTRY);
                })
                .toList();
        if (entries.isEmpty()) {
            throw new IllegalArgumentException("Extension package is missing " + PACKAGE_MANIFEST_ENTRY);
        }
        if (entries.size() > 1) {
            throw new IllegalArgumentException("Extension package contains multiple " + PACKAGE_MANIFEST_ENTRY + " files");
        }
        return entries.get(0);
    }

    private static Map<String, Object> readZipJsonObject(ZipFile zip, ZipEntry entry) throws java.io.IOException {
        try (var input = zip.getInputStream(entry)) {
            @SuppressWarnings("unchecked")
            Map<String, Object> value = new ObjectMapper().readValue(input, Map.class);
            return value == null ? Map.of() : value;
        }
    }

    private static String packageJarArtifactPath(Map<String, Object> packageManifest) {
        String path = manifestValue(packageJarArtifact(packageManifest), "path").replace('\\', '/');
        if (!path.isBlank()) {
            return path;
        }
        throw new IllegalArgumentException("Extension package has no java-jar artifact in " + PACKAGE_MANIFEST_ENTRY);
    }

    private static Map<String, Object> packageJarArtifact(Map<String, Object> packageManifest) {
        for (Map<String, Object> artifact : manifestObjectList(packageManifest, "artifacts")) {
            String type = manifestValue(artifact, "type");
            String path = manifestValue(artifact, "path").replace('\\', '/');
            if (!path.isBlank() && path.endsWith(".jar")
                    && (type.isBlank() || "java-jar".equals(type) || "jar".equals(type))) {
                return artifact;
            }
        }
        return Map.of();
    }

    private static String zipEntryRoot(String entryName) {
        int slash = entryName == null ? -1 : entryName.lastIndexOf('/');
        return slash < 0 ? "" : entryName.substring(0, slash + 1);
    }

    private static String safeZipEntryName(String root, String relative) {
        String cleanRoot = root == null ? "" : root;
        String cleanRelative = relative == null ? "" : relative.replace('\\', '/');
        if (cleanRelative.isBlank() || cleanRelative.startsWith("/") || cleanRelative.contains("\0")) {
            throw new IllegalArgumentException("Invalid extension package artifact path: " + relative);
        }
        for (String part : cleanRelative.split("/")) {
            if (part.isBlank() || ".".equals(part) || "..".equals(part)) {
                throw new IllegalArgumentException("Invalid extension package artifact path: " + relative);
            }
        }
        return cleanRoot + cleanRelative;
    }

    private static void deleteRecursively(Path root) throws java.io.IOException {
        if (root == null || !Files.exists(root)) {
            return;
        }
        try (var stream = Files.walk(root)) {
            for (Path path : stream.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }

    private static FeatureJarReport inspectJar(Path jarPath) {
        try (JarFile jar = new JarFile(jarPath.toFile())) {
            ServiceEntry serviceEntry = readServiceEntry(jar, SERVICE_ENTRY);
            ServiceEntry adapterServiceEntry = readServiceEntry(jar, ADAPTER_SERVICE_ENTRY);
            ServiceEntry modelFamilyServiceEntry = readServiceEntry(jar, MODEL_FAMILY_SERVICE_ENTRY);
            ServiceEntry gollekPluginServiceEntry = readServiceEntry(jar, GOLLEK_PLUGIN_SERVICE_ENTRY);
            ManifestEntry manifestEntry = readManifestEntry(jar);
            ManifestEntry pluginDescriptorEntry = readManifestEntry(jar, PLUGIN_DESCRIPTOR_ENTRY);
            Map<String, Boolean> classPresence = classPresence(
                    jar,
                    serviceEntry.providers(),
                    adapterServiceEntry.providers(),
                    modelFamilyServiceEntry.providers(),
                    gollekPluginServiceEntry.providers(),
                    manifestEntry.manifest(),
                    pluginDescriptorEntry.manifest());
            return new FeatureJarReport(
                    jarPath.toString(),
                    serviceEntry.present(),
                    serviceEntry.providers(),
                    adapterServiceEntry.present(),
                    adapterServiceEntry.providers(),
                    modelFamilyServiceEntry.present(),
                    modelFamilyServiceEntry.providers(),
                    gollekPluginServiceEntry.present(),
                    gollekPluginServiceEntry.providers(),
                    pluginDescriptorEntry.manifest(),
                    pluginDescriptorEntry.entryName(),
                    pluginDescriptorEntry.error(),
                    manifestEntry.manifest(),
                    manifestEntry.entryName(),
                    manifestEntry.error(),
                    classPresence,
                    Map.of(),
                    List.of(),
                    null);
        } catch (Exception e) {
            return new FeatureJarReport(jarPath.toString(), false, List.of(), false, List.of(), false, List.of(),
                    false, List.of(), Map.of(), "", null, Map.of(), "", null,
                    Map.of(), Map.of(), List.of(), e.getMessage());
        }
    }

    private static Path installLockPath(Path featureDir) {
        Path root = featureDir == null ? Path.of(".") : featureDir;
        return normalize(root.resolve(INSTALL_LOCKFILE));
    }

    private static Path legacyInstallLockPath(Path featureDir) {
        Path root = featureDir == null ? Path.of(".") : featureDir;
        return normalize(root.resolve(LEGACY_INSTALL_LOCKFILE));
    }

    private static Path installLockReadPath(Path featureDir) {
        Path preferred = installLockPath(featureDir);
        if (Files.isRegularFile(preferred)) {
            return preferred;
        }
        Path legacy = legacyInstallLockPath(featureDir);
        if (Files.isRegularFile(legacy)) {
            return legacy;
        }
        return preferred;
    }

    private static FeatureInstallLockState readInstallLock(Path featureDir) {
        if (featureDir == null) {
            return new FeatureInstallLockState(Map.of(), "");
        }
        Path lockfile = installLockReadPath(featureDir);
        if (!Files.isRegularFile(lockfile)) {
            return new FeatureInstallLockState(Map.of(), "");
        }
        try (var input = Files.newInputStream(lockfile)) {
            @SuppressWarnings("unchecked")
            Map<String, Object> root = new ObjectMapper().readValue(input, Map.class);
            Object rawFeatures = root == null ? null : firstNonNull(root.get("extensions"), root.get("features"));
            if (!(rawFeatures instanceof Map<?, ?> rawMap)) {
                return new FeatureInstallLockState(Map.of(), "Extension install lock has no 'extensions' object: " + lockfile);
            }
            Map<String, Map<String, Object>> features = new LinkedHashMap<>();
            rawMap.forEach((key, value) -> {
                if (key != null && value instanceof Map<?, ?> rawEntry) {
                    Map<String, Object> entry = new LinkedHashMap<>();
                    rawEntry.forEach((entryKey, entryValue) -> {
                        if (entryKey != null) {
                            entry.put(String.valueOf(entryKey), entryValue);
                        }
                    });
                    features.put(String.valueOf(key), entry);
                }
            });
            return new FeatureInstallLockState(Map.copyOf(features), "");
        } catch (Exception e) {
            return new FeatureInstallLockState(Map.of(), "Extension install lock is not readable: " + e.getMessage());
        }
    }

    private static FeatureJarReport applyInstallLock(
            Path featureDir,
            FeatureJarReport report,
            FeatureInstallLockState lockState) {
        if (report == null || featureDir == null || lockState == null) {
            return report;
        }
        Path jarPath = normalize(Path.of(report.path()));
        Path name = jarPath.getFileName();
        List<String> warnings = new ArrayList<>();
        if (lockState.error() != null && !lockState.error().isBlank()) {
            warnings.add(lockState.error());
        }
        Map<String, Object> entry = name == null ? null : lockState.features().get(name.toString());
        if (entry == null || entry.isEmpty()) {
            return report.withInstallLock(Map.of(), warnings);
        }
        String expectedHash = String.valueOf(entry.getOrDefault("sha256", "")).trim();
        if (!expectedHash.isBlank()) {
            try {
                String actualHash = sha256(jarPath);
                if (!expectedHash.equalsIgnoreCase(actualHash)) {
                    warnings.add("Extension jar checksum differs from install lock: expected "
                            + shortHash(expectedHash) + ", actual " + shortHash(actualHash));
                }
            } catch (Exception e) {
                warnings.add("Extension jar checksum could not be verified: " + e.getMessage());
            }
        }
        long expectedSize = longValue(entry.get("size_bytes"), -1L);
        if (expectedSize >= 0) {
            try {
                long actualSize = Files.size(jarPath);
                if (expectedSize != actualSize) {
                    warnings.add("Extension jar size differs from install lock: expected "
                            + expectedSize + " bytes, actual " + actualSize + " bytes");
                }
            } catch (Exception e) {
                warnings.add("Extension jar size could not be verified: " + e.getMessage());
            }
        }
        String lockedPath = String.valueOf(entry.getOrDefault("path", "")).trim();
        if (!lockedPath.isBlank()) {
            try {
                Path expectedPath = normalize(Path.of(lockedPath));
                if (!expectedPath.equals(jarPath)) {
                    warnings.add("Extension jar path differs from install lock: expected "
                            + expectedPath + ", actual " + jarPath);
                }
            } catch (InvalidPathException e) {
                warnings.add("Extension install lock path is invalid: " + lockedPath);
            }
        }
        appendSourcePackageWarnings(entry, warnings);
        appendTokenizerInstallLockWarnings(entry, installLockEntry(jarPath, report), warnings);
        return report.withInstallLock(entry, warnings);
    }

    private static void appendTokenizerInstallLockWarnings(
            Map<String, Object> lockEntry,
            Map<String, Object> currentEntry,
            List<String> warnings) {
        if (!hasTokenizerMetadata(currentEntry)) {
            return;
        }
        if (!hasTokenizerMetadata(lockEntry)) {
            warnings.add("Extension install lock is missing tokenizer metadata; run gollek extensions doctor --repair");
            return;
        }
        if (!tokenizerMetadataMatches(lockEntry, currentEntry)) {
            warnings.add("Extension install lock tokenizer metadata differs from jar metadata; run gollek extensions doctor --repair");
        }
    }

    private static boolean hasTokenizerMetadata(Map<String, Object> entry) {
        return entry != null
                && (!stringValue(entry.get("plugin_tokenizer_kind")).isBlank()
                || !manifestStringList(entry.get("plugin_tokenizer_kinds")).isEmpty()
                || !stringValue(entry.get("plugin_tokenizer_metadata_status")).isBlank()
                || !stringValue(entry.get("plugin_tokenizer_metadata_pending_reason")).isBlank()
                || !manifestStringList(entry.get("model_family_tokenizer_kinds")).isEmpty()
                || !manifestStringList(entry.get("model_family_tokenizer_metadata_statuses")).isEmpty()
                || !objectMap(entry.get("model_family_tokenizer_metadata_pending_reasons")).isEmpty()
                || !stringValue(entry.get("tokenizer_kind")).isBlank()
                || !manifestStringList(entry.get("tokenizer_kinds")).isEmpty()
                || !stringValue(entry.get("tokenizer_metadata_status")).isBlank()
                || !stringValue(entry.get("tokenizer_metadata_pending_reason")).isBlank());
    }

    private static boolean tokenizerMetadataMatches(Map<String, Object> left, Map<String, Object> right) {
        return stringValue(left.get("plugin_tokenizer_kind")).equals(stringValue(right.get("plugin_tokenizer_kind")))
                && manifestStringList(left.get("plugin_tokenizer_kinds")).equals(manifestStringList(right.get("plugin_tokenizer_kinds")))
                && stringValue(left.get("plugin_tokenizer_metadata_status")).equals(stringValue(right.get("plugin_tokenizer_metadata_status")))
                && stringValue(left.get("plugin_tokenizer_metadata_pending_reason")).equals(stringValue(right.get("plugin_tokenizer_metadata_pending_reason")))
                && manifestStringList(left.get("model_family_tokenizer_kinds")).equals(manifestStringList(right.get("model_family_tokenizer_kinds")))
                && manifestStringList(left.get("model_family_tokenizer_metadata_statuses")).equals(manifestStringList(right.get("model_family_tokenizer_metadata_statuses")))
                && objectMap(left.get("model_family_tokenizer_metadata_pending_reasons")).equals(objectMap(right.get("model_family_tokenizer_metadata_pending_reasons")))
                && stringValue(left.get("tokenizer_kind")).equals(stringValue(right.get("tokenizer_kind")))
                && manifestStringList(left.get("tokenizer_kinds")).equals(manifestStringList(right.get("tokenizer_kinds")))
                && stringValue(left.get("tokenizer_metadata_status")).equals(stringValue(right.get("tokenizer_metadata_status")))
                && stringValue(left.get("tokenizer_metadata_pending_reason")).equals(stringValue(right.get("tokenizer_metadata_pending_reason")));
    }

    private static String tokenizerInstallLockStatus(FeatureJarReport jar) {
        if (jar == null || !jar.hasInstallLock()) {
            return "not_applicable";
        }
        Map<String, Object> currentEntry = installLockEntry(normalize(Path.of(jar.path())), jar);
        if (!hasTokenizerMetadata(currentEntry)) {
            return "not_applicable";
        }
        if (!hasTokenizerMetadata(jar.installLock())) {
            return "missing";
        }
        return tokenizerMetadataMatches(jar.installLock(), currentEntry) ? "ok" : "stale";
    }

    private static void appendSourcePackageWarnings(Map<String, Object> lockEntry, List<String> warnings) {
        if (!(lockEntry.get("source_package") instanceof Map<?, ?> rawPackage)) {
            return;
        }
        String packagePathText = stringValue(rawPackage.get("path")).trim();
        if (packagePathText.isBlank()) {
            return;
        }
        Path packagePath;
        try {
            packagePath = normalize(Path.of(packagePathText));
        } catch (InvalidPathException e) {
            warnings.add("Extension source package path from install lock is invalid: " + packagePathText);
            return;
        }
        if (!Files.exists(packagePath)) {
            return;
        }
        if (!Files.isRegularFile(packagePath)) {
            warnings.add("Extension source package path from install lock is not a file: " + packagePath);
            return;
        }
        String expectedHash = stringValue(rawPackage.get("sha256")).trim();
        if (!expectedHash.isBlank()) {
            try {
                String actualHash = sha256(packagePath);
                if (!expectedHash.equalsIgnoreCase(actualHash)) {
                    warnings.add("Extension source package checksum differs from install lock: expected "
                            + shortHash(expectedHash) + ", actual " + shortHash(actualHash));
                }
            } catch (Exception e) {
                warnings.add("Extension source package checksum could not be verified: " + e.getMessage());
            }
        }
        long expectedSize = longValue(rawPackage.get("size_bytes"), -1L);
        if (expectedSize >= 0) {
            try {
                long actualSize = Files.size(packagePath);
                if (expectedSize != actualSize) {
                    warnings.add("Extension source package size differs from install lock: expected "
                            + expectedSize + " bytes, actual " + actualSize + " bytes");
                }
            } catch (Exception e) {
                warnings.add("Extension source package size could not be verified: " + e.getMessage());
            }
        }
        appendSourcePackageArtifactLockWarnings(lockEntry, rawPackage, warnings);
        appendSourcePackageStructureWarnings(packagePath, rawPackage, warnings);
    }

    private static void appendSourcePackageArtifactLockWarnings(
            Map<String, Object> lockEntry,
            Map<?, ?> rawPackage,
            List<String> warnings) {
        String expectedHash = stringValue(rawPackage.get("artifact_sha256"));
        String installedHash = stringValue(lockEntry.get("sha256"));
        if (!expectedHash.isBlank() && !installedHash.isBlank() && !expectedHash.equalsIgnoreCase(installedHash)) {
            warnings.add("Extension source package artifact checksum differs from installed jar lock: expected "
                    + shortHash(expectedHash) + ", installed " + shortHash(installedHash));
        }
        long expectedSize = longValue(rawPackage.get("artifact_size_bytes"), -1L);
        long installedSize = longValue(lockEntry.get("size_bytes"), -1L);
        if (expectedSize >= 0 && installedSize >= 0 && expectedSize != installedSize) {
            warnings.add("Extension source package artifact size differs from installed jar lock: expected "
                    + expectedSize + " bytes, installed " + installedSize + " bytes");
        }
    }

    private static void appendSourcePackageStructureWarnings(
            Path packagePath,
            Map<?, ?> rawPackage,
            List<String> warnings) {
        try (ZipFile zip = new ZipFile(packagePath.toFile())) {
            String expectedManifestEntry = stringValue(rawPackage.get("manifest_entry"));
            ZipEntry manifestEntry = null;
            if (!expectedManifestEntry.isBlank()) {
                manifestEntry = zip.getEntry(expectedManifestEntry);
                if (manifestEntry == null || manifestEntry.isDirectory()) {
                    warnings.add("Extension source package descriptor entry is missing: " + expectedManifestEntry);
                    manifestEntry = null;
                }
            }
            if (manifestEntry == null) {
                try {
                    manifestEntry = findPackageManifestEntry(zip);
                } catch (Exception e) {
                    warnings.add("Extension source package descriptor could not be found: " + e.getMessage());
                    return;
                }
            }

            Map<String, Object> packageManifest;
            try {
                packageManifest = readZipJsonObject(zip, manifestEntry);
            } catch (Exception e) {
                warnings.add("Extension source package descriptor could not be read: " + e.getMessage());
                return;
            }
            appendSourcePackageManifestWarning(rawPackage, packageManifest, "id", warnings);
            appendSourcePackageManifestWarning(rawPackage, packageManifest, "name", warnings);
            appendSourcePackageManifestWarning(rawPackage, packageManifest, "version", warnings);
            appendSourcePackageManifestWarning(rawPackage, packageManifest, "kind", warnings);
            appendSourcePackageRequiresWarning(rawPackage, packageManifest, warnings);

            String expectedArtifactEntry = stringValue(rawPackage.get("artifact_entry"));
            if (!expectedArtifactEntry.isBlank()) {
                ZipEntry artifactEntry = zip.getEntry(expectedArtifactEntry);
                if (artifactEntry == null || artifactEntry.isDirectory()) {
                    warnings.add("Extension source package jar artifact entry is missing: " + expectedArtifactEntry);
                } else {
                    appendSourcePackageArtifactEntryWarnings(zip, artifactEntry, rawPackage, warnings);
                }
            } else {
                try {
                    String inferredEntry = safeZipEntryName(
                            zipEntryRoot(manifestEntry.getName()),
                            packageJarArtifactPath(packageManifest));
                    ZipEntry artifactEntry = zip.getEntry(inferredEntry);
                    if (artifactEntry == null || artifactEntry.isDirectory()) {
                        warnings.add("Extension source package jar artifact entry is missing: " + inferredEntry);
                    } else {
                        appendSourcePackageArtifactEntryWarnings(zip, artifactEntry, rawPackage, warnings);
                    }
                } catch (Exception e) {
                    warnings.add("Extension source package jar artifact could not be inferred: " + e.getMessage());
                }
            }
        } catch (Exception e) {
            warnings.add("Extension source package structure could not be verified: " + e.getMessage());
        }
    }

    private static void appendSourcePackageArtifactEntryWarnings(
            ZipFile zip,
            ZipEntry artifactEntry,
            Map<?, ?> rawPackage,
            List<String> warnings) {
        long expectedSize = longValue(rawPackage.get("artifact_size_bytes"), -1L);
        long actualSize = artifactEntry.getSize();
        if (expectedSize >= 0 && actualSize >= 0 && expectedSize != actualSize) {
            warnings.add("Extension source package jar artifact size differs from install lock: expected "
                    + expectedSize + " bytes, actual " + actualSize + " bytes");
        }
        String expectedHash = stringValue(rawPackage.get("artifact_sha256"));
        if (!expectedHash.isBlank()) {
            try {
                String actualHash = sha256(zip, artifactEntry);
                if (!expectedHash.equalsIgnoreCase(actualHash)) {
                    warnings.add("Extension source package jar artifact checksum differs from install lock: expected "
                            + shortHash(expectedHash) + ", actual " + shortHash(actualHash));
                }
            } catch (Exception e) {
                warnings.add("Extension source package jar artifact checksum could not be verified: " + e.getMessage());
            }
        }
    }

    private static void appendSourcePackageManifestWarning(
            Map<?, ?> rawPackage,
            Map<String, Object> packageManifest,
            String key,
            List<String> warnings) {
        String expected = stringValue(rawPackage.get(key));
        if (expected.isBlank()) {
            return;
        }
        String actual = manifestValue(packageManifest, key);
        if (!expected.equals(actual)) {
            warnings.add("Extension source package " + key + " differs from install lock: expected "
                    + expected + ", actual " + firstNonBlank(actual, "<blank>"));
        }
    }

    private static void appendSourcePackageRequiresWarning(
            Map<?, ?> rawPackage,
            Map<String, Object> packageManifest,
            List<String> warnings) {
        Map<String, String> expected = stringMap(rawPackage.get("requires"));
        if (expected.isEmpty()) {
            return;
        }
        Map<String, String> actual = stringMap(packageManifest.get("requires"));
        if (!expected.equals(actual)) {
            warnings.add("Extension source package requires differs from install lock: expected "
                    + expected + ", actual " + actual);
        }
    }

    private static Map<String, String> stringMap(Object value) {
        if (!(value instanceof Map<?, ?> rawMap)) {
            return Map.of();
        }
        Map<String, String> result = new LinkedHashMap<>();
        rawMap.forEach((key, item) -> {
            if (key != null) {
                result.put(String.valueOf(key).trim(), stringValue(item));
            }
        });
        return Map.copyOf(result);
    }

    private static Map<String, Object> objectMap(Object value) {
        if (!(value instanceof Map<?, ?> rawMap)) {
            return Map.of();
        }
        Map<String, Object> result = new LinkedHashMap<>();
        rawMap.forEach((key, item) -> {
            if (key != null) {
                result.put(String.valueOf(key), item);
            }
        });
        return result;
    }

    private static List<FeatureJarReport> missingLockedJars(
            Path featureDir,
            Set<String> seenJars,
            FeatureInstallLockState lockState) {
        if (featureDir == null || lockState == null || lockState.features().isEmpty()) {
            return List.of();
        }
        List<FeatureJarReport> missing = new ArrayList<>();
        for (Map.Entry<String, Map<String, Object>> entry : lockState.features().entrySet()) {
            String filename = entry.getKey();
            if (filename == null || filename.isBlank() || seenJars.contains(filename)) {
                continue;
            }
            Path missingPath = normalize(featureDir.resolve(filename));
            missing.add(new FeatureJarReport(
                    missingPath.toString(),
                    false,
                    List.of(),
                    false,
                    List.of(),
                    false,
                    List.of(),
                    false,
                    List.of(),
                    Map.of(),
                    "",
                    null,
                    Map.of(),
                    "",
                    null,
                    Map.of(),
                    entry.getValue(),
                    List.of("Extension install lock references a missing jar: " + filename),
                    "Extension jar referenced by install lock is missing"));
        }
        return missing;
    }

    private static Map<String, Object> writeInstallLockEntry(
            Path featureDir,
            Path jarPath,
            FeatureJarReport report) throws java.io.IOException {
        return writeInstallLockEntry(featureDir, jarPath, report, null);
    }

    private static Map<String, Object> writeInstallLockEntry(
            Path featureDir,
            Path jarPath,
            FeatureJarReport report,
            FeaturePackageReport packageReport) throws java.io.IOException {
        Path rootDir = normalize(featureDir);
        Path target = normalize(jarPath);
        Files.createDirectories(rootDir);
        FeatureInstallLockState state = readInstallLock(rootDir);
        Map<String, Map<String, Object>> features = new LinkedHashMap<>(state.features());
        String filename = target.getFileName() == null ? target.toString() : target.getFileName().toString();
        Map<String, Object> entry = installLockEntry(target, report, packageReport);
        features.put(filename, entry);
        writeInstallLock(rootDir, features);
        return entry;
    }

    private static void removeInstallLockEntry(Path featureDir, Path jarPath) throws java.io.IOException {
        Path rootDir = normalize(featureDir);
        FeatureInstallLockState state = readInstallLock(rootDir);
        if (state.features().isEmpty()) {
            return;
        }
        Map<String, Map<String, Object>> features = new LinkedHashMap<>(state.features());
        Path name = jarPath.getFileName();
        if (name != null) {
            features.remove(name.toString());
        }
        writeInstallLock(rootDir, features);
    }

    private static FeatureMigrationReport migrateLegacyExtensions(
            Path sourceRoot,
            Path targetRoot,
            boolean force,
            boolean dryRun,
            boolean noBackup) throws java.io.IOException {
        Path source = normalize(sourceRoot);
        Path target = normalize(targetRoot);
        List<FeatureMigrationOperation> operations = new ArrayList<>();
        if (source.equals(target)) {
            operations.add(FeatureMigrationOperation.error(
                    source,
                    target,
                    "same_directory",
                    "Legacy and preferred extension directories are the same path"));
            return new FeatureMigrationReport(source, target, force, dryRun, operations);
        }
        if (!Files.exists(source)) {
            operations.add(FeatureMigrationOperation.skipped(
                    source,
                    target,
                    "source_missing",
                    "Legacy extension directory does not exist"));
            return new FeatureMigrationReport(source, target, force, dryRun, operations);
        }
        if (!Files.isDirectory(source)) {
            operations.add(FeatureMigrationOperation.error(
                    source,
                    target,
                    "source_not_directory",
                    "Legacy extension path is not a directory"));
            return new FeatureMigrationReport(source, target, force, dryRun, operations);
        }
        if (!dryRun) {
            Files.createDirectories(target);
        }
        try (var stream = Files.list(source)) {
            List<Path> jars = stream
                    .filter(path -> Files.isRegularFile(path) && isJar(path))
                    .map(FeaturesCommand::normalize)
                    .sorted()
                    .toList();
            for (Path jar : jars) {
                operations.add(migrateLegacyExtensionJar(source, target, jar, force, dryRun, noBackup));
            }
        }
        return new FeatureMigrationReport(source, target, force, dryRun, operations);
    }

    private static FeatureMigrationOperation migrateLegacyExtensionJar(
            Path sourceRoot,
            Path targetRoot,
            Path sourceJar,
            boolean force,
            boolean dryRun,
            boolean noBackup) {
        try {
            ensureInside(sourceRoot, sourceJar);
            FeatureJarReport sourceReport = inspectJar(sourceJar);
            if (sourceReport.error() != null && !sourceReport.error().isBlank()) {
                return FeatureMigrationOperation.error(
                        sourceJar,
                        targetRoot.resolve(sourceJar.getFileName()),
                        "unreadable",
                        "Extension jar is not readable: " + sourceReport.error(),
                        sourceReport);
            }
            String filename = sourceJar.getFileName() == null ? sourceJar.toString() : sourceJar.getFileName().toString();
            Path target = normalize(targetRoot.resolve(filename));
            ensureInside(targetRoot, target);
            boolean targetExists = Files.exists(target);
            boolean sameContent = targetExists && sameSha256(sourceJar, target);
            if (sameContent) {
                if (!dryRun) {
                    writeInstallLockEntry(targetRoot, target, inspectJar(target));
                }
                return FeatureMigrationOperation.unchanged(
                        sourceJar,
                        target,
                        dryRun,
                        "Target already contains the same extension jar",
                        sourceReport);
            }
            if (targetExists && !force) {
                return FeatureMigrationOperation.skipped(
                        sourceJar,
                        target,
                        "conflict",
                        "Target jar already exists and differs; pass --force to replace",
                        sourceReport);
            }

            FeatureBackupRecord backup = null;
            boolean wouldBackup = targetExists && force && !noBackup;
            if (!dryRun) {
                Files.createDirectories(targetRoot);
                if (wouldBackup) {
                    backup = createFeatureBackup(targetRoot, target, "migrate-replace");
                }
                if (targetExists) {
                    Files.copy(sourceJar, target, StandardCopyOption.REPLACE_EXISTING);
                } else {
                    Files.copy(sourceJar, target);
                }
                writeInstallLockEntry(targetRoot, target, inspectJar(target));
            }
            return FeatureMigrationOperation.copied(
                    sourceJar,
                    target,
                    targetExists,
                    dryRun,
                    backup,
                    dryRun && wouldBackup,
                    dryRun
                            ? (targetExists ? "Target jar would be replaced" : "Extension jar would be copied")
                            : (targetExists ? "Target jar replaced" : "Extension jar copied"),
                    sourceReport);
        } catch (Exception e) {
            Path target = sourceJar.getFileName() == null
                    ? targetRoot.resolve(sourceJar.toString())
                    : targetRoot.resolve(sourceJar.getFileName().toString());
            return FeatureMigrationOperation.error(sourceJar, target, "failed", e.getMessage());
        }
    }

    private static FeatureLegacyPruneReport pruneLegacyExtensions(
            Path sourceRoot,
            Path targetRoot,
            String rawTarget,
            boolean dryRun) throws java.io.IOException {
        return pruneLegacyExtensions(sourceRoot, targetRoot, rawTarget, dryRun, "prune-legacy");
    }

    private static FeatureLegacyPruneReport pruneLegacyExtensions(
            Path sourceRoot,
            Path targetRoot,
            String rawTarget,
            boolean dryRun,
            String backupReason) throws java.io.IOException {
        Path source = normalize(sourceRoot);
        Path target = normalize(targetRoot);
        List<FeatureLegacyPruneOperation> operations = new ArrayList<>();
        if (source.equals(target)) {
            operations.add(FeatureLegacyPruneOperation.error(
                    source,
                    null,
                    "same_directory",
                    "Legacy and preferred extension directories are the same path"));
            return new FeatureLegacyPruneReport(source, target, rawTarget, dryRun, operations);
        }
        if (!Files.exists(source)) {
            operations.add(FeatureLegacyPruneOperation.skipped(
                    source,
                    null,
                    "source_missing",
                    "Legacy extension directory does not exist",
                    null,
                    null,
                    List.of()));
            return new FeatureLegacyPruneReport(source, target, rawTarget, dryRun, operations);
        }
        if (!Files.isDirectory(source)) {
            operations.add(FeatureLegacyPruneOperation.error(
                    source,
                    null,
                    "source_not_directory",
                    "Legacy extension path is not a directory"));
            return new FeatureLegacyPruneReport(source, target, rawTarget, dryRun, operations);
        }
        if (!Files.isDirectory(target)) {
            operations.add(FeatureLegacyPruneOperation.skipped(
                    source,
                    null,
                    "target_missing",
                    "Preferred extension directory does not exist",
                    null,
                    null,
                    List.of()));
            return new FeatureLegacyPruneReport(source, target, rawTarget, dryRun, operations);
        }

        List<FeatureJarReport> preferredJars = installedJars(target).stream()
                .map(FeaturesCommand::inspectJar)
                .filter(report -> report.error() == null || report.error().isBlank())
                .toList();
        try (var stream = Files.list(source)) {
            List<Path> legacyJars = stream
                    .filter(path -> Files.isRegularFile(path) && isJar(path))
                    .map(FeaturesCommand::normalize)
                    .sorted()
                    .toList();
            for (Path jar : legacyJars) {
                operations.add(pruneLegacyExtensionJar(
                        source,
                        target,
                        jar,
                        preferredJars,
                        rawTarget,
                        dryRun,
                        backupReason));
            }
        }
        return new FeatureLegacyPruneReport(source, target, rawTarget, dryRun, operations);
    }

    private static FeatureLegacyPruneReport inspectLegacyMigrationStatus() {
        Path source = null;
        Path target = null;
        try {
            source = legacyInstallDirectory(null);
            target = installDirectory(null);
            return pruneLegacyExtensions(source, target, null, true);
        } catch (Exception e) {
            Path resolvedSource = source == null ? normalize(Path.of(".", ".gollek", "features")) : source;
            Path resolvedTarget = target == null ? normalize(Path.of(".", ".gollek", "extensions")) : target;
            return new FeatureLegacyPruneReport(
                    resolvedSource,
                    resolvedTarget,
                    "",
                    true,
                    List.of(FeatureLegacyPruneOperation.error(
                            resolvedSource,
                            null,
                            "legacy_status_failed",
                            "Legacy migration status could not be inspected: " + e.getMessage())));
        }
    }

    private static FeatureLegacyPruneOperation pruneLegacyExtensionJar(
            Path sourceRoot,
            Path targetRoot,
            Path sourceJar,
            List<FeatureJarReport> preferredJars,
            String rawTarget,
            boolean dryRun,
            String backupReason) {
        try {
            ensureInside(sourceRoot, sourceJar);
            FeatureJarReport sourceReport = inspectJar(sourceJar);
            if (sourceReport.error() != null && !sourceReport.error().isBlank()) {
                return FeatureLegacyPruneOperation.error(
                        sourceJar,
                        null,
                        "unreadable",
                        "Legacy extension jar is not readable: " + sourceReport.error(),
                        sourceReport);
            }
            if (rawTarget != null && !rawTarget.trim().isBlank()
                    && !matchesTarget(sourceRoot, sourceReport, rawTarget)) {
                return FeatureLegacyPruneOperation.skipped(
                        sourceJar,
                        null,
                        "target_filter",
                        "Legacy jar does not match requested target",
                        sourceReport,
                        null,
                        List.of());
            }

            FeatureJarMatch match = bestPreferredMatch(sourceReport, preferredJars);
            if (match == null) {
                return FeatureLegacyPruneOperation.skipped(
                        sourceJar,
                        null,
                        "no_preferred_match",
                        "No preferred extension jar claims the same identity",
                        sourceReport,
                        null,
                        List.of());
            }

            FeatureBackupRecord backup = null;
            if (!dryRun) {
                backup = createFeatureBackup(sourceRoot, sourceJar, firstNonBlank(backupReason, "prune-legacy"));
                Files.delete(sourceJar);
                removeInstallLockEntry(sourceRoot, sourceJar);
            }
            return FeatureLegacyPruneOperation.archived(
                    sourceJar,
                    normalize(Path.of(match.jar().path())),
                    dryRun,
                    backup,
                    sourceReport,
                    match.jar(),
                    match.reasons());
        } catch (Exception e) {
            return FeatureLegacyPruneOperation.error(sourceJar, null, "failed", e.getMessage());
        }
    }

    private static FeatureJarMatch bestPreferredMatch(FeatureJarReport source, List<FeatureJarReport> preferredJars) {
        if (source == null || preferredJars == null || preferredJars.isEmpty()) {
            return null;
        }
        Set<String> sourceClaims = extensionIdentityClaims(source);
        FeatureJarMatch best = null;
        for (FeatureJarReport preferred : preferredJars) {
            Set<String> preferredClaims = extensionIdentityClaims(preferred);
            List<String> reasons = new ArrayList<>();
            for (String claim : sourceClaims) {
                if (preferredClaims.contains(claim)) {
                    reasons.add(claim);
                }
            }
            if (reasons.isEmpty() || !hasStrongPruneReason(reasons)) {
                continue;
            }
            reasons.sort(String::compareTo);
            FeatureJarMatch match = new FeatureJarMatch(preferred, List.copyOf(reasons));
            if (best == null || match.reasons().size() > best.reasons().size()) {
                best = match;
            }
        }
        return best;
    }

    private static boolean hasStrongPruneReason(List<String> reasons) {
        for (String reason : reasons) {
            if (reason == null || reason.isBlank()) {
                continue;
            }
            if (reason.startsWith("sha256:") || !reason.startsWith("filename:")) {
                return true;
            }
        }
        return false;
    }

    private static Set<String> extensionIdentityClaims(FeatureJarReport jar) {
        Set<String> claims = new LinkedHashSet<>();
        if (jar == null) {
            return claims;
        }
        try {
            Path path = normalize(Path.of(jar.path()));
            String filename = path.getFileName() == null ? "" : path.getFileName().toString();
            if (!filename.isBlank()) {
                claims.add("filename:" + filename);
            }
            String hash = sha256(path);
            if (!hash.isBlank()) {
                claims.add("sha256:" + hash);
            }
        } catch (Exception ignored) {
            // Hash and filename matching are best-effort; manifest and provider claims still apply.
        }
        Map<String, Object> manifest = jar.manifest();
        addIdentityClaim(claims, "extension_id", manifestValue(manifest, "id"));
        for (String provider : jar.providers()) {
            addIdentityClaim(claims, "provider_class", provider);
        }
        for (String provider : jar.adapterProviders()) {
            addIdentityClaim(claims, "adapter_provider_class", provider);
        }
        for (Map<String, Object> capability : manifestCapabilities(manifest)) {
            String kind = manifestCapabilityKind(capability);
            String id = manifestValue(capability, "id");
            if (FeatureAdapterKinds.PIPELINE.equals(kind)) {
                addIdentityClaim(claims, "pipeline_id", id);
            } else if (FeatureAdapterKinds.ADAPTER.equals(kind)) {
                addIdentityClaim(claims, "adapter_id", id);
            } else {
                addIdentityClaim(claims, "feature_id", id);
            }
            addIdentityClaim(claims, "feature_id", id);
            addIdentityClaim(claims, "implementation_class", manifestValue(capability, "class"));
            addIdentityClaim(claims, "provider_id", manifestValue(capability, "provider"));
        }
        return claims;
    }

    private static void addIdentityClaim(Set<String> claims, String kind, String value) {
        if (value == null || value.isBlank()) {
            return;
        }
        claims.add(kind + ":" + value.trim());
    }

    private static boolean sameSha256(Path left, Path right) {
        try {
            return sha256(left).equalsIgnoreCase(sha256(right));
        } catch (Exception e) {
            return false;
        }
    }

    private static FeatureBackupRecord createFeatureBackup(Path featureDir, Path jarPath, String reason) throws java.io.IOException {
        Path rootDir = normalize(featureDir);
        Path source = normalize(jarPath);
        ensureInside(rootDir, source);
        if (!Files.isRegularFile(source)) {
            throw new java.io.IOException("Extension jar not found for backup: " + source);
        }
        String filename = source.getFileName() == null ? source.toString() : source.getFileName().toString();
        Path backupDir = backupDirectory(rootDir, filename);
        Files.createDirectories(backupDir);
        Instant createdAt = Instant.now();
        Path backupPath = uniqueBackupPath(backupDir, backupTimestamp(createdAt), filename);
        Files.copy(source, backupPath);
        FeatureJarReport report = inspectJar(backupPath);
        Map<String, Object> jar = report.toMap();
        FeatureBackupRecord record = new FeatureBackupRecord(
                rootDir.toString(),
                source.toString(),
                backupPath.toString(),
                backupPath.resolveSibling(backupPath.getFileName().toString() + ".json").toString(),
                filename,
                createdAt.toString(),
                firstNonBlank(reason, "backup"),
                jar);
        writeFeatureBackupMetadata(record);
        return record;
    }

    private static void writeFeatureBackupMetadata(FeatureBackupRecord record) throws java.io.IOException {
        Files.writeString(
                normalize(Path.of(record.metadataPath())),
                new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(record.toMap()) + System.lineSeparator(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static Path backupDirectory(Path featureDir, String filename) {
        Path rootDir = normalize(featureDir);
        Path dir = normalize(rootDir.resolve(FEATURE_BACKUP_DIR).resolve(filename));
        ensureInside(rootDir, dir);
        return dir;
    }

    private static Path uniqueBackupPath(Path backupDir, String timestamp, String filename) {
        Path candidate = normalize(backupDir.resolve(timestamp + "--" + filename));
        int counter = 1;
        while (Files.exists(candidate)) {
            candidate = normalize(backupDir.resolve(timestamp + "-" + counter + "--" + filename));
            counter++;
        }
        return candidate;
    }

    private static String backupTimestamp(Instant instant) {
        return instant.toString()
                .replace("-", "")
                .replace(":", "")
                .replace(".", "");
    }

    private static Path resolveRollbackTarget(Path root, String rawTarget) throws java.io.IOException {
        String target = rawTarget == null ? "" : rawTarget.trim();
        if (target.isBlank()) {
            throw new IllegalArgumentException("Rollback target is required");
        }
        List<FeatureJarReport> matches = installedJars(root).stream()
                .map(FeaturesCommand::inspectJar)
                .filter(report -> matchesTarget(root, report, target))
                .toList();
        if (matches.size() > 1) {
            String candidates = matches.stream()
                    .map(FeatureJarReport::path)
                    .reduce((left, right) -> left + ", " + right)
                    .orElse("");
            throw new IllegalArgumentException("Multiple extension jars matched; use an exact filename: " + candidates);
        }
        if (matches.size() == 1) {
            Path path = normalize(Path.of(matches.get(0).path()));
            ensureInside(root, path);
            return path;
        }
        String filename = target;
        try {
            Path targetPath = Path.of(target);
            if (targetPath.isAbsolute()) {
                Path normalized = normalize(targetPath);
                ensureInside(root, normalized);
                filename = normalized.getFileName() == null ? "" : normalized.getFileName().toString();
            } else if (!targetPath.getFileName().toString().equals(target)) {
                throw new IllegalArgumentException("Rollback target must be an installed feature or jar filename: " + target);
            }
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException("Rollback target is not a valid path or feature id: " + target);
        }
        if (filename.isBlank()) {
            throw new IllegalArgumentException("Rollback target does not include a jar filename: " + target);
        }
        if (!filename.endsWith(".jar")) {
            filename = filename + ".jar";
        }
        Path path = normalize(root.resolve(filename));
        ensureInside(root, path);
        return path;
    }

    private static Path selectFeatureBackup(Path root, String filename, String requested) throws java.io.IOException {
        List<Path> backups = featureBackupFiles(root, filename);
        if (backups.isEmpty()) {
            throw new IllegalArgumentException("No rollback backups found for: " + filename);
        }
        String value = requested == null ? "" : requested.trim();
        if (value.isBlank() || "latest".equalsIgnoreCase(value)) {
            return backups.get(0);
        }
        for (Path backup : backups) {
            String backupName = backup.getFileName() == null ? "" : backup.getFileName().toString();
            String stem = backupName.endsWith(".jar") ? backupName.substring(0, backupName.length() - 4) : backupName;
            if (value.equals(backupName) || value.equals(stem)) {
                return backup;
            }
        }
        try {
            Path backupPath = normalize(Path.of(value));
            if (Files.isRegularFile(backupPath) && isJar(backupPath) && isInsideAnyBackupRoot(root, backupPath)) {
                return backupPath;
            }
        } catch (InvalidPathException ignored) {
            // Fall through to a clear error.
        }
        throw new IllegalArgumentException("No rollback backup matched: " + value);
    }

    private static List<FeatureBackupReport> inspectFeatureBackups(Path root, String rawTarget) throws java.io.IOException {
        Path rootDir = normalize(root);
        if (backupRoots(rootDir).stream().noneMatch(Files::isDirectory)) {
            return List.of();
        }
        List<FeatureBackupReport> backups = new ArrayList<>();
        for (Path backup : allFeatureBackupFiles(rootDir)) {
            FeatureBackupReport report = inspectFeatureBackup(rootDir, backup);
            if (matchesBackupTarget(rootDir, report, rawTarget)) {
                backups.add(report);
            }
        }
        backups.sort(FeaturesCommand::compareBackupReportsNewestFirst);
        return List.copyOf(backups);
    }

    private static FeatureBackupReport inspectFeatureBackup(Path root, Path backupPath) {
        Path backup = normalize(backupPath);
        Map<String, Object> metadata = readFeatureBackupMetadata(backup);
        FeatureJarReport jar = inspectJar(backup);
        String filename = firstNonBlank(
                stringValue(metadata.get("filename")),
                backupTargetFilename(backup),
                backup.getFileName() == null ? "" : backup.getFileName().toString());
        String metadataPath = backup.resolveSibling(backup.getFileName().toString() + ".json").toString();
        String createdAt = firstNonBlank(stringValue(metadata.get("created_at")), backupCreatedAtFromName(backup));
        String reason = firstNonBlank(stringValue(metadata.get("reason")), "unknown");
        String sourcePath = firstNonBlank(stringValue(metadata.get("source_path")), normalize(root.resolve(filename)).toString());
        long size = -1L;
        String hash = "";
        String error = jar.error();
        try {
            size = Files.size(backup);
        } catch (Exception e) {
            error = firstNonBlank(error, "Backup size could not be read: " + e.getMessage());
        }
        try {
            hash = sha256(backup);
        } catch (Exception e) {
            error = firstNonBlank(error, "Backup checksum could not be read: " + e.getMessage());
        }
        return new FeatureBackupReport(
                root.toString(),
                backup.toString(),
                metadataPath,
                filename,
                createdAt,
                reason,
                sourcePath,
                Files.isRegularFile(Path.of(metadataPath)),
                size,
                hash,
                jar,
                error);
    }

    private static Map<String, Object> readFeatureBackupMetadata(Path backupPath) {
        Path metadata = backupPath.resolveSibling(backupPath.getFileName().toString() + ".json");
        if (!Files.isRegularFile(metadata)) {
            return Map.of();
        }
        try (var input = Files.newInputStream(metadata)) {
            @SuppressWarnings("unchecked")
            Map<String, Object> value = new ObjectMapper().readValue(input, Map.class);
            return value == null ? Map.of() : new LinkedHashMap<>(value);
        } catch (Exception ignored) {
            return Map.of();
        }
    }

    private static FeatureBackupPruneReport pruneFeatureBackups(
            List<FeatureBackupReport> backups,
            int keep,
            boolean dryRun) {
        Map<String, List<FeatureBackupReport>> byFilename = new LinkedHashMap<>();
        for (FeatureBackupReport backup : backups) {
            byFilename.computeIfAbsent(backup.filename(), ignored -> new ArrayList<>()).add(backup);
        }
        List<FeatureBackupPruneOperation> operations = new ArrayList<>();
        for (List<FeatureBackupReport> group : byFilename.values()) {
            group.sort(FeaturesCommand::compareBackupReportsNewestFirst);
            for (int index = 0; index < group.size(); index++) {
                FeatureBackupReport backup = group.get(index);
                if (index < keep) {
                    operations.add(new FeatureBackupPruneOperation(
                            backup.path(),
                            backup.metadataPath(),
                            backup.filename(),
                            "keep",
                            "unchanged",
                            "Kept by --keep=" + keep));
                    continue;
                }
                if (dryRun) {
                    operations.add(new FeatureBackupPruneOperation(
                            backup.path(),
                            backup.metadataPath(),
                            backup.filename(),
                            "prune",
                            "would_delete",
                            "Older than kept backup window"));
                    continue;
                }
                try {
                    Files.deleteIfExists(Path.of(backup.path()));
                    Files.deleteIfExists(Path.of(backup.metadataPath()));
                    operations.add(new FeatureBackupPruneOperation(
                            backup.path(),
                            backup.metadataPath(),
                            backup.filename(),
                            "prune",
                            "deleted",
                            "Older than kept backup window"));
                } catch (Exception e) {
                    operations.add(new FeatureBackupPruneOperation(
                            backup.path(),
                            backup.metadataPath(),
                            backup.filename(),
                            "prune",
                            "error",
                            e.getMessage()));
                }
            }
        }
        return new FeatureBackupPruneReport(dryRun, keep, operations);
    }

    private static boolean matchesBackupTarget(Path root, FeatureBackupReport backup, String rawTarget) {
        String target = rawTarget == null ? "" : rawTarget.trim();
        if (target.isBlank()) {
            return true;
        }
        if (target.equals(backup.filename())
                || target.equals(backup.path())
                || target.equals(backup.metadataPath())) {
            return true;
        }
        String stem = backup.filename().endsWith(".jar")
                ? backup.filename().substring(0, backup.filename().length() - 4)
                : backup.filename();
        if (target.equals(stem)) {
            return true;
        }
        Path backupPath = normalize(Path.of(backup.path()));
        String backupName = backupPath.getFileName() == null ? "" : backupPath.getFileName().toString();
        String backupStem = backupName.endsWith(".jar") ? backupName.substring(0, backupName.length() - 4) : backupName;
        if (target.equals(backupName) || target.equals(backupStem)) {
            return true;
        }
        try {
            if (normalize(Path.of(target)).equals(backupPath)) {
                return true;
            }
        } catch (InvalidPathException ignored) {
            // Fall through to manifest/provider matching.
        }
        return matchesTarget(root, backup.jar(), target);
    }

    private static FeatureBackupSelection selectFeatureBackupByMetadata(
            Path root,
            String rawTarget,
            String requested) throws java.io.IOException {
        List<Path> backups = allFeatureBackupFiles(root);
        if (backups.isEmpty()) {
            return null;
        }
        String requestedValue = requested == null ? "" : requested.trim();
        List<Path> matches = new ArrayList<>();
        for (Path backup : backups) {
            if (!requestedValue.isBlank() && !"latest".equalsIgnoreCase(requestedValue) && !matchesBackupName(backup, requestedValue)) {
                continue;
            }
            FeatureJarReport report = inspectJar(backup);
            if (matchesTarget(root, report, rawTarget)) {
                matches.add(backup);
            }
        }
        if (matches.isEmpty()) {
            return null;
        }
        Path backup = matches.stream()
                .sorted((left, right) -> right.getFileName().toString().compareTo(left.getFileName().toString()))
                .findFirst()
                .orElseThrow();
        Path parent = backup.getParent();
        String filename = parent == null || parent.getFileName() == null ? "" : parent.getFileName().toString();
        if (filename.isBlank()) {
            return null;
        }
        Path targetPath = normalize(root.resolve(filename));
        ensureInside(root, targetPath);
        return new FeatureBackupSelection(targetPath, backup);
    }

    private static int compareBackupReportsNewestFirst(FeatureBackupReport left, FeatureBackupReport right) {
        int created = right.createdAt().compareTo(left.createdAt());
        if (created != 0) {
            return created;
        }
        return right.path().compareTo(left.path());
    }

    private static boolean isRollbackFilenameTarget(String rawTarget, Path targetPath) {
        String target = rawTarget == null ? "" : rawTarget.trim();
        if (target.isBlank()) {
            return false;
        }
        String filename = targetPath == null || targetPath.getFileName() == null ? "" : targetPath.getFileName().toString();
        String stem = filename.endsWith(".jar") ? filename.substring(0, filename.length() - 4) : filename;
        if (target.equals(filename) || target.equals(stem) || target.endsWith(".jar")) {
            return true;
        }
        try {
            Path path = Path.of(target);
            return path.isAbsolute() || !path.getFileName().toString().equals(target);
        } catch (InvalidPathException ignored) {
            return false;
        }
    }

    private static boolean matchesBackupName(Path backup, String requested) {
        String value = requested == null ? "" : requested.trim();
        if (value.isBlank()) {
            return true;
        }
        String backupName = backup.getFileName() == null ? "" : backup.getFileName().toString();
        String stem = backupName.endsWith(".jar") ? backupName.substring(0, backupName.length() - 4) : backupName;
        if (value.equals(backupName) || value.equals(stem)) {
            return true;
        }
        try {
            return normalize(Path.of(value)).equals(backup);
        } catch (InvalidPathException ignored) {
            return false;
        }
    }

    private static List<Path> featureBackupFiles(Path root, String filename) throws java.io.IOException {
        List<Path> backups = new ArrayList<>();
        for (Path backupRoot : backupRoots(root)) {
            Path dir = normalize(backupRoot.resolve(filename));
            ensureInside(normalize(root), dir);
            if (!Files.isDirectory(dir)) {
                continue;
            }
            try (var stream = Files.list(dir)) {
                stream.filter(path -> Files.isRegularFile(path) && isJar(path))
                        .map(FeaturesCommand::normalize)
                        .forEach(backups::add);
            }
        }
        return backups.stream()
                .sorted((left, right) -> right.getFileName().toString().compareTo(left.getFileName().toString()))
                .toList();
    }

    private static String backupTargetFilename(Path backupPath) {
        Path parent = backupPath.getParent();
        return parent == null || parent.getFileName() == null ? "" : parent.getFileName().toString();
    }

    private static String backupCreatedAtFromName(Path backupPath) {
        String name = backupPath.getFileName() == null ? "" : backupPath.getFileName().toString();
        int marker = name.indexOf("--");
        if (marker <= 0) {
            return "";
        }
        return name.substring(0, marker);
    }

    private static List<Path> allFeatureBackupFiles(Path root) throws java.io.IOException {
        List<Path> backups = new ArrayList<>();
        for (Path backupRoot : backupRoots(root)) {
            if (!Files.isDirectory(backupRoot)) {
                continue;
            }
            try (var stream = Files.walk(backupRoot, 3)) {
                stream.filter(path -> Files.isRegularFile(path) && isJar(path))
                        .map(FeaturesCommand::normalize)
                        .forEach(backups::add);
            }
        }
        return backups.stream()
                .sorted((left, right) -> right.getFileName().toString().compareTo(left.getFileName().toString()))
                .toList();
    }

    private static Path backupRoot(Path featureDir) {
        Path rootDir = normalize(featureDir);
        Path backupRoot = normalize(rootDir.resolve(FEATURE_BACKUP_DIR));
        ensureInside(rootDir, backupRoot);
        return backupRoot;
    }

    private static Path legacyBackupRoot(Path featureDir) {
        Path rootDir = normalize(featureDir);
        Path backupRoot = normalize(rootDir.resolve(LEGACY_FEATURE_BACKUP_DIR));
        ensureInside(rootDir, backupRoot);
        return backupRoot;
    }

    private static List<Path> backupRoots(Path featureDir) {
        return List.of(backupRoot(featureDir), legacyBackupRoot(featureDir));
    }

    private static boolean isInsideAnyBackupRoot(Path featureDir, Path backupPath) {
        Path normalized = normalize(backupPath);
        return backupRoots(featureDir).stream().anyMatch(normalized::startsWith);
    }

    private static void writeInstallLock(
            Path featureDir,
            Map<String, Map<String, Object>> features) throws java.io.IOException {
        Map<String, Object> lock = new LinkedHashMap<>();
        lock.put("schema_version", 1);
        lock.put("updated_at", Instant.now().toString());
        lock.put("runtime", runtimeCompatibilityContext().toMap());
        lock.put("extensions", features);
        lock.put("features", features);
        Files.createDirectories(featureDir);
        Files.writeString(
                installLockPath(featureDir),
                new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(lock) + System.lineSeparator(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static Map<String, Object> installLockEntry(Path jarPath, FeatureJarReport report) {
        return installLockEntry(jarPath, report, null);
    }

    private static Map<String, Object> installLockEntry(
            Path jarPath,
            FeatureJarReport report,
            FeaturePackageReport packageReport) {
        return installLockEntry(jarPath, jarPath, report, packageReport);
    }

    private static Map<String, Object> installLockEntry(
            Path jarPath,
            Path digestPath,
            FeatureJarReport report,
            FeaturePackageReport packageReport) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("filename", jarPath.getFileName() == null ? "" : jarPath.getFileName().toString());
        entry.put("path", jarPath.toString());
        entry.put("installed_at", Instant.now().toString());
        entry.put("runtime", runtimeCompatibilityContext().toMap());
        try {
            entry.put("sha256", sha256(digestPath));
        } catch (Exception e) {
            entry.put("sha256_error", e.getMessage());
        }
        try {
            entry.put("size_bytes", Files.size(digestPath));
        } catch (Exception e) {
            entry.put("size_error", e.getMessage());
        }
        entry.put("providers", report == null ? List.of() : report.providers());
        entry.put("adapter_providers", report == null ? List.of() : report.adapterProviders());
        entry.put("model_family_providers", report == null ? List.of() : report.modelFamilyProviders());
        entry.put("gollek_plugin_providers", report == null ? List.of() : report.gollekPluginProviders());
        Map<String, Object> pluginDescriptor = report == null ? Map.of() : report.pluginDescriptor();
        entry.put("plugin_id", manifestValue(pluginDescriptor, "id"));
        entry.put("plugin_name", manifestValue(pluginDescriptor, "name"));
        entry.put("plugin_version", manifestValue(pluginDescriptor, "version"));
        entry.put("plugin_main_class", manifestValue(pluginDescriptor, "mainClass"));
        Map<String, Object> manifest = report == null ? Map.of() : report.manifest();
        entry.put("feature_id", manifestValue(manifest, "id"));
        entry.put("feature_name", manifestValue(manifest, "name"));
        entry.put("feature_version", manifestValue(manifest, "version"));
        entry.put("pipelines", manifestPipelineIds(manifest));
        entry.put("capabilities", manifestCapabilityIds(manifest));
        entry.put("capability_kinds", manifestCapabilityKinds(manifest));
        entry.put("model_family_ids", manifestModelFamilyIds(manifest));
        entry.putAll(tokenizerArtifactMetadata(pluginDescriptor, manifest));
        if (packageReport != null) {
            entry.put("source_package", packageReport.toLockMap());
        }
        return entry;
    }

    private static Map<String, Object> tokenizerArtifactMetadata(
            Map<String, Object> pluginDescriptor,
            Map<String, Object> manifest) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        String pluginTokenizerKind = pluginDescriptorTokenizerKind(pluginDescriptor);
        List<String> pluginTokenizerKinds = pluginDescriptorTokenizerKinds(pluginDescriptor);
        String pluginTokenizerMetadataStatus = pluginDescriptorTokenizerMetadataStatus(pluginDescriptor);
        String pluginTokenizerMetadataPendingReason = pluginDescriptorTokenizerMetadataPendingReason(pluginDescriptor);
        metadata.put("plugin_tokenizer_kind", pluginTokenizerKind);
        metadata.put("plugin_tokenizer_kinds", pluginTokenizerKinds);
        metadata.put("plugin_tokenizer_metadata_status", pluginTokenizerMetadataStatus);
        metadata.put("plugin_tokenizer_metadata_pending_reason", pluginTokenizerMetadataPendingReason);
        List<String> modelFamilyTokenizerKinds = manifestModelFamilyTokenizerKinds(manifest);
        List<String> modelFamilyTokenizerMetadataStatuses = manifestModelFamilyTokenizerMetadataStatuses(manifest);
        Map<String, Object> modelFamilyTokenizerMetadataPendingReasons =
                manifestModelFamilyTokenizerMetadataPendingReasons(manifest);
        List<String> tokenizerKinds = orderedDistinctStrings(mergeLists(pluginTokenizerKinds, modelFamilyTokenizerKinds));
        metadata.put("model_family_tokenizer_kinds", modelFamilyTokenizerKinds);
        metadata.put("model_family_tokenizer_metadata_statuses", modelFamilyTokenizerMetadataStatuses);
        metadata.put("model_family_tokenizer_metadata_pending_reasons", modelFamilyTokenizerMetadataPendingReasons);
        metadata.put("tokenizer_kind", tokenizerKinds.isEmpty() ? "" : tokenizerKinds.get(0));
        metadata.put("tokenizer_kinds", tokenizerKinds);
        metadata.put("tokenizer_metadata_status", tokenizerMetadataStatus(
                pluginTokenizerMetadataStatus,
                modelFamilyTokenizerMetadataStatuses,
                tokenizerKinds));
        metadata.put("tokenizer_metadata_pending_reason", tokenizerMetadataPendingReason(
                pluginTokenizerMetadataPendingReason,
                modelFamilyTokenizerMetadataPendingReasons));
        return metadata;
    }

    private static String pluginDescriptorTokenizerKind(Map<String, Object> pluginDescriptor) {
        Map<String, Object> properties = objectMap(pluginDescriptor == null ? null : pluginDescriptor.get("properties"));
        String tokenizerKind = manifestValue(properties, "tokenizerKind");
        if (!tokenizerKind.isBlank()) {
            return tokenizerKind;
        }
        List<String> tokenizerKinds = manifestStringList(properties.get("tokenizerKinds"));
        return tokenizerKinds.isEmpty() ? "" : tokenizerKinds.get(0);
    }

    private static List<String> pluginDescriptorTokenizerKinds(Map<String, Object> pluginDescriptor) {
        Map<String, Object> properties = objectMap(pluginDescriptor == null ? null : pluginDescriptor.get("properties"));
        List<String> tokenizerKinds = new ArrayList<>();
        String tokenizerKind = manifestValue(properties, "tokenizerKind");
        if (!tokenizerKind.isBlank()) {
            tokenizerKinds.add(tokenizerKind);
        }
        tokenizerKinds.addAll(manifestStringList(properties.get("tokenizerKinds")));
        return orderedDistinctStrings(tokenizerKinds);
    }

    private static String pluginDescriptorTokenizerMetadataStatus(Map<String, Object> pluginDescriptor) {
        Map<String, Object> properties = objectMap(pluginDescriptor == null ? null : pluginDescriptor.get("properties"));
        String status = normalizeTokenizerMetadataStatusToken(manifestValue(properties, "tokenizerMetadataStatus"));
        if (!status.isBlank()) {
            return status;
        }
        return pluginDescriptorTokenizerKinds(pluginDescriptor).isEmpty() ? "" : "ready";
    }

    private static String pluginDescriptorTokenizerMetadataPendingReason(Map<String, Object> pluginDescriptor) {
        Map<String, Object> properties = objectMap(pluginDescriptor == null ? null : pluginDescriptor.get("properties"));
        return manifestValue(properties, "tokenizerMetadataPendingReason");
    }

    private static List<String> manifestModelFamilyTokenizerKinds(Map<String, Object> manifest) {
        List<String> tokenizerKinds = new ArrayList<>();
        for (Map<String, Object> family : manifestModelFamilies(manifest)) {
            String tokenizerKind = manifestValue(family, "tokenizer_kind");
            if (!tokenizerKind.isBlank()) {
                tokenizerKinds.add(tokenizerKind);
            }
            tokenizerKinds.addAll(manifestStringList(family.get("tokenizer_kinds")));
        }
        return orderedDistinctStrings(tokenizerKinds);
    }

    private static List<String> manifestModelFamilyTokenizerMetadataStatuses(Map<String, Object> manifest) {
        List<String> statuses = new ArrayList<>();
        for (Map<String, Object> family : manifestModelFamilies(manifest)) {
            String status = normalizeTokenizerMetadataStatusToken(manifestValue(family, "tokenizer_metadata_status"));
            if (status.isBlank()
                    && (!manifestValue(family, "tokenizer_kind").isBlank()
                    || !manifestStringList(family.get("tokenizer_kinds")).isEmpty())) {
                status = "ready";
            }
            if (!status.isBlank()) {
                statuses.add(status);
            }
        }
        return orderedDistinctStrings(statuses);
    }

    private static Map<String, Object> manifestModelFamilyTokenizerMetadataPendingReasons(Map<String, Object> manifest) {
        Map<String, Object> reasons = new LinkedHashMap<>();
        int index = 0;
        for (Map<String, Object> family : manifestModelFamilies(manifest)) {
            String reason = manifestValue(family, "tokenizer_metadata_pending_reason");
            if (!reason.isBlank()) {
                String familyId = firstNonBlank(
                        manifestValue(family, "id"),
                        manifestValue(family, "class"),
                        "family_" + index);
                reasons.put(familyId, reason);
            }
            index++;
        }
        return Map.copyOf(reasons);
    }

    private static String tokenizerMetadataStatus(
            String pluginStatus,
            List<String> modelFamilyStatuses,
            List<String> tokenizerKinds) {
        if (!pluginStatus.isBlank()) {
            return pluginStatus;
        }
        if (modelFamilyStatuses != null && modelFamilyStatuses.contains("pending")) {
            return "pending";
        }
        if (modelFamilyStatuses != null && modelFamilyStatuses.contains("ready")) {
            return "ready";
        }
        return tokenizerKinds == null || tokenizerKinds.isEmpty() ? "" : "ready";
    }

    private static String tokenizerMetadataPendingReason(
            String pluginReason,
            Map<String, Object> modelFamilyReasons) {
        if (!pluginReason.isBlank()) {
            return pluginReason;
        }
        if (modelFamilyReasons != null) {
            for (Object reason : modelFamilyReasons.values()) {
                String text = stringValue(reason);
                if (!text.isBlank()) {
                    return text;
                }
            }
        }
        return "";
    }

    private static String normalizeTokenizerMetadataStatusToken(String status) {
        String normalized = normalizeFeatureId(status);
        return switch (normalized) {
            case "ready", "pending" -> normalized;
            default -> "";
        };
    }

    private static List<String> mergeLists(List<String> first, List<String> second) {
        List<String> merged = new ArrayList<>();
        if (first != null) {
            merged.addAll(first);
        }
        if (second != null) {
            merged.addAll(second);
        }
        return merged;
    }

    private static List<String> orderedDistinctStrings(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        LinkedHashSet<String> distinct = new LinkedHashSet<>();
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                distinct.add(value.trim());
            }
        }
        return List.copyOf(distinct);
    }

    private static String sha256(Path path) throws java.io.IOException {
        try (var input = Files.newInputStream(path)) {
            return sha256(input);
        }
    }

    private static String sha256(ZipFile zip, ZipEntry entry) throws java.io.IOException {
        try (var input = zip.getInputStream(entry)) {
            return sha256(input);
        }
    }

    private static String sha256(java.io.InputStream input) throws java.io.IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                if (read > 0) {
                    digest.update(buffer, 0, read);
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private static String shortHash(String hash) {
        if (hash == null) {
            return "";
        }
        String text = hash.trim();
        return text.length() <= 12 ? text : text.substring(0, 12);
    }

    private static long longValue(Object value, long fallback) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        if (value != null) {
            try {
                return Long.parseLong(String.valueOf(value).trim());
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private static FeatureJarValidation validateArtifact(
            FeatureJarReport report,
            FeaturePackageReport packageReport,
            boolean strict) {
        FeatureJarValidation jarValidation = validateJar(report, false);
        List<String> warnings = new ArrayList<>(jarValidation.warnings());
        List<String> errors = new ArrayList<>(jarValidation.errors());
        warnings.addAll(packageJarConsistencyWarnings(packageReport, report));
        if (strict && errors.isEmpty() && !warnings.isEmpty()) {
            errors.addAll(warnings);
        }
        String status = errors.isEmpty()
                ? warnings.isEmpty() ? "ok" : "warning"
                : "error";
        return new FeatureJarValidation(errors.isEmpty(), status, warnings, errors);
    }

    private static List<String> packageJarConsistencyWarnings(
            FeaturePackageReport packageReport,
            FeatureJarReport report) {
        if (packageReport == null || report == null || !report.hasManifest()) {
            return List.of();
        }
        List<String> warnings = new ArrayList<>();
        Map<String, Object> packageManifest = packageReport.manifest() == null
                ? Map.of()
                : packageReport.manifest();
        Map<String, Object> jarManifest = report.manifest();
        appendPackageJarFieldWarning(packageManifest, jarManifest, "id", warnings);
        appendPackageJarFieldWarning(packageManifest, jarManifest, "name", warnings);
        appendPackageJarFieldWarning(packageManifest, jarManifest, "version", warnings);
        appendPackageJarRequiresWarnings(packageManifest, jarManifest, warnings);
        Map<String, Object> artifact = packageJarArtifact(packageManifest);
        String expectedManifest = manifestValue(artifact, "manifest");
        if (!expectedManifest.isBlank()
                && report.manifestEntryName() != null
                && !expectedManifest.equals(report.manifestEntryName())) {
            warnings.add("Extension package artifact manifest differs from jar manifest entry: package "
                    + expectedManifest + ", jar " + report.manifestEntryName());
        }
        return List.copyOf(warnings);
    }

    private static void appendPackageJarFieldWarning(
            Map<String, Object> packageManifest,
            Map<String, Object> jarManifest,
            String key,
            List<String> warnings) {
        String packageValue = manifestValue(packageManifest, key);
        if (packageValue.isBlank()) {
            return;
        }
        String jarValue = manifestValue(jarManifest, key);
        if (!packageValue.equals(jarValue)) {
            warnings.add("Extension package " + key + " differs from jar manifest " + key + ": package "
                    + packageValue + ", jar " + firstNonBlank(jarValue, "<blank>"));
        }
    }

    private static void appendPackageJarRequiresWarnings(
            Map<String, Object> packageManifest,
            Map<String, Object> jarManifest,
            List<String> warnings) {
        Map<String, String> packageRequires = stringMap(packageManifest.get("requires"));
        if (packageRequires.isEmpty()) {
            return;
        }
        Map<String, String> jarRequires = stringMap(jarManifest.get("requires"));
        for (Map.Entry<String, String> entry : packageRequires.entrySet()) {
            String key = entry.getKey();
            String packageValue = entry.getValue();
            if (!jarRequires.containsKey(key)) {
                warnings.add("Extension package requires." + key + " is missing from jar manifest requires");
                continue;
            }
            String jarValue = jarRequires.get(key);
            if (!packageValue.equals(jarValue)) {
                warnings.add("Extension package requires." + key
                        + " differs from jar manifest requires." + key + ": package "
                        + packageValue + ", jar " + firstNonBlank(jarValue, "<blank>"));
            }
        }
    }

    private static FeatureJarValidation validateJar(FeatureJarReport report, boolean strict) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();
        if (report.error() != null && !report.error().isBlank()) {
            errors.add("Extension jar is not readable: " + report.error());
        }
        boolean hasManifestPipelines = report.hasManifest() && !manifestPipelines(report.manifest()).isEmpty();
        boolean hasManifestModelFamilies = report.hasManifest() && !manifestModelFamilies(report.manifest()).isEmpty();
        List<Map<String, Object>> manifestCapabilities = report.hasManifest()
                ? manifestCapabilities(report.manifest())
                : List.of();
        boolean hasManifestCapabilities = !manifestCapabilities.isEmpty();
        boolean hasNonPipelineManifestCapabilities = manifestCapabilities.stream()
                .map(FeaturesCommand::manifestCapabilityKind)
                .anyMatch(kind -> !FeatureAdapterKinds.PIPELINE.equals(kind));
        if (!report.hasServiceEntry()) {
            if (hasManifestPipelines) {
                errors.add("Extension jar declares pipelines but has no " + SERVICE_ENTRY);
            } else if (report.hasAdapterServiceEntry() || report.hasModelFamilyServiceEntry()) {
                // Adapter-only and model-family-only extensions are valid without a ModelPipeline service entry.
            } else if (hasNonPipelineManifestCapabilities || hasManifestCapabilities) {
                String loadingHint = hasManifestModelFamilies
                        ? "; model-family manifests need " + MODEL_FAMILY_SERVICE_ENTRY + " to load at runtime"
                        : "; manifest-only adapters need a domain registry to load them";
                warnings.add("Extension jar has no " + SERVICE_ENTRY
                        + ", " + ADAPTER_SERVICE_ENTRY
                        + ", or " + MODEL_FAMILY_SERVICE_ENTRY
                        + loadingHint);
            } else {
                errors.add("Extension jar has no " + SERVICE_ENTRY
                        + ", " + ADAPTER_SERVICE_ENTRY
                        + ", or " + MODEL_FAMILY_SERVICE_ENTRY);
            }
        } else if (report.providers().isEmpty()) {
            errors.add("Extension jar service entry has no providers");
        } else {
            validateServiceProviders(report.providers(), report.classPresence(),
                    "Extension service provider", warnings, errors);
        }
        if (report.hasAdapterServiceEntry()) {
            if (report.adapterProviders().isEmpty()) {
                errors.add("Extension adapter service entry has no providers");
            } else {
                validateServiceProviders(report.adapterProviders(), report.classPresence(),
                        "Extension adapter provider", warnings, errors);
            }
        }
        if (report.hasModelFamilyServiceEntry()) {
            if (report.modelFamilyProviders().isEmpty()) {
                errors.add("Extension model-family service entry has no providers");
            } else {
                validateServiceProviders(report.modelFamilyProviders(), report.classPresence(),
                        "Extension model-family provider", warnings, errors);
            }
        }
        if (report.hasGollekPluginServiceEntry()) {
            if (report.gollekPluginProviders().isEmpty()) {
                errors.add("Extension GollekPlugin service entry has no providers");
            } else {
                validateServiceProviders(report.gollekPluginProviders(), report.classPresence(),
                        "Extension GollekPlugin provider", warnings, errors);
            }
        }
        if (hasManifestModelFamilies) {
            if (!report.hasGollekPluginServiceEntry()) {
                warnings.add("Model-family plugin jar has no " + GOLLEK_PLUGIN_SERVICE_ENTRY
                        + "; plugin-core cannot discover it as a detachable GollekPlugin");
            }
            if (!report.hasPluginDescriptor()) {
                warnings.add("Model-family plugin jar has no " + PLUGIN_DESCRIPTOR_ENTRY
                        + "; plugin-core JarPluginLoader needs a descriptor for plugin-dir installs");
            }
        }
        validatePluginDescriptor(report, warnings, errors);
        if (report.manifestError() != null && !report.manifestError().isBlank()) {
            errors.add("Extension manifest is not valid JSON: " + report.manifestError());
        } else if (!report.hasManifest()) {
            warnings.add("Extension jar has no " + MANIFEST_ENTRY + " or legacy " + LEGACY_MANIFEST_ENTRY);
        } else {
            validateManifest(report, warnings, errors);
        }
        if (report.installLockWarnings() != null && !report.installLockWarnings().isEmpty()) {
            warnings.addAll(report.installLockWarnings());
        }
        if (strict && errors.isEmpty() && !warnings.isEmpty()) {
            errors.addAll(warnings);
        }
        String status = errors.isEmpty()
                ? warnings.isEmpty() ? "ok" : "warning"
                : "error";
        return new FeatureJarValidation(errors.isEmpty(), status, warnings, errors);
    }

    private static void validatePluginDescriptor(
            FeatureJarReport report,
            List<String> warnings,
            List<String> errors) {
        if (report.pluginDescriptorError() != null && !report.pluginDescriptorError().isBlank()) {
            errors.add("Plugin descriptor is not valid JSON: " + report.pluginDescriptorError());
            return;
        }
        if (!report.hasPluginDescriptor()) {
            return;
        }
        Map<String, Object> descriptor = report.pluginDescriptor();
        String id = manifestValue(descriptor, "id");
        if (id.isBlank()) {
            warnings.add("Plugin descriptor is missing 'id'");
        }
        String mainClass = manifestValue(descriptor, "mainClass");
        if (mainClass.isBlank()) {
            warnings.add("Plugin descriptor is missing 'mainClass'");
        } else if (!isJavaClassName(mainClass)) {
            errors.add("Plugin descriptor mainClass is not a valid Java class name: " + mainClass);
        } else if (Boolean.FALSE.equals(report.classPresence().get(mainClass))) {
            errors.add("Plugin descriptor mainClass not found in jar: " + mainClass);
        }
        if (!mainClass.isBlank()
                && report.hasGollekPluginServiceEntry()
                && !report.gollekPluginProviders().contains(mainClass)) {
            warnings.add("Plugin descriptor mainClass is not listed in GollekPlugin service providers: " + mainClass);
        }
    }

    private static void validateServiceProviders(
            List<String> providers,
            Map<String, Boolean> classPresence,
            String label,
            List<String> warnings,
            List<String> errors) {
        Set<String> seen = new LinkedHashSet<>();
        Set<String> duplicates = new LinkedHashSet<>();
        for (String provider : providers) {
            if (!seen.add(provider)) {
                duplicates.add(provider);
            }
            if (!isJavaClassName(provider)) {
                errors.add(label + " is not a valid Java class name: " + provider);
            } else if (Boolean.FALSE.equals(classPresence.get(provider))) {
                errors.add(label + " class not found in jar: " + provider);
            }
        }
        duplicates.forEach(provider -> warnings.add(label + " is duplicated: " + provider));
    }

    private static void validateManifest(
            FeatureJarReport report,
            List<String> warnings,
            List<String> errors) {
        Map<String, Object> manifest = report.manifest();
        warnIfMissing(manifest, warnings, "id");
        warnIfMissing(manifest, warnings, "name");
        warnIfMissing(manifest, warnings, "version");
        String featureId = manifestValue(manifest, "id");
        if (!featureId.isBlank() && !isFeatureId(featureId)) {
            warnings.add("Extension manifest id should use lowercase letters, numbers, dot, underscore, or dash: " + featureId);
        }
        validateManifestRequires(manifest, warnings, errors);
        List<Map<String, Object>> capabilities = manifestCapabilities(manifest);
        if (capabilities.isEmpty()) {
            warnings.add("Extension manifest has no pipelines, model_families, capabilities, or adapters");
            return;
        }
        validateManifestCapabilities(capabilities, report, warnings);
        List<String> pipelineClasses = manifestPipelineClasses(manifest);
        if (!pipelineClasses.isEmpty()) {
            for (String provider : report.providers()) {
                if (!pipelineClasses.contains(provider)) {
                    warnings.add("Service provider is not listed in manifest pipelines: " + provider);
                }
            }
            for (String pipelineClass : pipelineClasses) {
                if (!report.providers().contains(pipelineClass)) {
                    warnings.add("Manifest pipeline class is not listed in service providers: " + pipelineClass);
                }
            }
        }
        List<String> capabilityClasses = manifestCapabilityClasses(manifest);
        if (!capabilityClasses.isEmpty() && report.hasAdapterServiceEntry()) {
            for (String provider : report.adapterProviders()) {
                if (!capabilityClasses.contains(provider)) {
                    warnings.add("Adapter service provider is not listed in manifest capabilities/adapters: " + provider);
                }
            }
        }
        List<String> modelFamilyClasses = manifestModelFamilyClasses(manifest);
        if (!modelFamilyClasses.isEmpty()) {
            for (String provider : report.modelFamilyProviders()) {
                if (!modelFamilyClasses.contains(provider)) {
                    warnings.add("Model-family service provider is not listed in manifest model_families: " + provider);
                }
            }
            for (String modelFamilyClass : modelFamilyClasses) {
                if (!report.modelFamilyProviders().contains(modelFamilyClass)) {
                    warnings.add("Manifest model-family class is not listed in service providers: " + modelFamilyClass);
                }
            }
            for (String provider : report.gollekPluginProviders()) {
                if (!modelFamilyClasses.contains(provider)) {
                    warnings.add("GollekPlugin service provider is not listed in manifest model_families: " + provider);
                }
            }
            for (String modelFamilyClass : modelFamilyClasses) {
                if (report.hasGollekPluginServiceEntry()
                        && !report.gollekPluginProviders().contains(modelFamilyClass)) {
                    warnings.add("Manifest model-family class is not listed in GollekPlugin service providers: "
                            + modelFamilyClass);
                }
            }
            String pluginMainClass = manifestValue(report.pluginDescriptor(), "mainClass");
            if (!pluginMainClass.isBlank() && !modelFamilyClasses.contains(pluginMainClass)) {
                warnings.add("Plugin descriptor mainClass is not listed in manifest model_families: "
                        + pluginMainClass);
            }
        }
    }

    private static void validateManifestRequires(
            Map<String, Object> manifest,
            List<String> warnings,
            List<String> errors) {
        Object rawRequires = manifest == null ? null : manifest.get("requires");
        if (rawRequires == null) {
            warnings.add("Extension manifest has no 'requires' compatibility metadata");
            return;
        }
        if (!(rawRequires instanceof Map<?, ?> rawMap)) {
            warnings.add("Extension manifest 'requires' should be an object");
            return;
        }

        Map<String, Object> requires = new LinkedHashMap<>();
        rawMap.forEach((key, value) -> {
            if (key != null) {
                requires.put(String.valueOf(key).trim(), value);
            }
        });
        FeatureRuntimeContext runtime = runtimeCompatibilityContext();
        boolean hasGollekRequirement = false;
        hasGollekRequirement |= validateVersionRequirement(
                requires, "gollek", runtime.gollekVersion(), "Gollek", warnings, errors);
        hasGollekRequirement |= validateVersionRequirement(
                requires, "gollek_runtime", runtime.gollekVersion(), "Gollek runtime", warnings, errors);
        hasGollekRequirement |= validateVersionRequirement(
                requires, "gollek_cli", runtime.gollekVersion(), "Gollek CLI", warnings, errors);
        hasGollekRequirement |= validateVersionRequirement(
                requires, "gollek_spi", runtime.gollekVersion(), "Gollek SPI", warnings, errors);
        if (!hasGollekRequirement) {
            warnings.add("Extension manifest 'requires' does not declare Gollek runtime or SPI compatibility");
        }
        validateVersionRequirement(requires, "java", runtime.javaVersion(), "Java", warnings, errors);
    }

    private static boolean validateVersionRequirement(
            Map<String, Object> requires,
            String key,
            String actual,
            String label,
            List<String> warnings,
            List<String> errors) {
        if (requires == null || !requires.containsKey(key)) {
            return false;
        }
        String requirement = String.valueOf(requires.get(key) == null ? "" : requires.get(key)).trim();
        if (requirement.isBlank()) {
            warnings.add("Extension manifest requires." + key + " is blank");
            return true;
        }
        VersionRequirementCheck check = checkVersionRequirement(actual, requirement);
        if (!check.understood()) {
            warnings.add("Extension requires " + key + " version '" + requirement
                    + "' but Gollek could not parse this constraint; current " + label + " version is " + actual);
        } else if (!check.compatible()) {
            errors.add("Extension requires " + key + " " + requirement
                    + " but current " + label + " version is " + actual);
        }
        return true;
    }

    private static void validateManifestPipelines(
            List<Map<String, Object>> pipelines,
            FeatureJarReport report,
            List<String> warnings) {
        validateManifestCapabilities(pipelines, report, warnings);
    }

    private static void validateManifestCapabilities(
            List<Map<String, Object>> capabilities,
            FeatureJarReport report,
            List<String> warnings) {
        Set<String> pipelineIds = new LinkedHashSet<>();
        Set<String> duplicateIds = new LinkedHashSet<>();
        for (int i = 0; i < capabilities.size(); i++) {
            Map<String, Object> capability = capabilities.get(i);
            String source = firstNonBlank(manifestValue(capability, "_manifest_source"), "capabilities");
            String kind = manifestCapabilityKind(capability);
            String prefix = "Extension manifest " + source + "[" + i + "]";
            String id = manifestValue(capability, "id");
            if (id.isBlank()) {
                warnings.add(prefix + " is missing 'id'");
            } else {
                String duplicateKey = kind + ":" + id;
                if (!pipelineIds.add(duplicateKey)) {
                    duplicateIds.add(duplicateKey);
                }
                if (!isFeatureId(id)) {
                    warnings.add(prefix + " id should use lowercase letters, numbers, dot, underscore, or dash: " + id);
                }
            }
            String rawKind = firstNonBlank(
                    manifestValue(capability, "kind"),
                    manifestValue(capability, "type"),
                    manifestValue(capability, "domain"));
            if (!rawKind.isBlank() && !rawKind.equals(kind)) {
                warnings.add(prefix + " kind should normalize cleanly to lowercase letters, numbers, dot, or dash: " + rawKind);
            }
            String className = manifestValue(capability, "class");
            if (className.isBlank()) {
                if (FeatureAdapterKinds.PIPELINE.equals(kind) || MODEL_FAMILY_KIND.equals(kind)) {
                    warnings.add(prefix + " is missing 'class'");
                }
            } else if (!isJavaClassName(className)) {
                warnings.add(prefix + " class is not a valid Java class name: " + className);
            } else if (Boolean.FALSE.equals(report.classPresence().get(className))) {
                warnings.add(prefix + " class not found in jar: " + className);
            }
            if (FeatureAdapterKinds.PIPELINE.equals(kind)) {
                validateManifestModalities(capability, "inputs", prefix, warnings);
                validateManifestModalities(capability, "outputs", prefix, warnings);
            }
        }
        duplicateIds.forEach(id -> warnings.add("Extension manifest capability id is duplicated for kind: " + id));
    }

    private static void validateManifestModalities(
            Map<String, Object> pipeline,
            String key,
            String prefix,
            List<String> warnings) {
        List<String> values = manifestStringList(pipeline.get(key));
        if (values.isEmpty()) {
            warnings.add(prefix + " has no '" + key + "' modalities");
            return;
        }
        for (String value : values) {
            String normalized = value.replace('-', '_').toUpperCase(Locale.ROOT);
            try {
                ModalityType.valueOf(normalized);
            } catch (IllegalArgumentException e) {
                warnings.add(prefix + " has unknown " + key + " modality: " + value);
            }
        }
    }

    private static void warnIfMissing(Map<String, Object> manifest, List<String> warnings, String key) {
        if (manifestValue(manifest, key).isBlank()) {
            warnings.add("Extension manifest is missing '" + key + "'");
        }
    }

    private static ServiceEntry readServiceEntry(JarFile jar, String entryName) throws java.io.IOException {
        var entry = jar.getEntry(entryName);
        if (entry == null) {
            return new ServiceEntry(false, List.of());
        }
        try (var input = jar.getInputStream(entry)) {
            String serviceText = new String(input.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
            List<String> providers = serviceText.lines()
                    .map(String::trim)
                    .filter(line -> !line.isEmpty())
                    .filter(line -> !line.startsWith("#"))
                    .toList();
            return new ServiceEntry(true, providers);
        }
    }

    private static ManifestEntry readManifestEntry(JarFile jar) {
        ManifestEntry preferred = readManifestEntry(jar, MANIFEST_ENTRY);
        if (preferred.hasEntry()) {
            return preferred;
        }
        return readManifestEntry(jar, LEGACY_MANIFEST_ENTRY);
    }

    private static ManifestEntry readManifestEntry(JarFile jar, String entryName) {
        var entry = jar.getEntry(entryName);
        if (entry == null) {
            return new ManifestEntry(Map.of(), "", null);
        }
        try (var input = jar.getInputStream(entry)) {
            @SuppressWarnings("unchecked")
            Map<String, Object> manifest = new ObjectMapper().readValue(input, Map.class);
            return new ManifestEntry(manifest == null ? Map.of() : manifest, entryName, null);
        } catch (Exception e) {
            return new ManifestEntry(Map.of(), entryName, e.getMessage());
        }
    }

    private static Map<String, Boolean> classPresence(
            JarFile jar,
            List<String> providers,
            List<String> adapterProviders,
            List<String> modelFamilyProviders,
            List<String> gollekPluginProviders,
            Map<String, Object> manifest,
            Map<String, Object> pluginDescriptor) {
        Set<String> declared = new LinkedHashSet<>();
        if (providers != null) {
            declared.addAll(providers);
        }
        if (adapterProviders != null) {
            declared.addAll(adapterProviders);
        }
        if (modelFamilyProviders != null) {
            declared.addAll(modelFamilyProviders);
        }
        if (gollekPluginProviders != null) {
            declared.addAll(gollekPluginProviders);
        }
        declared.addAll(manifestCapabilityClasses(manifest));
        String pluginMainClass = manifestValue(pluginDescriptor, "mainClass");
        if (!pluginMainClass.isBlank()) {
            declared.add(pluginMainClass);
        }
        if (declared.isEmpty()) {
            return Map.of();
        }
        Set<String> classes = new LinkedHashSet<>();
        jar.stream()
                .map(entry -> entry == null ? "" : entry.getName())
                .filter(name -> name.endsWith(".class"))
                .map(FeaturesCommand::classNameFromEntry)
                .forEach(classes::add);
        Map<String, Boolean> result = new LinkedHashMap<>();
        for (String className : declared) {
            if (className != null && !className.isBlank()) {
                result.put(className, classes.contains(className));
            }
        }
        return Map.copyOf(result);
    }

    private static String classNameFromEntry(String entryName) {
        String withoutSuffix = entryName.substring(0, entryName.length() - ".class".length());
        return withoutSuffix.replace('/', '.').replace('\\', '.');
    }

    private static List<Path> runtimeFeatureRoots() {
        return runtimeFeatureRoots(List.of());
    }

    private static List<Path> configuredFeatureRoots(List<String> configuredRoots) {
        List<Path> roots = new ArrayList<>();
        if (configuredRoots != null) {
            for (String root : configuredRoots) {
                addConfiguredRoots(roots, root);
            }
        }
        return roots.stream()
                .map(FeaturesCommand::normalize)
                .distinct()
                .toList();
    }

    private static List<Path> runtimeFeatureRoots(List<String> extraRoots) {
        List<Path> roots = new ArrayList<>();
        if (extraRoots != null) {
            for (String root : extraRoots) {
                addConfiguredRoots(roots, root);
            }
        }
        addConfiguredRoots(roots, System.getProperty(EXTENSION_PATH_PROPERTY));
        addConfiguredRoots(roots, System.getenv(EXTENSION_PATH_ENV));
        String home = System.getProperty("user.home");
        if (home != null && !home.isBlank()) {
            roots.add(Path.of(home, ".gollek", "extensions"));
        }
        return roots.stream()
                .map(FeaturesCommand::normalize)
                .distinct()
                .toList();
    }

    private static Path installDirectory(String configured) {
        if (configured != null && !configured.isBlank()) {
            return normalize(Path.of(configured.trim()));
        }
        String home = System.getProperty("user.home");
        if (home == null || home.isBlank()) {
            throw new IllegalStateException("Cannot resolve user home for default extension directory");
        }
        return normalize(Path.of(home, ".gollek", "extensions"));
    }

    private static Path installTargetDirectory(String outputDir, String pluginDir, FeatureJarReport report) {
        if (pluginDir != null && !pluginDir.isBlank()) {
            return pluginDirectory(pluginDir);
        }
        if ((outputDir == null || outputDir.isBlank()) && modelFamilyOnly(report)) {
            return pluginDirectory(null);
        }
        return installDirectory(outputDir);
    }

    private static Path pluginDirectory(String configured) {
        if (configured != null && !configured.isBlank()) {
            return normalize(Path.of(configured.trim()));
        }
        return normalize(ExternalPluginClasspath.defaultPluginDirectory());
    }

    private static boolean pluginInstallRequested(String pluginDir, String outputDir, FeatureJarReport report) {
        return pluginDir != null && !pluginDir.isBlank()
                || ((outputDir == null || outputDir.isBlank()) && modelFamilyOnly(report));
    }

    private static List<String> modelFamilyPluginInstallReadinessErrors(FeatureJarReport report) {
        if (report == null) {
            return List.of();
        }
        List<String> errors = new ArrayList<>();
        if (!report.hasGollekPluginServiceEntry()) {
            errors.add("Model-family plugin install requires " + GOLLEK_PLUGIN_SERVICE_ENTRY
                    + " so gollek-plugin-core can discover it");
        } else if (report.gollekPluginProviders().isEmpty()) {
            errors.add("Model-family plugin install requires at least one GollekPlugin service provider");
        }
        if (!report.hasPluginDescriptor()) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " with mainClass for gollek-plugin-core JarPluginLoader");
        } else {
            String mainClass = manifestValue(report.pluginDescriptor(), "mainClass");
            if (mainClass.isBlank()) {
                errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                        + " mainClass for gollek-plugin-core JarPluginLoader");
            } else if (report.hasGollekPluginServiceEntry()
                    && !report.gollekPluginProviders().contains(mainClass)) {
                errors.add("Model-family plugin install requires plugin.json mainClass to be listed in "
                        + GOLLEK_PLUGIN_SERVICE_ENTRY + ": " + mainClass);
            }
        }
        return List.copyOf(errors);
    }

    private static boolean modelFamilyOnly(FeatureJarReport report) {
        return report != null
                && report.hasModelFamilyServiceEntry()
                && !report.hasServiceEntry()
                && !report.hasAdapterServiceEntry();
    }

    private static Path legacyInstallDirectory(String configured) {
        if (configured != null && !configured.isBlank()) {
            return normalize(Path.of(configured.trim()));
        }
        String home = System.getProperty("user.home");
        if (home == null || home.isBlank()) {
            throw new IllegalStateException("Cannot resolve user home for legacy extension directory");
        }
        return normalize(Path.of(home, ".gollek", "features"));
    }

    private static void addConfiguredRoots(List<Path> roots, String value) {
        if (value == null || value.isBlank()) {
            return;
        }
        String normalizedSeparators = value.contains(File.pathSeparator)
                ? value
                : value.replace(',', File.pathSeparatorChar);
        String[] parts = normalizedSeparators.split(Pattern.quote(File.pathSeparator));
        for (String part : parts) {
            String text = part == null ? "" : part.trim();
            if (text.isEmpty()) {
                continue;
            }
            try {
                roots.add(Path.of(text));
            } catch (InvalidPathException ignored) {
                // The doctor output will simply omit invalid paths.
            }
        }
    }

    private static void printDoctor(FeatureDoctorReport report, boolean strict) {
        FeatureRuntimeContext runtime = runtimeCompatibilityContext();
        System.out.println("Extension Path Doctor");
        System.out.println("---------------------");
        System.out.printf("Runtime: Gollek %s, Java %s%n", runtime.gollekVersion(), runtime.javaVersion());
        System.out.printf("Extension loading: %s%n",
                runtime.dynamicExtensionLoading() ? "JVM dynamic jars enabled" : "native image AOT-only");
        if (runtime.nativeImage()) {
            System.out.println("Native image note: extension jars can be inspected here, but runtime pipelines must be included ahead of time.");
        }
        if (!runtime.ignoredLegacyExtensionPaths().isEmpty()) {
            System.out.println("Ignored legacy extension settings:");
            for (String ignored : runtime.ignoredLegacyExtensionPaths()) {
                System.out.printf("  warning   %s%n", ignored);
            }
            System.out.printf("  hint      Use %s or -D%s; legacy feature paths are migration-only.%n",
                    EXTENSION_PATH_ENV, EXTENSION_PATH_PROPERTY);
        }
        if (report.roots().isEmpty()) {
            System.out.println("No extension roots configured.");
        } else {
            System.out.println("Roots:");
            for (FeatureRootReport root : report.roots()) {
                String kind = root.directory() ? "directory" : root.regularFile() ? "jar" : "missing";
                System.out.printf("  %-9s %s%n", kind, root.path());
            }
        }
        FeatureCollisionReport collisions = report.collisionReport();
        if (!collisions.items().isEmpty()) {
            System.out.println("Collisions:");
            for (FeatureCollisionItem collision : collisions.items()) {
                System.out.printf("  warning   %-22s %s%n", collision.kind(), collision.value());
                for (FeatureCollisionClaim claim : collision.claims()) {
                    System.out.printf("    %s (%s)%n", claim.path(), claim.source());
                }
            }
        }
        if (report.jars().isEmpty()) {
            System.out.println("Jars: none found");
            return;
        }
        System.out.println("Jars:");
        for (FeatureJarReport jar : report.jars()) {
            FeatureJarValidation validation = validateJar(jar, strict);
            String status = validation.status();
            System.out.printf("  %-10s %s%n", status, jar.path());
            if (jar.error() != null) {
                System.out.printf("    error: %s%n", jar.error());
            }
            if (jar.hasManifest()) {
                System.out.printf("    manifest: %s%n", manifestSummary(jar.manifest()));
                if (jar.manifestEntryName() != null && !jar.manifestEntryName().isBlank()) {
                    System.out.printf("    manifest entry: %s%n", jar.manifestEntryName());
                }
                String capabilities = manifestCapabilitiesSummary(jar.manifest());
                if (!capabilities.isBlank()) {
                    System.out.printf("    capabilities: %s%n", capabilities);
                }
            } else if (jar.manifestError() != null) {
                System.out.printf("    manifest error: %s%n", jar.manifestError());
            }
            if (jar.hasInstallLock()) {
                String hash = String.valueOf(jar.installLock().getOrDefault("sha256", ""));
                System.out.printf("    lock: tracked%s%n", hash.isBlank() ? "" : " sha256=" + shortHash(hash));
                printTokenizerLockMetadata(jar, "    ");
            } else {
                printTokenizerArtifactMetadata(jar, "    ");
            }
            for (String provider : jar.providers()) {
                System.out.printf("    provider: %s%n", provider);
            }
            for (String provider : jar.adapterProviders()) {
                System.out.printf("    adapter provider: %s%n", provider);
            }
            for (String provider : jar.modelFamilyProviders()) {
                System.out.printf("    model-family provider: %s%n", provider);
            }
            if (modelFamilyOnly(jar)) {
                List<String> pluginInstallErrors = modelFamilyPluginInstallReadinessErrors(jar);
                System.out.printf("    plugin install: %s%n",
                        pluginInstallErrors.isEmpty() ? "ready" : "not-ready");
                for (String error : pluginInstallErrors) {
                    System.out.printf("    plugin install error: %s%n", error);
                }
            }
            for (String warning : validation.warnings()) {
                System.out.printf("    warning: %s%n", warning);
            }
            for (String error : validation.errors()) {
                System.out.printf("    error: %s%n", error);
            }
        }
    }

    private static void printLegacyMigrationStatus(FeatureLegacyPruneReport report) {
        if (report == null || !report.shouldPrintMigrationStatus()) {
            return;
        }
        FeatureLegacyPruneSummary summary = report.summary();
        System.out.println("Legacy migration:");
        System.out.printf("  source    %s (migration-only, not a runtime extension path)%n", report.sourceRoot());
        if (summary.wouldArchive() > 0) {
            System.out.printf("  info      %d legacy jar(s) are already represented in %s%n",
                    summary.wouldArchive(), report.targetRoot());
            System.out.println("  next      gollek extensions finalize-migration --dry-run");
        }
        int unpaired = report.unpairedLegacyJarCount();
        if (unpaired > 0) {
            System.out.printf("  info      %d legacy jar(s) are not installed in %s%n",
                    unpaired, report.targetRoot());
            System.out.println("  next      gollek extensions migrate --dry-run");
        }
        if (summary.error() > 0) {
            System.out.printf("  warning   %d legacy migration issue(s) need attention%n", summary.error());
        }
    }

    private static void printRepair(FeatureRepairReport report) {
        FeatureRepairSummary summary = report.summary();
        System.out.println("Extension Lock Repair");
        System.out.println("---------------------");
        System.out.printf("%s: %d operation(s), %d changed, %d unchanged, %d skipped, %d error%n",
                report.dryRun() ? "Dry run" : "Applied",
                summary.operations(),
                summary.changed(),
                summary.unchanged(),
                summary.skipped(),
                summary.error());
        for (FeatureRepairOperation operation : report.operations()) {
            if ("unchanged".equals(operation.status())) {
                continue;
            }
            System.out.printf("  %-12s %-18s %s%n",
                    operation.status(),
                    operation.action(),
                    operation.path());
            if (operation.message() != null && !operation.message().isBlank()) {
                System.out.printf("    %s%n", operation.message());
            }
        }
    }

    private static void printMigration(FeatureMigrationReport report) {
        FeatureMigrationSummary summary = report.summary();
        System.out.println("Extension Migration");
        System.out.println("-------------------");
        System.out.printf("From: %s%n", report.sourceRoot());
        System.out.printf("To:   %s%n", report.targetRoot());
        System.out.printf("%s: %d operation(s), %d copied, %d replaced, %d unchanged, %d skipped, %d error%n",
                report.dryRun() ? "Dry run" : "Applied",
                summary.operations(),
                summary.copied(),
                summary.replaced(),
                summary.unchanged(),
                summary.skipped(),
                summary.error());
        if (report.operations().isEmpty()) {
            System.out.println("No legacy extension jars found.");
            return;
        }
        for (FeatureMigrationOperation operation : report.operations()) {
            System.out.printf("  %-12s %-16s %s%n",
                    operation.status(),
                    operation.action(),
                    operation.target());
            if (operation.message() != null && !operation.message().isBlank()) {
                System.out.printf("    %s%n", operation.message());
            }
            if (operation.backup() != null) {
                System.out.printf("    backup: %s%n", operation.backup().path());
            } else if (operation.wouldBackup()) {
                System.out.println("    backup: would create rollback backup before replacing target jar");
            }
        }
    }

    private static void printLegacyPrune(FeatureLegacyPruneReport report) {
        printLegacyArchiveReport(report, "Legacy Extension Prune", "----------------------");
    }

    private static void printFinalizeMigration(FeatureLegacyPruneReport report) {
        printLegacyArchiveReport(report, "Extension Migration Finalize", "----------------------------");
    }

    private static void printLegacyArchiveReport(FeatureLegacyPruneReport report, String title, String rule) {
        FeatureLegacyPruneSummary summary = report.summary();
        System.out.println(title);
        System.out.println(rule);
        System.out.printf("From: %s%n", report.sourceRoot());
        System.out.printf("To:   %s%n", report.targetRoot());
        if (report.target() != null && !report.target().isBlank()) {
            System.out.printf("Target: %s%n", report.target());
        }
        System.out.printf("%s: %d operation(s), %d archived, %d would archive, %d skipped, %d error%n",
                report.dryRun() ? "Dry run" : "Applied",
                summary.operations(),
                summary.archived(),
                summary.wouldArchive(),
                summary.skipped(),
                summary.error());
        if (report.operations().isEmpty()) {
            System.out.println("No legacy extension jars found.");
            return;
        }
        for (FeatureLegacyPruneOperation operation : report.operations()) {
            System.out.printf("  %-14s %-18s %s%n",
                    operation.status(),
                    operation.action(),
                    operation.source());
            if (operation.matched() != null && !operation.matched().isBlank()) {
                System.out.printf("    preferred: %s%n", operation.matched());
            }
            if (operation.message() != null && !operation.message().isBlank()) {
                System.out.printf("    %s%n", operation.message());
            }
            if (operation.matchReasons() != null && !operation.matchReasons().isEmpty()) {
                System.out.printf("    matches: %s%n", String.join(", ", operation.matchReasons()));
            }
            if (operation.backup() != null) {
                System.out.printf("    backup: %s%n", operation.backup().path());
            } else if (operation.wouldBackup()) {
                System.out.println("    backup: would create rollback backup before archiving legacy jar");
            }
        }
    }

    private static void printBackups(
            Path root,
            String target,
            List<FeatureBackupReport> backups,
            FeatureBackupSummary summary) {
        System.out.println("Extension Backups");
        System.out.println("-----------------");
        System.out.printf("Root: %s%n", root);
        if (target != null && !target.isBlank()) {
            System.out.printf("Target: %s%n", target);
        }
        System.out.printf("Summary: %d backup(s), %d jar filename(s), %.1f KB%n",
                summary.count(),
                summary.filenames(),
                summary.sizeBytes() / 1024.0);
        if (backups.isEmpty()) {
            System.out.println("Backups: none found");
            return;
        }
        System.out.printf("%-24s %-28s %-16s %-16s %s%n",
                "CREATED", "FILENAME", "FEATURE", "REASON", "PATH");
        System.out.println("-".repeat(112));
        for (FeatureBackupReport backup : backups) {
            System.out.printf("%-24s %-28s %-16s %-16s %s%n",
                    truncate(backup.createdAt(), 24),
                    truncate(backup.filename(), 28),
                    truncate(manifestValue(backup.jar().manifest(), "id"), 16),
                    truncate(backup.reason(), 16),
                    backup.path());
            if (backup.error() != null && !backup.error().isBlank()) {
                System.out.printf("  error: %s%n", backup.error());
            }
        }
    }

    private static void printBackupPrune(FeatureBackupPruneReport report) {
        FeatureBackupPruneSummary summary = report.summary();
        System.out.println("Extension Backup Prune");
        System.out.println("----------------------");
        System.out.printf("%s: keep=%d, %d operation(s), %d kept, %d deleted, %d would delete, %d error%n",
                report.dryRun() ? "Dry run" : "Applied",
                report.keep(),
                summary.operations(),
                summary.kept(),
                summary.deleted(),
                summary.wouldDelete(),
                summary.error());
        for (FeatureBackupPruneOperation operation : report.operations()) {
            if ("unchanged".equals(operation.status())) {
                continue;
            }
            System.out.printf("  %-12s %-16s %s%n",
                    operation.status(),
                    operation.filename(),
                    operation.path());
            if (operation.message() != null && !operation.message().isBlank()) {
                System.out.printf("    %s%n", operation.message());
            }
        }
    }

    private static void printPackageInspect(FeaturePackageReport report) {
        if (report == null) {
            return;
        }
        Map<String, Object> manifest = report.manifest();
        System.out.println("Extension Package");
        System.out.println("-----------------");
        System.out.printf("path:     %s%n", report.path());
        System.out.printf("package:  %s%n", firstNonBlank(manifestValue(manifest, "id"), "unknown"));
        String version = manifestValue(manifest, "version");
        if (!version.isBlank()) {
            System.out.printf("version:  %s%n", version);
        }
        String requiredSpi = "";
        Object requires = manifest == null ? null : manifest.get("requires");
        if (requires instanceof Map<?, ?> requiresMap) {
            Object spi = requiresMap.get("gollek_spi");
            requiredSpi = spi == null ? "" : String.valueOf(spi).trim();
        }
        if (!requiredSpi.isBlank()) {
            System.out.printf("requires: Gollek SPI %s%n", requiredSpi);
        }
        if (report.sizeBytes() >= 0) {
            System.out.printf("size:     %d bytes%n", report.sizeBytes());
        }
        if (report.sha256() != null && !report.sha256().isBlank()) {
            System.out.printf("sha256:   %s%n", report.sha256());
        }
        System.out.printf("artifact: %s%n", report.artifactEntryName());
        if (report.artifactSizeBytes() >= 0) {
            System.out.printf("artifact size:   %d bytes%n", report.artifactSizeBytes());
        }
        if (report.artifactSha256() != null && !report.artifactSha256().isBlank()) {
            System.out.printf("artifact sha256: %s%n", report.artifactSha256());
        }
    }

    private static void printInspect(FeatureJarReport report, FeatureJarValidation validation) {
        FeatureRuntimeContext runtime = runtimeCompatibilityContext();
        System.out.println("Extension Jar Inspect");
        System.out.println("---------------------");
        System.out.printf("status: %s%n", validation.status());
        System.out.printf("runtime: Gollek %s, Java %s%n", runtime.gollekVersion(), runtime.javaVersion());
        System.out.printf("path:   %s%n", report.path());
        if (report.hasManifest()) {
            System.out.printf("manifest: %s%n", manifestSummary(report.manifest()));
            if (report.manifestEntryName() != null && !report.manifestEntryName().isBlank()) {
                System.out.printf("manifest entry: %s%n", report.manifestEntryName());
            }
            String capabilities = manifestCapabilitiesSummary(report.manifest());
            if (!capabilities.isBlank()) {
                System.out.printf("capabilities: %s%n", capabilities);
            }
        } else if (report.manifestError() != null) {
            System.out.printf("manifest error: %s%n", report.manifestError());
        } else {
            System.out.println("manifest: missing");
        }
        if (report.hasInstallLock()) {
            String hash = String.valueOf(report.installLock().getOrDefault("sha256", ""));
            System.out.printf("lock: tracked%s%n", hash.isBlank() ? "" : " sha256=" + shortHash(hash));
            printTokenizerLockMetadata(report, "");
        } else {
            printTokenizerArtifactMetadata(report, "");
        }
        if (report.providers().isEmpty()) {
            System.out.println("providers: none");
        } else {
            for (String provider : report.providers()) {
                System.out.printf("provider: %s%n", provider);
            }
        }
        if (report.adapterProviders().isEmpty()) {
            System.out.println("adapter providers: none");
        } else {
            for (String provider : report.adapterProviders()) {
                System.out.printf("adapter provider: %s%n", provider);
            }
        }
        if (report.modelFamilyProviders().isEmpty()) {
            System.out.println("model-family providers: none");
        } else {
            for (String provider : report.modelFamilyProviders()) {
                System.out.printf("model-family provider: %s%n", provider);
            }
        }
        if (report.gollekPluginProviders().isEmpty()) {
            System.out.println("gollek plugin providers: none");
        } else {
            for (String provider : report.gollekPluginProviders()) {
                System.out.printf("gollek plugin provider: %s%n", provider);
            }
        }
        if (report.hasPluginDescriptor()) {
            System.out.printf("plugin descriptor: %s%n", report.pluginDescriptorEntryName());
            String mainClass = manifestValue(report.pluginDescriptor(), "mainClass");
            if (!mainClass.isBlank()) {
                System.out.printf("plugin mainClass: %s%n", mainClass);
            }
        } else {
            System.out.println("plugin descriptor: none");
        }
        for (String warning : validation.warnings()) {
            System.out.printf("warning: %s%n", warning);
        }
        for (String error : validation.errors()) {
            System.out.printf("error: %s%n", error);
        }
    }

    private static String manifestSummary(Map<String, Object> manifest) {
        String id = manifestValue(manifest, "id");
        String name = manifestValue(manifest, "name");
        String version = manifestValue(manifest, "version");
        List<String> parts = new ArrayList<>();
        if (!id.isBlank()) {
            parts.add(id);
        }
        if (!name.isBlank()) {
            parts.add("(" + name + ")");
        }
        if (!version.isBlank()) {
            parts.add("v" + version);
        }
        return parts.isEmpty() ? "present" : String.join(" ", parts);
    }

    private static void printTokenizerLockMetadata(FeatureJarReport jar, String indent) {
        if (jar == null || !jar.hasInstallLock()) {
            return;
        }
        String tokenizerMetadata = tokenizerMetadataLockSummary(jar.installLock());
        if (!tokenizerMetadata.isBlank()) {
            System.out.printf("%stokenizer metadata: %s%n", indent, tokenizerMetadata);
        }
        String tokenizerLockStatus = tokenizerInstallLockStatus(jar);
        if (!"not_applicable".equals(tokenizerLockStatus)) {
            System.out.printf("%stokenizer lock: %s%n", indent, tokenizerLockStatus);
        }
    }

    private static void printTokenizerArtifactMetadata(FeatureJarReport jar, String indent) {
        String tokenizerMetadata = tokenizerMetadataArtifactSummary(jar);
        if (!tokenizerMetadata.isBlank()) {
            System.out.printf("%stokenizer metadata: %s%n", indent, tokenizerMetadata);
        }
    }

    private static String tokenizerMetadataArtifactSummary(FeatureJarReport jar) {
        if (jar == null || jar.error() != null) {
            return "";
        }
        try {
            return tokenizerMetadataLockSummary(installLockEntry(normalize(Path.of(jar.path())), jar));
        } catch (Exception e) {
            return "";
        }
    }

    private static String tokenizerMetadataLockSummary(Map<String, Object> lockEntry) {
        if (lockEntry == null || lockEntry.isEmpty()) {
            return "";
        }
        String status = stringValue(lockEntry.get("tokenizer_metadata_status"));
        String reason = stringValue(lockEntry.get("tokenizer_metadata_pending_reason"));
        List<String> kinds = manifestStringList(lockEntry.get("tokenizer_kinds"));
        if (status.isBlank() && !kinds.isEmpty()) {
            status = "ready";
        }
        if (status.isBlank() && kinds.isEmpty()) {
            return "";
        }
        List<String> parts = new ArrayList<>();
        if (!status.isBlank()) {
            parts.add(status);
        }
        if (!kinds.isEmpty()) {
            parts.add("kinds=" + String.join(",", kinds));
        }
        String summary = String.join(" ", parts);
        return reason.isBlank() ? summary : summary + " (" + reason + ")";
    }

    private static String manifestCapabilitiesSummary(Map<String, Object> manifest) {
        List<String> parts = new ArrayList<>();
        for (Map<String, Object> capability : manifestCapabilities(manifest)) {
            String id = manifestValue(capability, "id");
            String kind = manifestCapabilityKind(capability);
            if (!id.isBlank()) {
                parts.add(kind + ":" + id);
            } else {
                parts.add(kind);
            }
        }
        return String.join(", ", parts);
    }

    private static String manifestValue(Map<String, Object> manifest, String key) {
        Object value = manifest == null ? null : manifest.get(key);
        return value == null ? "" : String.valueOf(value).trim();
    }

    private static void putManifestValue(Map<String, Object> target, Map<String, Object> source, String key) {
        Object value = source == null ? null : source.get(key);
        if (value != null && !String.valueOf(value).isBlank()) {
            target.put(key, value);
        }
    }

    private static List<Map<String, Object>> manifestPipelines(Map<String, Object> manifest) {
        return manifestObjectList(manifest, "pipelines");
    }

    private static List<Map<String, Object>> manifestModelFamilies(Map<String, Object> manifest) {
        return manifestObjectList(manifest, "model_families");
    }

    private static List<Map<String, Object>> manifestCapabilities(Map<String, Object> manifest) {
        if (manifest == null || manifest.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> capabilities = new ArrayList<>();
        for (Map<String, Object> pipeline : manifestObjectList(manifest, "pipelines")) {
            Map<String, Object> capability = new LinkedHashMap<>(pipeline);
            capability.putIfAbsent("kind", FeatureAdapterKinds.PIPELINE);
            capability.put("_manifest_source", "pipelines");
            capabilities.add(capability);
        }
        for (Map<String, Object> family : manifestModelFamilies(manifest)) {
            Map<String, Object> item = new LinkedHashMap<>(family);
            item.putIfAbsent("kind", MODEL_FAMILY_KIND);
            item.put("_manifest_source", "model_families");
            capabilities.add(item);
        }
        for (Map<String, Object> capability : manifestObjectList(manifest, "capabilities")) {
            Map<String, Object> item = new LinkedHashMap<>(capability);
            item.putIfAbsent("kind", FeatureAdapterKinds.CAPABILITY);
            item.put("_manifest_source", "capabilities");
            capabilities.add(item);
        }
        for (Map<String, Object> adapter : manifestObjectList(manifest, "adapters")) {
            Map<String, Object> item = new LinkedHashMap<>(adapter);
            item.putIfAbsent("kind", FeatureAdapterKinds.ADAPTER);
            item.put("_manifest_source", "adapters");
            capabilities.add(item);
        }

        Set<String> seen = new LinkedHashSet<>();
        List<Map<String, Object>> deduped = new ArrayList<>();
        for (Map<String, Object> capability : capabilities) {
            String key = manifestCapabilityKind(capability) + "|"
                    + manifestValue(capability, "id") + "|"
                    + manifestValue(capability, "class") + "|"
                    + manifestValue(capability, "provider");
            if (seen.add(key)) {
                deduped.add(capability);
            }
        }
        return List.copyOf(deduped);
    }

    private static List<Map<String, Object>> manifestObjectList(Map<String, Object> manifest, String key) {
        Object value = manifest == null ? null : manifest.get(key);
        if (!(value instanceof List<?> rawItems)) {
            return List.of();
        }
        List<Map<String, Object>> items = new ArrayList<>();
        for (Object rawItem : rawItems) {
            if (rawItem instanceof Map<?, ?> rawMap) {
                Map<String, Object> item = new LinkedHashMap<>();
                rawMap.forEach((rawKey, rawValue) -> {
                    if (rawKey != null) {
                        item.put(String.valueOf(rawKey), rawValue);
                    }
                });
                items.add(item);
            }
        }
        return List.copyOf(items);
    }

    private static String manifestCapabilityKind(Map<String, Object> capability) {
        String kind = firstNonBlank(
                manifestValue(capability, "kind"),
                manifestValue(capability, "type"),
                manifestValue(capability, "domain"));
        String source = manifestValue(capability, "_manifest_source");
        if (kind.isBlank() && "pipelines".equals(source)) {
            kind = FeatureAdapterKinds.PIPELINE;
        } else if (kind.isBlank() && "model_families".equals(source)) {
            kind = MODEL_FAMILY_KIND;
        } else if (kind.isBlank() && "adapters".equals(source)) {
            kind = FeatureAdapterKinds.ADAPTER;
        }
        return FeatureAdapterKinds.normalize(kind);
    }

    private static List<String> manifestStringList(Object value) {
        if (value == null) {
            return List.of();
        }
        List<String> result = new ArrayList<>();
        if (value instanceof Iterable<?> items) {
            for (Object item : items) {
                if (item != null && !String.valueOf(item).isBlank()) {
                    result.add(String.valueOf(item).trim());
                }
            }
        } else if (value instanceof String text) {
            for (String item : text.split(",")) {
                if (!item.isBlank()) {
                    result.add(item.trim());
                }
            }
        } else if (!String.valueOf(value).isBlank()) {
            result.add(String.valueOf(value).trim());
        }
        return List.copyOf(result);
    }

    private static FeatureRuntimeContext runtimeCompatibilityContext() {
        return new FeatureRuntimeContext(
                runtimeGollekVersion(),
                firstNonBlank(System.getProperty("java.specification.version"), System.getProperty("java.version"), ""),
                isNativeImageRuntime(),
                !isNativeImageRuntime(),
                ignoredLegacyExtensionPathSettings());
    }

    private static List<String> ignoredLegacyExtensionPathSettings() {
        List<String> ignored = new ArrayList<>();
        addIgnoredLegacySetting(ignored, "-D" + LEGACY_FEATURE_PATH_PROPERTY, System.getProperty(LEGACY_FEATURE_PATH_PROPERTY));
        addIgnoredLegacySetting(ignored, LEGACY_FEATURE_PATH_ENV, System.getenv(LEGACY_FEATURE_PATH_ENV));
        return List.copyOf(ignored);
    }

    private static void addIgnoredLegacySetting(List<String> ignored, String name, String value) {
        if (hasUsableText(value)) {
            ignored.add(name + "=" + value.trim());
        }
    }

    private static String runtimeGollekVersion() {
        String quarkusVersion = System.getProperty("quarkus.application.version");
        if (hasUsableText(quarkusVersion)) {
            return quarkusVersion.trim();
        }
        try (var input = FeaturesCommand.class.getResourceAsStream("/META-INF/gollek-version.properties")) {
            if (input != null) {
                var props = new java.util.Properties();
                props.load(input);
                String version = props.getProperty("version");
                if (hasUsableText(version)) {
                    return version.trim();
                }
            }
        } catch (Exception ignored) {
            // Fall through to Maven metadata.
        }
        try (var input = FeaturesCommand.class.getResourceAsStream(
                "/META-INF/maven/tech.kayys.gollek/gollek-cli/pom.properties")) {
            if (input != null) {
                var props = new java.util.Properties();
                props.load(input);
                String version = props.getProperty("version");
                if (hasUsableText(version)) {
                    return version.trim();
                }
            }
        } catch (Exception ignored) {
            // Fall through to default.
        }
        return DEFAULT_RUNTIME_VERSION;
    }

    private static boolean hasUsableText(String value) {
        return value != null && !value.isBlank() && !value.contains("${");
    }

    private static boolean isNativeImageRuntime() {
        String imageCode = System.getProperty("org.graalvm.nativeimage.imagecode", "");
        String imageKind = System.getProperty("org.graalvm.nativeimage.kind", "");
        return "runtime".equalsIgnoreCase(imageCode) || !imageKind.isBlank();
    }

    private static VersionRequirementCheck checkVersionRequirement(String actual, String requirement) {
        String trimmed = requirement == null ? "" : requirement.trim();
        if (trimmed.isBlank() || "*".equals(trimmed)) {
            return new VersionRequirementCheck(true, true);
        }
        if (trimmed.contains("||")) {
            return new VersionRequirementCheck(false, false);
        }
        for (String rawPart : trimmed.split(",")) {
            String part = rawPart.trim();
            if (part.isEmpty()) {
                continue;
            }
            VersionRequirementCheck check = checkSingleVersionRequirement(actual, part);
            if (!check.understood() || !check.compatible()) {
                return check;
            }
        }
        return new VersionRequirementCheck(true, true);
    }

    private static VersionRequirementCheck checkSingleVersionRequirement(String actual, String requirement) {
        String operator = "";
        String expected = requirement;
        for (String candidate : List.of(">=", "<=", "==", "=", ">", "<")) {
            if (requirement.startsWith(candidate)) {
                operator = candidate;
                expected = requirement.substring(candidate.length()).trim();
                break;
            }
        }
        if (expected.isBlank()) {
            return new VersionRequirementCheck(false, false);
        }
        if (operator.isBlank() && hasVersionWildcard(expected)) {
            return new VersionRequirementCheck(versionWildcardMatches(actual, expected), true);
        }
        SimpleVersion actualVersion = SimpleVersion.parse(actual);
        SimpleVersion expectedVersion = SimpleVersion.parse(expected);
        if (!actualVersion.parsed() || !expectedVersion.parsed()) {
            if (operator.isBlank() || "=".equals(operator) || "==".equals(operator)) {
                return new VersionRequirementCheck(normalizeVersionText(actual).equals(normalizeVersionText(expected)), true);
            }
            return new VersionRequirementCheck(false, false);
        }
        int comparison = actualVersion.compareTo(expectedVersion);
        boolean compatible = switch (operator) {
            case ">" -> comparison > 0;
            case ">=" -> comparison >= 0;
            case "<" -> comparison < 0;
            case "<=" -> comparison <= 0;
            case "=", "==" -> comparison == 0;
            default -> false;
        };
        if (operator.isBlank()) {
            compatible = comparison == 0;
        }
        return new VersionRequirementCheck(compatible, true);
    }

    private static boolean hasVersionWildcard(String value) {
        return value != null && (value.contains("*") || value.toLowerCase(Locale.ROOT).contains("x"));
    }

    private static boolean versionWildcardMatches(String actual, String expected) {
        List<String> actualParts = versionCore(actual);
        List<String> expectedParts = versionCore(expected);
        if (actualParts.isEmpty() || expectedParts.isEmpty()) {
            return false;
        }
        for (int i = 0; i < expectedParts.size(); i++) {
            String expectedPart = expectedParts.get(i).toLowerCase(Locale.ROOT);
            if ("*".equals(expectedPart) || "x".equals(expectedPart)) {
                return true;
            }
            if (i >= actualParts.size() || !expectedPart.equals(actualParts.get(i))) {
                return false;
            }
        }
        return actualParts.size() == expectedParts.size();
    }

    private static List<String> versionCore(String value) {
        String normalized = normalizeVersionText(value);
        if (normalized.isBlank()) {
            return List.of();
        }
        String core = normalized.split("[-+]", 2)[0];
        List<String> parts = new ArrayList<>();
        for (String part : core.split("\\.")) {
            String text = part.trim().toLowerCase(Locale.ROOT);
            if (!text.isBlank()) {
                parts.add(text);
            }
        }
        return parts;
    }

    private static String normalizeVersionText(String value) {
        String text = value == null ? "" : value.trim();
        if (text.startsWith("v") || text.startsWith("V")) {
            text = text.substring(1);
        }
        return text;
    }

    private static boolean matchesTarget(Path root, FeatureJarReport report, String rawTarget) {
        String target = rawTarget == null ? "" : rawTarget.trim();
        if (target.isEmpty()) {
            return false;
        }
        Path path = normalize(Path.of(report.path()));
        String filename = path.getFileName() == null ? "" : path.getFileName().toString();
        String stem = filename.endsWith(".jar") ? filename.substring(0, filename.length() - 4) : filename;
        if (target.equals(filename) || target.equals(stem) || target.equals(path.toString())) {
            return true;
        }
        try {
            Path targetPath = normalize(Path.of(target));
            if (targetPath.equals(path) && targetPath.startsWith(root)) {
                return true;
            }
        } catch (InvalidPathException ignored) {
            // Fall through to manifest/provider matching.
        }
        if (target.equals(manifestValue(report.manifest(), "id"))) {
            return true;
        }
        Map<String, Object> pluginDescriptor = report.pluginDescriptor();
        if (target.equals(manifestValue(pluginDescriptor, "id"))) {
            return true;
        }
        String pluginMainClass = manifestValue(pluginDescriptor, "mainClass");
        if (target.equals(pluginMainClass) || target.equals(simpleClassName(pluginMainClass))) {
            return true;
        }
        for (String provider : report.providers()) {
            if (target.equals(provider) || target.equals(simpleClassName(provider))) {
                return true;
            }
        }
        for (String provider : report.adapterProviders()) {
            if (target.equals(provider) || target.equals(simpleClassName(provider))) {
                return true;
            }
        }
        for (String provider : report.modelFamilyProviders()) {
            if (target.equals(provider) || target.equals(simpleClassName(provider))) {
                return true;
            }
        }
        for (String provider : report.gollekPluginProviders()) {
            if (target.equals(provider) || target.equals(simpleClassName(provider))) {
                return true;
            }
        }
        for (String familyId : manifestModelFamilyIds(report.manifest())) {
            if (target.equals(familyId) || target.equals("model-family/" + familyId)) {
                return true;
            }
        }
        for (String capabilityId : manifestCapabilityIds(report.manifest())) {
            if (target.equals(capabilityId)) {
                return true;
            }
        }
        return false;
    }

    private static List<String> manifestPipelineIds(Map<String, Object> manifest) {
        List<String> ids = new ArrayList<>();
        for (Map<String, Object> pipeline : manifestPipelines(manifest)) {
            Object id = pipeline.get("id");
            if (id != null && !String.valueOf(id).isBlank()) {
                ids.add(String.valueOf(id).trim());
            }
        }
        return ids;
    }

    private static List<String> manifestPipelineClasses(Map<String, Object> manifest) {
        List<String> classes = new ArrayList<>();
        for (Map<String, Object> pipeline : manifestPipelines(manifest)) {
            Object className = pipeline.get("class");
            if (className != null && !String.valueOf(className).isBlank()) {
                classes.add(String.valueOf(className).trim());
            }
        }
        return classes;
    }

    private static List<String> manifestModelFamilyClasses(Map<String, Object> manifest) {
        List<String> classes = new ArrayList<>();
        for (Map<String, Object> family : manifestModelFamilies(manifest)) {
            Object className = family.get("class");
            if (className != null && !String.valueOf(className).isBlank()) {
                classes.add(String.valueOf(className).trim());
            }
        }
        return classes;
    }

    private static List<String> manifestModelFamilyIds(Map<String, Object> manifest) {
        List<String> ids = new ArrayList<>();
        for (Map<String, Object> family : manifestModelFamilies(manifest)) {
            Object id = family.get("id");
            if (id != null && !String.valueOf(id).isBlank()) {
                ids.add(String.valueOf(id).trim());
            }
        }
        return List.copyOf(new ArrayList<>(new LinkedHashSet<>(ids)));
    }

    private static List<String> manifestCapabilityIds(Map<String, Object> manifest) {
        List<String> ids = new ArrayList<>();
        for (Map<String, Object> capability : manifestCapabilities(manifest)) {
            Object id = capability.get("id");
            if (id != null && !String.valueOf(id).isBlank()) {
                ids.add(String.valueOf(id).trim());
            }
        }
        return List.copyOf(new ArrayList<>(new LinkedHashSet<>(ids)));
    }

    private static List<String> manifestCapabilityClasses(Map<String, Object> manifest) {
        List<String> classes = new ArrayList<>();
        for (Map<String, Object> capability : manifestCapabilities(manifest)) {
            Object className = capability.get("class");
            if (className != null && !String.valueOf(className).isBlank()) {
                classes.add(String.valueOf(className).trim());
            }
        }
        return List.copyOf(new ArrayList<>(new LinkedHashSet<>(classes)));
    }

    private static List<String> manifestCapabilityKinds(Map<String, Object> manifest) {
        List<String> kinds = new ArrayList<>();
        for (Map<String, Object> capability : manifestCapabilities(manifest)) {
            kinds.add(manifestCapabilityKind(capability));
        }
        return List.copyOf(new ArrayList<>(new LinkedHashSet<>(kinds)));
    }

    private static boolean isFeatureId(String value) {
        return value != null && value.matches("[a-z0-9][a-z0-9._-]*");
    }

    private static boolean isJavaClassName(String value) {
        if (value == null || value.isBlank() || value.contains("..")) {
            return false;
        }
        String[] parts = value.split("\\.");
        if (parts.length == 0) {
            return false;
        }
        for (String part : parts) {
            if (part.isBlank() || !Character.isJavaIdentifierStart(part.charAt(0))) {
                return false;
            }
            for (int i = 1; i < part.length(); i++) {
                if (!Character.isJavaIdentifierPart(part.charAt(i))) {
                    return false;
                }
            }
        }
        return true;
    }

    private static String simpleClassName(String className) {
        int dot = className == null ? -1 : className.lastIndexOf('.');
        return dot < 0 ? String.valueOf(className) : className.substring(dot + 1);
    }

    private static boolean isJar(Path path) {
        String name = path.getFileName() == null ? "" : path.getFileName().toString();
        return name.endsWith(".jar");
    }

    private static boolean isZip(Path path) {
        String name = path.getFileName() == null ? "" : path.getFileName().toString();
        return name.endsWith(".zip");
    }

    private static Path normalize(Path path) {
        return path.toAbsolutePath().normalize();
    }

    private static void ensureInside(Path root, Path target) {
        Path normalizedRoot = normalize(root);
        Path normalizedTarget = normalize(target);
        if (!normalizedTarget.startsWith(normalizedRoot)) {
            throw new IllegalArgumentException("Path escapes extension directory: " + normalizedTarget);
        }
    }

    private static String codeSource(Class<?> type) {
        try {
            CodeSource codeSource = type.getProtectionDomain().getCodeSource();
            if (codeSource == null || codeSource.getLocation() == null) {
                return "";
            }
            return Path.of(codeSource.getLocation().toURI()).toAbsolutePath().normalize().toString();
        } catch (Exception e) {
            return "";
        }
    }

    private static String metadataText(Map<String, Object> metadata, String... keys) {
        if (metadata == null || metadata.isEmpty()) {
            return "";
        }
        for (String key : keys) {
            Object value = metadata.get(key);
            if (value != null) {
                String text = String.valueOf(value).trim();
                if (!text.isEmpty()) {
                    return text;
                }
            }
        }
        return "";
    }

    private static String truncate(String value, int width) {
        if (value == null) {
            return "";
        }
        if (value.length() <= width) {
            return value;
        }
        return value.substring(0, Math.max(0, width - 3)) + "...";
    }

    private static void printObjectJson(Map<String, Object> value) {
        try {
            System.out.println(objectJson(value));
        } catch (Exception e) {
            System.err.println("Failed to write extension JSON: " + e.getMessage());
        }
    }

    private static String objectJson(Map<String, Object> value) throws java.io.IOException {
        return new ObjectMapper()
                .writerWithDefaultPrettyPrinter()
                .writeValueAsString(value);
    }

    private static void printErrorJson(String action, String error) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("action", action);
        value.put("error", error == null ? "" : error);
        printObjectJson(value);
    }

    private record FeatureRow(
            String id,
            String name,
            String version,
            String kind,
            String family,
            List<String> inputs,
            List<String> outputs,
            List<String> tags,
            Map<String, Object> metadata,
            int priority,
            String source,
            String className,
            String codeSource) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("id", id);
            value.put("name", name);
            value.put("version", version);
            value.put("kind", kind);
            value.put("family", family);
            value.put("inputs", inputs);
            value.put("outputs", outputs);
            value.put("tags", tags);
            value.put("metadata", metadata);
            value.put("priority", priority);
            value.put("source", source);
            value.put("class", className);
            value.put("code_source", codeSource);
            return value;
        }
    }

    private record FeatureRootReport(
            String path,
            boolean exists,
            boolean directory,
            boolean regularFile) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("path", path);
            value.put("exists", exists);
            value.put("directory", directory);
            value.put("regular_file", regularFile);
            return value;
        }
    }

    private record FeatureJarReport(
            String path,
            boolean hasServiceEntry,
            List<String> providers,
            boolean hasAdapterServiceEntry,
            List<String> adapterProviders,
            boolean hasModelFamilyServiceEntry,
            List<String> modelFamilyProviders,
            boolean hasGollekPluginServiceEntry,
            List<String> gollekPluginProviders,
            Map<String, Object> pluginDescriptor,
            String pluginDescriptorEntryName,
            String pluginDescriptorError,
            Map<String, Object> manifest,
            String manifestEntryName,
            String manifestError,
            Map<String, Boolean> classPresence,
            Map<String, Object> installLock,
            List<String> installLockWarnings,
            String error) {
        boolean hasManifest() {
            return manifest != null && !manifest.isEmpty();
        }

        boolean hasPluginDescriptor() {
            return pluginDescriptor != null && !pluginDescriptor.isEmpty();
        }

        boolean hasInstallLock() {
            return installLock != null && !installLock.isEmpty();
        }

        FeatureJarReport withPath(String displayPath) {
            return new FeatureJarReport(
                    displayPath,
                    hasServiceEntry,
                    providers,
                    hasAdapterServiceEntry,
                    adapterProviders,
                    hasModelFamilyServiceEntry,
                    modelFamilyProviders,
                    hasGollekPluginServiceEntry,
                    gollekPluginProviders,
                    pluginDescriptor,
                    pluginDescriptorEntryName,
                    pluginDescriptorError,
                    manifest,
                    manifestEntryName,
                    manifestError,
                    classPresence,
                    installLock,
                    installLockWarnings,
                    error);
        }

        FeatureJarReport withInstallLock(Map<String, Object> lock, List<String> warnings) {
            return new FeatureJarReport(
                    path,
                    hasServiceEntry,
                    providers,
                    hasAdapterServiceEntry,
                    adapterProviders,
                    hasModelFamilyServiceEntry,
                    modelFamilyProviders,
                    hasGollekPluginServiceEntry,
                    gollekPluginProviders,
                    pluginDescriptor,
                    pluginDescriptorEntryName,
                    pluginDescriptorError,
                    manifest,
                    manifestEntryName,
                    manifestError,
                    classPresence,
                    lock == null ? Map.of() : Map.copyOf(lock),
                    warnings == null ? List.of() : List.copyOf(warnings),
                    error);
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("path", path);
            value.put("has_service_entry", hasServiceEntry);
            value.put("providers", providers);
            value.put("has_adapter_service_entry", hasAdapterServiceEntry);
            value.put("adapter_providers", adapterProviders);
            value.put("has_model_family_service_entry", hasModelFamilyServiceEntry);
            value.put("model_family_providers", modelFamilyProviders);
            value.put("has_gollek_plugin_service_entry", hasGollekPluginServiceEntry);
            value.put("gollek_plugin_providers", gollekPluginProviders);
            boolean pluginInstallCandidate = modelFamilyOnly(this);
            value.put("plugin_install_candidate", pluginInstallCandidate);
            if (pluginInstallCandidate) {
                List<String> pluginInstallErrors = modelFamilyPluginInstallReadinessErrors(this);
                value.put("plugin_install_ready", pluginInstallErrors.isEmpty());
                value.put("plugin_install_errors", pluginInstallErrors);
            }
            value.put("has_plugin_descriptor", hasPluginDescriptor());
            if (pluginDescriptorEntryName != null && !pluginDescriptorEntryName.isBlank()) {
                value.put("plugin_descriptor_entry", pluginDescriptorEntryName);
            }
            if (hasPluginDescriptor()) {
                value.put("plugin_descriptor", pluginDescriptor);
            }
            if (pluginDescriptorError != null && !pluginDescriptorError.isBlank()) {
                value.put("plugin_descriptor_error", pluginDescriptorError);
            }
            value.put("has_manifest", hasManifest());
            if (manifestEntryName != null && !manifestEntryName.isBlank()) {
                value.put("manifest_entry", manifestEntryName);
            }
            if (hasManifest()) {
                value.put("manifest", manifest);
            }
            if (manifestError != null && !manifestError.isBlank()) {
                value.put("manifest_error", manifestError);
            }
            if (classPresence != null && !classPresence.isEmpty()) {
                value.put("class_presence", classPresence);
            }
            Map<String, Object> artifactTokenizerMetadata = tokenizerArtifactMetadata(pluginDescriptor, manifest);
            if (hasTokenizerMetadata(artifactTokenizerMetadata)) {
                value.put("artifact_tokenizer_metadata", artifactTokenizerMetadata);
            }
            if (hasInstallLock()) {
                value.put("install_lock", installLock);
                String tokenizerLockStatus = tokenizerInstallLockStatus(this);
                if (!"not_applicable".equals(tokenizerLockStatus)) {
                    value.put("tokenizer_lock_status", tokenizerLockStatus);
                }
            }
            if (installLockWarnings != null && !installLockWarnings.isEmpty()) {
                value.put("install_lock_warnings", installLockWarnings);
            }
            if (error != null && !error.isBlank()) {
                value.put("error", error);
            }
            return value;
        }
    }

    private record FeaturePackageReport(
            String path,
            Map<String, Object> manifest,
            String manifestEntryName,
            String artifactEntryName,
            String installFilename,
            long sizeBytes,
            String sha256,
            long artifactSizeBytes,
            String artifactSha256) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("path", path);
            value.put("manifest_entry", manifestEntryName);
            value.put("artifact_entry", artifactEntryName);
            value.put("install_filename", installFilename);
            if (artifactSizeBytes >= 0) {
                value.put("artifact_size_bytes", artifactSizeBytes);
            }
            if (artifactSha256 != null && !artifactSha256.isBlank()) {
                value.put("artifact_sha256", artifactSha256);
            }
            if (sizeBytes >= 0) {
                value.put("size_bytes", sizeBytes);
            }
            if (sha256 != null && !sha256.isBlank()) {
                value.put("sha256", sha256);
            }
            value.put("manifest", manifest == null ? Map.of() : manifest);
            return value;
        }

        Map<String, Object> toLockMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            Map<String, Object> packageManifest = manifest == null ? Map.of() : manifest;
            value.put("path", path);
            value.put("manifest_entry", manifestEntryName);
            value.put("artifact_entry", artifactEntryName);
            value.put("install_filename", installFilename);
            if (artifactSizeBytes >= 0) {
                value.put("artifact_size_bytes", artifactSizeBytes);
            }
            if (artifactSha256 != null && !artifactSha256.isBlank()) {
                value.put("artifact_sha256", artifactSha256);
            }
            if (sizeBytes >= 0) {
                value.put("size_bytes", sizeBytes);
            }
            if (sha256 != null && !sha256.isBlank()) {
                value.put("sha256", sha256);
            }
            value.put("id", manifestValue(packageManifest, "id"));
            value.put("name", manifestValue(packageManifest, "name"));
            value.put("version", manifestValue(packageManifest, "version"));
            value.put("kind", manifestValue(packageManifest, "kind"));
            Object requires = packageManifest.get("requires");
            if (requires instanceof Map<?, ?> rawRequires) {
                Map<String, Object> copied = new LinkedHashMap<>();
                rawRequires.forEach((key, item) -> {
                    if (key != null) {
                        copied.put(String.valueOf(key), item);
                    }
                });
                value.put("requires", copied);
            }
            return value;
        }
    }

    private static final class FeatureInstallSource implements AutoCloseable {
        private final Path sourcePath;
        private final Path jarPath;
        private final FeaturePackageReport packageReport;
        private final Path tempDir;

        private FeatureInstallSource(
                Path sourcePath,
                Path jarPath,
                FeaturePackageReport packageReport,
                Path tempDir) {
            this.sourcePath = sourcePath;
            this.jarPath = jarPath;
            this.packageReport = packageReport;
            this.tempDir = tempDir;
        }

        Path jarPath() {
            return jarPath;
        }

        FeaturePackageReport packageReport() {
            return packageReport;
        }

        String defaultInstalledName() {
            if (packageReport != null && packageReport.installFilename() != null && !packageReport.installFilename().isBlank()) {
                return packageReport.installFilename();
            }
            return sourcePath.getFileName() == null ? sourcePath.toString() : sourcePath.getFileName().toString();
        }

        String displayJarPath() {
            if (packageReport == null) {
                return jarPath.toString();
            }
            return sourcePath + "!" + packageReport.artifactEntryName();
        }

        @Override
        public void close() throws java.io.IOException {
            deleteRecursively(tempDir);
        }
    }

    private record FeatureDoctorReport(
            List<FeatureRootReport> roots,
            List<FeatureJarReport> jars) {
        Map<String, Object> toMap() {
            return toMap(false);
        }

        Map<String, Object> toMap(boolean strict) {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("runtime", runtimeCompatibilityContext().toMap());
            value.put("roots", roots.stream().map(FeatureRootReport::toMap).toList());
            value.put("summary", validationSummary(strict).toMap());
            value.put("collisions", collisionReport().toMap());
            value.put("jars", jars.stream().map(jar -> jarWithValidation(jar, strict)).toList());
            return value;
        }

        private Map<String, Object> jarWithValidation(FeatureJarReport jar, boolean strict) {
            Map<String, Object> value = jar.toMap();
            value.put("validation", validateJar(jar, strict).toMap());
            return value;
        }

        private FeatureDoctorSummary validationSummary(boolean strict) {
            int ok = 0;
            int warning = 0;
            int error = 0;
            int pluginInstallCandidates = 0;
            int pluginInstallReady = 0;
            int pluginInstallNotReady = 0;
            int tokenizerLockOk = 0;
            int tokenizerLockMissing = 0;
            int tokenizerLockStale = 0;
            for (FeatureJarReport jar : jars) {
                String status = validateJar(jar, strict).status();
                if ("ok".equals(status)) {
                    ok++;
                } else if ("warning".equals(status)) {
                    warning++;
                } else {
                    error++;
                }
                if (modelFamilyOnly(jar)) {
                    pluginInstallCandidates++;
                    if (modelFamilyPluginInstallReadinessErrors(jar).isEmpty()) {
                        pluginInstallReady++;
                    } else {
                        pluginInstallNotReady++;
                    }
                }
                String tokenizerLockStatus = tokenizerInstallLockStatus(jar);
                if ("ok".equals(tokenizerLockStatus)) {
                    tokenizerLockOk++;
                } else if ("missing".equals(tokenizerLockStatus)) {
                    tokenizerLockMissing++;
                } else if ("stale".equals(tokenizerLockStatus)) {
                    tokenizerLockStale++;
                }
            }
            int collisions = collisionReport().items().size();
            if (collisions > 0) {
                if (strict) {
                    error += collisions;
                } else {
                    warning += collisions;
                }
            }
            int ignoredLegacyPaths = runtimeCompatibilityContext().ignoredLegacyExtensionPaths().size();
            if (ignoredLegacyPaths > 0) {
                if (strict) {
                    error += ignoredLegacyPaths;
                } else {
                    warning += ignoredLegacyPaths;
                }
            }
            return new FeatureDoctorSummary(
                    roots.size(),
                    jars.size(),
                    ok,
                    warning,
                    error,
                    collisions,
                    ignoredLegacyPaths,
                    pluginInstallCandidates,
                    pluginInstallReady,
                    pluginInstallNotReady,
                    tokenizerLockOk,
                    tokenizerLockMissing,
                    tokenizerLockStale);
        }

        private FeatureCollisionReport collisionReport() {
            return detectFeatureCollisions(jars);
        }
    }

    private record FeatureInstallLockState(
            Map<String, Map<String, Object>> features,
            String error) {
    }

    private record FeatureBackupRecord(
            String root,
            String sourcePath,
            String path,
            String metadataPath,
            String filename,
            String createdAt,
            String reason,
            Map<String, Object> jar) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("root", root);
            value.put("source_path", sourcePath);
            value.put("path", path);
            value.put("metadata_path", metadataPath);
            value.put("filename", filename);
            value.put("created_at", createdAt);
            value.put("reason", reason);
            value.put("jar", jar);
            return value;
        }
    }

    private record FeatureBackupReport(
            String root,
            String path,
            String metadataPath,
            String filename,
            String createdAt,
            String reason,
            String sourcePath,
            boolean hasMetadata,
            long sizeBytes,
            String sha256,
            FeatureJarReport jar,
            String error) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("root", root);
            value.put("path", path);
            value.put("metadata_path", metadataPath);
            value.put("filename", filename);
            value.put("created_at", createdAt);
            value.put("reason", reason);
            value.put("source_path", sourcePath);
            value.put("has_metadata", hasMetadata);
            if (sizeBytes >= 0) {
                value.put("size_bytes", sizeBytes);
            }
            if (sha256 != null && !sha256.isBlank()) {
                value.put("sha256", sha256);
            }
            value.put("jar", jar.toMap());
            if (error != null && !error.isBlank()) {
                value.put("error", error);
            }
            return value;
        }
    }

    private record FeatureBackupSummary(
            int count,
            int filenames,
            long sizeBytes) {
        static FeatureBackupSummary from(List<FeatureBackupReport> backups) {
            Set<String> filenames = new LinkedHashSet<>();
            long size = 0L;
            for (FeatureBackupReport backup : backups) {
                filenames.add(backup.filename());
                if (backup.sizeBytes() > 0) {
                    size += backup.sizeBytes();
                }
            }
            return new FeatureBackupSummary(backups.size(), filenames.size(), size);
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("count", count);
            value.put("filenames", filenames);
            value.put("size_bytes", sizeBytes);
            return value;
        }
    }

    private record FeatureBackupPruneReport(
            boolean dryRun,
            int keep,
            List<FeatureBackupPruneOperation> operations) {
        FeatureBackupPruneSummary summary() {
            int kept = 0;
            int deleted = 0;
            int wouldDelete = 0;
            int error = 0;
            for (FeatureBackupPruneOperation operation : operations) {
                if ("unchanged".equals(operation.status())) {
                    kept++;
                } else if ("deleted".equals(operation.status())) {
                    deleted++;
                } else if ("would_delete".equals(operation.status())) {
                    wouldDelete++;
                } else if ("error".equals(operation.status())) {
                    error++;
                }
            }
            return new FeatureBackupPruneSummary(operations.size(), kept, deleted, wouldDelete, error);
        }

        boolean hasErrors() {
            return summary().error() > 0;
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("dry_run", dryRun);
            value.put("keep", keep);
            value.put("summary", summary().toMap());
            value.put("operations", operations.stream().map(FeatureBackupPruneOperation::toMap).toList());
            return value;
        }
    }

    private record FeatureBackupPruneSummary(
            int operations,
            int kept,
            int deleted,
            int wouldDelete,
            int error) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("operations", operations);
            value.put("kept", kept);
            value.put("deleted", deleted);
            value.put("would_delete", wouldDelete);
            value.put("error", error);
            return value;
        }
    }

    private record FeatureBackupPruneOperation(
            String path,
            String metadataPath,
            String filename,
            String action,
            String status,
            String message) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("path", path);
            value.put("metadata_path", metadataPath);
            value.put("filename", filename);
            value.put("action", action);
            value.put("status", status);
            if (message != null && !message.isBlank()) {
                value.put("message", message);
            }
            return value;
        }
    }

    private record FeatureBackupSelection(
            Path targetPath,
            Path backupPath) {
    }

    private static final class FeatureRepairPlan {
        private final Path root;
        private final Set<Path> jars = new LinkedHashSet<>();
        private boolean fullDirectory;

        private FeatureRepairPlan(Path root) {
            this.root = root;
        }
    }

    private record FeatureJarMatch(
            FeatureJarReport jar,
            List<String> reasons) {
    }

    private record FeatureLegacyPruneReport(
            Path sourceRoot,
            Path targetRoot,
            String target,
            boolean dryRun,
            List<FeatureLegacyPruneOperation> operations) {
        FeatureLegacyPruneSummary summary() {
            int archived = 0;
            int wouldArchive = 0;
            int skipped = 0;
            int error = 0;
            for (FeatureLegacyPruneOperation operation : operations) {
                String status = operation.status();
                if ("archived".equals(status)) {
                    archived++;
                } else if ("would_archive".equals(status)) {
                    wouldArchive++;
                } else if ("skipped".equals(status)) {
                    skipped++;
                } else if ("error".equals(status)) {
                    error++;
                }
            }
            return new FeatureLegacyPruneSummary(operations.size(), archived, wouldArchive, skipped, error);
        }

        boolean hasErrors() {
            return summary().error() > 0;
        }

        boolean shouldPrintMigrationStatus() {
            FeatureLegacyPruneSummary summary = summary();
            return summary.wouldArchive() > 0 || summary.error() > 0 || unpairedLegacyJarCount() > 0;
        }

        int unpairedLegacyJarCount() {
            int count = 0;
            for (FeatureLegacyPruneOperation operation : operations) {
                if ("skipped".equals(operation.status())
                        && "no_preferred_match".equals(operation.action())
                        && operation.hasLegacyJar()) {
                    count++;
                }
            }
            return count;
        }

        Map<String, Object> toStatusMap() {
            FeatureLegacyPruneSummary summary = summary();
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("source_root", sourceRoot.toString());
            value.put("target_root", targetRoot.toString());
            value.put("runtime_scanned", false);
            value.put("status", migrationStatus());
            value.put("summary", summary.toMap());
            value.put("unpaired_legacy_jars", unpairedLegacyJarCount());
            value.put("next_step", migrationNextStep());
            value.put("operations", operations.stream()
                    .filter(FeatureLegacyPruneOperation::hasLegacyJar)
                    .map(FeatureLegacyPruneOperation::toStatusMap)
                    .toList());
            return value;
        }

        private String migrationStatus() {
            FeatureLegacyPruneSummary summary = summary();
            if (summary.error() > 0) {
                return "needs_attention";
            }
            if (summary.wouldArchive() > 0) {
                return "cleanup_available";
            }
            if (unpairedLegacyJarCount() > 0) {
                return "migration_available";
            }
            return "clean";
        }

        private String migrationNextStep() {
            FeatureLegacyPruneSummary summary = summary();
            if (summary.error() > 0) {
                return "Inspect the legacy migration issue, then rerun gollek extensions migrate --dry-run.";
            }
            if (summary.wouldArchive() > 0) {
                return "gollek extensions finalize-migration --dry-run";
            }
            if (unpairedLegacyJarCount() > 0) {
                return "gollek extensions migrate --dry-run";
            }
            return "";
        }

        Map<String, Object> toMap() {
            return toMap("prune-legacy");
        }

        Map<String, Object> toMap(String action) {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("action", action == null || action.isBlank() ? "prune-legacy" : action);
            value.put("source_root", sourceRoot.toString());
            value.put("target_root", targetRoot.toString());
            value.put("target", target == null ? "" : target);
            value.put("dry_run", dryRun);
            value.put("lockfile", installLockPath(sourceRoot).toString());
            value.put("summary", summary().toMap());
            value.put("operations", operations.stream().map(FeatureLegacyPruneOperation::toMap).toList());
            return value;
        }
    }

    private record FeatureLegacyPruneSummary(
            int operations,
            int archived,
            int wouldArchive,
            int skipped,
            int error) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("operations", operations);
            value.put("archived", archived);
            value.put("would_archive", wouldArchive);
            value.put("skipped", skipped);
            value.put("error", error);
            return value;
        }
    }

    private record FeatureLegacyPruneOperation(
            String source,
            String matched,
            String action,
            String status,
            String message,
            boolean backupCreated,
            boolean wouldBackup,
            FeatureBackupRecord backup,
            FeatureJarReport jar,
            FeatureJarReport matchedJar,
            List<String> matchReasons) {
        static FeatureLegacyPruneOperation archived(
                Path source,
                Path matched,
                boolean dryRun,
                FeatureBackupRecord backup,
                FeatureJarReport jar,
                FeatureJarReport matchedJar,
                List<String> matchReasons) {
            return new FeatureLegacyPruneOperation(
                    source.toString(),
                    matched == null ? "" : matched.toString(),
                    "archive",
                    dryRun ? "would_archive" : "archived",
                    dryRun ? "Legacy jar would be archived because a preferred extension already claims it"
                            : "Legacy jar archived because a preferred extension already claims it",
                    backup != null,
                    dryRun,
                    backup,
                    jar,
                    matchedJar,
                    matchReasons == null ? List.of() : List.copyOf(matchReasons));
        }

        static FeatureLegacyPruneOperation skipped(
                Path source,
                Path matched,
                String action,
                String message,
                FeatureJarReport jar,
                FeatureJarReport matchedJar,
                List<String> matchReasons) {
            return new FeatureLegacyPruneOperation(
                    source.toString(),
                    matched == null ? "" : matched.toString(),
                    action,
                    "skipped",
                    message,
                    false,
                    false,
                    null,
                    jar,
                    matchedJar,
                    matchReasons == null ? List.of() : List.copyOf(matchReasons));
        }

        static FeatureLegacyPruneOperation error(Path source, Path matched, String action, String message) {
            return error(source, matched, action, message, null);
        }

        static FeatureLegacyPruneOperation error(
                Path source,
                Path matched,
                String action,
                String message,
                FeatureJarReport jar) {
            return new FeatureLegacyPruneOperation(
                    source.toString(),
                    matched == null ? "" : matched.toString(),
                    action,
                    "error",
                    message,
                    false,
                    false,
                    null,
                    jar,
                    null,
                    List.of());
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("source", source);
            value.put("matched", matched);
            value.put("action", action);
            value.put("status", status);
            if (message != null && !message.isBlank()) {
                value.put("message", message);
            }
            value.put("backup_created", backupCreated);
            value.put("would_backup", wouldBackup);
            if (backup != null) {
                value.put("backup", backup.toMap());
            }
            if (jar != null) {
                value.put("jar", jar.toMap());
            }
            if (matchedJar != null) {
                value.put("matched_jar", matchedJar.toMap());
            }
            if (matchReasons != null && !matchReasons.isEmpty()) {
                value.put("match_reasons", matchReasons);
            }
            return value;
        }

        boolean hasLegacyJar() {
            return jar != null || "would_archive".equals(status) || "archived".equals(status)
                    || "no_preferred_match".equals(action) || "unreadable".equals(action);
        }

        Map<String, Object> toStatusMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("source", source);
            value.put("matched", matched);
            value.put("action", action);
            value.put("status", status);
            if (message != null && !message.isBlank()) {
                value.put("message", message);
            }
            if (matchReasons != null && !matchReasons.isEmpty()) {
                value.put("match_reasons", matchReasons);
            }
            return value;
        }
    }

    private record FeatureMigrationReport(
            Path sourceRoot,
            Path targetRoot,
            boolean force,
            boolean dryRun,
            List<FeatureMigrationOperation> operations) {
        FeatureMigrationSummary summary() {
            int copied = 0;
            int replaced = 0;
            int unchanged = 0;
            int skipped = 0;
            int error = 0;
            for (FeatureMigrationOperation operation : operations) {
                String status = operation.status();
                if ("copied".equals(status) || "would_copy".equals(status)) {
                    copied++;
                } else if ("replaced".equals(status) || "would_replace".equals(status)) {
                    replaced++;
                } else if ("unchanged".equals(status)) {
                    unchanged++;
                } else if ("skipped".equals(status)) {
                    skipped++;
                } else if ("error".equals(status)) {
                    error++;
                }
            }
            return new FeatureMigrationSummary(operations.size(), copied, replaced, unchanged, skipped, error);
        }

        boolean hasErrors() {
            return summary().error() > 0;
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("action", "migrate");
            value.put("source_root", sourceRoot.toString());
            value.put("target_root", targetRoot.toString());
            value.put("force", force);
            value.put("dry_run", dryRun);
            value.put("lockfile", installLockPath(targetRoot).toString());
            value.put("summary", summary().toMap());
            value.put("operations", operations.stream().map(FeatureMigrationOperation::toMap).toList());
            return value;
        }
    }

    private record FeatureMigrationSummary(
            int operations,
            int copied,
            int replaced,
            int unchanged,
            int skipped,
            int error) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("operations", operations);
            value.put("copied", copied);
            value.put("replaced", replaced);
            value.put("unchanged", unchanged);
            value.put("skipped", skipped);
            value.put("error", error);
            return value;
        }
    }

    private record FeatureMigrationOperation(
            String source,
            String target,
            String action,
            String status,
            String message,
            boolean backupCreated,
            boolean wouldBackup,
            FeatureBackupRecord backup,
            FeatureJarReport jar) {
        static FeatureMigrationOperation copied(
                Path source,
                Path target,
                boolean replaced,
                boolean dryRun,
                FeatureBackupRecord backup,
                boolean wouldBackup,
                String message,
                FeatureJarReport jar) {
            String action = replaced ? "replace" : "copy";
            String status;
            if (dryRun) {
                status = replaced ? "would_replace" : "would_copy";
            } else {
                status = replaced ? "replaced" : "copied";
            }
            String resolvedMessage = message == null || message.isBlank()
                    ? (replaced ? "Target jar replaced" : "Extension jar copied")
                    : message;
            return new FeatureMigrationOperation(
                    source.toString(),
                    target.toString(),
                    action,
                    status,
                    resolvedMessage,
                    backup != null,
                    wouldBackup,
                    backup,
                    jar);
        }

        static FeatureMigrationOperation unchanged(
                Path source,
                Path target,
                boolean dryRun,
                String message,
                FeatureJarReport jar) {
            return new FeatureMigrationOperation(
                    source.toString(),
                    target.toString(),
                    "keep",
                    "unchanged",
                    message,
                    false,
                    false,
                    null,
                    jar);
        }

        static FeatureMigrationOperation skipped(Path source, Path target, String action, String message) {
            return skipped(source, target, action, message, null);
        }

        static FeatureMigrationOperation skipped(
                Path source,
                Path target,
                String action,
                String message,
                FeatureJarReport jar) {
            return new FeatureMigrationOperation(
                    source.toString(),
                    target.toString(),
                    action,
                    "skipped",
                    message,
                    false,
                    false,
                    null,
                    jar);
        }

        static FeatureMigrationOperation error(Path source, Path target, String action, String message) {
            return error(source, target, action, message, null);
        }

        static FeatureMigrationOperation error(
                Path source,
                Path target,
                String action,
                String message,
                FeatureJarReport jar) {
            return new FeatureMigrationOperation(
                    source.toString(),
                    target.toString(),
                    action,
                    "error",
                    message,
                    false,
                    false,
                    null,
                    jar);
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("source", source);
            value.put("target", target);
            value.put("action", action);
            value.put("status", status);
            if (message != null && !message.isBlank()) {
                value.put("message", message);
            }
            value.put("backup_created", backupCreated);
            value.put("would_backup", wouldBackup);
            if (backup != null) {
                value.put("backup", backup.toMap());
            }
            if (jar != null) {
                value.put("jar", jar.toMap());
            }
            return value;
        }
    }

    private record FeatureRepairReport(
            boolean dryRun,
            List<FeatureRepairOperation> operations) {
        FeatureRepairSummary summary() {
            int changed = 0;
            int unchanged = 0;
            int skipped = 0;
            int error = 0;
            for (FeatureRepairOperation operation : operations) {
                String status = operation.status();
                if ("changed".equals(status) || "would_change".equals(status)) {
                    changed++;
                } else if ("unchanged".equals(status)) {
                    unchanged++;
                } else if ("skipped".equals(status)) {
                    skipped++;
                } else if ("error".equals(status)) {
                    error++;
                }
            }
            return new FeatureRepairSummary(operations.size(), changed, unchanged, skipped, error);
        }

        boolean hasErrors() {
            return summary().error() > 0;
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("dry_run", dryRun);
            value.put("summary", summary().toMap());
            value.put("operations", operations.stream().map(FeatureRepairOperation::toMap).toList());
            return value;
        }
    }

    private record FeatureRepairSummary(
            int operations,
            int changed,
            int unchanged,
            int skipped,
            int error) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("operations", operations);
            value.put("changed", changed);
            value.put("unchanged", unchanged);
            value.put("skipped", skipped);
            value.put("error", error);
            return value;
        }
    }

    private record FeatureRepairOperation(
            String root,
            String path,
            String action,
            String status,
            String message) {
        static FeatureRepairOperation error(Path root, Path path, String action, String message) {
            return new FeatureRepairOperation(root.toString(), path.toString(), action, "error", message);
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("root", root);
            value.put("path", path);
            value.put("action", action);
            value.put("status", status);
            if (message != null && !message.isBlank()) {
                value.put("message", message);
            }
            return value;
        }
    }

    private record FeatureCollisionReport(
            List<FeatureCollisionItem> items) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("count", items.size());
            value.put("status", items.isEmpty() ? "ok" : "warning");
            value.put("items", items.stream().map(FeatureCollisionItem::toMap).toList());
            return value;
        }
    }

    private record FeatureCollisionItem(
            String kind,
            String value,
            List<FeatureCollisionClaim> claims) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("kind", kind);
            value.put("value", value());
            value.put("severity", "warning");
            value.put("claims", claims.stream().map(FeatureCollisionClaim::toMap).toList());
            return value;
        }
    }

    private record FeatureCollisionClaim(
            String path,
            String source) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("path", path);
            value.put("source", source);
            return value;
        }
    }

    private record FeatureDoctorSummary(
            int roots,
            int jars,
            int ok,
            int warning,
            int error,
            int collisions,
            int ignoredLegacyPaths,
            int pluginInstallCandidates,
            int pluginInstallReady,
            int pluginInstallNotReady,
            int tokenizerLockOk,
            int tokenizerLockMissing,
            int tokenizerLockStale) {
        int exitCode() {
            return error == 0 ? 0 : 1;
        }

        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("roots", roots);
            value.put("jars", jars);
            value.put("ok", ok);
            value.put("warning", warning);
            value.put("error", error);
            value.put("collisions", collisions);
            value.put("ignored_legacy_paths", ignoredLegacyPaths);
            value.put("plugin_install_candidates", pluginInstallCandidates);
            value.put("plugin_install_ready", pluginInstallReady);
            value.put("plugin_install_not_ready", pluginInstallNotReady);
            value.put("tokenizer_lock_ok", tokenizerLockOk);
            value.put("tokenizer_lock_missing", tokenizerLockMissing);
            value.put("tokenizer_lock_stale", tokenizerLockStale);
            return value;
        }
    }

    private record FeatureRuntimeContext(
            String gollekVersion,
            String javaVersion,
            boolean nativeImage,
            boolean dynamicExtensionLoading,
            List<String> ignoredLegacyExtensionPaths) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("gollek_version", gollekVersion);
            value.put("java_version", javaVersion);
            value.put("native_image", nativeImage);
            value.put("dynamic_extension_loading", dynamicExtensionLoading);
            value.put("extension_loading_mode", dynamicExtensionLoading ? "jvm-dynamic" : "aot-only");
            value.put("ignored_legacy_extension_paths",
                    ignoredLegacyExtensionPaths == null ? List.of() : ignoredLegacyExtensionPaths);
            return value;
        }
    }

    private record VersionRequirementCheck(
            boolean compatible,
            boolean understood) {
    }

    private record SimpleVersion(
            int major,
            int minor,
            int patch,
            boolean parsed) implements Comparable<SimpleVersion> {
        static SimpleVersion parse(String value) {
            List<String> parts = versionCore(value);
            if (parts.isEmpty()) {
                return new SimpleVersion(0, 0, 0, false);
            }
            int major = parseVersionPart(parts, 0);
            int minor = parseVersionPart(parts, 1);
            int patch = parseVersionPart(parts, 2);
            boolean parsed = major >= 0
                    && (parts.size() < 2 || minor >= 0)
                    && (parts.size() < 3 || patch >= 0);
            return new SimpleVersion(
                    Math.max(major, 0),
                    Math.max(minor, 0),
                    Math.max(patch, 0),
                    parsed);
        }

        private static int parseVersionPart(List<String> parts, int index) {
            if (index >= parts.size()) {
                return 0;
            }
            String value = parts.get(index);
            if (!value.matches("\\d+")) {
                return -1;
            }
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                return -1;
            }
        }

        @Override
        public int compareTo(SimpleVersion other) {
            int majorCompare = Integer.compare(major, other.major);
            if (majorCompare != 0) {
                return majorCompare;
            }
            int minorCompare = Integer.compare(minor, other.minor);
            if (minorCompare != 0) {
                return minorCompare;
            }
            return Integer.compare(patch, other.patch);
        }
    }

    private record ServiceEntry(
            boolean present,
            List<String> providers) {
    }

    private record ManifestEntry(
            Map<String, Object> manifest,
            String entryName,
            String error) {
        boolean hasEntry() {
            return (entryName != null && !entryName.isBlank())
                    || (error != null && !error.isBlank())
                    || (manifest != null && !manifest.isEmpty());
        }
    }

    private record FeatureJarValidation(
            boolean valid,
            String status,
            List<String> warnings,
            List<String> errors) {
        Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("valid", valid);
            value.put("status", status);
            value.put("warnings", warnings);
            value.put("errors", errors);
            return value;
        }
    }

    // --- DUMMY CLASSES FOR SPI MIGRATION ---
    public static class FeatureAdapterKinds {
        public static final String PIPELINE = "pipeline";
        public static final String RUNNER = "runner";
        public static final String TRAINING = "training";
        public static final String BACKEND = "backend";
        public static final String CONVERTER = "converter";
        public static final String EXPORTER = "exporter";
        public static final String DATASET = "dataset";
        public static final String EVALUATOR = "evaluator";
        public static final String TOOLING = "tooling";
        public static final String ADAPTER = "adapter";
        public static final String OPTIMIZATION = "optimization";
        public static final String QUANTIZATION = "quantization";
        public static final String CAPABILITY = "capability";
        public static String normalize(String kind) { return kind; }
    }
    public enum ModalityType {
        TEXT, IMAGE, AUDIO, VIDEO, STRUCTURED;
        
    }
    public enum ModelFamilyCapability {
        CHAT, INSTRUCT, COMPLETION;
        
    }
    public interface ModelPipeline {
        String id();
        String version();
        String description();
        int priority();
        PipelineDescriptor descriptor();
    }
    public interface PipelineDescriptor {
        String name();
        String version();
        String description();
        String family();
        java.util.List<ModalityType> inputModalities();
        java.util.List<ModalityType> outputModalities();
        java.util.List<String> tags();
        java.util.Map<String, Object> metadata();
    }
    public interface ModelPipelineRegistry {
        java.util.List<ModelPipeline> all();
        java.util.Optional<ModelPipeline> get(String id);
    }

}
