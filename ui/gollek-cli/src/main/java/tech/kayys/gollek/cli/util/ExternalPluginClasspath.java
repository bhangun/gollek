/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.jar.JarFile;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Utility for parsing optional external plugin classpath entries.
 */
public final class ExternalPluginClasspath {
    public static final String MODEL_FAMILY_SERVICE_ENTRY =
            "META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin";
    public static final String GOLLEK_PLUGIN_SERVICE_ENTRY =
            "META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin";
    public static final String UNIFIED_MULTIMODAL_RUNTIME_SERVICE_ENTRY =
            "META-INF/services/tech.kayys.gollek.spi.multimodal.UnifiedMultimodalRuntime";
    public static final String PLUGIN_DESCRIPTOR_ENTRY = "plugin.json";
    public static final String OPTION_PLUGIN_CLASSPATH = "--plugin-classpath";
    public static final String OPTION_EXTERNAL_PLUGIN_CLASSPATH = "--external-plugin-classpath";
    public static final String OPTION_PLUGIN_DIR = "--plugin-dir";
    public static final String OPTION_EXTERNAL_PLUGIN_DIR = "--external-plugin-dir";
    public static final String PROPERTY_PLUGIN_CLASSPATH = "gollek.plugin.classpath";
    public static final String PROPERTY_PLUGIN_DIRS = "gollek.plugin.dirs";
    public static final String PROPERTY_PLUGIN_DIR = "gollek.plugin.dir";
    public static final String PROPERTY_PLUGIN_AUTOLOAD_DEFAULT_DIR = "gollek.plugin.autoloadDefaultDir";
    public static final String ENV_PLUGIN_CLASSPATH = "GOLLEK_PLUGIN_CLASSPATH";
    public static final String ENV_PLUGIN_DIRS = "GOLLEK_PLUGIN_DIRS";
    public static final String ENV_PLUGIN_DIR = "GOLLEK_PLUGIN_DIR";
    public static final String ENV_PLUGIN_AUTOLOAD_DEFAULT_DIR = "GOLLEK_PLUGIN_AUTOLOAD_DEFAULT_DIR";
    public static final String TOKENIZER_METADATA_STATUS_NOT_APPLICABLE = "not_applicable";
    public static final String TOKENIZER_METADATA_STATUS_READY = "ready";
    public static final String TOKENIZER_METADATA_STATUS_PENDING = "pending";
    public static final String TOKENIZER_METADATA_STATUS_MISSING = "missing";
    public static final String TOKENIZER_METADATA_STATUS_INVALID = "invalid";
    public static final List<String> SUPPORTED_TOKENIZER_KINDS = List.of(
            "hf-bpe",
            "sentencepiece",
            "wordpiece",
            "byte-level",
            "char",
            "processor",
            "tokenizer-json",
            "custom-bpe",
            "moses-bpe",
            "japanese-segmentation",
            "jieba",
            "tekken",
            "tiktoken",
            "jamba",
            "entity-aware",
            "composite-tokenizer",
            "audio-processor",
            "vision-processor",
            "vision-text-processor");
    public static final String SUPPORTED_TOKENIZER_KINDS_DESCRIPTION =
            String.join(", ", SUPPORTED_TOKENIZER_KINDS);
    public static final String MODEL_FAMILY_OPTION_DESCRIPTION =
            "Attach external model-family plugin classes or jars (repeat, comma, or path-separated entries)";
    public static final String MODEL_FAMILY_AND_EXTENSION_OPTION_DESCRIPTION =
            "Attach external model-family/extension plugin classes or jars (repeat, comma, or path-separated entries)";
    public static final String PLUGIN_DIRECTORY_OPTION_DESCRIPTION =
            "Attach jars/classes from plugin directories such as ~/.gollek/plugins (repeat, comma, or path-separated)";
    private static final Pattern PLUGIN_ID_PATTERN =
            Pattern.compile("\"id\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern PLUGIN_MAIN_CLASS_PATTERN =
            Pattern.compile("\"mainClass\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern PLUGIN_EXTENSION_POINT_PATTERN =
            Pattern.compile("\"extensionPoint\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern PLUGIN_FAMILIES_PATTERN =
            Pattern.compile("\"families\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL);
    private static final Pattern PLUGIN_BUNDLE_PROFILE_PATTERN =
            Pattern.compile("\"bundleProfile\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern PLUGIN_TOKENIZER_KIND_PATTERN =
            Pattern.compile("\"tokenizerKind\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern PLUGIN_TOKENIZER_KINDS_PATTERN =
            Pattern.compile("\"tokenizerKinds\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL);
    private static final Pattern PLUGIN_TOKENIZER_METADATA_STATUS_PATTERN =
            Pattern.compile("\"tokenizerMetadataStatus\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern PLUGIN_TOKENIZER_METADATA_PENDING_REASON_PATTERN =
            Pattern.compile("\"tokenizerMetadataPendingReason\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern JSON_STRING_PATTERN =
            Pattern.compile("\"([^\"]+)\"");

    private ExternalPluginClasspath() {
    }

    public record PluginDirectoryReport(
            List<Path> commandDirectories,
            List<Path> configuredDirectories,
            Path defaultDirectory,
            boolean defaultDirectoryExists,
            boolean defaultDirectoryAutoloadEnabled) {
        public PluginDirectoryReport {
            commandDirectories = normalize(commandDirectories);
            configuredDirectories = normalize(configuredDirectories);
            defaultDirectory = defaultDirectory == null
                    ? defaultPluginDirectory().toAbsolutePath().normalize()
                    : defaultDirectory.toAbsolutePath().normalize();
        }

        public List<Path> activeDirectories() {
            LinkedHashMap<String, Path> paths = new LinkedHashMap<>();
            addAll(paths, commandDirectories);
            addAll(paths, configuredDirectories);
            return List.copyOf(paths.values());
        }

        public boolean defaultDirectoryActive() {
            return activeDirectories().contains(defaultDirectory);
        }

        private static List<Path> normalize(List<Path> values) {
            if (values == null || values.isEmpty()) {
                return List.of();
            }
            return values.stream()
                    .filter(path -> path != null)
                    .map(path -> path.toAbsolutePath().normalize())
                    .toList();
        }
    }

    public record PluginDirectoryJarReport(
            Path path,
            boolean hasModelFamilyServiceEntry,
            boolean hasGollekPluginServiceEntry,
            List<String> gollekPluginProviders,
            boolean hasUnifiedMultimodalRuntimeServiceEntry,
            List<String> unifiedMultimodalRuntimeProviders,
            List<String> unifiedMultimodalRuntimeMissingProviderClasses,
            boolean hasPluginDescriptor,
            String pluginDescriptorId,
            String pluginExtensionPoint,
            List<String> pluginFamilies,
            String pluginBundleProfile,
            String pluginTokenizerKind,
            List<String> pluginTokenizerKinds,
            String pluginTokenizerMetadataDescriptorStatus,
            String pluginTokenizerMetadataPendingReason,
            String pluginMainClass,
            boolean pluginInstallCandidate,
            boolean pluginInstallReady,
            List<String> pluginInstallErrors,
            String inspectionError) {
        public PluginDirectoryJarReport {
            path = path == null ? null : path.toAbsolutePath().normalize();
            gollekPluginProviders = List.copyOf(gollekPluginProviders == null
                    ? List.of()
                    : gollekPluginProviders);
            unifiedMultimodalRuntimeProviders = List.copyOf(unifiedMultimodalRuntimeProviders == null
                    ? List.of()
                    : unifiedMultimodalRuntimeProviders);
            unifiedMultimodalRuntimeMissingProviderClasses = List.copyOf(
                    unifiedMultimodalRuntimeMissingProviderClasses == null
                            ? List.of()
                            : unifiedMultimodalRuntimeMissingProviderClasses);
            pluginDescriptorId = pluginDescriptorId == null ? "" : pluginDescriptorId;
            pluginExtensionPoint = pluginExtensionPoint == null ? "" : pluginExtensionPoint;
            pluginFamilies = List.copyOf(pluginFamilies == null ? List.of() : pluginFamilies);
            pluginBundleProfile = pluginBundleProfile == null ? "" : pluginBundleProfile;
            pluginTokenizerKind = pluginTokenizerKind == null ? "" : pluginTokenizerKind;
            pluginTokenizerKinds = List.copyOf(pluginTokenizerKinds == null
                    ? List.of()
                    : pluginTokenizerKinds);
            pluginTokenizerMetadataDescriptorStatus = pluginTokenizerMetadataDescriptorStatus == null
                    ? ""
                    : pluginTokenizerMetadataDescriptorStatus;
            pluginTokenizerMetadataPendingReason = pluginTokenizerMetadataPendingReason == null
                    ? ""
                    : pluginTokenizerMetadataPendingReason;
            pluginMainClass = pluginMainClass == null ? "" : pluginMainClass;
            pluginInstallErrors = List.copyOf(pluginInstallErrors == null ? List.of() : pluginInstallErrors);
            inspectionError = inspectionError == null ? "" : inspectionError;
        }

        public String pluginTokenizerMetadataStatus() {
            if (!pluginInstallCandidate) {
                return TOKENIZER_METADATA_STATUS_NOT_APPLICABLE;
            }
            if (hasTokenizerMetadataError()) {
                return TOKENIZER_METADATA_STATUS_INVALID;
            }
            if (TOKENIZER_METADATA_STATUS_PENDING.equals(normalizedTokenizerMetadataStatus(
                    pluginTokenizerMetadataDescriptorStatus))) {
                return TOKENIZER_METADATA_STATUS_PENDING;
            }
            if (pluginTokenizerKinds.isEmpty()) {
                return TOKENIZER_METADATA_STATUS_MISSING;
            }
            return TOKENIZER_METADATA_STATUS_READY;
        }

        public boolean unifiedRuntimeReady() {
            return hasUnifiedMultimodalRuntimeServiceEntry && unifiedRuntimeErrors().isEmpty();
        }

        public List<String> unifiedRuntimeErrors() {
            if (!hasUnifiedMultimodalRuntimeServiceEntry) {
                return List.of();
            }
            if (unifiedMultimodalRuntimeProviders.isEmpty()) {
                return List.of("Unified multimodal runtime service entry has no providers");
            }
            if (!unifiedMultimodalRuntimeMissingProviderClasses.isEmpty()) {
                return List.of("Unified multimodal runtime provider classes are missing from jar: "
                        + String.join(", ", unifiedMultimodalRuntimeMissingProviderClasses));
            }
            return List.of();
        }

        private boolean hasTokenizerMetadataError() {
            return pluginInstallErrors.stream()
                    .anyMatch(error -> error.contains("properties.tokenizerKind")
                            || error.contains("properties.tokenizerKinds")
                            || error.contains("properties.tokenizerMetadataStatus")
                            || error.contains("properties.tokenizerMetadataPendingReason"));
        }
    }

    public record PluginDirectoryInspection(
            List<Path> activeDirectories,
            List<PluginDirectoryJarReport> jars,
            List<String> errors) {
        public PluginDirectoryInspection {
            activeDirectories = List.copyOf(activeDirectories == null ? List.of() : activeDirectories);
            jars = List.copyOf(jars == null ? List.of() : jars);
            errors = List.copyOf(errors == null ? List.of() : errors);
        }

        public int jarCount() {
            return jars.size();
        }

        public int modelFamilyPluginCandidates() {
            return (int) jars.stream()
                    .filter(PluginDirectoryJarReport::pluginInstallCandidate)
                    .count();
        }

        public int unifiedRuntimePluginCandidates() {
            return (int) jars.stream()
                    .filter(PluginDirectoryJarReport::hasUnifiedMultimodalRuntimeServiceEntry)
                    .count();
        }

        public int unifiedRuntimeReady() {
            return (int) jars.stream()
                    .filter(PluginDirectoryJarReport::unifiedRuntimeReady)
                    .count();
        }

        public int unifiedRuntimeNotReady() {
            return (int) jars.stream()
                    .filter(PluginDirectoryJarReport::hasUnifiedMultimodalRuntimeServiceEntry)
                    .filter(report -> !report.unifiedRuntimeReady())
                    .count();
        }

        public int pluginInstallReady() {
            return (int) jars.stream()
                    .filter(PluginDirectoryJarReport::pluginInstallCandidate)
                    .filter(PluginDirectoryJarReport::pluginInstallReady)
                    .count();
        }

        public int pluginInstallNotReady() {
            return (int) jars.stream()
                    .filter(PluginDirectoryJarReport::pluginInstallCandidate)
                    .filter(report -> !report.pluginInstallReady())
                    .count();
        }

        public int pluginTokenizerMetadataReady() {
            return (int) jars.stream()
                    .filter(report -> TOKENIZER_METADATA_STATUS_READY.equals(report.pluginTokenizerMetadataStatus()))
                    .count();
        }

        public int pluginTokenizerMetadataPending() {
            return (int) jars.stream()
                    .filter(report -> TOKENIZER_METADATA_STATUS_PENDING.equals(report.pluginTokenizerMetadataStatus()))
                    .count();
        }

        public int pluginTokenizerMetadataMissing() {
            return (int) jars.stream()
                    .filter(report -> TOKENIZER_METADATA_STATUS_MISSING.equals(report.pluginTokenizerMetadataStatus()))
                    .count();
        }

        public int pluginTokenizerMetadataInvalid() {
            return (int) jars.stream()
                    .filter(report -> TOKENIZER_METADATA_STATUS_INVALID.equals(report.pluginTokenizerMetadataStatus()))
                    .count();
        }

        public List<Path> pluginTokenizerMetadataReadyJars() {
            return pluginTokenizerMetadataJars(TOKENIZER_METADATA_STATUS_READY);
        }

        public List<Path> pluginTokenizerMetadataPendingJars() {
            return pluginTokenizerMetadataJars(TOKENIZER_METADATA_STATUS_PENDING);
        }

        public List<Path> pluginTokenizerMetadataMissingJars() {
            return pluginTokenizerMetadataJars(TOKENIZER_METADATA_STATUS_MISSING);
        }

        public List<Path> pluginTokenizerMetadataInvalidJars() {
            return pluginTokenizerMetadataJars(TOKENIZER_METADATA_STATUS_INVALID);
        }

        public List<Path> unifiedRuntimePluginJars() {
            return jars.stream()
                    .filter(PluginDirectoryJarReport::hasUnifiedMultimodalRuntimeServiceEntry)
                    .map(PluginDirectoryJarReport::path)
                    .filter(path -> path != null)
                    .toList();
        }

        public List<Path> unifiedRuntimeReadyJars() {
            return jars.stream()
                    .filter(PluginDirectoryJarReport::unifiedRuntimeReady)
                    .map(PluginDirectoryJarReport::path)
                    .filter(path -> path != null)
                    .toList();
        }

        public List<Path> unifiedRuntimeNotReadyJars() {
            return jars.stream()
                    .filter(PluginDirectoryJarReport::hasUnifiedMultimodalRuntimeServiceEntry)
                    .filter(report -> !report.unifiedRuntimeReady())
                    .map(PluginDirectoryJarReport::path)
                    .filter(path -> path != null)
                    .toList();
        }

        private List<Path> pluginTokenizerMetadataJars(String status) {
            return jars.stream()
                    .filter(report -> status.equals(report.pluginTokenizerMetadataStatus()))
                    .map(PluginDirectoryJarReport::path)
                    .filter(path -> path != null)
                    .toList();
        }
    }

    public record PluginClasspathArguments(
            List<String> classpathEntries,
            List<String> pluginDirectoryEntries) {
        public PluginClasspathArguments {
            classpathEntries = List.copyOf(classpathEntries == null ? List.of() : classpathEntries);
            pluginDirectoryEntries = List.copyOf(pluginDirectoryEntries == null ? List.of() : pluginDirectoryEntries);
        }
    }

    public static List<Path> parse(String[] args, int startIndex) {
        if (args == null || args.length <= startIndex) {
            return List.of();
        }
        List<String> entries = new ArrayList<>();
        for (int i = Math.max(0, startIndex); i < args.length; i++) {
            entries.add(args[i]);
        }
        return parse(entries);
    }

    public static List<Path> parse(List<String> entries) {
        if (entries == null || entries.isEmpty()) {
            return List.of();
        }
        List<Path> paths = new ArrayList<>();
        for (String entry : entries) {
            if (entry == null) {
                continue;
            }
            for (String item : splitPathListEntry(entry)) {
                if (!item.isBlank()) {
                    paths.add(expandUserHome(item));
                }
            }
        }
        return List.copyOf(paths);
    }

    public static PluginClasspathArguments parseArguments(String[] args, int startIndex) {
        if (args == null || args.length <= startIndex) {
            return new PluginClasspathArguments(List.of(), List.of());
        }
        List<String> classpathEntries = new ArrayList<>();
        List<String> pluginDirectoryEntries = new ArrayList<>();
        for (int i = Math.max(0, startIndex); i < args.length; i++) {
            String arg = args[i];
            if (arg == null || arg.isBlank()) {
                continue;
            }
            String inlineClasspath = inlineOptionValue(
                    arg,
                    OPTION_PLUGIN_CLASSPATH,
                    OPTION_EXTERNAL_PLUGIN_CLASSPATH);
            if (inlineClasspath != null) {
                classpathEntries.add(inlineClasspath);
                continue;
            }
            if (matchesOptionName(arg, OPTION_PLUGIN_CLASSPATH, OPTION_EXTERNAL_PLUGIN_CLASSPATH)) {
                classpathEntries.add(requiredOptionValue(args, ++i, arg));
                continue;
            }
            String inlinePluginDirectory = inlineOptionValue(
                    arg,
                    OPTION_PLUGIN_DIR,
                    OPTION_EXTERNAL_PLUGIN_DIR);
            if (inlinePluginDirectory != null) {
                pluginDirectoryEntries.add(inlinePluginDirectory);
                continue;
            }
            if (matchesOptionName(arg, OPTION_PLUGIN_DIR, OPTION_EXTERNAL_PLUGIN_DIR)) {
                pluginDirectoryEntries.add(requiredOptionValue(args, ++i, arg));
                continue;
            }
            if ("--".equals(arg)) {
                for (i++; i < args.length; i++) {
                    if (args[i] != null && !args[i].isBlank()) {
                        classpathEntries.add(args[i]);
                    }
                }
                break;
            }
            if (arg.startsWith("--")) {
                throw new IllegalArgumentException("Unknown external plugin classpath option: " + arg);
            }
            classpathEntries.add(arg);
        }
        return new PluginClasspathArguments(classpathEntries, pluginDirectoryEntries);
    }

    public static List<Path> resolve(List<String> classpathEntries, List<String> pluginDirectoryEntries) {
        LinkedHashMap<String, Path> paths = new LinkedHashMap<>();
        addAll(paths, parse(classpathEntries));
        addAll(paths, scanPluginDirectories(parse(pluginDirectoryEntries)));
        addAll(paths, persistentClasspath());
        return List.copyOf(paths.values());
    }

    public static List<Path> resolve(String[] args, int startIndex) {
        PluginClasspathArguments parsed = parseArguments(args, startIndex);
        return resolve(parsed.classpathEntries(), parsed.pluginDirectoryEntries());
    }

    public static List<Path> persistentClasspath() {
        LinkedHashMap<String, Path> paths = new LinkedHashMap<>();
        addAll(paths, parse(configuredEntries(PROPERTY_PLUGIN_CLASSPATH, ENV_PLUGIN_CLASSPATH)));
        addAll(paths, scanPluginDirectories(configuredPluginDirectories()));
        return List.copyOf(paths.values());
    }

    public static List<Path> configuredPluginDirectories() {
        List<String> entries = new ArrayList<>();
        entries.addAll(configuredEntries(PROPERTY_PLUGIN_DIRS, ENV_PLUGIN_DIRS));
        entries.addAll(configuredEntries(PROPERTY_PLUGIN_DIR, ENV_PLUGIN_DIR));
        if (autoloadDefaultPluginDirectory()) {
            Path defaultDirectory = defaultPluginDirectory();
            if (Files.exists(defaultDirectory)) {
                entries.add(defaultDirectory.toString());
            }
        }
        return parse(entries);
    }

    public static PluginDirectoryReport pluginDirectoryReport(List<String> commandDirectoryEntries) {
        Path defaultDirectory = defaultPluginDirectory().toAbsolutePath().normalize();
        return new PluginDirectoryReport(
                parse(commandDirectoryEntries),
                configuredPluginDirectories(),
                defaultDirectory,
                Files.exists(defaultDirectory),
                autoloadDefaultPluginDirectory());
    }

    public static PluginDirectoryInspection inspectPluginDirectories(List<String> commandDirectoryEntries) {
        PluginDirectoryReport directoryReport = pluginDirectoryReport(commandDirectoryEntries);
        List<PluginDirectoryJarReport> jars = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        for (Path activeDirectory : directoryReport.activeDirectories()) {
            inspectPluginDirectory(activeDirectory, jars, errors);
        }
        return new PluginDirectoryInspection(directoryReport.activeDirectories(), jars, errors);
    }

    public static PluginDirectoryInspection inspectPluginClasspath(List<Path> pluginClasspath) {
        List<Path> entries = pluginClasspath == null ? List.of() : pluginClasspath.stream()
                .filter(path -> path != null)
                .map(path -> path.toAbsolutePath().normalize())
                .distinct()
                .toList();
        List<PluginDirectoryJarReport> jars = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        for (Path entry : entries) {
            if (!Files.exists(entry)) {
                errors.add("External plugin classpath entry does not exist: " + entry);
                continue;
            }
            if (Files.isRegularFile(entry)) {
                if (isJar(entry)) {
                    jars.add(inspectPluginJar(entry));
                }
                continue;
            }
            if (Files.isDirectory(entry) && !isServiceProviderDirectory(entry)) {
                inspectPluginDirectory(entry, jars, errors);
            }
        }
        return new PluginDirectoryInspection(entries, jars, errors);
    }

    public static Path defaultPluginDirectory() {
        return Path.of(System.getProperty("user.home", "."), ".gollek", "plugins");
    }

    public static List<Path> scanPluginDirectories(List<Path> pluginDirectories) {
        if (pluginDirectories == null || pluginDirectories.isEmpty()) {
            return List.of();
        }

        LinkedHashMap<String, Path> paths = new LinkedHashMap<>();
        for (Path pluginDirectory : pluginDirectories) {
            if (pluginDirectory == null) {
                continue;
            }
            Path normalized = pluginDirectory.toAbsolutePath().normalize();
            if (!Files.exists(normalized)) {
                throw new IllegalArgumentException("External plugin directory does not exist: " + normalized);
            }
            if (Files.isRegularFile(normalized)) {
                if (isJar(normalized)) {
                    add(paths, normalized);
                    continue;
                }
                throw new IllegalArgumentException("External plugin directory is not a directory or jar: " + normalized);
            }
            if (!Files.isDirectory(normalized)) {
                throw new IllegalArgumentException("External plugin directory is not a directory: " + normalized);
            }
            if (isServiceProviderDirectory(normalized)) {
                add(paths, normalized);
                continue;
            }
            scanPluginDirectoryChildren(normalized, paths);
        }
        return List.copyOf(paths.values());
    }

    public static URLClassLoader classLoader(List<Path> classpath, Class<?> parentAnchor) {
        if (classpath == null || classpath.isEmpty()) {
            return null;
        }
        URL[] urls = classpath.stream()
                .map(path -> {
                    try {
                        if (!Files.exists(path)) {
                            throw new IllegalArgumentException("External plugin classpath does not exist: " + path);
                        }
                        return path.toUri().toURL();
                    } catch (IllegalArgumentException e) {
                        throw e;
                    } catch (Exception e) {
                        throw new IllegalArgumentException("Invalid external plugin classpath: " + path, e);
                    }
                })
                .toArray(URL[]::new);
        return new URLClassLoader(urls, parentAnchor.getClassLoader());
    }

    public static List<String> display(List<Path> classpath) {
        if (classpath == null || classpath.isEmpty()) {
            return List.of();
        }
        return classpath.stream()
                .map(path -> path.toAbsolutePath().normalize().toString())
                .toList();
    }

    private static boolean matchesOptionName(String arg, String... names) {
        for (String name : names) {
            if (name.equals(arg)) {
                return true;
            }
        }
        return false;
    }

    private static String inlineOptionValue(String arg, String... names) {
        for (String name : names) {
            String prefix = name + "=";
            if (arg.startsWith(prefix)) {
                String value = arg.substring(prefix.length()).trim();
                if (value.isBlank()) {
                    throw new IllegalArgumentException("Missing value for " + name);
                }
                return value;
            }
        }
        return null;
    }

    private static String requiredOptionValue(String[] args, int valueIndex, String option) {
        if (args == null || valueIndex >= args.length) {
            throw new IllegalArgumentException("Missing value for " + option);
        }
        String value = args[valueIndex];
        if (value == null || value.isBlank() || value.startsWith("--")) {
            throw new IllegalArgumentException("Missing value for " + option);
        }
        return value.trim();
    }

    private static List<String> splitPathListEntry(String entry) {
        if (entry == null || entry.isBlank()) {
            return List.of();
        }
        List<String> items = new ArrayList<>();
        for (String commaPart : entry.split(",")) {
            for (String pathPart : commaPart.split(Pattern.quote(File.pathSeparator))) {
                String item = pathPart.trim();
                if (!item.isBlank()) {
                    items.add(item);
                }
            }
        }
        return items;
    }

    private static List<String> configuredEntries(String propertyName, String environmentName) {
        List<String> entries = new ArrayList<>();
        addConfiguredEntry(entries, System.getProperty(propertyName));
        addConfiguredEntry(entries, System.getenv(environmentName));
        return entries;
    }

    private static void addConfiguredEntry(List<String> entries, String value) {
        if (value != null && !value.isBlank()) {
            entries.add(value);
        }
    }

    public static boolean autoloadDefaultPluginDirectory() {
        String property = System.getProperty(PROPERTY_PLUGIN_AUTOLOAD_DEFAULT_DIR);
        if (property != null && !property.isBlank()) {
            return Boolean.parseBoolean(property);
        }
        String environment = System.getenv(ENV_PLUGIN_AUTOLOAD_DEFAULT_DIR);
        return environment != null && Boolean.parseBoolean(environment);
    }

    private static void scanPluginDirectoryChildren(Path pluginDirectory, LinkedHashMap<String, Path> paths) {
        try (Stream<Path> children = Files.list(pluginDirectory)) {
            children
                    .sorted()
                    .forEach(child -> {
                        if (Files.isRegularFile(child) && isJar(child)) {
                            add(paths, child);
                            return;
                        }
                        if (Files.isDirectory(child)) {
                            Path classesDirectory = child.resolve("classes");
                            if (isServiceProviderDirectory(child)) {
                                add(paths, child);
                            } else if (isServiceProviderDirectory(classesDirectory)) {
                                add(paths, classesDirectory);
                            }
                        }
                    });
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to scan external plugin directory: " + pluginDirectory, e);
        }
    }

    private static void inspectPluginDirectory(
            Path pluginDirectory,
            List<PluginDirectoryJarReport> jars,
            List<String> errors) {
        Path normalized = pluginDirectory == null ? null : pluginDirectory.toAbsolutePath().normalize();
        if (normalized == null) {
            return;
        }
        if (!Files.exists(normalized)) {
            errors.add("Plugin directory does not exist: " + normalized);
            return;
        }
        if (Files.isRegularFile(normalized)) {
            if (isJar(normalized)) {
                jars.add(inspectPluginJar(normalized));
            } else {
                errors.add("Plugin directory entry is not a jar or directory: " + normalized);
            }
            return;
        }
        if (!Files.isDirectory(normalized)) {
            errors.add("Plugin directory entry is not a directory: " + normalized);
            return;
        }
        try (Stream<Path> children = Files.list(normalized)) {
            children
                    .sorted()
                    .filter(Files::isRegularFile)
                    .filter(ExternalPluginClasspath::isJar)
                    .map(ExternalPluginClasspath::inspectPluginJar)
                    .forEach(jars::add);
        } catch (Exception e) {
            errors.add("Failed to inspect plugin directory " + normalized + ": " + e.getMessage());
        }
    }

    private static PluginDirectoryJarReport inspectPluginJar(Path jarPath) {
        Path normalized = jarPath.toAbsolutePath().normalize();
        try (JarFile jar = new JarFile(normalized.toFile())) {
            boolean hasModelFamilyService = jar.getEntry(MODEL_FAMILY_SERVICE_ENTRY) != null;
            boolean hasGollekPluginService = jar.getEntry(GOLLEK_PLUGIN_SERVICE_ENTRY) != null;
            boolean hasUnifiedRuntimeService = jar.getEntry(UNIFIED_MULTIMODAL_RUNTIME_SERVICE_ENTRY) != null;
            boolean hasPluginDescriptor = jar.getEntry(PLUGIN_DESCRIPTOR_ENTRY) != null;
            List<String> gollekPluginProviders = hasGollekPluginService
                    ? readServiceProviders(jar, GOLLEK_PLUGIN_SERVICE_ENTRY)
                    : List.of();
            List<String> unifiedRuntimeProviders = hasUnifiedRuntimeService
                    ? readServiceProviders(jar, UNIFIED_MULTIMODAL_RUNTIME_SERVICE_ENTRY)
                    : List.of();
            List<String> missingUnifiedRuntimeProviders = hasUnifiedRuntimeService
                    ? missingProviderClasses(jar, unifiedRuntimeProviders)
                    : List.of();
            String pluginDescriptor = hasPluginDescriptor ? readJarEntry(jar, PLUGIN_DESCRIPTOR_ENTRY) : "";
            String pluginDescriptorId = readPluginStringField(pluginDescriptor, PLUGIN_ID_PATTERN);
            String pluginExtensionPoint = readPluginStringField(pluginDescriptor, PLUGIN_EXTENSION_POINT_PATTERN);
            List<String> pluginFamilies = readPluginStringArray(pluginDescriptor, PLUGIN_FAMILIES_PATTERN);
            String pluginBundleProfile = readPluginStringField(pluginDescriptor, PLUGIN_BUNDLE_PROFILE_PATTERN);
            String pluginTokenizerKind = readPluginStringField(pluginDescriptor, PLUGIN_TOKENIZER_KIND_PATTERN);
            List<String> pluginTokenizerKinds = pluginTokenizerKinds(
                    pluginTokenizerKind,
                    readPluginStringArray(pluginDescriptor, PLUGIN_TOKENIZER_KINDS_PATTERN));
            String pluginTokenizerMetadataStatus =
                    readPluginStringField(pluginDescriptor, PLUGIN_TOKENIZER_METADATA_STATUS_PATTERN);
            String pluginTokenizerMetadataPendingReason =
                    readPluginStringField(pluginDescriptor, PLUGIN_TOKENIZER_METADATA_PENDING_REASON_PATTERN);
            String pluginMainClass = readPluginStringField(pluginDescriptor, PLUGIN_MAIN_CLASS_PATTERN);
            List<String> readinessErrors = modelFamilyPluginReadinessErrors(
                    hasModelFamilyService,
                    hasGollekPluginService,
                    gollekPluginProviders,
                    hasPluginDescriptor,
                    pluginDescriptorId,
                    pluginExtensionPoint,
                    pluginFamilies,
                    pluginTokenizerKinds,
                    pluginTokenizerMetadataStatus,
                    pluginTokenizerMetadataPendingReason,
                    pluginMainClass);
            return new PluginDirectoryJarReport(
                    normalized,
                    hasModelFamilyService,
                    hasGollekPluginService,
                    gollekPluginProviders,
                    hasUnifiedRuntimeService,
                    unifiedRuntimeProviders,
                    missingUnifiedRuntimeProviders,
                    hasPluginDescriptor,
                    pluginDescriptorId,
                    pluginExtensionPoint,
                    pluginFamilies,
                    pluginBundleProfile,
                    pluginTokenizerKind,
                    pluginTokenizerKinds,
                    pluginTokenizerMetadataStatus,
                    pluginTokenizerMetadataPendingReason,
                    pluginMainClass,
                    hasModelFamilyService,
                    hasModelFamilyService && readinessErrors.isEmpty(),
                    readinessErrors,
                    "");
        } catch (Exception e) {
            return new PluginDirectoryJarReport(
                    normalized,
                    false,
                    false,
                    List.of(),
                    false,
                    List.of(),
                    List.of(),
                    false,
                    "",
                    "",
                    List.of(),
                    "",
                    "",
                    List.of(),
                    "",
                    "",
                    "",
                    false,
                    false,
                    List.of("Failed to inspect plugin jar: " + e.getMessage()),
                    e.getMessage());
        }
    }

    private static List<String> modelFamilyPluginReadinessErrors(
            boolean hasModelFamilyService,
            boolean hasGollekPluginService,
            List<String> gollekPluginProviders,
            boolean hasPluginDescriptor,
            String pluginDescriptorId,
            String pluginExtensionPoint,
            List<String> pluginFamilies,
            List<String> pluginTokenizerKinds,
            String pluginTokenizerMetadataStatus,
            String pluginTokenizerMetadataPendingReason,
            String pluginMainClass) {
        if (!hasModelFamilyService) {
            return List.of();
        }
        List<String> errors = new ArrayList<>();
        String normalizedTokenizerMetadataStatus =
                normalizedTokenizerMetadataStatus(pluginTokenizerMetadataStatus);
        if (!hasGollekPluginService) {
            errors.add("Model-family plugin install requires " + GOLLEK_PLUGIN_SERVICE_ENTRY
                    + " so gollek-plugin-core can discover it");
        } else if (gollekPluginProviders.isEmpty()) {
            errors.add("Model-family plugin install requires at least one GollekPlugin service provider");
        }
        if (!hasPluginDescriptor) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " with mainClass for gollek-plugin-core JarPluginLoader");
        } else if (!pluginDescriptorId.startsWith("model-family/")) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " id to start with model-family/");
        } else if (!"model-family".equals(pluginExtensionPoint)) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.extensionPoint=model-family");
        } else if (pluginFamilies.isEmpty()) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.families with at least one family id");
        } else if (!normalizedTokenizerMetadataStatus.isBlank()
                && !TOKENIZER_METADATA_STATUS_READY.equals(normalizedTokenizerMetadataStatus)
                && !TOKENIZER_METADATA_STATUS_PENDING.equals(normalizedTokenizerMetadataStatus)) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.tokenizerMetadataStatus to be ready or pending when declared");
        } else if (TOKENIZER_METADATA_STATUS_READY.equals(normalizedTokenizerMetadataStatus)
                && pluginTokenizerKinds.isEmpty()) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.tokenizerKind/tokenizerKinds when properties.tokenizerMetadataStatus=ready");
        } else if (TOKENIZER_METADATA_STATUS_PENDING.equals(normalizedTokenizerMetadataStatus)
                && !pluginTokenizerKinds.isEmpty()) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.tokenizerMetadataStatus=pending to omit tokenizerKind/tokenizerKinds");
        } else if (TOKENIZER_METADATA_STATUS_PENDING.equals(normalizedTokenizerMetadataStatus)
                && (pluginTokenizerMetadataPendingReason == null
                || pluginTokenizerMetadataPendingReason.isBlank())) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.tokenizerMetadataPendingReason when tokenizerMetadataStatus=pending");
        } else if (!TOKENIZER_METADATA_STATUS_PENDING.equals(normalizedTokenizerMetadataStatus)
                && pluginTokenizerMetadataPendingReason != null
                && !pluginTokenizerMetadataPendingReason.isBlank()) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.tokenizerMetadataPendingReason only when tokenizerMetadataStatus=pending");
        } else if (pluginTokenizerKinds.stream().anyMatch(kind -> !knownTokenizerKind(kind))) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " properties.tokenizerKind/tokenizerKinds to be one of "
                    + SUPPORTED_TOKENIZER_KINDS_DESCRIPTION + " when declared");
        } else if (pluginMainClass.isBlank()) {
            errors.add("Model-family plugin install requires root " + PLUGIN_DESCRIPTOR_ENTRY
                    + " mainClass for gollek-plugin-core JarPluginLoader");
        } else if (hasGollekPluginService && !gollekPluginProviders.contains(pluginMainClass)) {
            errors.add("Model-family plugin install requires plugin.json mainClass to be listed in "
                    + GOLLEK_PLUGIN_SERVICE_ENTRY + ": " + pluginMainClass);
        }
        return List.copyOf(errors);
    }

    private static String normalizedTokenizerMetadataStatus(String status) {
        return status == null ? "" : status.strip().toLowerCase(Locale.ROOT);
    }

    private static boolean knownTokenizerKind(String tokenizerKind) {
        return SUPPORTED_TOKENIZER_KINDS.contains(tokenizerKind.strip().toLowerCase(Locale.ROOT));
    }

    private static List<String> pluginTokenizerKinds(String primaryKind, List<String> extraKinds) {
        LinkedHashMap<String, String> kinds = new LinkedHashMap<>();
        addTokenizerKind(kinds, primaryKind);
        if (extraKinds != null) {
            extraKinds.forEach(kind -> addTokenizerKind(kinds, kind));
        }
        return List.copyOf(kinds.values());
    }

    private static void addTokenizerKind(LinkedHashMap<String, String> kinds, String kind) {
        if (kind == null || kind.isBlank()) {
            return;
        }
        String value = kind.strip();
        kinds.putIfAbsent(value.toLowerCase(), value);
    }

    private static List<String> readServiceProviders(JarFile jar, String entryName) throws IOException {
        return readJarEntry(jar, entryName)
                .lines()
                .map(String::strip)
                .filter(line -> !line.isBlank())
                .filter(line -> !line.startsWith("#"))
                .distinct()
                .toList();
    }

    private static List<String> missingProviderClasses(JarFile jar, List<String> providers) {
        if (providers == null || providers.isEmpty()) {
            return List.of();
        }
        return providers.stream()
                .filter(provider -> provider != null && !provider.isBlank())
                .filter(provider -> jar.getEntry(provider.replace('.', '/') + ".class") == null)
                .toList();
    }

    private static String readPluginStringField(String json, Pattern pattern) {
        if (json == null || json.isBlank()) {
            return "";
        }
        Matcher matcher = pattern.matcher(json);
        return matcher.find() ? matcher.group(1).strip() : "";
    }

    private static List<String> readPluginStringArray(String json, Pattern pattern) {
        if (json == null || json.isBlank()) {
            return List.of();
        }
        Matcher arrayMatcher = pattern.matcher(json);
        if (!arrayMatcher.find()) {
            return List.of();
        }
        Matcher stringMatcher = JSON_STRING_PATTERN.matcher(arrayMatcher.group(1));
        List<String> values = new ArrayList<>();
        while (stringMatcher.find()) {
            String value = stringMatcher.group(1).strip();
            if (!value.isBlank()) {
                values.add(value);
            }
        }
        return List.copyOf(values);
    }

    private static String readJarEntry(JarFile jar, String entryName) throws IOException {
        var entry = jar.getEntry(entryName);
        if (entry == null) {
            return "";
        }
        try (var input = jar.getInputStream(entry)) {
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static boolean isServiceProviderDirectory(Path directory) {
        return directory != null && Files.isDirectory(directory.resolve("META-INF/services"));
    }

    private static boolean isJar(Path path) {
        String filename = path == null || path.getFileName() == null ? "" : path.getFileName().toString();
        return filename.endsWith(".jar");
    }

    private static Path expandUserHome(String value) {
        String trimmed = value == null ? "" : value.trim();
        if ("~".equals(trimmed)) {
            return Path.of(System.getProperty("user.home", "."));
        }
        if (trimmed.startsWith("~" + File.separator) || trimmed.startsWith("~/")) {
            return Path.of(System.getProperty("user.home", "."), trimmed.substring(2));
        }
        return Path.of(trimmed);
    }

    private static void addAll(LinkedHashMap<String, Path> paths, List<Path> values) {
        if (values == null || values.isEmpty()) {
            return;
        }
        values.forEach(path -> add(paths, path));
    }

    private static void add(LinkedHashMap<String, Path> paths, Path path) {
        if (path == null) {
            return;
        }
        Path normalized = path.toAbsolutePath().normalize();
        paths.putIfAbsent(normalized.toString(), normalized);
    }
}
