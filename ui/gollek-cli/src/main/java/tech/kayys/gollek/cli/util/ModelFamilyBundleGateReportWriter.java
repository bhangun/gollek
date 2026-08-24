/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import tech.kayys.gollek.cli.util.ModelFamilyBundleGateReportFields.Root;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyContractViolation;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Writes the model-family bundle release gate as a build artifact for CI archival.
 */
public final class ModelFamilyBundleGateReportWriter {
    private static final int SCHEMA_VERSION = ModelFamilyBundleGateReportFields.SCHEMA_VERSION;
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    private ModelFamilyBundleGateReportWriter() {
    }

    public static void main(String[] args) throws Exception {
        Path output = args.length == 0
                ? Path.of("build/reports/gollek/model-family-bundle-gate.json")
                : Path.of(args[0]);
        Files.createDirectories(output.toAbsolutePath().getParent());
        try (ExternalPluginClasspathScope pluginScope =
                ExternalPluginClasspathScope.open(args, 1, ModelFamilyBundleGateReportWriter.class)) {
            JSON.writeValue(output.toFile(), buildReport(pluginScope.discoveryClassLoader(), pluginScope.classpath()));
        }
        System.out.println("Model-family bundle gate report: " + output.toAbsolutePath());
    }

    static Map<String, Object> buildReport() {
        return buildReport(null);
    }

    static Map<String, Object> buildReport(ClassLoader pluginClassLoader) {
        return buildReport(pluginClassLoader, List.of());
    }

    static Map<String, Object> buildReport(ClassLoader pluginClassLoader, List<Path> pluginClasspath) {
        PluginAvailabilityChecker checker = new PluginAvailabilityChecker();
        ModelFamilyPluginRegistry registry = checker.getModelFamilyPluginRegistry(pluginClassLoader);
        List<String> discoveredFamilyIds = PluginAvailabilityChecker.modelFamilyPluginIds(registry);
        ModelFamilyBundleManifest manifest = checker.getModelFamilyBundleManifest();
        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        manifest,
                        new LinkedHashSet<>(discoveredFamilyIds));
        List<ModelFamilyContractViolation> contractViolations = registry.contractViolations();
        ModelFamilyBundleGate gate = PluginAvailabilityChecker.modelFamilyBundleGate(
                manifest,
                new LinkedHashSet<>(discoveredFamilyIds),
                registry);
        Map<String, Object> contract = contractReport(contractViolations);

        Map<String, Object> report = new LinkedHashMap<>();
        report.put(Root.SCHEMA_VERSION, SCHEMA_VERSION);
        report.put(Root.GENERATED_AT, Instant.now().toString());
        report.put(Root.EXTERNAL_PLUGIN_CLASSPATH, ExternalPluginClasspath.display(pluginClasspath));
        report.put(Root.MODEL_FAMILY_REGISTRY_SCOPE, pluginClassLoader == null ? "global" : "scoped");
        report.put(Root.GATE, ModelFamilyBundleGateReports.gate(gate));
        report.put(Root.AVAILABILITY, ModelFamilyBundleAvailabilityReports.availability(availability));
        report.put(Root.CONTRACT, contract);
        report.put(Root.CONTRACT_VALIDATION, ModelFamilyContractViolationReports.validationReport(contract));
        report.put(Root.MANIFEST, ModelFamilyBundleManifestReports.manifest(manifest));
        report.put(Root.DISCOVERED_FAMILIES, discoveredFamilyIds);
        report.put(Root.FAMILIES, registry.all().stream()
                .flatMap(p -> { try { return java.util.stream.Stream.of(p.supportReport()); } catch (Exception e) { return java.util.stream.Stream.empty(); } })
                .map(ModelFamilyBundleInventoryReports::family)
                .toList());
        report.put(Root.RUNTIME_MANIFESTS, registry.all().stream()
                .flatMap(p -> { try { return java.util.stream.Stream.of(p.runtimeManifest()); } catch (Exception e) { return java.util.stream.Stream.empty(); } })
                .map(ModelFamilyBundleInventoryReports::runtimeManifest)
                .toList());
        report.put(Root.RUNTIME_COMPATIBILITY, ModelFamilyRuntimeCompatibilityReports.compatibility(
                manifest,
                registry));
        return report;
    }

    private static Map<String, Object> contractReport(List<ModelFamilyContractViolation> violations) {
        return ModelFamilyContractViolationReports.summary(violations);
    }
}
