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
import tech.kayys.gollek.cli.util.PluginGatesReportFields.Root;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeRegistry;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Writes a single CI artifact for all Gollek plugin release gates.
 */
public final class PluginGatesReportWriter {
    private static final int SCHEMA_VERSION = PluginGatesReportFields.SCHEMA_VERSION;
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    private PluginGatesReportWriter() {
    }

    public static void main(String[] args) throws Exception {
        Path output = args.length == 0
                ? Path.of("build/reports/gollek/plugin-gates.json")
                : Path.of(args[0]);
        Files.createDirectories(output.toAbsolutePath().getParent());
        try (ExternalPluginClasspathScope pluginScope =
                ExternalPluginClasspathScope.open(args, 1, PluginGatesReportWriter.class)) {
            JSON.writeValue(output.toFile(), buildReport(pluginScope.discoveryClassLoader(), pluginScope.classpath()));
        }
        System.out.println("Gollek plugin gates report: " + output.toAbsolutePath());
    }

    static Map<String, Object> buildReport() {
        return buildReport(null);
    }

    static Map<String, Object> buildReport(ClassLoader pluginClassLoader) {
        return buildReport(pluginClassLoader, List.of());
    }

    static Map<String, Object> buildReport(ClassLoader pluginClassLoader, List<Path> pluginClasspath) {
        Map<String, Object> extensionAvailability =
                ExtensionAvailabilityGateReportWriter.buildReport(pluginClassLoader, pluginClasspath);
        Map<String, Object> modelFamilyBundle =
                ModelFamilyBundleGateReportWriter.buildReport(pluginClassLoader, pluginClasspath);
        ExternalPluginClasspath.PluginDirectoryInspection pluginDirectoryInspection =
                ExternalPluginClasspath.inspectPluginClasspath(pluginClasspath);
        UnifiedRuntimeRegistry unifiedRuntimeRegistry = UnifiedRuntimeRegistry.discover(pluginClassLoader);
        PluginAvailabilityChecker checker = new PluginAvailabilityChecker();
        ModelFamilyPluginRegistry modelFamilyRegistry = checker.getModelFamilyPluginRegistry(pluginClassLoader);
        ModelFamilyBundleManifest manifest = checker.getModelFamilyBundleManifest();
        List<UnifiedRuntimeRequirementCompatibility> unifiedRuntimeRequirements =
                UnifiedRuntimeRequirementResolver.evaluate(
                        modelFamilyRegistry,
                        unifiedRuntimeRegistry,
                        manifest.families());
        Map<String, Object> unifiedRuntimes = UnifiedRuntimeRegistryReports.report(
                unifiedRuntimeRegistry,
                unifiedRuntimeRequirements);
        Map<String, Object> runnerRouteContracts = RunnerRouteContractBundleReports.section();
        Map<String, Object> routeBenchmarkCacheReportContract = RouteBenchmarkCacheReportContract.schemaSection();
        PluginGates gate = RouteBenchmarkCacheReportContract.applySchemaGateFromSection(
                RunnerRouteContractBundleReports.applyGateFromSection(
                        PluginGates.withUnifiedRuntimeRequirementReportContract(
                                PluginGates.withUnifiedRuntimeRequirements(
                                        PluginGates.withUnifiedRuntimeReadiness(
                                                PluginGates.withPluginDirectoryReadiness(
                                                        PluginGatesReportReader.gatesFromReports(
                                                                extensionAvailability,
                                                                modelFamilyBundle),
                                                        pluginDirectoryInspection),
                                                unifiedRuntimeRegistry),
                                        unifiedRuntimeRequirements),
                                unifiedRuntimes),
                        runnerRouteContracts),
                routeBenchmarkCacheReportContract);

        Map<String, Object> report = new LinkedHashMap<>();
        report.put(Root.SCHEMA_VERSION, SCHEMA_VERSION);
        report.put(Root.GENERATED_AT, Instant.now().toString());
        report.put(Root.EXTERNAL_PLUGIN_CLASSPATH, ExternalPluginClasspath.display(pluginClasspath));
        report.put(Root.GATE, PluginGatesReports.gate(gate));
        report.put(Root.RUNNER_ROUTE_CONTRACTS, runnerRouteContracts);
        report.put(Root.ROUTE_BENCHMARK_CACHE_REPORT_CONTRACT, routeBenchmarkCacheReportContract);
        report.put(Root.PLUGIN_DIRECTORY_READINESS, PluginDirectoryReadinessReports.report(pluginDirectoryInspection));
        report.put(Root.UNIFIED_RUNTIMES, unifiedRuntimes);
        report.put(Root.EXTENSION_AVAILABILITY, extensionAvailability);
        report.put(Root.MODEL_FAMILY_BUNDLE, modelFamilyBundle);
        return report;
    }

    static PluginGates gatesFromReport(Map<String, Object> report) {
        return PluginGatesReportReader.gatesFromReport(report);
    }
}
