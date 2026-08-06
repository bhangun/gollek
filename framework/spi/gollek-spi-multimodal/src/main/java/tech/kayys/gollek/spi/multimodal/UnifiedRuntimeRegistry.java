/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.spi.multimodal;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;

/**
 * ServiceLoader-backed registry for detachable unified multimodal runtimes.
 */
public final class UnifiedRuntimeRegistry {
    private static final String UNKNOWN_RUNTIME_ID = "unknown-unified-runtime";

    private final List<UnifiedRuntimeReport> reports;

    private UnifiedRuntimeRegistry(List<UnifiedRuntimeReport> reports) {
        this.reports = List.copyOf(reports == null ? List.of() : reports);
    }

    public static UnifiedRuntimeRegistry of(Collection<? extends UnifiedMultimodalRuntime> runtimes) {
        if (runtimes == null || runtimes.isEmpty()) {
            return new UnifiedRuntimeRegistry(List.of());
        }
        return new UnifiedRuntimeRegistry(runtimes.stream()
                .map(UnifiedRuntimeRegistry::report)
                .toList());
    }

    public static UnifiedRuntimeRegistry discover() {
        return discover((ClassLoader) null);
    }

    public static UnifiedRuntimeRegistry discover(ClassLoader classLoader) {
        ServiceLoader<UnifiedMultimodalRuntime> loader = classLoader == null
                ? ServiceLoader.load(UnifiedMultimodalRuntime.class)
                : ServiceLoader.load(UnifiedMultimodalRuntime.class, classLoader);
        return discover(loader);
    }

    public List<UnifiedRuntimeReport> reports() {
        return reports;
    }

    public Optional<UnifiedRuntimeReport> firstSupportingModelType(String modelType) {
        return reportsSupportingModelType(modelType).stream().findFirst();
    }

    public List<UnifiedRuntimeReport> reportsSupportingModelType(String modelType) {
        return reports.stream()
                .filter(report -> report.supportsModelType(modelType))
                .toList();
    }

    public List<UnifiedRuntimeManifestViolation> modelTypeConflicts() {
        Map<String, List<String>> runtimeIdsByModelType = new LinkedHashMap<>();
        for (UnifiedRuntimeReport report : reports) {
            if (report.manifest() == null) {
                continue;
            }
            for (String modelType : report.manifest().modelTypes()) {
                List<String> runtimeIds = new ArrayList<>(runtimeIdsByModelType.getOrDefault(modelType, List.of()));
                runtimeIds.add(report.runtimeId());
                runtimeIdsByModelType.put(modelType, List.copyOf(runtimeIds));
            }
        }
        List<UnifiedRuntimeManifestViolation> conflicts = new ArrayList<>();
        runtimeIdsByModelType.forEach((modelType, runtimeIds) -> {
            List<String> distinctRuntimeIds = runtimeIds.stream().distinct().toList();
            if (distinctRuntimeIds.size() > 1) {
                conflicts.add(new UnifiedRuntimeManifestViolation(
                        String.join(",", distinctRuntimeIds),
                        "duplicate_model_type_claim",
                        "model_type '" + modelType + "' is claimed by "
                                + String.join(", ", distinctRuntimeIds)));
            }
        });
        return List.copyOf(conflicts);
    }

    private static UnifiedRuntimeRegistry discover(ServiceLoader<UnifiedMultimodalRuntime> loader) {
        List<UnifiedRuntimeReport> reports = new ArrayList<>();
        Iterator<UnifiedMultimodalRuntime> iterator = loader.iterator();
        while (true) {
            UnifiedMultimodalRuntime runtime;
            try {
                if (!iterator.hasNext()) {
                    break;
                }
                runtime = iterator.next();
            } catch (ServiceConfigurationError error) {
                reports.add(serviceLoaderError(error));
                continue;
            }
            reports.add(report(runtime));
        }
        return new UnifiedRuntimeRegistry(reports);
    }

    private static UnifiedRuntimeReport report(UnifiedMultimodalRuntime runtime) {
        if (runtime == null) {
            return new UnifiedRuntimeReport(
                    UNKNOWN_RUNTIME_ID,
                    null,
                    List.of(new UnifiedRuntimeManifestViolation(
                            UNKNOWN_RUNTIME_ID,
                            "runtime_null",
                            "unified multimodal runtime instance must not be null")),
                    "runtime instance is null");
        }
        try {
            UnifiedRuntimeManifest manifest = runtime.manifest();
            List<UnifiedRuntimeManifestViolation> violations =
                    UnifiedRuntimeManifestValidator.validate(manifest);
            String runtimeId = manifest == null ? UNKNOWN_RUNTIME_ID : manifest.runtimeId();
            return new UnifiedRuntimeReport(runtimeId, manifest, violations, "manifest loaded");
        } catch (LinkageError | RuntimeException error) {
            String diagnostics = error.getClass().getSimpleName() + ": " + error.getMessage();
            return new UnifiedRuntimeReport(
                    UNKNOWN_RUNTIME_ID,
                    null,
                    List.of(new UnifiedRuntimeManifestViolation(
                            UNKNOWN_RUNTIME_ID,
                            "manifest_unavailable",
                            "runtime manifest could not be loaded: " + diagnostics)),
                    diagnostics);
        }
    }

    private static UnifiedRuntimeReport serviceLoaderError(ServiceConfigurationError error) {
        String diagnostics = error.getClass().getSimpleName() + ": " + error.getMessage();
        return new UnifiedRuntimeReport(
                UNKNOWN_RUNTIME_ID,
                null,
                List.of(new UnifiedRuntimeManifestViolation(
                        UNKNOWN_RUNTIME_ID,
                        "runtime_service_loader_error",
                        "ServiceLoader could not load unified runtime provider: " + diagnostics)),
                diagnostics);
    }

    public record UnifiedRuntimeReport(
            String runtimeId,
            UnifiedRuntimeManifest manifest,
            List<UnifiedRuntimeManifestViolation> violations,
            String diagnostics) {

        public UnifiedRuntimeReport {
            runtimeId = runtimeId == null || runtimeId.isBlank() ? UNKNOWN_RUNTIME_ID : runtimeId.trim();
            violations = List.copyOf(violations == null ? List.of() : violations);
            diagnostics = diagnostics == null || diagnostics.isBlank() ? "unavailable" : diagnostics.trim();
        }

        public boolean manifestAvailable() {
            return manifest != null;
        }

        public boolean valid() {
            return violations.isEmpty();
        }

        public boolean supportsModelType(String modelType) {
            return manifest != null && manifest.supportsModelType(modelType);
        }
    }
}
