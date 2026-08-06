package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.sdk.route.RunnerRouteReportFields;
import tech.kayys.gollek.sdk.route.RunnerRouteReportFields.Report;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Structured explanation of the CLI runner/provider routing decision.
 */
public record RunnerRouteReport(
        String requestedRunner,
        String normalizedRunner,
        String selectionSource,
        boolean runnerExplicit,
        String requestedProvider,
        boolean providerExplicit,
        String requestedFormat,
        String policyProvider,
        String policyFormat,
        String effectiveProvider,
        String effectiveFormat,
        boolean runtimeRedirected,
        String runtimeRedirectFromProvider,
        String runtimeRedirectFromFormat,
        String runtimeRedirectToProvider,
        String runtimeRedirectToFormat,
        String runtimeRedirectReason,
        boolean runtimeRedirectCacheHit,
        String runtimeRedirectCacheKind,
        boolean preferAlternateRuntime,
        boolean forceGguf,
        String mode,
        String routeProfileStatus,
        String routeProfileSource,
        String routeProfileProvider,
        String routeProfileFormat,
        String routeProfileReason,
        String routeProfileAdvice) {

    public static RunnerRouteReport from(
            String requestedRunner,
            String requestedProvider,
            boolean providerExplicit,
            String requestedFormat,
            RunnerRoutePolicy.Selection selection) {
        String normalizedRunner = selection.runner();
        RunnerRoutePerformanceProfile profile = RunnerRoutePerformanceProfile.unavailable();
        return new RunnerRouteReport(
                textOrDefault(requestedRunner, RunnerRoutePolicy.AUTO),
                normalizedRunner,
                selectionSource(requestedRunner, providerExplicit, requestedFormat),
                selection.explicit(),
                blankToNull(requestedProvider),
                providerExplicit,
                blankToNull(requestedFormat),
                blankToNull(selection.providerId()),
                blankToNull(selection.format()),
                blankToNull(selection.providerId()),
                blankToNull(selection.format()),
                false,
                null,
                null,
                null,
                null,
                null,
                false,
                null,
                selection.preferAlternateRuntime(),
                selection.forceGguf(),
                modeFor(selection),
                profile.status(),
                profile.source(),
                profile.provider(),
                profile.format(),
                profile.reason(),
                profile.advice());
    }

    public RunnerRouteReport withEffectiveRoute(String provider, String format) {
        RunnerRouteReport next = new RunnerRouteReport(
                requestedRunner,
                normalizedRunner,
                selectionSource,
                runnerExplicit,
                requestedProvider,
                providerExplicit,
                requestedFormat,
                policyProvider,
                policyFormat,
                blankToNull(provider),
                blankToNull(format),
                runtimeRedirected,
                runtimeRedirectFromProvider,
                runtimeRedirectFromFormat,
                runtimeRedirectToProvider,
                runtimeRedirectToFormat,
                runtimeRedirectReason,
                runtimeRedirectCacheHit,
                runtimeRedirectCacheKind,
                preferAlternateRuntime,
                forceGguf,
                mode,
                routeProfileStatus,
                routeProfileSource,
                routeProfileProvider,
                routeProfileFormat,
                routeProfileReason,
                routeProfileAdvice);
        return next.withRouteProfile(RunnerRoutePerformanceProfile.from(next));
    }

    public RunnerRouteReport withRuntimeRedirect(
            String fromProvider,
            String fromFormat,
            String toProvider,
            String toFormat,
            String reason) {
        return withRuntimeRedirect(fromProvider, fromFormat, toProvider, toFormat, reason, false, null);
    }

    public RunnerRouteReport withRuntimeRedirect(
            String fromProvider,
            String fromFormat,
            String toProvider,
            String toFormat,
            String reason,
            boolean cacheHit,
            String cacheKind) {
        String normalizedReason = mergedRedirectReason(runtimeRedirectReason, reason);
        String normalizedCacheKind = mergedRedirectReason(runtimeRedirectCacheKind, cacheKind);
        RunnerRouteReport next = new RunnerRouteReport(
                requestedRunner,
                normalizedRunner,
                selectionSource,
                runnerExplicit,
                requestedProvider,
                providerExplicit,
                requestedFormat,
                policyProvider,
                policyFormat,
                blankToNull(toProvider),
                blankToNull(toFormat),
                true,
                runtimeRedirected ? runtimeRedirectFromProvider : blankToNull(fromProvider),
                runtimeRedirected ? runtimeRedirectFromFormat : blankToNull(fromFormat),
                blankToNull(toProvider),
                blankToNull(toFormat),
                normalizedReason,
                runtimeRedirectCacheHit || cacheHit,
                normalizedCacheKind,
                preferAlternateRuntime,
                forceGguf,
                mode,
                routeProfileStatus,
                routeProfileSource,
                routeProfileProvider,
                routeProfileFormat,
                routeProfileReason,
                routeProfileAdvice);
        return next.withRouteProfile(RunnerRoutePerformanceProfile.from(next));
    }

    public RunnerRouteReport withRouteProfile(RunnerRoutePerformanceProfile profile) {
        RunnerRoutePerformanceProfile normalized = profile == null
                ? RunnerRoutePerformanceProfile.unavailable()
                : profile;
        return new RunnerRouteReport(
                requestedRunner,
                normalizedRunner,
                selectionSource,
                runnerExplicit,
                requestedProvider,
                providerExplicit,
                requestedFormat,
                policyProvider,
                policyFormat,
                effectiveProvider,
                effectiveFormat,
                runtimeRedirected,
                runtimeRedirectFromProvider,
                runtimeRedirectFromFormat,
                runtimeRedirectToProvider,
                runtimeRedirectToFormat,
                runtimeRedirectReason,
                runtimeRedirectCacheHit,
                runtimeRedirectCacheKind,
                preferAlternateRuntime,
                forceGguf,
                mode,
                blankToNull(normalized.status()),
                blankToNull(normalized.source()),
                blankToNull(normalized.provider()),
                blankToNull(normalized.format()),
                blankToNull(normalized.reason()),
                blankToNull(normalized.advice()));
    }

    public Map<String, Object> toMap() {
        Map<String, Object> report = new LinkedHashMap<>();
        put(report, Report.REQUESTED_RUNNER, requestedRunner);
        put(report, Report.NORMALIZED_RUNNER, normalizedRunner);
        put(report, Report.SELECTION_SOURCE, selectionSource);
        report.put(Report.RUNNER_EXPLICIT, runnerExplicit);
        put(report, Report.REQUESTED_PROVIDER, requestedProvider);
        report.put(Report.PROVIDER_EXPLICIT, providerExplicit);
        put(report, Report.REQUESTED_FORMAT, requestedFormat);
        put(report, Report.POLICY_PROVIDER, policyProvider);
        put(report, Report.POLICY_FORMAT, policyFormat);
        put(report, Report.EFFECTIVE_PROVIDER, effectiveProvider);
        put(report, Report.EFFECTIVE_FORMAT, effectiveFormat);
        report.put(Report.RUNTIME_REDIRECTED, runtimeRedirected);
        put(report, Report.RUNTIME_REDIRECT_FROM_PROVIDER, runtimeRedirectFromProvider);
        put(report, Report.RUNTIME_REDIRECT_FROM_FORMAT, runtimeRedirectFromFormat);
        put(report, Report.RUNTIME_REDIRECT_TO_PROVIDER, runtimeRedirectToProvider);
        put(report, Report.RUNTIME_REDIRECT_TO_FORMAT, runtimeRedirectToFormat);
        put(report, Report.RUNTIME_REDIRECT_REASON, runtimeRedirectReason);
        report.put(Report.RUNTIME_REDIRECT_CACHE_HIT, runtimeRedirectCacheHit);
        put(report, Report.RUNTIME_REDIRECT_CACHE_KIND, runtimeRedirectCacheKind);
        report.put(Report.PREFER_ALTERNATE_RUNTIME, preferAlternateRuntime);
        report.put(Report.FORCE_GGUF, forceGguf);
        put(report, Report.MODE, mode);
        report.put(Report.AUTO_DETECTED, !runnerExplicit && !providerExplicit);
        report.put(Report.PROVIDER_INFERRED, requestedProvider == null && effectiveProvider != null);
        put(report, Report.ROUTE_PROFILE_STATUS, routeProfileStatus);
        put(report, Report.ROUTE_PROFILE_SOURCE, routeProfileSource);
        put(report, Report.ROUTE_PROFILE_PROVIDER, routeProfileProvider);
        put(report, Report.ROUTE_PROFILE_FORMAT, routeProfileFormat);
        put(report, Report.ROUTE_PROFILE_REASON, routeProfileReason);
        put(report, Report.ROUTE_PROFILE_ADVICE, routeProfileAdvice);
        return report;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> report = toMap();
        Map<String, Object> validation = RunnerRouteReportContract.reportValidationReport(report);
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put(RunnerRouteReportFields.METADATA_ROOT, report);
        metadata.put(RunnerRouteReportFields.VALIDATION_METADATA_ROOT, validation);
        report.forEach((key, value) -> metadata.put(RunnerRouteReportFields.metadataKey(key), value));
        validation.forEach((key, value) -> metadata.put(RunnerRouteReportFields.validationMetadataKey(key), value));
        if (requestedProvider != null) {
            metadata.put(Report.REQUESTED_PROVIDER, requestedProvider);
        }
        if (effectiveProvider != null) {
            metadata.put(Report.EFFECTIVE_PROVIDER, effectiveProvider);
        }
        return metadata;
    }

    private static String mergedRedirectReason(String existing, String next) {
        String normalizedNext = blankToNull(next);
        String normalizedExisting = blankToNull(existing);
        if (normalizedExisting == null) {
            return normalizedNext;
        }
        if (normalizedNext == null || normalizedExisting.equals(normalizedNext)) {
            return normalizedExisting;
        }
        return normalizedExisting + " | " + normalizedNext;
    }

    private static String modeFor(RunnerRoutePolicy.Selection selection) {
        if (!selection.explicit()) {
            return RunnerRoutePolicy.AUTO;
        }
        if (RunnerRoutePolicy.HYBRID.equals(selection.runner())) {
            return RunnerRoutePolicy.HYBRID;
        }
        return "fixed";
    }

    private static String selectionSource(
            String requestedRunner,
            boolean providerExplicit,
            String requestedFormat) {
        if (blankToNull(requestedRunner) != null) {
            return RunnerRouteReportFields.SelectionSource.RUNNER_CLI;
        }
        if (providerExplicit) {
            return RunnerRouteReportFields.SelectionSource.PROVIDER_CLI;
        }
        if (blankToNull(requestedFormat) != null) {
            return RunnerRouteReportFields.SelectionSource.FORMAT_CLI;
        }
        return RunnerRouteReportFields.SelectionSource.AUTO;
    }

    private static void put(Map<String, Object> target, String key, String value) {
        if (value != null && !value.isBlank()) {
            target.put(key, value);
        }
    }

    private static String textOrDefault(String value, String fallback) {
        String text = blankToNull(value);
        return text != null ? text : fallback;
    }

    private static String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
