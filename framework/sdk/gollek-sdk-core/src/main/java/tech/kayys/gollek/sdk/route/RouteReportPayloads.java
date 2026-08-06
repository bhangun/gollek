package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RouteReportPayloadContract;
import tech.kayys.gollek.sdk.route.RouteReportPayloadFields;
import tech.kayys.gollek.sdk.route.RouteReportPayloadFields.Payload;
import tech.kayys.gollek.sdk.route.RouteReportPayloadFields.RouteArtifactCacheState;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticContract;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields;
import tech.kayys.gollek.sdk.route.RunnerRouteReportFields;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

public final class RouteReportPayloads {
    private RouteReportPayloads() {
    }

    public static Map<String, Object> routeReportPayload(
            RunnerRouteReport report,
            String requestedModel,
            String effectiveModel,
            String localPath,
            String provider,
            String format) {
        return routeReportPayload(
                report,
                requestedModel,
                effectiveModel,
                localPath,
                provider,
                format,
                false,
                false,
                false);
    }

    public static Map<String, Object> routeReportPayload(
            RunnerRouteReport report,
            String requestedModel,
            String effectiveModel,
            String localPath,
            String provider,
            String format,
            boolean mutationAllowed,
            boolean repositoryResolutionSkipped,
            boolean requireLocal) {
        RoutePreflightReport preflight = RoutePreflightReport.evaluate(
                requestedModel,
                effectiveModel,
                localPath,
                provider,
                format,
                mutationAllowed,
                requireLocal);
        return routeReportPayload(
                report,
                requestedModel,
                effectiveModel,
                localPath,
                provider,
                format,
                mutationAllowed,
                repositoryResolutionSkipped,
                preflight);
    }

    public static Map<String, Object> routeReportPayload(
            RunnerRouteReport report,
            String requestedModel,
            String effectiveModel,
            String localPath,
            String provider,
            String format,
            boolean mutationAllowed,
            boolean repositoryResolutionSkipped,
            RoutePreflightReport preflight) {
        Map<String, Object> metadata = report == null ? Map.of() : report.toMetadata();
        List<Map<String, Object>> preflightProblems = preflight.problemMaps();
        List<Map<String, Object>> nextActions = preflight.nextActionMaps();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put(Payload.TYPE, RouteReportPayloadContract.PAYLOAD_TYPE);
        payload.put(Payload.DRY_RUN, true);
        payload.put(Payload.MUTATION_ALLOWED, mutationAllowed);
        payload.put(Payload.RESOLUTION_MODE, mutationAllowed ? "allow-pull" : "local-only");
        payload.put(Payload.RESOLVED_LOCAL, hasText(localPath));
        payload.put(Payload.RESOLUTION_STATUS, resolutionStatus(localPath, repositoryResolutionSkipped));
        payload.put(Payload.REQUIRE_LOCAL, preflight.requireLocal());
        payload.put(Payload.PREFLIGHT_PASSED, preflight.passed());
        payload.put(Payload.PREFLIGHT_STATUS, preflight.status());
        payload.put(Payload.PREFLIGHT_PROBLEM_COUNT, preflight.problemCount());
        payload.put(Payload.PREFLIGHT_PROBLEM_CODES, problemCodes(preflightProblems));
        payload.put(Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES,
                missingRuntimeCapabilities(preflightProblems));
        payload.put(Payload.PREFLIGHT_PROBLEMS, preflightProblems);
        payload.put(Payload.EXIT_CODE, preflight.exitCode());
        payload.put(Payload.NEXT_ACTION_COUNT, nextActions.size());
        payload.put(Payload.NEXT_ACTION_KINDS, actionKinds(nextActions));
        payload.put(Payload.NEXT_ACTIONS, nextActions);
        payload.put(RoutePreflightDiagnosticFields.VALIDATION_ROOT,
                RoutePreflightDiagnosticContract.diagnosticsValidationReport(
                        payload.get(Payload.PREFLIGHT_PROBLEMS),
                        payload.get(Payload.NEXT_ACTIONS)));
        putIfText(payload, Payload.REQUESTED_MODEL, requestedModel);
        putIfText(payload, Payload.MODEL, effectiveModel);
        putIfText(payload, Payload.LOCAL_PATH, localPath);
        putIfText(payload, Payload.PROVIDER, provider);
        putIfText(payload, Payload.FORMAT, format);
        Object route = metadata.get(RunnerRouteReportFields.METADATA_ROOT);
        Object validation = metadata.get(RunnerRouteReportFields.VALIDATION_METADATA_ROOT);
        Map<?, ?> routeMap = route instanceof Map<?, ?> map ? map : Map.of();
        addRouteRunnerSummary(payload, routeMap);
        addRouteSelectionSourceSummary(payload, routeMap);
        addRouteRedirectSummary(payload, routeMap);
        addRouteArtifactCacheSummary(payload, routeMap);
        addRouteProfileSummary(payload, routeMap);
        payload.put(RunnerRouteReportFields.METADATA_ROOT, routeMap);
        payload.put(RunnerRouteReportFields.VALIDATION_METADATA_ROOT,
                validation instanceof Map<?, ?> ? validation : Map.of());
        payload.put(RouteReportPayloadFields.VALIDATION_ROOT,
                RouteReportPayloadContract.payloadValidationReport(payload));
        return payload;
    }

    public static String resolutionStatus(String localPath, boolean repositoryResolutionSkipped) {
        if (hasText(localPath)) {
            return "resolved";
        }
        return repositoryResolutionSkipped ? "not_local" : "unresolved";
    }

    private static void addRouteRunnerSummary(Map<String, Object> payload, Map<?, ?> route) {
        Object runner = route.get(RunnerRouteReportFields.Report.NORMALIZED_RUNNER);
        if (runner instanceof String text && RunnerRoutePolicy.SUPPORTED_RUNNERS.contains(text)) {
            payload.put(Payload.ROUTE_RUNNER, text);
            return;
        }
        payload.put(Payload.ROUTE_RUNNER, RunnerRoutePolicy.AUTO);
    }

    private static void addRouteSelectionSourceSummary(Map<String, Object> payload, Map<?, ?> route) {
        Object source = route.get(RunnerRouteReportFields.Report.SELECTION_SOURCE);
        if (source instanceof String text && RunnerRouteReportFields.selectionSources().contains(text)) {
            payload.put(Payload.ROUTE_SELECTION_SOURCE, text);
            return;
        }
        payload.put(Payload.ROUTE_SELECTION_SOURCE, RunnerRouteReportFields.SelectionSource.AUTO);
    }

    private static void addRouteRedirectSummary(Map<String, Object> payload, Map<?, ?> route) {
        boolean redirected = Boolean.TRUE.equals(route.get(RunnerRouteReportFields.Report.RUNTIME_REDIRECTED));
        payload.put(Payload.ROUTE_REDIRECTED, redirected);
        Object reason = route.get(RunnerRouteReportFields.Report.RUNTIME_REDIRECT_REASON);
        if (reason instanceof String text && !text.isBlank()) {
            payload.put(Payload.ROUTE_REDIRECT_REASON, text);
        }
    }

    private static void addRouteArtifactCacheSummary(Map<String, Object> payload, Map<?, ?> route) {
        boolean cacheHit = Boolean.TRUE.equals(route.get(
                RunnerRouteReportFields.Report.RUNTIME_REDIRECT_CACHE_HIT));
        boolean runtimeRedirected = Boolean.TRUE.equals(route.get(
                RunnerRouteReportFields.Report.RUNTIME_REDIRECTED));
        payload.put(Payload.ROUTE_ARTIFACT_CACHE_HIT, cacheHit);
        payload.put(Payload.ROUTE_ARTIFACT_CACHE_STATE, routeArtifactCacheState(cacheHit, runtimeRedirected));
        Object cacheKind = route.get(RunnerRouteReportFields.Report.RUNTIME_REDIRECT_CACHE_KIND);
        if (cacheKind instanceof String text && !text.isBlank()) {
            payload.put(Payload.ROUTE_ARTIFACT_CACHE_KIND, text);
        }
    }

    private static void addRouteProfileSummary(Map<String, Object> payload, Map<?, ?> route) {
        Object status = route.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_STATUS);
        if (status instanceof String text && RunnerRouteReportFields.routeProfileStatuses().contains(text)) {
            payload.put(Payload.ROUTE_PROFILE_STATUS, text);
        } else {
            payload.put(Payload.ROUTE_PROFILE_STATUS, RunnerRouteReportFields.RouteProfileStatus.UNAVAILABLE);
        }

        Object source = route.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_SOURCE);
        if (source instanceof String text && RunnerRouteReportFields.routeProfileSources().contains(text)) {
            payload.put(Payload.ROUTE_PROFILE_SOURCE, text);
        } else {
            payload.put(Payload.ROUTE_PROFILE_SOURCE, RunnerRouteReportFields.RouteProfileSource.NONE);
        }

        copyText(payload, Payload.ROUTE_PROFILE_PROVIDER, route,
                RunnerRouteReportFields.Report.ROUTE_PROFILE_PROVIDER);
        copyText(payload, Payload.ROUTE_PROFILE_FORMAT, route,
                RunnerRouteReportFields.Report.ROUTE_PROFILE_FORMAT);
        copyText(payload, Payload.ROUTE_PROFILE_REASON, route,
                RunnerRouteReportFields.Report.ROUTE_PROFILE_REASON);
        copyText(payload, Payload.ROUTE_PROFILE_ADVICE, route,
                RunnerRouteReportFields.Report.ROUTE_PROFILE_ADVICE);
    }

    private static List<String> missingRuntimeCapabilities(List<Map<String, Object>> preflightProblems) {
        LinkedHashSet<String> capabilities = new LinkedHashSet<>();
        for (Map<String, Object> problem : preflightProblems) {
            Object detailsValue = problem.get(RoutePreflightDiagnosticFields.Problem.DETAILS);
            if (!(detailsValue instanceof Map<?, ?> details)) {
                continue;
            }
            Object capability = details.get(
                    RoutePreflightDiagnosticFields.ProblemDetail.MISSING_RUNTIME_CAPABILITY);
            if (capability instanceof String text && !text.isBlank()) {
                capabilities.add(text);
            }
        }
        return List.copyOf(capabilities);
    }

    private static List<String> problemCodes(List<Map<String, Object>> preflightProblems) {
        LinkedHashSet<String> codes = new LinkedHashSet<>();
        for (Map<String, Object> problem : preflightProblems) {
            Object code = problem.get(RoutePreflightDiagnosticFields.Problem.CODE);
            if (code instanceof String text && !text.isBlank()) {
                codes.add(text);
            }
        }
        return List.copyOf(codes);
    }

    private static List<String> actionKinds(List<Map<String, Object>> nextActions) {
        LinkedHashSet<String> kinds = new LinkedHashSet<>();
        for (Map<String, Object> action : nextActions) {
            Object kind = action.get(RoutePreflightDiagnosticFields.Action.KIND);
            if (kind instanceof String text && !text.isBlank()) {
                kinds.add(text);
            }
        }
        return List.copyOf(kinds);
    }

    private static String routeArtifactCacheState(boolean cacheHit, boolean runtimeRedirected) {
        if (cacheHit) {
            return RouteArtifactCacheState.HIT;
        }
        return runtimeRedirected ? RouteArtifactCacheState.MISS : RouteArtifactCacheState.NOT_APPLICABLE;
    }

    private static void putIfText(Map<String, Object> payload, String key, String value) {
        if (hasText(value)) {
            payload.put(key, value);
        }
    }

    private static void copyText(Map<String, Object> payload, String payloadKey, Map<?, ?> route, String routeKey) {
        Object value = route.get(routeKey);
        if (value instanceof String text && hasText(text)) {
            payload.put(payloadKey, text);
        }
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }
}
