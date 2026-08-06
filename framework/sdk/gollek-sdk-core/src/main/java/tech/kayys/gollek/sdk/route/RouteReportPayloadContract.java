/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RouteReportPayloadFields.Payload;
import tech.kayys.gollek.sdk.route.RouteReportPayloadFields.RouteArtifactCacheState;
import tech.kayys.gollek.sdk.route.RouteReportPayloadFields.Schema;
import tech.kayys.gollek.sdk.route.RouteReportPayloadFields.Validation;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight validator for top-level route-report payloads.
 */
public final class RouteReportPayloadContract {
    public static final String PAYLOAD_TYPE = "gollek_route_report";

    private RouteReportPayloadContract() {
    }

    public static Map<String, Object> schema() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put(Schema.CONTRACT_ID, RouteReportPayloadFields.CONTRACT_ID);
        schema.put(Schema.SCHEMA_VERSION, RouteReportPayloadFields.SCHEMA_VERSION);
        schema.put(Schema.SCHEMA_FINGERPRINT, RouteReportPayloadFields.schemaFingerprint());
        schema.put(Schema.VALIDATION_ROOT, RouteReportPayloadFields.VALIDATION_ROOT);
        schema.put(Schema.PAYLOAD_FIELDS, RouteReportPayloadFields.payloadFields());
        schema.put(Schema.REQUIRED_PAYLOAD_FIELDS, RouteReportPayloadFields.requiredPayloadFields());
        schema.put(Schema.OPTIONAL_PAYLOAD_FIELDS, RouteReportPayloadFields.optionalPayloadFields());
        schema.put(Schema.VALIDATION_FIELDS, RouteReportPayloadFields.validationFields());
        return schema;
    }

    public static List<String> validateSchema(Map<?, ?> schema) {
        if (schema == null) {
            return List.of("route report payload schema is missing");
        }
        List<String> problems = new ArrayList<>();
        requireValue(problems, "schema." + Schema.CONTRACT_ID,
                schema.get(Schema.CONTRACT_ID), RouteReportPayloadFields.CONTRACT_ID);
        requireValue(problems, "schema." + Schema.SCHEMA_VERSION,
                schema.get(Schema.SCHEMA_VERSION), RouteReportPayloadFields.SCHEMA_VERSION);
        requireValue(problems, "schema." + Schema.SCHEMA_FINGERPRINT,
                schema.get(Schema.SCHEMA_FINGERPRINT), RouteReportPayloadFields.schemaFingerprint());
        requireValue(problems, "schema." + Schema.VALIDATION_ROOT,
                schema.get(Schema.VALIDATION_ROOT), RouteReportPayloadFields.VALIDATION_ROOT);
        requireList(problems, Schema.PAYLOAD_FIELDS,
                schema.get(Schema.PAYLOAD_FIELDS), RouteReportPayloadFields.payloadFields());
        requireList(problems, Schema.REQUIRED_PAYLOAD_FIELDS,
                schema.get(Schema.REQUIRED_PAYLOAD_FIELDS), RouteReportPayloadFields.requiredPayloadFields());
        requireList(problems, Schema.OPTIONAL_PAYLOAD_FIELDS,
                schema.get(Schema.OPTIONAL_PAYLOAD_FIELDS), RouteReportPayloadFields.optionalPayloadFields());
        requireList(problems, Schema.VALIDATION_FIELDS,
                schema.get(Schema.VALIDATION_FIELDS), RouteReportPayloadFields.validationFields());
        return List.copyOf(problems);
    }

    public static Map<String, Object> schemaValidationReport(Map<?, ?> schema) {
        return validationReport(validateSchema(schema));
    }

    public static List<String> validatePayload(Map<?, ?> payload) {
        if (payload == null) {
            return List.of("route report payload is missing");
        }
        List<String> problems = new ArrayList<>();
        List<String> allowedFields = RouteReportPayloadFields.payloadFields();
        for (Object rawKey : payload.keySet()) {
            String key = String.valueOf(rawKey);
            if (!allowedFields.contains(key)) {
                problems.add("route report payload contains unknown field: " + key);
            }
        }
        for (String required : RouteReportPayloadFields.requiredPayloadFields()) {
            if (!payload.containsKey(required)) {
                problems.add("route report payload missing required field: " + required);
            }
        }
        requireValue(problems, "payload." + Payload.TYPE, payload.get(Payload.TYPE), PAYLOAD_TYPE);
        requireValue(problems, "payload." + Payload.DRY_RUN, payload.get(Payload.DRY_RUN), true);
        requireRouteRunner(problems, payload.get(Payload.ROUTE_RUNNER));
        requireRouteSelectionSource(problems, payload.get(Payload.ROUTE_SELECTION_SOURCE));
        requireBooleanPayload(problems, Payload.ROUTE_REDIRECTED,
                payload.get(Payload.ROUTE_REDIRECTED));
        requireOptionalString(problems, Payload.ROUTE_REDIRECT_REASON,
                payload.get(Payload.ROUTE_REDIRECT_REASON));
        requireBooleanPayload(problems, Payload.ROUTE_ARTIFACT_CACHE_HIT,
                payload.get(Payload.ROUTE_ARTIFACT_CACHE_HIT));
        requireCacheState(problems, payload.get(Payload.ROUTE_ARTIFACT_CACHE_STATE));
        requireOptionalString(problems, Payload.ROUTE_ARTIFACT_CACHE_KIND,
                payload.get(Payload.ROUTE_ARTIFACT_CACHE_KIND));
        requireRouteProfileStatus(problems, payload.get(Payload.ROUTE_PROFILE_STATUS));
        requireRouteProfileSource(problems, payload.get(Payload.ROUTE_PROFILE_SOURCE));
        requireOptionalString(problems, Payload.ROUTE_PROFILE_PROVIDER,
                payload.get(Payload.ROUTE_PROFILE_PROVIDER));
        requireOptionalString(problems, Payload.ROUTE_PROFILE_FORMAT,
                payload.get(Payload.ROUTE_PROFILE_FORMAT));
        requireOptionalString(problems, Payload.ROUTE_PROFILE_REASON,
                payload.get(Payload.ROUTE_PROFILE_REASON));
        requireOptionalString(problems, Payload.ROUTE_PROFILE_ADVICE,
                payload.get(Payload.ROUTE_PROFILE_ADVICE));
        requirePreflightProblemCount(
                problems,
                payload.get(Payload.PREFLIGHT_PROBLEM_COUNT));
        requirePreflightProblemCountMatchesProblems(
                problems,
                payload.get(Payload.PREFLIGHT_PROBLEM_COUNT),
                payload.get(Payload.PREFLIGHT_PROBLEMS));
        requirePreflightProblemCodes(
                problems,
                payload.get(Payload.PREFLIGHT_PROBLEM_CODES));
        requirePreflightProblemCodeSummaryMatchesProblems(
                problems,
                payload.get(Payload.PREFLIGHT_PROBLEM_CODES),
                payload.get(Payload.PREFLIGHT_PROBLEMS));
        requireMissingRuntimeCapabilities(
                problems,
                payload.get(Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES));
        requireMissingRuntimeCapabilitySummaryMatchesProblems(
                problems,
                payload.get(Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES),
                payload.get(Payload.PREFLIGHT_PROBLEMS));
        requireNextActionCount(
                problems,
                payload.get(Payload.NEXT_ACTION_COUNT));
        requireNextActionCountMatchesActions(
                problems,
                payload.get(Payload.NEXT_ACTION_COUNT),
                payload.get(Payload.NEXT_ACTIONS));
        requireNextActionKinds(
                problems,
                payload.get(Payload.NEXT_ACTION_KINDS));
        requireNextActionKindSummaryMatchesActions(
                problems,
                payload.get(Payload.NEXT_ACTION_KINDS),
                payload.get(Payload.NEXT_ACTIONS));
        if (payload.containsKey(RouteReportPayloadFields.VALIDATION_ROOT)) {
            addPrefixed(problems, "payload.",
                    validateValidationReport(payload.get(RouteReportPayloadFields.VALIDATION_ROOT)));
        }
        addPrefixed(problems, "payload.",
                RoutePreflightDiagnosticContract.validateProblems(payload.get(Payload.PREFLIGHT_PROBLEMS)));
        addPrefixed(problems, "payload.",
                RoutePreflightDiagnosticContract.validateActions(payload.get(Payload.NEXT_ACTIONS)));
        addPrefixed(problems, "payload.",
                RoutePreflightDiagnosticContract.validateValidationReport(
                        payload.get(RoutePreflightDiagnosticFields.VALIDATION_ROOT)));
        requireMapValue(problems, "payload." + RunnerRouteReportFields.METADATA_ROOT,
                payload.get(RunnerRouteReportFields.METADATA_ROOT));
        requireMapValue(problems, "payload." + RunnerRouteReportFields.VALIDATION_METADATA_ROOT,
                payload.get(RunnerRouteReportFields.VALIDATION_METADATA_ROOT));
        Object route = payload.get(RunnerRouteReportFields.METADATA_ROOT);
        if (route instanceof Map<?, ?> routeMap) {
            RunnerRouteReportContract.validateReport(routeMap).stream()
                    .map(problem -> "payload." + RunnerRouteReportFields.METADATA_ROOT + ": " + problem)
                    .forEach(problems::add);
            requireRouteSelectionSourceMatchesRoute(
                    problems,
                    payload.get(Payload.ROUTE_SELECTION_SOURCE),
                    routeMap.get(RunnerRouteReportFields.Report.SELECTION_SOURCE));
            requireRouteRunnerMatchesRoute(
                    problems,
                    payload.get(Payload.ROUTE_RUNNER),
                    routeMap.get(RunnerRouteReportFields.Report.NORMALIZED_RUNNER));
            requireRouteRedirectedMatchesRoute(
                    problems,
                    payload.get(Payload.ROUTE_REDIRECTED),
                    routeMap.get(RunnerRouteReportFields.Report.RUNTIME_REDIRECTED));
            requireRouteRedirectReasonMatchesRoute(
                    problems,
                    payload.get(Payload.ROUTE_REDIRECT_REASON),
                    routeMap.get(RunnerRouteReportFields.Report.RUNTIME_REDIRECT_REASON));
            requireRouteProfileFieldMatchesRoute(
                    problems,
                    Payload.ROUTE_PROFILE_STATUS,
                    payload.get(Payload.ROUTE_PROFILE_STATUS),
                    routeMap.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_STATUS));
            requireRouteProfileFieldMatchesRoute(
                    problems,
                    Payload.ROUTE_PROFILE_SOURCE,
                    payload.get(Payload.ROUTE_PROFILE_SOURCE),
                    routeMap.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_SOURCE));
            requireRouteProfileFieldMatchesRoute(
                    problems,
                    Payload.ROUTE_PROFILE_PROVIDER,
                    payload.get(Payload.ROUTE_PROFILE_PROVIDER),
                    routeMap.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_PROVIDER));
            requireRouteProfileFieldMatchesRoute(
                    problems,
                    Payload.ROUTE_PROFILE_FORMAT,
                    payload.get(Payload.ROUTE_PROFILE_FORMAT),
                    routeMap.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_FORMAT));
            requireRouteProfileFieldMatchesRoute(
                    problems,
                    Payload.ROUTE_PROFILE_REASON,
                    payload.get(Payload.ROUTE_PROFILE_REASON),
                    routeMap.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_REASON));
            requireRouteProfileFieldMatchesRoute(
                    problems,
                    Payload.ROUTE_PROFILE_ADVICE,
                    payload.get(Payload.ROUTE_PROFILE_ADVICE),
                    routeMap.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_ADVICE));
        }
        return List.copyOf(problems);
    }

    public static Map<String, Object> payloadValidationReport(Map<?, ?> payload) {
        return validationReport(validatePayload(payload));
    }

    public static List<String> validateValidationReport(Object value) {
        if (!(value instanceof Map<?, ?> validation)) {
            return List.of(RouteReportPayloadFields.VALIDATION_ROOT + " must be an object");
        }
        List<String> problems = new ArrayList<>();
        for (Object rawKey : validation.keySet()) {
            String key = String.valueOf(rawKey);
            if (!RouteReportPayloadFields.validationFields().contains(key)) {
                problems.add(RouteReportPayloadFields.VALIDATION_ROOT
                        + " contains unknown field: " + key);
            }
        }
        requireValue(problems, RouteReportPayloadFields.VALIDATION_ROOT + "."
                        + Validation.CONTRACT_ID,
                validation.get(Validation.CONTRACT_ID), RouteReportPayloadFields.CONTRACT_ID);
        requireValue(problems, RouteReportPayloadFields.VALIDATION_ROOT + "."
                        + Validation.SCHEMA_VERSION,
                validation.get(Validation.SCHEMA_VERSION), RouteReportPayloadFields.SCHEMA_VERSION);
        requireValue(problems, RouteReportPayloadFields.VALIDATION_ROOT + "."
                        + Validation.SCHEMA_FINGERPRINT,
                validation.get(Validation.SCHEMA_FINGERPRINT), RouteReportPayloadFields.schemaFingerprint());
        requireBoolean(problems, Validation.PASSED, validation.get(Validation.PASSED));
        requireBoolean(problems, Validation.FAILED, validation.get(Validation.FAILED));
        requireInteger(problems, Validation.PROBLEM_COUNT, validation.get(Validation.PROBLEM_COUNT));
        requireListValue(problems, Validation.PROBLEMS, validation.get(Validation.PROBLEMS));
        requireValidationReportConsistency(
                problems,
                validation.get(Validation.PASSED),
                validation.get(Validation.FAILED),
                validation.get(Validation.PROBLEM_COUNT),
                validation.get(Validation.PROBLEMS));
        return List.copyOf(problems);
    }

    private static Map<String, Object> validationReport(List<String> problems) {
        Map<String, Object> validation = new LinkedHashMap<>();
        validation.put(Validation.CONTRACT_ID, RouteReportPayloadFields.CONTRACT_ID);
        validation.put(Validation.SCHEMA_VERSION, RouteReportPayloadFields.SCHEMA_VERSION);
        validation.put(Validation.SCHEMA_FINGERPRINT, RouteReportPayloadFields.schemaFingerprint());
        validation.put(Validation.PASSED, problems.isEmpty());
        validation.put(Validation.FAILED, !problems.isEmpty());
        validation.put(Validation.PROBLEM_COUNT, problems.size());
        validation.put(Validation.PROBLEMS, problems);
        return validation;
    }

    private static void addPrefixed(List<String> problems, String prefix, List<String> additions) {
        additions.stream()
                .map(problem -> prefix + problem)
                .forEach(problems::add);
    }

    private static void requireValue(
            List<String> problems,
            String path,
            Object actual,
            Object expected) {
        if (!Objects.equals(expected, actual)) {
            problems.add(path + " expected " + expected + " but was " + actual);
        }
    }

    private static void requireList(
            List<String> problems,
            String field,
            Object actual,
            List<String> expected) {
        String path = "schema." + field;
        if (!(actual instanceof List<?> values)) {
            problems.add(path + " must be a list");
            return;
        }
        List<String> normalized = values.stream()
                .map(String::valueOf)
                .toList();
        if (!Objects.equals(expected, normalized)) {
            problems.add(path + " expected " + expected + " but was " + normalized);
        }
    }

    private static void requireMapValue(List<String> problems, String path, Object value) {
        if (!(value instanceof Map<?, ?>)) {
            problems.add(path + " must be an object");
        }
    }

    private static void requireBooleanPayload(List<String> problems, String field, Object actual) {
        if (!(actual instanceof Boolean)) {
            problems.add("payload." + field + " must be a boolean");
        }
    }

    private static void requireRouteRunner(List<String> problems, Object actual) {
        if (!(actual instanceof String runner) || !RunnerRoutePolicyFields.supportedRunners().contains(runner)) {
            problems.add("payload." + Payload.ROUTE_RUNNER
                    + " has unknown value: " + actual);
        }
    }

    private static void requireRouteSelectionSource(List<String> problems, Object actual) {
        if (!(actual instanceof String source) || !RunnerRouteReportFields.selectionSources().contains(source)) {
            problems.add("payload." + Payload.ROUTE_SELECTION_SOURCE
                    + " has unknown value: " + actual);
        }
    }

    private static void requireRouteProfileStatus(List<String> problems, Object actual) {
        if (!(actual instanceof String status)
                || !RunnerRouteReportFields.routeProfileStatuses().contains(status)) {
            problems.add("payload." + Payload.ROUTE_PROFILE_STATUS
                    + " has unknown value: " + actual);
        }
    }

    private static void requireRouteProfileSource(List<String> problems, Object actual) {
        if (!(actual instanceof String source)
                || !RunnerRouteReportFields.routeProfileSources().contains(source)) {
            problems.add("payload." + Payload.ROUTE_PROFILE_SOURCE
                    + " has unknown value: " + actual);
        }
    }

    private static void requireRouteRunnerMatchesRoute(
            List<String> problems,
            Object payloadRunner,
            Object routeRunner) {
        if (routeRunner instanceof String runner
                && RunnerRoutePolicyFields.supportedRunners().contains(runner)
                && !Objects.equals(payloadRunner, runner)) {
            problems.add("payload." + Payload.ROUTE_RUNNER
                    + " must match " + RunnerRouteReportFields.METADATA_ROOT + "."
                    + RunnerRouteReportFields.Report.NORMALIZED_RUNNER);
        }
    }

    private static void requireRouteSelectionSourceMatchesRoute(
            List<String> problems,
            Object payloadSource,
            Object routeSource) {
        if (routeSource instanceof String source
                && RunnerRouteReportFields.selectionSources().contains(source)
                && !Objects.equals(payloadSource, source)) {
            problems.add("payload." + Payload.ROUTE_SELECTION_SOURCE
                    + " must match " + RunnerRouteReportFields.METADATA_ROOT + "."
                    + RunnerRouteReportFields.Report.SELECTION_SOURCE);
        }
    }

    private static void requireRouteRedirectedMatchesRoute(
            List<String> problems,
            Object payloadRedirected,
            Object routeRedirected) {
        if (routeRedirected instanceof Boolean redirected && !Objects.equals(payloadRedirected, redirected)) {
            problems.add("payload." + Payload.ROUTE_REDIRECTED
                    + " must match " + RunnerRouteReportFields.METADATA_ROOT + "."
                    + RunnerRouteReportFields.Report.RUNTIME_REDIRECTED);
        }
    }

    private static void requireRouteRedirectReasonMatchesRoute(
            List<String> problems,
            Object payloadReason,
            Object routeReason) {
        if (routeReason instanceof String reason
                && !reason.isBlank()
                && !Objects.equals(payloadReason, reason)) {
            problems.add("payload." + Payload.ROUTE_REDIRECT_REASON
                    + " must match " + RunnerRouteReportFields.METADATA_ROOT + "."
                    + RunnerRouteReportFields.Report.RUNTIME_REDIRECT_REASON);
        }
    }

    private static void requireRouteProfileFieldMatchesRoute(
            List<String> problems,
            String payloadField,
            Object payloadValue,
            Object routeValue) {
        if (routeValue instanceof String text
                && !text.isBlank()
                && !Objects.equals(payloadValue, text)) {
            problems.add("payload." + payloadField
                    + " must match " + RunnerRouteReportFields.METADATA_ROOT + "."
                    + payloadField);
        }
    }

    private static void requireCacheState(List<String> problems, Object actual) {
        if (!(actual instanceof String state)
                || !(RouteArtifactCacheState.HIT.equals(state)
                || RouteArtifactCacheState.MISS.equals(state)
                || RouteArtifactCacheState.NOT_APPLICABLE.equals(state))) {
            problems.add("payload." + Payload.ROUTE_ARTIFACT_CACHE_STATE
                    + " has unknown value: " + actual);
        }
    }

    private static void requirePreflightProblemCount(List<String> problems, Object actual) {
        requirePayloadInteger(problems, Payload.PREFLIGHT_PROBLEM_COUNT, actual);
    }

    private static void requirePreflightProblemCountMatchesProblems(
            List<String> problems,
            Object countValue,
            Object problemValue) {
        Integer actual = payloadInteger(countValue);
        if (actual == null || !(problemValue instanceof List<?> preflightProblems)) {
            return;
        }
        int expected = preflightProblems.size();
        if (actual != expected) {
            problems.add("payload." + Payload.PREFLIGHT_PROBLEM_COUNT
                    + " must match preflight_problems size: expected "
                    + expected + " but was " + actual);
        }
    }

    private static void requirePreflightProblemCodes(List<String> problems, Object actual) {
        if (!(actual instanceof List<?> values)) {
            problems.add("payload." + Payload.PREFLIGHT_PROBLEM_CODES + " must be a list");
            return;
        }
        List<String> allowed = RoutePreflightDiagnosticFields.problemCodes();
        for (int index = 0; index < values.size(); index++) {
            Object value = values.get(index);
            if (!(value instanceof String code) || !allowed.contains(code)) {
                problems.add("payload." + Payload.PREFLIGHT_PROBLEM_CODES
                        + "[" + index + "] has unknown value: " + value);
            }
        }
    }

    private static void requirePreflightProblemCodeSummaryMatchesProblems(
            List<String> problems,
            Object summaryValue,
            Object problemValue) {
        if (!(summaryValue instanceof List<?> summary)) {
            return;
        }
        List<String> expected = preflightProblemCodes(problemValue);
        List<String> actual = summary.stream()
                .map(String::valueOf)
                .toList();
        if (!Objects.equals(expected, actual)) {
            problems.add("payload." + Payload.PREFLIGHT_PROBLEM_CODES
                    + " must match preflight_problems.code: expected "
                    + expected + " but was " + actual);
        }
    }

    private static List<String> preflightProblemCodes(Object problemValue) {
        if (!(problemValue instanceof List<?> problems)) {
            return List.of();
        }
        LinkedHashSet<String> codes = new LinkedHashSet<>();
        for (Object problem : problems) {
            if (!(problem instanceof Map<?, ?> problemMap)) {
                continue;
            }
            Object code = problemMap.get(RoutePreflightDiagnosticFields.Problem.CODE);
            if (code instanceof String text && !text.isBlank()) {
                codes.add(text);
            }
        }
        return List.copyOf(codes);
    }

    private static void requireMissingRuntimeCapabilities(List<String> problems, Object actual) {
        if (!(actual instanceof List<?> values)) {
            problems.add("payload." + Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES + " must be a list");
            return;
        }
        List<String> allowed = RoutePreflightDiagnosticFields.missingRuntimeCapabilities();
        for (int index = 0; index < values.size(); index++) {
            Object value = values.get(index);
            if (!(value instanceof String capability) || !allowed.contains(capability)) {
                problems.add("payload." + Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES
                        + "[" + index + "] has unknown value: " + value);
            }
        }
    }

    private static void requireMissingRuntimeCapabilitySummaryMatchesProblems(
            List<String> problems,
            Object summaryValue,
            Object problemValue) {
        if (!(summaryValue instanceof List<?> summary)) {
            return;
        }
        List<String> expected = missingRuntimeCapabilities(problemValue);
        List<String> actual = summary.stream()
                .map(String::valueOf)
                .toList();
        if (!Objects.equals(expected, actual)) {
            problems.add("payload." + Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES
                    + " must match preflight_problems.details.missingRuntimeCapability: expected "
                    + expected + " but was " + actual);
        }
    }

    private static List<String> missingRuntimeCapabilities(Object problemValue) {
        if (!(problemValue instanceof List<?> problems)) {
            return List.of();
        }
        LinkedHashSet<String> capabilities = new LinkedHashSet<>();
        for (Object problem : problems) {
            if (!(problem instanceof Map<?, ?> problemMap)) {
                continue;
            }
            Object detailsValue = problemMap.get(RoutePreflightDiagnosticFields.Problem.DETAILS);
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

    private static void requireNextActionCount(List<String> problems, Object actual) {
        requirePayloadInteger(problems, Payload.NEXT_ACTION_COUNT, actual);
    }

    private static void requireNextActionCountMatchesActions(
            List<String> problems,
            Object countValue,
            Object actionValue) {
        Integer actual = payloadInteger(countValue);
        if (actual == null || !(actionValue instanceof List<?> nextActions)) {
            return;
        }
        int expected = nextActions.size();
        if (actual != expected) {
            problems.add("payload." + Payload.NEXT_ACTION_COUNT
                    + " must match next_actions size: expected "
                    + expected + " but was " + actual);
        }
    }

    private static void requireNextActionKinds(List<String> problems, Object actual) {
        if (!(actual instanceof List<?> values)) {
            problems.add("payload." + Payload.NEXT_ACTION_KINDS + " must be a list");
            return;
        }
        List<String> allowed = RoutePreflightDiagnosticFields.actionKinds();
        for (int index = 0; index < values.size(); index++) {
            Object value = values.get(index);
            if (!(value instanceof String kind) || !allowed.contains(kind)) {
                problems.add("payload." + Payload.NEXT_ACTION_KINDS
                        + "[" + index + "] has unknown value: " + value);
            }
        }
    }

    private static void requireNextActionKindSummaryMatchesActions(
            List<String> problems,
            Object summaryValue,
            Object actionValue) {
        if (!(summaryValue instanceof List<?> summary)) {
            return;
        }
        List<String> expected = nextActionKinds(actionValue);
        List<String> actual = summary.stream()
                .map(String::valueOf)
                .toList();
        if (!Objects.equals(expected, actual)) {
            problems.add("payload." + Payload.NEXT_ACTION_KINDS
                    + " must match next_actions.kind: expected "
                    + expected + " but was " + actual);
        }
    }

    private static List<String> nextActionKinds(Object actionValue) {
        if (!(actionValue instanceof List<?> actions)) {
            return List.of();
        }
        LinkedHashSet<String> kinds = new LinkedHashSet<>();
        for (Object action : actions) {
            if (!(action instanceof Map<?, ?> actionMap)) {
                continue;
            }
            Object kind = actionMap.get(RoutePreflightDiagnosticFields.Action.KIND);
            if (kind instanceof String text && !text.isBlank()) {
                kinds.add(text);
            }
        }
        return List.copyOf(kinds);
    }

    private static void requirePayloadInteger(List<String> problems, String field, Object actual) {
        if (payloadInteger(actual) == null) {
            problems.add("payload." + field + " must be an integer");
        }
    }

    private static Integer payloadInteger(Object actual) {
        if (!(actual instanceof Number number) || number.intValue() != number.doubleValue()) {
            return null;
        }
        return number.intValue();
    }

    private static void requireOptionalString(List<String> problems, String field, Object actual) {
        if (actual != null && !(actual instanceof String)) {
            problems.add("payload." + field + " must be a string");
        }
    }

    private static void requireBoolean(List<String> problems, String field, Object actual) {
        if (!(actual instanceof Boolean)) {
            problems.add(RouteReportPayloadFields.VALIDATION_ROOT + "." + field + " must be a boolean");
        }
    }

    private static void requireInteger(List<String> problems, String field, Object actual) {
        if (!(actual instanceof Number number) || number.intValue() != number.doubleValue()) {
            problems.add(RouteReportPayloadFields.VALIDATION_ROOT + "." + field + " must be an integer");
        }
    }

    private static void requireListValue(List<String> problems, String field, Object actual) {
        if (!(actual instanceof List<?>)) {
            problems.add(RouteReportPayloadFields.VALIDATION_ROOT + "." + field + " must be a list");
        }
    }

    private static void requireValidationReportConsistency(
            List<String> problems,
            Object passedValue,
            Object failedValue,
            Object countValue,
            Object problemsValue) {
        if (!(problemsValue instanceof List<?> validationProblems)) {
            return;
        }
        Integer count = validationInteger(countValue);
        int expectedCount = validationProblems.size();
        if (count != null && count != expectedCount) {
            problems.add(RouteReportPayloadFields.VALIDATION_ROOT + "." + Validation.PROBLEM_COUNT
                    + " must match problems size: expected " + expectedCount + " but was " + count);
        }
        boolean expectedPassed = expectedCount == 0;
        requireValidationBooleanMatches(
                problems,
                Validation.PASSED,
                passedValue,
                expectedPassed);
        requireValidationBooleanMatches(
                problems,
                Validation.FAILED,
                failedValue,
                !expectedPassed);
    }

    private static void requireValidationBooleanMatches(
            List<String> problems,
            String field,
            Object actual,
            boolean expected) {
        if (actual instanceof Boolean value && value != expected) {
            problems.add(RouteReportPayloadFields.VALIDATION_ROOT + "." + field
                    + " must match problems emptiness: expected " + expected + " but was " + value);
        }
    }

    private static Integer validationInteger(Object actual) {
        if (!(actual instanceof Number number) || number.intValue() != number.doubleValue()) {
            return null;
        }
        return number.intValue();
    }
}
