/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RunnerRouteReportFields.Schema;
import tech.kayys.gollek.sdk.route.RunnerRouteReportFields.Validation;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight validator for reusable runner/provider route reports.
 */
public final class RunnerRouteReportContract {
    private RunnerRouteReportContract() {
    }

    public static Map<String, Object> schema() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put(Schema.CONTRACT_ID, RunnerRouteReportFields.CONTRACT_ID);
        schema.put(Schema.SCHEMA_VERSION, RunnerRouteReportFields.SCHEMA_VERSION);
        schema.put(Schema.SCHEMA_FINGERPRINT, RunnerRouteReportFields.schemaFingerprint());
        schema.put(Schema.METADATA_ROOT, RunnerRouteReportFields.METADATA_ROOT);
        schema.put(Schema.METADATA_PREFIX, RunnerRouteReportFields.METADATA_PREFIX);
        schema.put(Schema.VALIDATION_METADATA_ROOT, RunnerRouteReportFields.VALIDATION_METADATA_ROOT);
        schema.put(Schema.VALIDATION_METADATA_PREFIX, RunnerRouteReportFields.VALIDATION_METADATA_PREFIX);
        schema.put(Schema.REPORT_FIELDS, RunnerRouteReportFields.reportFields());
        schema.put(Schema.REQUIRED_REPORT_FIELDS, RunnerRouteReportFields.requiredReportFields());
        schema.put(Schema.SELECTION_SOURCES, RunnerRouteReportFields.selectionSources());
        schema.put(Schema.ROUTE_PROFILE_STATUSES, RunnerRouteReportFields.routeProfileStatuses());
        schema.put(Schema.ROUTE_PROFILE_SOURCES, RunnerRouteReportFields.routeProfileSources());
        return schema;
    }

    public static List<String> validateSchema(Map<?, ?> schema) {
        if (schema == null) {
            return List.of("runner route report schema is missing");
        }
        List<String> problems = new ArrayList<>();
        requireValue(problems, "schema." + Schema.CONTRACT_ID,
                schema.get(Schema.CONTRACT_ID), RunnerRouteReportFields.CONTRACT_ID);
        requireValue(problems, "schema." + Schema.SCHEMA_VERSION,
                schema.get(Schema.SCHEMA_VERSION), RunnerRouteReportFields.SCHEMA_VERSION);
        requireValue(problems, "schema." + Schema.SCHEMA_FINGERPRINT,
                schema.get(Schema.SCHEMA_FINGERPRINT), RunnerRouteReportFields.schemaFingerprint());
        requireValue(problems, "schema." + Schema.METADATA_ROOT,
                schema.get(Schema.METADATA_ROOT), RunnerRouteReportFields.METADATA_ROOT);
        requireValue(problems, "schema." + Schema.METADATA_PREFIX,
                schema.get(Schema.METADATA_PREFIX), RunnerRouteReportFields.METADATA_PREFIX);
        requireValue(problems, "schema." + Schema.VALIDATION_METADATA_ROOT,
                schema.get(Schema.VALIDATION_METADATA_ROOT), RunnerRouteReportFields.VALIDATION_METADATA_ROOT);
        requireValue(problems, "schema." + Schema.VALIDATION_METADATA_PREFIX,
                schema.get(Schema.VALIDATION_METADATA_PREFIX), RunnerRouteReportFields.VALIDATION_METADATA_PREFIX);
        requireList(problems, Schema.REPORT_FIELDS,
                schema.get(Schema.REPORT_FIELDS), RunnerRouteReportFields.reportFields());
        requireList(problems, Schema.REQUIRED_REPORT_FIELDS,
                schema.get(Schema.REQUIRED_REPORT_FIELDS), RunnerRouteReportFields.requiredReportFields());
        requireList(problems, Schema.SELECTION_SOURCES,
                schema.get(Schema.SELECTION_SOURCES), RunnerRouteReportFields.selectionSources());
        requireList(problems, Schema.ROUTE_PROFILE_STATUSES,
                schema.get(Schema.ROUTE_PROFILE_STATUSES), RunnerRouteReportFields.routeProfileStatuses());
        requireList(problems, Schema.ROUTE_PROFILE_SOURCES,
                schema.get(Schema.ROUTE_PROFILE_SOURCES), RunnerRouteReportFields.routeProfileSources());
        return List.copyOf(problems);
    }

    public static Map<String, Object> schemaValidationReport(Map<?, ?> schema) {
        List<String> problems = validateSchema(schema);
        return validationReport(problems);
    }

    public static List<String> validateReport(Map<?, ?> report) {
        if (report == null) {
            return List.of("runner route report is missing");
        }
        List<String> problems = new ArrayList<>();
        List<String> allowedFields = RunnerRouteReportFields.reportFields();
        for (Object rawKey : report.keySet()) {
            String key = String.valueOf(rawKey);
            if (!allowedFields.contains(key)) {
                problems.add("runner route report contains unknown field: " + key);
            }
        }
        for (String required : RunnerRouteReportFields.requiredReportFields()) {
            if (!report.containsKey(required)) {
                problems.add("runner route report missing required field: " + required);
            }
        }
        requireSelectionSource(problems, report.get(RunnerRouteReportFields.Report.SELECTION_SOURCE));
        requireRouteProfileStatus(problems, report.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_STATUS));
        requireRouteProfileSource(problems, report.get(RunnerRouteReportFields.Report.ROUTE_PROFILE_SOURCE));
        return List.copyOf(problems);
    }

    public static Map<String, Object> reportValidationReport(Map<?, ?> report) {
        List<String> problems = validateReport(report);
        return validationReport(problems);
    }

    private static Map<String, Object> validationReport(List<String> problems) {
        Map<String, Object> validation = new LinkedHashMap<>();
        validation.put(Validation.CONTRACT_ID, RunnerRouteReportFields.CONTRACT_ID);
        validation.put(Validation.SCHEMA_VERSION, RunnerRouteReportFields.SCHEMA_VERSION);
        validation.put(Validation.SCHEMA_FINGERPRINT, RunnerRouteReportFields.schemaFingerprint());
        validation.put(Validation.PASSED, problems.isEmpty());
        validation.put(Validation.FAILED, !problems.isEmpty());
        validation.put(Validation.PROBLEM_COUNT, problems.size());
        validation.put(Validation.PROBLEMS, problems);
        return validation;
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

    private static void requireSelectionSource(List<String> problems, Object actual) {
        if (!(actual instanceof String source) || !RunnerRouteReportFields.selectionSources().contains(source)) {
            problems.add("runner route report selection_source has unknown value: " + actual);
        }
    }

    private static void requireRouteProfileStatus(List<String> problems, Object actual) {
        if (!(actual instanceof String status)
                || !RunnerRouteReportFields.routeProfileStatuses().contains(status)) {
            problems.add("runner route report route_profile_status has unknown value: " + actual);
        }
    }

    private static void requireRouteProfileSource(List<String> problems, Object actual) {
        if (!(actual instanceof String source)
                || !RunnerRouteReportFields.routeProfileSources().contains(source)) {
            problems.add("runner route report route_profile_source has unknown value: " + actual);
        }
    }
}
