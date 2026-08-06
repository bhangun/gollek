/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Action;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Problem;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Schema;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Validation;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight validator for route preflight problem and action entries.
 */
public final class RoutePreflightDiagnosticContract {
    private RoutePreflightDiagnosticContract() {
    }

    public static Map<String, Object> schema() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put(Schema.CONTRACT_ID, RoutePreflightDiagnosticFields.CONTRACT_ID);
        schema.put(Schema.SCHEMA_VERSION, RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        schema.put(Schema.SCHEMA_FINGERPRINT, RoutePreflightDiagnosticFields.schemaFingerprint());
        schema.put(Schema.VALIDATION_ROOT, RoutePreflightDiagnosticFields.VALIDATION_ROOT);
        schema.put(Schema.PROBLEM_FIELDS, RoutePreflightDiagnosticFields.problemFields());
        schema.put(Schema.REQUIRED_PROBLEM_FIELDS, RoutePreflightDiagnosticFields.requiredProblemFields());
        schema.put(Schema.PROBLEM_DETAIL_FIELDS, RoutePreflightDiagnosticFields.problemDetailFields());
        schema.put(Schema.EXECUTION_PROFILE_FIELDS, RoutePreflightDiagnosticFields.executionProfileFields());
        schema.put(Schema.HEADER_INSPECTION_FIELDS, RoutePreflightDiagnosticFields.headerInspectionFields());
        schema.put(Schema.TENSOR_INVENTORY_FIELDS, RoutePreflightDiagnosticFields.tensorInventoryFields());
        schema.put(Schema.COMPONENT_READINESS_FIELDS, RoutePreflightDiagnosticFields.componentReadinessFields());
        schema.put(Schema.INPUT_MODES, RoutePreflightDiagnosticFields.inputModes());
        schema.put(Schema.RUNTIME_CAPABILITIES, RoutePreflightDiagnosticFields.runtimeCapabilities());
        schema.put(Schema.MISSING_RUNTIME_CAPABILITIES, RoutePreflightDiagnosticFields.missingRuntimeCapabilities());
        schema.put(Schema.ACTION_FIELDS, RoutePreflightDiagnosticFields.actionFields());
        schema.put(Schema.REQUIRED_ACTION_FIELDS, RoutePreflightDiagnosticFields.requiredActionFields());
        schema.put(Schema.ACTION_DETAIL_FIELDS, RoutePreflightDiagnosticFields.actionDetailFields());
        schema.put(Schema.VALIDATION_FIELDS, RoutePreflightDiagnosticFields.validationFields());
        schema.put(Schema.PROBLEM_CODES, RoutePreflightDiagnosticFields.problemCodes());
        schema.put(Schema.ACTION_KINDS, RoutePreflightDiagnosticFields.actionKinds());
        return schema;
    }

    public static List<String> validateSchema(Map<?, ?> schema) {
        if (schema == null) {
            return List.of("route preflight diagnostic schema is missing");
        }
        List<String> problems = new ArrayList<>();
        requireValue(problems, "schema." + Schema.CONTRACT_ID,
                schema.get(Schema.CONTRACT_ID), RoutePreflightDiagnosticFields.CONTRACT_ID);
        requireValue(problems, "schema." + Schema.SCHEMA_VERSION,
                schema.get(Schema.SCHEMA_VERSION), RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        requireValue(problems, "schema." + Schema.SCHEMA_FINGERPRINT,
                schema.get(Schema.SCHEMA_FINGERPRINT), RoutePreflightDiagnosticFields.schemaFingerprint());
        requireValue(problems, "schema." + Schema.VALIDATION_ROOT,
                schema.get(Schema.VALIDATION_ROOT), RoutePreflightDiagnosticFields.VALIDATION_ROOT);
        requireList(problems, Schema.PROBLEM_FIELDS,
                schema.get(Schema.PROBLEM_FIELDS), RoutePreflightDiagnosticFields.problemFields());
        requireList(problems, Schema.REQUIRED_PROBLEM_FIELDS,
                schema.get(Schema.REQUIRED_PROBLEM_FIELDS), RoutePreflightDiagnosticFields.requiredProblemFields());
        requireList(problems, Schema.PROBLEM_DETAIL_FIELDS,
                schema.get(Schema.PROBLEM_DETAIL_FIELDS), RoutePreflightDiagnosticFields.problemDetailFields());
        requireList(problems, Schema.EXECUTION_PROFILE_FIELDS,
                schema.get(Schema.EXECUTION_PROFILE_FIELDS), RoutePreflightDiagnosticFields.executionProfileFields());
        requireList(problems, Schema.HEADER_INSPECTION_FIELDS,
                schema.get(Schema.HEADER_INSPECTION_FIELDS), RoutePreflightDiagnosticFields.headerInspectionFields());
        requireList(problems, Schema.TENSOR_INVENTORY_FIELDS,
                schema.get(Schema.TENSOR_INVENTORY_FIELDS), RoutePreflightDiagnosticFields.tensorInventoryFields());
        requireList(problems, Schema.COMPONENT_READINESS_FIELDS,
                schema.get(Schema.COMPONENT_READINESS_FIELDS),
                RoutePreflightDiagnosticFields.componentReadinessFields());
        requireList(problems, Schema.INPUT_MODES,
                schema.get(Schema.INPUT_MODES), RoutePreflightDiagnosticFields.inputModes());
        requireList(problems, Schema.RUNTIME_CAPABILITIES,
                schema.get(Schema.RUNTIME_CAPABILITIES), RoutePreflightDiagnosticFields.runtimeCapabilities());
        requireList(problems, Schema.MISSING_RUNTIME_CAPABILITIES,
                schema.get(Schema.MISSING_RUNTIME_CAPABILITIES),
                RoutePreflightDiagnosticFields.missingRuntimeCapabilities());
        requireList(problems, Schema.ACTION_FIELDS,
                schema.get(Schema.ACTION_FIELDS), RoutePreflightDiagnosticFields.actionFields());
        requireList(problems, Schema.REQUIRED_ACTION_FIELDS,
                schema.get(Schema.REQUIRED_ACTION_FIELDS), RoutePreflightDiagnosticFields.requiredActionFields());
        requireList(problems, Schema.ACTION_DETAIL_FIELDS,
                schema.get(Schema.ACTION_DETAIL_FIELDS), RoutePreflightDiagnosticFields.actionDetailFields());
        requireList(problems, Schema.VALIDATION_FIELDS,
                schema.get(Schema.VALIDATION_FIELDS), RoutePreflightDiagnosticFields.validationFields());
        requireList(problems, Schema.PROBLEM_CODES,
                schema.get(Schema.PROBLEM_CODES), RoutePreflightDiagnosticFields.problemCodes());
        requireList(problems, Schema.ACTION_KINDS,
                schema.get(Schema.ACTION_KINDS), RoutePreflightDiagnosticFields.actionKinds());
        return List.copyOf(problems);
    }

    public static Map<String, Object> schemaValidationReport(Map<?, ?> schema) {
        return validationReport(validateSchema(schema));
    }

    public static Map<String, Object> diagnosticsValidationReport(Object problemsValue, Object actionsValue) {
        List<String> problems = new ArrayList<>();
        problems.addAll(validateProblems(problemsValue));
        problems.addAll(validateActions(actionsValue));
        return validationReport(problems);
    }

    public static List<String> validateValidationReport(Object value) {
        if (!(value instanceof Map<?, ?> validation)) {
            return List.of(RoutePreflightDiagnosticFields.VALIDATION_ROOT + " must be an object");
        }
        List<String> problems = new ArrayList<>();
        for (Object rawKey : validation.keySet()) {
            String key = String.valueOf(rawKey);
            if (!RoutePreflightDiagnosticFields.validationFields().contains(key)) {
                problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT
                        + " contains unknown field: " + key);
            }
        }
        requireValue(problems, RoutePreflightDiagnosticFields.VALIDATION_ROOT + "."
                        + Validation.CONTRACT_ID,
                validation.get(Validation.CONTRACT_ID), RoutePreflightDiagnosticFields.CONTRACT_ID);
        requireValue(problems, RoutePreflightDiagnosticFields.VALIDATION_ROOT + "."
                        + Validation.SCHEMA_VERSION,
                validation.get(Validation.SCHEMA_VERSION), RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        requireValue(problems, RoutePreflightDiagnosticFields.VALIDATION_ROOT + "."
                        + Validation.SCHEMA_FINGERPRINT,
                validation.get(Validation.SCHEMA_FINGERPRINT), RoutePreflightDiagnosticFields.schemaFingerprint());
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

    public static List<String> validateProblems(Object value) {
        List<String> problems = new ArrayList<>(validateEntryList(
                "preflight_problems",
                value,
                RoutePreflightDiagnosticFields.problemFields(),
                RoutePreflightDiagnosticFields.requiredProblemFields(),
                Problem.CODE,
                RoutePreflightDiagnosticFields.problemCodes()));
        if (!(value instanceof List<?> entries)) {
            return problems;
        }
        for (int index = 0; index < entries.size(); index++) {
            Object entry = entries.get(index);
            if (entry instanceof Map<?, ?> entryMap && entryMap.containsKey(Problem.DETAILS)) {
                validateDiagnosticDetails(
                        problems,
                        "preflight_problems[" + index + "]." + Problem.DETAILS,
                        entryMap.get(Problem.DETAILS),
                        RoutePreflightDiagnosticFields.problemDetailFields());
            }
        }
        return List.copyOf(problems);
    }

    public static List<String> validateActions(Object value) {
        List<String> problems = new ArrayList<>(validateEntryList(
                "next_actions",
                value,
                RoutePreflightDiagnosticFields.actionFields(),
                RoutePreflightDiagnosticFields.requiredActionFields(),
                Action.KIND,
                RoutePreflightDiagnosticFields.actionKinds()));
        if (!(value instanceof List<?> actions)) {
            return problems;
        }
        for (int index = 0; index < actions.size(); index++) {
            Object action = actions.get(index);
            if (action instanceof Map<?, ?> actionMap
                    && !(actionMap.get(Action.ARGV) instanceof List<?>)) {
                problems.add("next_actions[" + index + "]." + Action.ARGV + " must be a list");
            }
            if (action instanceof Map<?, ?> actionMap && actionMap.containsKey(Action.DETAILS)) {
                validateDiagnosticDetails(
                        problems,
                        "next_actions[" + index + "]." + Action.DETAILS,
                        actionMap.get(Action.DETAILS),
                        RoutePreflightDiagnosticFields.actionDetailFields());
            }
        }
        return List.copyOf(problems);
    }

    private static List<String> validateEntryList(
            String name,
            Object value,
            List<String> allowedFields,
            List<String> requiredFields,
            String discriminatorField,
            List<String> allowedDiscriminators) {
        if (!(value instanceof List<?> entries)) {
            return List.of(name + " must be a list");
        }
        List<String> problems = new ArrayList<>();
        for (int index = 0; index < entries.size(); index++) {
            Object entry = entries.get(index);
            String path = name + "[" + index + "]";
            if (!(entry instanceof Map<?, ?> entryMap)) {
                problems.add(path + " must be an object");
                continue;
            }
            validateEntry(problems, path, entryMap, allowedFields, requiredFields,
                    discriminatorField, allowedDiscriminators);
        }
        return List.copyOf(problems);
    }

    private static void validateEntry(
            List<String> problems,
            String path,
            Map<?, ?> entry,
            List<String> allowedFields,
            List<String> requiredFields,
            String discriminatorField,
            List<String> allowedDiscriminators) {
        for (Object rawKey : entry.keySet()) {
            String key = String.valueOf(rawKey);
            if (!allowedFields.contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        for (String required : requiredFields) {
            if (!entry.containsKey(required)) {
                problems.add(path + " missing required field: " + required);
            }
        }
        Object discriminator = entry.get(discriminatorField);
        if (discriminator != null && !allowedDiscriminators.contains(String.valueOf(discriminator))) {
            problems.add(path + "." + discriminatorField + " has unknown value: " + discriminator);
        }
    }

    private static void validateDiagnosticDetails(
            List<String> problems,
            String path,
            Object value,
            List<String> allowedFields) {
        if (!(value instanceof Map<?, ?> details)) {
            problems.add(path + " must be an object");
            return;
        }
        for (Object rawKey : details.keySet()) {
            String key = String.valueOf(rawKey);
            if (!allowedFields.contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        Object missingCapability = details.get(
                RoutePreflightDiagnosticFields.ProblemDetail.MISSING_RUNTIME_CAPABILITY);
        if (missingCapability != null
                && !RoutePreflightDiagnosticFields.missingRuntimeCapabilities()
                        .contains(String.valueOf(missingCapability))) {
            problems.add(path + "."
                    + RoutePreflightDiagnosticFields.ProblemDetail.MISSING_RUNTIME_CAPABILITY
                    + " has unknown value: " + missingCapability);
        }
        validateKnownStringList(
                problems,
                path,
                details,
                RoutePreflightDiagnosticFields.ProblemDetail.SUPPORTED_INPUT_MODES,
                RoutePreflightDiagnosticFields.inputModes());
        validateKnownStringList(
                problems,
                path,
                details,
                RoutePreflightDiagnosticFields.ProblemDetail.BLOCKED_INPUT_MODES,
                RoutePreflightDiagnosticFields.inputModes());
        validateKnownStringList(
                problems,
                path,
                details,
                RoutePreflightDiagnosticFields.ProblemDetail.BLOCKED_CAPABILITIES,
                RoutePreflightDiagnosticFields.missingRuntimeCapabilities());
        validateKnownStringList(
                problems,
                path,
                details,
                RoutePreflightDiagnosticFields.ProblemDetail.READY_RUNTIME_CAPABILITIES,
                RoutePreflightDiagnosticFields.runtimeCapabilities());
        validateKnownStringList(
                problems,
                path,
                details,
                RoutePreflightDiagnosticFields.ProblemDetail.MISSING_RUNTIME_CAPABILITIES,
                RoutePreflightDiagnosticFields.missingRuntimeCapabilities());
        validateHeaderInspection(
                problems,
                path + "." + RoutePreflightDiagnosticFields.ProblemDetail.HEADER_INSPECTION,
                details.get(RoutePreflightDiagnosticFields.ProblemDetail.HEADER_INSPECTION));
        validateTensorInventory(
                problems,
                path + "." + RoutePreflightDiagnosticFields.ProblemDetail.TENSOR_INVENTORY,
                details.get(RoutePreflightDiagnosticFields.ProblemDetail.TENSOR_INVENTORY));
        validateComponentReadiness(
                problems,
                path + "." + RoutePreflightDiagnosticFields.ProblemDetail.COMPONENT_READINESS,
                details.get(RoutePreflightDiagnosticFields.ProblemDetail.COMPONENT_READINESS));
        validateExecutionProfile(
                problems,
                path + "." + RoutePreflightDiagnosticFields.ProblemDetail.EXECUTION_PROFILE,
                details.get(RoutePreflightDiagnosticFields.ProblemDetail.EXECUTION_PROFILE));
    }

    private static void validateExecutionProfile(
            List<String> problems,
            String path,
            Object value) {
        if (value == null) {
            return;
        }
        if (!(value instanceof Map<?, ?> executionProfile)) {
            problems.add(path + " must be an object");
            return;
        }
        for (Object rawKey : executionProfile.keySet()) {
            String key = String.valueOf(rawKey);
            if (!RoutePreflightDiagnosticFields.executionProfileFields().contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        for (String field : RoutePreflightDiagnosticFields.executionProfileFields()) {
            requireBooleanAtPath(problems, path + "." + field, executionProfile.get(field));
        }
    }

    private static void validateHeaderInspection(
            List<String> problems,
            String path,
            Object value) {
        if (value == null) {
            return;
        }
        if (!(value instanceof Map<?, ?> headerInspection)) {
            problems.add(path + " must be an object");
            return;
        }
        for (Object rawKey : headerInspection.keySet()) {
            String key = String.valueOf(rawKey);
            if (!RoutePreflightDiagnosticFields.headerInspectionFields().contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        for (String field : RoutePreflightDiagnosticFields.headerInspectionFields()) {
            requireNonNegativeInteger(problems, path + "." + field, headerInspection.get(field));
        }
    }

    private static void validateTensorInventory(
            List<String> problems,
            String path,
            Object value) {
        if (value == null) {
            return;
        }
        if (!(value instanceof Map<?, ?> tensorInventory)) {
            problems.add(path + " must be an object");
            return;
        }
        for (Object rawKey : tensorInventory.keySet()) {
            String key = String.valueOf(rawKey);
            if (!RoutePreflightDiagnosticFields.tensorInventoryFields().contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        for (String field : RoutePreflightDiagnosticFields.tensorInventoryFields()) {
            requireNonNegativeInteger(problems, path + "." + field, tensorInventory.get(field));
        }
    }

    private static void validateComponentReadiness(
            List<String> problems,
            String path,
            Object value) {
        if (value == null) {
            return;
        }
        if (!(value instanceof Map<?, ?> componentReadiness)) {
            problems.add(path + " must be an object");
            return;
        }
        for (Object rawKey : componentReadiness.keySet()) {
            String key = String.valueOf(rawKey);
            if (!RoutePreflightDiagnosticFields.componentReadinessFields().contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        for (String field : RoutePreflightDiagnosticFields.componentReadinessFields()) {
            requireBooleanAtPath(problems, path + "." + field, componentReadiness.get(field));
        }
    }

    private static void requireNonNegativeInteger(List<String> problems, String path, Object value) {
        if (!isIntegerNumber(value)) {
            problems.add(path + " must be a non-negative integer");
            return;
        }
        if (((Number) value).longValue() < 0) {
            problems.add(path + " must be a non-negative integer");
        }
    }

    private static boolean isIntegerNumber(Object value) {
        if (!(value instanceof Number number)) {
            return false;
        }
        if (number instanceof Byte || number instanceof Short || number instanceof Integer || number instanceof Long) {
            return true;
        }
        double doubleValue = number.doubleValue();
        return Double.isFinite(doubleValue) && Math.rint(doubleValue) == doubleValue;
    }

    private static void validateKnownStringList(
            List<String> problems,
            String path,
            Map<?, ?> details,
            String field,
            List<String> allowedValues) {
        Object value = details.get(field);
        if (value == null) {
            return;
        }
        if (!(value instanceof List<?> entries)) {
            problems.add(path + "." + field + " must be a list");
            return;
        }
        for (Object entry : entries) {
            if (!allowedValues.contains(String.valueOf(entry))) {
                problems.add(path + "." + field + " has unknown value: " + entry);
            }
        }
    }

    private static Map<String, Object> validationReport(List<String> problems) {
        Map<String, Object> validation = new LinkedHashMap<>();
        validation.put(Validation.CONTRACT_ID, RoutePreflightDiagnosticFields.CONTRACT_ID);
        validation.put(Validation.SCHEMA_VERSION, RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        validation.put(Validation.SCHEMA_FINGERPRINT, RoutePreflightDiagnosticFields.schemaFingerprint());
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

    private static void requireBoolean(List<String> problems, String field, Object actual) {
        if (!(actual instanceof Boolean)) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field + " must be a boolean");
        }
    }

    private static void requireBooleanAtPath(List<String> problems, String path, Object actual) {
        if (!(actual instanceof Boolean)) {
            problems.add(path + " must be a boolean");
        }
    }

    private static void requireInteger(List<String> problems, String field, Object actual) {
        if (!(actual instanceof Number number) || number.intValue() != number.doubleValue()) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field + " must be an integer");
        }
    }

    private static void requireListValue(List<String> problems, String field, Object actual) {
        if (!(actual instanceof List<?>)) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field + " must be a list");
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
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + Validation.PROBLEM_COUNT
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
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field
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
