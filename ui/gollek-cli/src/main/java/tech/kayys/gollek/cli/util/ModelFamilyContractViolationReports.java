/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.cli.util.ModelFamilyContractViolationReportFields.Report;
import tech.kayys.gollek.cli.util.ModelFamilyContractViolationReportFields.Schema;
import tech.kayys.gollek.cli.util.ModelFamilyContractViolationReportFields.Validation;
import tech.kayys.gollek.cli.util.ModelFamilyContractViolationReportFields.Violation;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyContractViolation;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Machine-readable summaries for model-family contract violations.
 */
public final class ModelFamilyContractViolationReports {
    public static final String CONTRACT_ID = ModelFamilyContractViolationReportFields.CONTRACT_ID;
    public static final int SCHEMA_VERSION = ModelFamilyContractViolationReportFields.SCHEMA_VERSION;

    private static final List<String> CATEGORY_KEYS = List.of(
            "descriptor",
            "bundleProfile",
            "origin",
            "capabilityShape",
            "tokenizerMetadata",
            "tokenizerDescriptor",
            "architectureAdapter",
            "directSafetensorMetadata",
            "unifiedRuntimeRequirement",
            "supportReport",
            "duplicateClaim",
            "pluginSurface",
            "unknown");

    private static final List<String> REPORT_FIELDS = ModelFamilyContractViolationReportFields.reportFields();
    private static final List<String> VIOLATION_FIELDS = ModelFamilyContractViolationReportFields.violationFields();

    public static final String SCHEMA_FINGERPRINT = schemaFingerprintValue();

    private ModelFamilyContractViolationReports() {
    }

    public static Map<String, Object> summary(List<ModelFamilyContractViolation> violations) {
        List<ModelFamilyContractViolation> normalized = normalizeViolations(violations);
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(Report.CONTRACT_ID, CONTRACT_ID);
        report.put(Report.SCHEMA_VERSION, SCHEMA_VERSION);
        report.put(Report.SCHEMA_FINGERPRINT, SCHEMA_FINGERPRINT);
        report.put(Report.SCHEMA, schema());
        report.put(Report.CATEGORY_KEYS, categoryKeys());
        report.put(Report.REMEDIATION_CATALOG, remediationCatalog());
        report.put(Report.PASSED, normalized.isEmpty());
        report.put(Report.FAILED, !normalized.isEmpty());
        report.put(Report.STATUS, normalized.isEmpty() ? "passed" : "failed");
        report.put(Report.VIOLATION_COUNT, normalized.size());
        report.put(Report.CATEGORY_COUNTS, categories(normalized));
        report.put(Report.CATEGORIES_WITH_VIOLATIONS, categoriesWithViolations(normalized));
        report.put(Report.CATEGORY_REMEDIATION_HINTS, categoryRemediationHints(normalized));
        report.put(Report.REMEDIATION_HINTS, remediationHints(normalized));
        report.put(Report.AFFECTED_FAMILY_COUNT, affectedFamilies(normalized).size());
        report.put(Report.AFFECTED_FAMILIES, affectedFamilies(normalized));
        report.put(Report.SUMMARIES, normalized.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList());
        report.put(Report.VIOLATIONS, normalized.stream()
                .map(ModelFamilyContractViolationReports::violationReport)
                .toList());
        return report;
    }

    public static Map<String, Object> schema() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put(Schema.CONTRACT_ID, CONTRACT_ID);
        schema.put(Schema.SCHEMA_VERSION, SCHEMA_VERSION);
        schema.put(Schema.SCHEMA_FINGERPRINT, SCHEMA_FINGERPRINT);
        schema.put(Schema.REPORT_FIELDS, REPORT_FIELDS);
        schema.put(Schema.VIOLATION_FIELDS, VIOLATION_FIELDS);
        schema.put(Schema.CATEGORY_KEYS, categoryKeys());
        return schema;
    }

    public static List<String> validateSummary(Map<?, ?> report) {
        List<String> problems = new ArrayList<>();
        if (report == null) {
            return List.of("report is missing");
        }

        requireString(report, Report.CONTRACT_ID, CONTRACT_ID, problems);
        requireInteger(report, Report.SCHEMA_VERSION, SCHEMA_VERSION, problems);
        requireString(report, Report.SCHEMA_FINGERPRINT, SCHEMA_FINGERPRINT, problems);

        Map<?, ?> schema = mapValue(report.get(Report.SCHEMA));
        if (schema == null) {
            problems.add("schema is missing or not an object");
        } else {
            requireString(schema, Schema.CONTRACT_ID, CONTRACT_ID, problems);
            requireInteger(schema, Schema.SCHEMA_VERSION, SCHEMA_VERSION, problems);
            requireString(schema, Schema.SCHEMA_FINGERPRINT, SCHEMA_FINGERPRINT, problems);
            requireStringList(schema, Schema.REPORT_FIELDS, REPORT_FIELDS, problems);
            requireStringList(schema, Schema.VIOLATION_FIELDS, VIOLATION_FIELDS, problems);
            requireStringList(schema, Schema.CATEGORY_KEYS, categoryKeys(), problems);
        }

        requireStringList(report, Report.CATEGORY_KEYS, categoryKeys(), problems);
        requireCategoryMap(report, Report.REMEDIATION_CATALOG, problems);
        requireCategoryMap(report, Report.CATEGORY_COUNTS, problems);
        validateViolationRows(report.get(Report.VIOLATIONS), problems);
        return List.copyOf(problems);
    }

    public static Map<String, Object> validationReport(Map<?, ?> report) {
        List<String> problems = validateSummary(report);
        Map<String, Object> validation = new LinkedHashMap<>();
        validation.put(Validation.CONTRACT_ID, CONTRACT_ID);
        validation.put(Validation.SCHEMA_VERSION, SCHEMA_VERSION);
        validation.put(Validation.SCHEMA_FINGERPRINT, SCHEMA_FINGERPRINT);
        validation.put(Validation.PASSED, problems.isEmpty());
        validation.put(Validation.FAILED, !problems.isEmpty());
        validation.put(Validation.PROBLEM_COUNT, problems.size());
        validation.put(Validation.PROBLEMS, problems);
        return validation;
    }

    public static List<String> categoryKeys() {
        return CATEGORY_KEYS;
    }

    public static Map<String, String> remediationCatalog() {
        Map<String, String> catalog = new LinkedHashMap<>();
        for (String category : CATEGORY_KEYS) {
            catalog.put(category, remediationHint(category));
        }
        return catalog;
    }

    public static Map<String, Integer> categories(List<ModelFamilyContractViolation> violations) {
        Map<String, Integer> categories = emptyCategories();
        for (ModelFamilyContractViolation violation : normalizeViolations(violations)) {
            String category = category(violation);
            categories.put(category, categories.get(category) + 1);
        }
        return categories;
    }

    public static Map<String, String> categoryRemediationHints(List<ModelFamilyContractViolation> violations) {
        Map<String, String> hints = new LinkedHashMap<>();
        for (String category : categoriesWithViolations(normalizeViolations(violations))) {
            hints.put(category, remediationHint(category));
        }
        return hints;
    }

    public static List<String> remediationHints(List<ModelFamilyContractViolation> violations) {
        return List.copyOf(categoryRemediationHints(violations).values());
    }

    public static Map<String, Object> violationReport(ModelFamilyContractViolation violation) {
        ModelFamilyContractViolation normalized = violation == null
                ? new ModelFamilyContractViolation("unknown", "contract_violation", "contract violation")
                : violation;
        Map<String, Object> report = new LinkedHashMap<>();
        String category = category(normalized);
        report.put(Violation.FAMILY_ID, normalized.familyId());
        report.put(Violation.CODE, normalized.code());
        report.put(Violation.CATEGORY, category);
        report.put(Violation.REMEDIATION_HINT, remediationHint(category));
        report.put(Violation.SUMMARY, normalized.summary());
        return report;
    }

    public static String category(ModelFamilyContractViolation violation) {
        return category(violation == null ? "" : violation.code());
    }

    public static String category(String code) {
        String value = code == null ? "" : code.strip().toLowerCase(Locale.ROOT);
        if (value.isBlank()) {
            return "unknown";
        }
        if (value.contains("bundle_profile")) {
            return "bundleProfile";
        }
        if (value.contains("origin")) {
            return "origin";
        }
        if (value.startsWith("tokenizer_metadata_")) {
            return "tokenizerMetadata";
        }
        if (value.startsWith("tokenizer_")) {
            return "tokenizerDescriptor";
        }
        if (value.contains("unified_runtime_requirement")) {
            return "unifiedRuntimeRequirement";
        }
        if (value.startsWith("support_report_")) {
            return "supportReport";
        }
        if (value.startsWith("architecture_adapter_")
                || "direct_safetensor_without_adapter".equals(value)) {
            return "architectureAdapter";
        }
        if (value.contains("scoped_direct_safetensor")) {
            return "directSafetensorMetadata";
        }
        if ("duplicate_model_type_claim".equals(value)) {
            return "duplicateClaim";
        }
        if (value.contains("capability") || value.contains("multimodal")) {
            return "capabilityShape";
        }
        if (value.startsWith("plugin_")
                || value.startsWith("descriptor_")
                || "plugin_null".equals(value)) {
            return "pluginSurface";
        }
        if (value.contains("family_id")
                || value.contains("model_type")
                || value.contains("claims")) {
            return "descriptor";
        }
        return "unknown";
    }

    public static String remediationHint(String category) {
        return switch (category == null ? "" : category) {
            case "descriptor" -> "Normalize descriptor id/model_type claims and ensure each family claims a "
                    + "supported Transformers model_type or architecture class.";
            case "bundleProfile" -> "Set metadata.bundle_profile to core, optional, metadata_only, or experimental "
                    + "so bundle presets can attach or detach the family safely.";
            case "origin" -> "Point metadata.origin to 3rdparty/transformers/src/transformers/models/<family>, "
                    + "external/<namespace>, or legacy/<namespace>.";
            case "capabilityShape" -> "Align descriptor capabilities with concrete modalities and published "
                    + "tokenizer/runtime surfaces.";
            case "tokenizerMetadata" -> "Use tokenizer_metadata_status=ready with tokenizer descriptors, or pending "
                    + "with tokenizer_metadata_pending_reason and no descriptors.";
            case "tokenizerDescriptor" -> "Publish reusable tokenizerDescriptors() with safe file groups and valid "
                    + "tokenizer ids, or mark tokenizer metadata pending.";
            case "architectureAdapter" -> "Add architectureAdapters() for DIRECT_SAFETENSOR_INFERENCE, or remove "
                    + "the direct capability until the adapter is ready.";
            case "directSafetensorMetadata" -> "Keep scoped direct SafeTensor caveat metadata explicit, pending, "
                    + "experimental, or not-ready.";
            case "unifiedRuntimeRequirement" -> "Mirror unified runtime requirements in descriptor metadata, claim "
                    + "the model type, and list required input modalities.";
            case "supportReport" -> "Let supportReport() derive from the plugin surface, or keep custom report "
                    + "fields synchronized with descriptor, tokenizer, and adapter metadata.";
            case "duplicateClaim" -> "Ensure only one model-family plugin claims each model_type in the active "
                    + "bundle.";
            case "pluginSurface" -> "Keep plugin id, descriptor(), service registration, and plugin-core metadata "
                    + "aligned with model-family/<familyId>.";
            default -> "Inspect the violation code and update the model-family descriptor, tokenizer, adapter, or "
                    + "runtime metadata before packaging for production.";
        };
    }

    private static List<String> categoriesWithViolations(List<ModelFamilyContractViolation> violations) {
        return categories(violations).entrySet().stream()
                .filter(entry -> entry.getValue() > 0)
                .map(Map.Entry::getKey)
                .toList();
    }

    private static List<String> affectedFamilies(List<ModelFamilyContractViolation> violations) {
        Set<String> families = new LinkedHashSet<>();
        for (ModelFamilyContractViolation violation : normalizeViolations(violations)) {
            if (!violation.familyId().isBlank()) {
                families.add(violation.familyId());
            }
        }
        return List.copyOf(families);
    }

    private static List<ModelFamilyContractViolation> normalizeViolations(
            List<ModelFamilyContractViolation> violations) {
        return violations == null ? List.of() : violations;
    }

    private static Map<String, Integer> emptyCategories() {
        Map<String, Integer> categories = new LinkedHashMap<>();
        for (String key : CATEGORY_KEYS) {
            categories.put(key, 0);
        }
        return categories;
    }

    private static void validateViolationRows(Object value, List<String> problems) {
        if (!(value instanceof List<?> rows)) {
            problems.add(Report.VIOLATIONS + " is missing or not an array");
            return;
        }
        for (int index = 0; index < rows.size(); index++) {
            Object row = rows.get(index);
            if (!(row instanceof Map<?, ?> violation)) {
                problems.add(Report.VIOLATIONS + "[" + index + "] is not an object");
                continue;
            }
            for (String field : VIOLATION_FIELDS) {
                if (!violation.containsKey(field)) {
                    problems.add(Report.VIOLATIONS + "[" + index + "]." + field + " is missing");
                }
            }
        }
    }

    private static void requireString(Map<?, ?> report, String field, String expected, List<String> problems) {
        Object value = report.get(field);
        if (!(value instanceof String actual) || !expected.equals(actual)) {
            problems.add(field + " must be " + expected);
        }
    }

    private static void requireInteger(Map<?, ?> report, String field, int expected, List<String> problems) {
        Object value = report.get(field);
        if (!(value instanceof Number actual) || actual.intValue() != expected) {
            problems.add(field + " must be " + expected);
        }
    }

    private static void requireStringList(
            Map<?, ?> report,
            String field,
            List<String> expected,
            List<String> problems) {
        List<String> actual = stringList(report.get(field));
        if (!expected.equals(actual)) {
            problems.add(field + " must match " + expected);
        }
    }

    private static void requireCategoryMap(Map<?, ?> report, String field, List<String> problems) {
        Map<?, ?> values = mapValue(report.get(field));
        if (values == null) {
            problems.add(field + " is missing or not an object");
            return;
        }
        for (String category : CATEGORY_KEYS) {
            if (!values.containsKey(category)) {
                problems.add(field + "." + category + " is missing");
            }
        }
    }

    private static Map<?, ?> mapValue(Object value) {
        return value instanceof Map<?, ?> map ? map : null;
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof List<?> values)) {
            return null;
        }
        List<String> strings = new ArrayList<>();
        for (Object item : values) {
            if (!(item instanceof String text)) {
                return null;
            }
            strings.add(text);
        }
        return List.copyOf(strings);
    }

    private static String schemaFingerprintValue() {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            updateDigest(digest, CONTRACT_ID);
            updateDigest(digest, Integer.toString(SCHEMA_VERSION));
            updateDigest(digest, String.join(",", REPORT_FIELDS));
            updateDigest(digest, String.join(",", VIOLATION_FIELDS));
            updateDigest(digest, String.join(",", CATEGORY_KEYS));
            return "sha256:" + HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                    "SHA-256 is required for model-family contract report schemas",
                    exception);
        }
    }

    private static void updateDigest(MessageDigest digest, String value) {
        digest.update(value.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) '\n');
    }
}
