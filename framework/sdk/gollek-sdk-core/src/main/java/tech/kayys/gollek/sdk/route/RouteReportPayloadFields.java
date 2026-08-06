/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.sdk.route;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

/**
 * Stable JSON field names for the top-level `gollek run --route-report-json` payload.
 */
public final class RouteReportPayloadFields {
    public static final String CONTRACT_ID = "gollek.route-report.payload";
    public static final int SCHEMA_VERSION = 13;
    public static final String VALIDATION_ROOT = "route_report_validation";

    private RouteReportPayloadFields() {
    }

    public static final class Schema {
        public static final String CONTRACT_ID = "contractId";
        public static final String SCHEMA_VERSION = "schemaVersion";
        public static final String SCHEMA_FINGERPRINT = "schemaFingerprint";
        public static final String VALIDATION_ROOT = "validationRoot";
        public static final String PAYLOAD_FIELDS = "payloadFields";
        public static final String REQUIRED_PAYLOAD_FIELDS = "requiredPayloadFields";
        public static final String OPTIONAL_PAYLOAD_FIELDS = "optionalPayloadFields";
        public static final String VALIDATION_FIELDS = "validationFields";

        private Schema() {
        }
    }

    public static final class Validation {
        public static final String CONTRACT_ID = Schema.CONTRACT_ID;
        public static final String SCHEMA_VERSION = Schema.SCHEMA_VERSION;
        public static final String SCHEMA_FINGERPRINT = Schema.SCHEMA_FINGERPRINT;
        public static final String PASSED = "passed";
        public static final String FAILED = "failed";
        public static final String PROBLEM_COUNT = "problemCount";
        public static final String PROBLEMS = "problems";

        private Validation() {
        }
    }

    public static final class Payload {
        public static final String TYPE = "type";
        public static final String DRY_RUN = "dry_run";
        public static final String MUTATION_ALLOWED = "mutation_allowed";
        public static final String RESOLUTION_MODE = "resolution_mode";
        public static final String RESOLVED_LOCAL = "resolved_local";
        public static final String RESOLUTION_STATUS = "resolution_status";
        public static final String REQUIRE_LOCAL = "require_local";
        public static final String PREFLIGHT_PASSED = "preflight_passed";
        public static final String PREFLIGHT_STATUS = "preflight_status";
        public static final String PREFLIGHT_PROBLEM_COUNT = "preflight_problem_count";
        public static final String PREFLIGHT_PROBLEM_CODES = "preflight_problem_codes";
        public static final String PREFLIGHT_MISSING_RUNTIME_CAPABILITIES =
                "preflight_missing_runtime_capabilities";
        public static final String PREFLIGHT_PROBLEMS = "preflight_problems";
        public static final String EXIT_CODE = "exit_code";
        public static final String NEXT_ACTION_COUNT = "next_action_count";
        public static final String NEXT_ACTION_KINDS = "next_action_kinds";
        public static final String NEXT_ACTIONS = "next_actions";
        public static final String REQUESTED_MODEL = "requested_model";
        public static final String MODEL = "model";
        public static final String LOCAL_PATH = "local_path";
        public static final String PROVIDER = "provider";
        public static final String FORMAT = "format";
        public static final String ROUTE_RUNNER = "route_runner";
        public static final String ROUTE_SELECTION_SOURCE = "route_selection_source";
        public static final String ROUTE_REDIRECTED = "route_redirected";
        public static final String ROUTE_REDIRECT_REASON = "route_redirect_reason";
        public static final String ROUTE_ARTIFACT_CACHE_HIT = "route_artifact_cache_hit";
        public static final String ROUTE_ARTIFACT_CACHE_KIND = "route_artifact_cache_kind";
        public static final String ROUTE_ARTIFACT_CACHE_STATE = "route_artifact_cache_state";
        public static final String ROUTE_PROFILE_STATUS = "route_profile_status";
        public static final String ROUTE_PROFILE_SOURCE = "route_profile_source";
        public static final String ROUTE_PROFILE_PROVIDER = "route_profile_provider";
        public static final String ROUTE_PROFILE_FORMAT = "route_profile_format";
        public static final String ROUTE_PROFILE_REASON = "route_profile_reason";
        public static final String ROUTE_PROFILE_ADVICE = "route_profile_advice";

        private Payload() {
        }
    }

    public static final class RouteArtifactCacheState {
        public static final String HIT = "hit";
        public static final String MISS = "miss";
        public static final String NOT_APPLICABLE = "not_applicable";

        private RouteArtifactCacheState() {
        }
    }

    public static List<String> payloadFields() {
        return List.of(
                Payload.TYPE,
                Payload.DRY_RUN,
                Payload.MUTATION_ALLOWED,
                Payload.RESOLUTION_MODE,
                Payload.RESOLVED_LOCAL,
                Payload.RESOLUTION_STATUS,
                Payload.REQUIRE_LOCAL,
                Payload.PREFLIGHT_PASSED,
                Payload.PREFLIGHT_STATUS,
                Payload.PREFLIGHT_PROBLEM_COUNT,
                Payload.PREFLIGHT_PROBLEM_CODES,
                Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES,
                Payload.PREFLIGHT_PROBLEMS,
                Payload.EXIT_CODE,
                Payload.NEXT_ACTION_COUNT,
                Payload.NEXT_ACTION_KINDS,
                Payload.NEXT_ACTIONS,
                RoutePreflightDiagnosticFields.VALIDATION_ROOT,
                Payload.REQUESTED_MODEL,
                Payload.MODEL,
                Payload.LOCAL_PATH,
                Payload.PROVIDER,
                Payload.FORMAT,
                Payload.ROUTE_RUNNER,
                Payload.ROUTE_SELECTION_SOURCE,
                Payload.ROUTE_REDIRECTED,
                Payload.ROUTE_REDIRECT_REASON,
                Payload.ROUTE_ARTIFACT_CACHE_HIT,
                Payload.ROUTE_ARTIFACT_CACHE_KIND,
                Payload.ROUTE_ARTIFACT_CACHE_STATE,
                Payload.ROUTE_PROFILE_STATUS,
                Payload.ROUTE_PROFILE_SOURCE,
                Payload.ROUTE_PROFILE_PROVIDER,
                Payload.ROUTE_PROFILE_FORMAT,
                Payload.ROUTE_PROFILE_REASON,
                Payload.ROUTE_PROFILE_ADVICE,
                RunnerRouteReportFields.METADATA_ROOT,
                RunnerRouteReportFields.VALIDATION_METADATA_ROOT,
                VALIDATION_ROOT);
    }

    public static List<String> requiredPayloadFields() {
        return List.of(
                Payload.TYPE,
                Payload.DRY_RUN,
                Payload.MUTATION_ALLOWED,
                Payload.RESOLUTION_MODE,
                Payload.RESOLVED_LOCAL,
                Payload.RESOLUTION_STATUS,
                Payload.REQUIRE_LOCAL,
                Payload.PREFLIGHT_PASSED,
                Payload.PREFLIGHT_STATUS,
                Payload.PREFLIGHT_PROBLEM_COUNT,
                Payload.PREFLIGHT_PROBLEM_CODES,
                Payload.PREFLIGHT_MISSING_RUNTIME_CAPABILITIES,
                Payload.PREFLIGHT_PROBLEMS,
                Payload.EXIT_CODE,
                Payload.NEXT_ACTION_COUNT,
                Payload.NEXT_ACTION_KINDS,
                Payload.NEXT_ACTIONS,
                RoutePreflightDiagnosticFields.VALIDATION_ROOT,
                Payload.ROUTE_RUNNER,
                Payload.ROUTE_SELECTION_SOURCE,
                Payload.ROUTE_REDIRECTED,
                Payload.ROUTE_ARTIFACT_CACHE_HIT,
                Payload.ROUTE_ARTIFACT_CACHE_STATE,
                Payload.ROUTE_PROFILE_STATUS,
                Payload.ROUTE_PROFILE_SOURCE,
                RunnerRouteReportFields.METADATA_ROOT,
                RunnerRouteReportFields.VALIDATION_METADATA_ROOT);
    }

    public static List<String> optionalPayloadFields() {
        return List.of(
                Payload.REQUESTED_MODEL,
                Payload.MODEL,
                Payload.LOCAL_PATH,
                Payload.PROVIDER,
                Payload.FORMAT,
                Payload.ROUTE_REDIRECT_REASON,
                Payload.ROUTE_ARTIFACT_CACHE_KIND,
                Payload.ROUTE_PROFILE_PROVIDER,
                Payload.ROUTE_PROFILE_FORMAT,
                Payload.ROUTE_PROFILE_REASON,
                Payload.ROUTE_PROFILE_ADVICE,
                VALIDATION_ROOT);
    }

    public static List<String> validationFields() {
        return List.of(
                Validation.CONTRACT_ID,
                Validation.SCHEMA_VERSION,
                Validation.SCHEMA_FINGERPRINT,
                Validation.PASSED,
                Validation.FAILED,
                Validation.PROBLEM_COUNT,
                Validation.PROBLEMS);
    }

    public static String schemaFingerprint() {
        String payload = String.join("\n",
                "contractId=" + CONTRACT_ID,
                "schemaVersion=" + SCHEMA_VERSION,
                "validationRoot=" + VALIDATION_ROOT,
                "payloadFields=" + String.join(",", payloadFields()),
                "requiredPayloadFields=" + String.join(",", requiredPayloadFields()),
                "optionalPayloadFields=" + String.join(",", optionalPayloadFields()),
                "validationFields=" + String.join(",", validationFields()));
        return "sha256:" + sha256(payload);
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(digest.length * 2);
            for (byte raw : digest) {
                int valueByte = raw & 0xff;
                if (valueByte < 16) {
                    hex.append('0');
                }
                hex.append(Integer.toHexString(valueByte));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is required for route report schema fingerprints.", error);
        }
    }
}
