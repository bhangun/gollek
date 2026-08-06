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
 * Stable JSON field names for runner/provider route reports.
 */
public final class RunnerRouteReportFields {
    public static final String CONTRACT_ID = "gollek.runner-route.report";
    public static final int SCHEMA_VERSION = 6;
    public static final String METADATA_ROOT = "runner_route";
    public static final String METADATA_PREFIX = METADATA_ROOT + ".";
    public static final String VALIDATION_METADATA_ROOT = "runner_route_validation";
    public static final String VALIDATION_METADATA_PREFIX = VALIDATION_METADATA_ROOT + ".";

    private RunnerRouteReportFields() {
    }

    public static final class Schema {
        public static final String CONTRACT_ID = "contractId";
        public static final String SCHEMA_VERSION = "schemaVersion";
        public static final String SCHEMA_FINGERPRINT = "schemaFingerprint";
        public static final String METADATA_ROOT = "metadataRoot";
        public static final String METADATA_PREFIX = "metadataPrefix";
        public static final String VALIDATION_METADATA_ROOT = "validationMetadataRoot";
        public static final String VALIDATION_METADATA_PREFIX = "validationMetadataPrefix";
        public static final String REPORT_FIELDS = "reportFields";
        public static final String REQUIRED_REPORT_FIELDS = "requiredReportFields";
        public static final String SELECTION_SOURCES = "selectionSources";
        public static final String ROUTE_PROFILE_STATUSES = "routeProfileStatuses";
        public static final String ROUTE_PROFILE_SOURCES = "routeProfileSources";

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

    public static final class Report {
        public static final String REQUESTED_RUNNER = "requested_runner";
        public static final String NORMALIZED_RUNNER = "normalized_runner";
        public static final String SELECTION_SOURCE = "selection_source";
        public static final String RUNNER_EXPLICIT = "runner_explicit";
        public static final String REQUESTED_PROVIDER = "requested_provider";
        public static final String PROVIDER_EXPLICIT = "provider_explicit";
        public static final String REQUESTED_FORMAT = "requested_format";
        public static final String POLICY_PROVIDER = "policy_provider";
        public static final String POLICY_FORMAT = "policy_format";
        public static final String EFFECTIVE_PROVIDER = "effective_provider";
        public static final String EFFECTIVE_FORMAT = "effective_format";
        public static final String RUNTIME_REDIRECTED = "runtime_redirected";
        public static final String RUNTIME_REDIRECT_FROM_PROVIDER = "runtime_redirect_from_provider";
        public static final String RUNTIME_REDIRECT_FROM_FORMAT = "runtime_redirect_from_format";
        public static final String RUNTIME_REDIRECT_TO_PROVIDER = "runtime_redirect_to_provider";
        public static final String RUNTIME_REDIRECT_TO_FORMAT = "runtime_redirect_to_format";
        public static final String RUNTIME_REDIRECT_REASON = "runtime_redirect_reason";
        public static final String RUNTIME_REDIRECT_CACHE_HIT = "runtime_redirect_cache_hit";
        public static final String RUNTIME_REDIRECT_CACHE_KIND = "runtime_redirect_cache_kind";
        public static final String PREFER_ALTERNATE_RUNTIME = "prefer_alternate_runtime";
        public static final String FORCE_GGUF = "force_gguf";
        public static final String MODE = "mode";
        public static final String AUTO_DETECTED = "auto_detected";
        public static final String PROVIDER_INFERRED = "provider_inferred";
        public static final String ROUTE_PROFILE_STATUS = "route_profile_status";
        public static final String ROUTE_PROFILE_SOURCE = "route_profile_source";
        public static final String ROUTE_PROFILE_PROVIDER = "route_profile_provider";
        public static final String ROUTE_PROFILE_FORMAT = "route_profile_format";
        public static final String ROUTE_PROFILE_REASON = "route_profile_reason";
        public static final String ROUTE_PROFILE_ADVICE = "route_profile_advice";

        private Report() {
        }
    }

    public static final class SelectionSource {
        public static final String AUTO = "auto";
        public static final String RUNNER_CLI = "runner_cli";
        public static final String PROVIDER_CLI = "provider_cli";
        public static final String FORMAT_CLI = "format_cli";

        private SelectionSource() {
        }
    }

    public static final class RouteProfileStatus {
        public static final String UNAVAILABLE = "unavailable";
        public static final String SELECTED = "selected";
        public static final String REDIRECTED = "redirected";
        public static final String CANDIDATE = "candidate";

        private RouteProfileStatus() {
        }
    }

    public static final class RouteProfileSource {
        public static final String NONE = "none";
        public static final String SELECTED_ROUTE = "selected_route";
        public static final String RUNTIME_REDIRECT = "runtime_redirect";
        public static final String ARTIFACT_CACHE = "artifact_cache";
        public static final String HEURISTIC = "heuristic";
        public static final String BENCHMARK_CACHE = "benchmark_cache";

        private RouteProfileSource() {
        }
    }

    public static List<String> reportFields() {
        return List.of(
                Report.REQUESTED_RUNNER,
                Report.NORMALIZED_RUNNER,
                Report.SELECTION_SOURCE,
                Report.RUNNER_EXPLICIT,
                Report.REQUESTED_PROVIDER,
                Report.PROVIDER_EXPLICIT,
                Report.REQUESTED_FORMAT,
                Report.POLICY_PROVIDER,
                Report.POLICY_FORMAT,
                Report.EFFECTIVE_PROVIDER,
                Report.EFFECTIVE_FORMAT,
                Report.RUNTIME_REDIRECTED,
                Report.RUNTIME_REDIRECT_FROM_PROVIDER,
                Report.RUNTIME_REDIRECT_FROM_FORMAT,
                Report.RUNTIME_REDIRECT_TO_PROVIDER,
                Report.RUNTIME_REDIRECT_TO_FORMAT,
                Report.RUNTIME_REDIRECT_REASON,
                Report.RUNTIME_REDIRECT_CACHE_HIT,
                Report.RUNTIME_REDIRECT_CACHE_KIND,
                Report.PREFER_ALTERNATE_RUNTIME,
                Report.FORCE_GGUF,
                Report.MODE,
                Report.AUTO_DETECTED,
                Report.PROVIDER_INFERRED,
                Report.ROUTE_PROFILE_STATUS,
                Report.ROUTE_PROFILE_SOURCE,
                Report.ROUTE_PROFILE_PROVIDER,
                Report.ROUTE_PROFILE_FORMAT,
                Report.ROUTE_PROFILE_REASON,
                Report.ROUTE_PROFILE_ADVICE);
    }

    public static List<String> selectionSources() {
        return List.of(
                SelectionSource.AUTO,
                SelectionSource.RUNNER_CLI,
                SelectionSource.PROVIDER_CLI,
                SelectionSource.FORMAT_CLI);
    }

    public static List<String> routeProfileStatuses() {
        return List.of(
                RouteProfileStatus.UNAVAILABLE,
                RouteProfileStatus.SELECTED,
                RouteProfileStatus.REDIRECTED,
                RouteProfileStatus.CANDIDATE);
    }

    public static List<String> routeProfileSources() {
        return List.of(
                RouteProfileSource.NONE,
                RouteProfileSource.SELECTED_ROUTE,
                RouteProfileSource.RUNTIME_REDIRECT,
                RouteProfileSource.ARTIFACT_CACHE,
                RouteProfileSource.HEURISTIC,
                RouteProfileSource.BENCHMARK_CACHE);
    }

    public static List<String> requiredReportFields() {
        return List.of(
                Report.REQUESTED_RUNNER,
                Report.NORMALIZED_RUNNER,
                Report.SELECTION_SOURCE,
                Report.RUNNER_EXPLICIT,
                Report.PROVIDER_EXPLICIT,
                Report.RUNTIME_REDIRECTED,
                Report.RUNTIME_REDIRECT_CACHE_HIT,
                Report.PREFER_ALTERNATE_RUNTIME,
                Report.FORCE_GGUF,
                Report.MODE,
                Report.AUTO_DETECTED,
                Report.PROVIDER_INFERRED,
                Report.ROUTE_PROFILE_STATUS,
                Report.ROUTE_PROFILE_SOURCE);
    }

    public static String metadataKey(String field) {
        return METADATA_PREFIX + field;
    }

    public static String validationMetadataKey(String field) {
        return VALIDATION_METADATA_PREFIX + field;
    }

    public static String schemaFingerprint() {
        String payload = String.join("\n",
                "contractId=" + CONTRACT_ID,
                "schemaVersion=" + SCHEMA_VERSION,
                "metadataRoot=" + METADATA_ROOT,
                "metadataPrefix=" + METADATA_PREFIX,
                "validationMetadataRoot=" + VALIDATION_METADATA_ROOT,
                "validationMetadataPrefix=" + VALIDATION_METADATA_PREFIX,
                "reportFields=" + String.join(",", reportFields()),
                "requiredReportFields=" + String.join(",", requiredReportFields()),
                "selectionSources=" + String.join(",", selectionSources()),
                "routeProfileStatuses=" + String.join(",", routeProfileStatuses()),
                "routeProfileSources=" + String.join(",", routeProfileSources()));
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
            throw new IllegalStateException("SHA-256 is required for report schema fingerprints.", error);
        }
    }
}
