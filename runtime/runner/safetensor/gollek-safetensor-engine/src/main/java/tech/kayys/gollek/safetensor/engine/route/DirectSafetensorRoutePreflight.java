/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.route;
import tech.kayys.gollek.sdk.route.*;

import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ActionKind;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.MissingRuntimeCapability;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemDetail;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemCode;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Adds direct safetensor execution-route guards to dry route preflight reports.
 */
public final class DirectSafetensorRoutePreflight {
    private DirectSafetensorRoutePreflight() {
    }

    public static RoutePreflightReport applyGemma4UnifiedValidation(
            RoutePreflightReport preflight,
            String provider,
            String requestedModelId,
            String localPath) {
        return applyGemma4UnifiedValidation(
                preflight,
                provider,
                requestedModelId,
                localPath,
                false,
                false);
    }

    public static RoutePreflightReport applyGemma4UnifiedValidation(
            RoutePreflightReport preflight,
            String provider,
            String requestedModelId,
            String localPath,
            boolean hasVisionInput,
            boolean ocrMode) {
        RoutePreflightReport base = preflight == null
                ? new RoutePreflightReport(false, List.of(), List.of())
                : preflight;
        DirectSafetensorRoutePolicy.RouteValidation multimodalValidation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedMultimodalExecutionRoute(
                        provider,
                        requestedModelId,
                        localPath,
                        hasVisionInput,
                        ocrMode);
        if (!multimodalValidation.allowed()) {
            return withRouteValidationProblems(base, multimodalValidation);
        }

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        provider,
                        requestedModelId,
                        localPath);
        if (validation.allowed()) {
            return base;
        }
        return withRouteValidationProblems(base, validation);
    }

    private static RoutePreflightReport withRouteValidationProblems(
            RoutePreflightReport preflight,
            DirectSafetensorRoutePolicy.RouteValidation validation) {
        String problemCode = directRouteProblemCode(validation);
        Map<String, Object> diagnosticDetails = directRouteProblemDetails(validation, problemCode);
        RoutePreflightReport withProblems = preflight.withAdditionalProblems(validation.messages().stream()
                .map(message -> new RoutePreflightProblem(
                        problemCode,
                        "error",
                        message,
                        diagnosticDetails))
                .toList());
        return withProblems.withAdditionalActions(directRouteValidationActions(
                validation,
                problemCode,
                diagnosticDetails));
    }

    private static List<RoutePreflightAction> directRouteValidationActions(
            DirectSafetensorRoutePolicy.RouteValidation validation,
            String problemCode,
            Map<String, Object> diagnosticDetails) {
        String joinedMessages = joinedMessages(validation);
        String description;
        if (ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problemCode)) {
            description = "Inspect attached model-family and runtime plugins for Gemma 4 packed-expert routing support.";
        } else if (ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING.equals(problemCode)) {
            description = "Inspect attached model-family and unified runtime plugins for Gemma 4 multimodal embedder support.";
        } else if (ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING.equals(problemCode)) {
            description = "Inspect attached provider and runtime plugins for Gemma 4 mobile QAT loader support.";
        } else if (ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH.equals(problemCode)) {
            description = "Inspect the Gemma 4 safetensor headers and direct text-adapter tensor coverage.";
        } else if (joinedMessages.contains("provider") || joinedMessages.contains("runner")) {
            description = "Inspect attached provider, runner, and model-family plugins before serving this safetensor route.";
        } else {
            description = "Inspect attached provider, runner, and model-family plugins before serving this safetensor route.";
        }
        return List.of(new RoutePreflightAction(
                ActionKind.INSPECT_MODULES,
                problemCode,
                description,
                List.of("gollek", "modules", "--json"),
                diagnosticDetails));
    }

    private static String directRouteProblemCode(DirectSafetensorRoutePolicy.RouteValidation validation) {
        String joinedMessages = joinedMessages(validation);
        if (joinedMessages.contains("mobile qat")) {
            return ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING;
        }
        if (joinedMessages.contains("multimodal safetensor runtime is not enabled")) {
            return ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING;
        }
        if (joinedMessages.contains("packed moe") || joinedMessages.contains("packed expert")) {
            return ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING;
        }
        if (joinedMessages.contains("multimodal")
                || joinedMessages.contains("vision")
                || joinedMessages.contains("audio")
                || joinedMessages.contains("video")
                || joinedMessages.contains("unified safetensor runtime is not enabled")) {
            return ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING;
        }
        if (joinedMessages.contains("text preflight")
                || joinedMessages.contains("safetensors header")
                || joinedMessages.contains("no .safetensors")
                || joinedMessages.contains("missing tensor")
                || joinedMessages.contains("tensor shape")) {
            return ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH;
        }
        return ProblemCode.DIRECT_ROUTE_VALIDATION_FAILED;
    }

    private static Map<String, Object> directRouteProblemDetails(
            DirectSafetensorRoutePolicy.RouteValidation validation,
            String problemCode) {
        String missingCapability = missingRuntimeCapability(problemCode);
        if (missingCapability == null) {
            return Map.of();
        }
        String joinedMessages = joinedMessages(validation);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.MODEL_FAMILY, "gemma4");
        details.put(ProblemDetail.RUNTIME_ROUTE, "safetensor_direct");
        details.put(ProblemDetail.CHECKPOINT_PROFILE, checkpointProfile(problemCode));
        String requestInputMode = requestInputMode(joinedMessages);
        if (requestInputMode != null) {
            details.put(ProblemDetail.REQUEST_INPUT_MODE, requestInputMode);
        }
        details.put(ProblemDetail.MISSING_RUNTIME_CAPABILITY, missingCapability);
        details.put(ProblemDetail.BLOCKED_CAPABILITY, missingCapability);
        details.put(ProblemDetail.RECOMMENDED_RUNTIME, recommendedRuntime(problemCode));
        details.putAll(validation.details());
        return details;
    }

    private static String missingRuntimeCapability(String problemCode) {
        if (ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH.equals(problemCode)) {
            return MissingRuntimeCapability.GEMMA4_TEXT_SAFETENSOR_HEADERS;
        }
        if (ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING.equals(problemCode)) {
            return MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER;
        }
        if (ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problemCode)) {
            return MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER;
        }
        if (ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING.equals(problemCode)) {
            return MissingRuntimeCapability.GEMMA4_MOBILE_QAT_LOADER;
        }
        return null;
    }

    private static String checkpointProfile(String problemCode) {
        if (ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING.equals(problemCode)) {
            return "gemma4_mobile_qat";
        }
        if (ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problemCode)) {
            return "gemma4_packed_moe";
        }
        if (ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH.equals(problemCode)) {
            return "gemma4_text";
        }
        return "gemma4_unified";
    }

    private static String recommendedRuntime(String problemCode) {
        if (ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH.equals(problemCode)) {
            return "gemma4_text_safetensor_header_contract";
        }
        if (ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING.equals(problemCode)) {
            return "gemma4_unified_multimodal_embedder_runtime";
        }
        if (ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problemCode)) {
            return "gemma4_packed_moe_router_runtime";
        }
        if (ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING.equals(problemCode)) {
            return "gemma4_mobile_qat_loader";
        }
        return "safetensor_direct_runtime";
    }

    private static String requestInputMode(String joinedMessages) {
        if (joinedMessages.contains("--ocr")) {
            return "ocr";
        }
        if (joinedMessages.contains("--image")) {
            return "image";
        }
        return null;
    }

    private static String joinedMessages(DirectSafetensorRoutePolicy.RouteValidation validation) {
        return validation == null || validation.messages() == null
                ? ""
                : String.join("\n", validation.messages()).toLowerCase(Locale.ROOT);
    }
}
