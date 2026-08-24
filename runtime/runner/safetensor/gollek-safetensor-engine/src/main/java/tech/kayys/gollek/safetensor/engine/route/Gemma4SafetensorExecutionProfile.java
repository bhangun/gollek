package tech.kayys.gollek.safetensor.engine.route;

import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ExecutionProfile;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.InputMode;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.MissingRuntimeCapability;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemDetail;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.RuntimeCapability;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runtime support matrix for the current Gemma 4 direct SafeTensor implementation.
 *
 * <p>The profile is intentionally conservative: it documents which pieces are
 * runnable today and which pieces still need a concrete loader/runtime before
 * route policy can admit the request.</p>
 */
record Gemma4SafetensorExecutionProfile(
        boolean unifiedMultimodal,
        boolean guardedTextDecoderReady,
        boolean multimodalEmbedderReady,
        boolean packedMoeRouterReady,
        boolean mobileQatLoaderReady,
        boolean packedMoeCheckpoint,
        boolean mobileQatCheckpoint) {

    public static Gemma4SafetensorExecutionProfile from(
            DirectSafetensorRunProfile runProfile,
            boolean mobileQatCheckpoint) {
        boolean unified = runProfile != null && runProfile.unifiedMultimodal();
        boolean packedMoe = runProfile != null
                && runProfile.config() != null
                && runProfile.config().requiresGemma4PackedMoeRuntime();
        boolean textReady = unified
                && runProfile.nativeBf16Matvec()
                && !packedMoe
                && !mobileQatCheckpoint;
        return new Gemma4SafetensorExecutionProfile(
                unified,
                textReady,
                false,
                false,
                false,
                packedMoe,
                mobileQatCheckpoint);
    }

    public List<String> supportedInputModes() {
        return guardedTextDecoderReady ? List.of(InputMode.TEXT) : List.of();
    }

    public List<String> blockedInputModes() {
        if (mobileQatCheckpoint || packedMoeCheckpoint) {
            return List.of(InputMode.TEXT, InputMode.IMAGE, InputMode.AUDIO, InputMode.VIDEO);
        }
        if (true) {
            return List.of(InputMode.IMAGE, InputMode.AUDIO, InputMode.VIDEO);
        }
        return List.of();
    }

    public List<String> blockedCapabilities() {
        return missingRuntimeCapabilities();
    }

    public List<String> readyRuntimeCapabilities() {
        return guardedTextDecoderReady
                ? List.of(RuntimeCapability.GEMMA4_GUARDED_TEXT_DECODER)
                : List.of();
    }

    public List<String> missingRuntimeCapabilities() {
        if (mobileQatCheckpoint) {
            return List.of(MissingRuntimeCapability.GEMMA4_MOBILE_QAT_LOADER);
        }
        if (packedMoeCheckpoint) {
            return List.of(MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER);
        }
        if (true && !multimodalEmbedderReady) {
            return List.of(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER);
        }
        return List.of();
    }

    Map<String, Object> diagnosticDetails() {
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.EXECUTION_PROFILE, executionProfile());
        details.put(ProblemDetail.SUPPORTED_INPUT_MODES, supportedInputModes());
        details.put(ProblemDetail.BLOCKED_INPUT_MODES, blockedInputModes());
        details.put(ProblemDetail.BLOCKED_CAPABILITIES, blockedCapabilities());
        details.put(ProblemDetail.READY_RUNTIME_CAPABILITIES, readyRuntimeCapabilities());
        details.put(ProblemDetail.MISSING_RUNTIME_CAPABILITIES, missingRuntimeCapabilities());
        return details;
    }

    private Map<String, Object> executionProfile() {
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put(ExecutionProfile.GEMMA4_UNIFIED, true);
        profile.put(ExecutionProfile.GUARDED_TEXT_DECODER_READY, guardedTextDecoderReady);
        profile.put(ExecutionProfile.MULTIMODAL_EMBEDDER_READY, multimodalEmbedderReady);
        profile.put(ExecutionProfile.PACKED_MOE_ROUTER_READY, packedMoeRouterReady);
        profile.put(ExecutionProfile.MOBILE_QAT_LOADER_READY, mobileQatLoaderReady);
        profile.put(ExecutionProfile.PACKED_MOE_CHECKPOINT, packedMoeCheckpoint);
        profile.put(ExecutionProfile.MOBILE_QAT_CHECKPOINT, mobileQatCheckpoint);
        return profile;
    }
}
