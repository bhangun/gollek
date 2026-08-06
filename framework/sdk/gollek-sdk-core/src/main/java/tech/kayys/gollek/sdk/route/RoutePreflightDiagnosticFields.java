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
 * Stable JSON field names for route preflight problems and next actions.
 */
public final class RoutePreflightDiagnosticFields {
    public static final String CONTRACT_ID = "gollek.route-preflight.diagnostics";
    public static final int SCHEMA_VERSION = 16;
    public static final String VALIDATION_ROOT = "route_preflight_diagnostics_validation";

    private RoutePreflightDiagnosticFields() {
    }

    public static final class Schema {
        public static final String CONTRACT_ID = "contractId";
        public static final String SCHEMA_VERSION = "schemaVersion";
        public static final String SCHEMA_FINGERPRINT = "schemaFingerprint";
        public static final String VALIDATION_ROOT = "validationRoot";
        public static final String PROBLEM_FIELDS = "problemFields";
        public static final String REQUIRED_PROBLEM_FIELDS = "requiredProblemFields";
        public static final String PROBLEM_DETAIL_FIELDS = "problemDetailFields";
        public static final String EXECUTION_PROFILE_FIELDS = "executionProfileFields";
        public static final String HEADER_INSPECTION_FIELDS = "headerInspectionFields";
        public static final String TENSOR_INVENTORY_FIELDS = "tensorInventoryFields";
        public static final String COMPONENT_READINESS_FIELDS = "componentReadinessFields";
        public static final String INPUT_MODES = "inputModes";
        public static final String RUNTIME_CAPABILITIES = "runtimeCapabilities";
        public static final String MISSING_RUNTIME_CAPABILITIES = "missingRuntimeCapabilities";
        public static final String ACTION_FIELDS = "actionFields";
        public static final String REQUIRED_ACTION_FIELDS = "requiredActionFields";
        public static final String ACTION_DETAIL_FIELDS = "actionDetailFields";
        public static final String VALIDATION_FIELDS = "validationFields";
        public static final String PROBLEM_CODES = "problemCodes";
        public static final String ACTION_KINDS = "actionKinds";

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

    public static final class Problem {
        public static final String CODE = "code";
        public static final String SEVERITY = "severity";
        public static final String MESSAGE = "message";
        public static final String DETAILS = "details";

        private Problem() {
        }
    }

    /**
     * Stable optional fields for structured route preflight problem/action details.
     */
    public static final class ProblemDetail {
        public static final String MODEL_FAMILY = "modelFamily";
        public static final String RUNTIME_ROUTE = "runtimeRoute";
        public static final String CHECKPOINT_PROFILE = "checkpointProfile";
        public static final String REQUEST_INPUT_MODE = "requestInputMode";
        public static final String MISSING_RUNTIME_CAPABILITY = "missingRuntimeCapability";
        public static final String BLOCKED_CAPABILITY = "blockedCapability";
        public static final String RECOMMENDED_RUNTIME = "recommendedRuntime";
        public static final String HEADER_PREFLIGHT = "headerPreflight";
        public static final String HEADER_INSPECTION = "headerInspection";
        public static final String TENSOR_INVENTORY = "tensorInventory";
        public static final String COMPONENT_READINESS = "componentReadiness";
        public static final String DETECTED_PROJECTORS = "detectedProjectors";
        public static final String DETECTED_PACKED_MOE = "detectedPackedMoe";
        public static final String EXECUTION_PROFILE = "executionProfile";
        public static final String SUPPORTED_INPUT_MODES = "supportedInputModes";
        public static final String BLOCKED_INPUT_MODES = "blockedInputModes";
        public static final String BLOCKED_CAPABILITIES = "blockedCapabilities";
        public static final String READY_RUNTIME_CAPABILITIES = "readyRuntimeCapabilities";
        public static final String MISSING_RUNTIME_CAPABILITIES = "missingRuntimeCapabilities";

        private ProblemDetail() {
        }
    }

    /**
     * Stable fields inside the header-only SafeTensors inspection detail object.
     */
    public static final class HeaderInspection {
        public static final String SAFETENSOR_FILE_COUNT = "safetensorFileCount";
        public static final String TENSOR_COUNT = "tensorCount";
        public static final String HEADER_BYTES_READ = "headerBytesRead";
        public static final String PAYLOAD_BYTES_LOADED = "payloadBytesLoaded";

        private HeaderInspection() {
        }
    }

    /**
     * Stable fields inside the Gemma 4 SafeTensor tensor-inventory summary.
     */
    public static final class TensorInventory {
        public static final String TEXT_DECODER_TENSORS = "textDecoderTensors";
        public static final String EMBEDDING_TENSORS = "embeddingTensors";
        public static final String LOGITS_HEAD_TENSORS = "logitsHeadTensors";
        public static final String MULTIMODAL_PROJECTOR_TENSORS = "multimodalProjectorTensors";
        public static final String VISION_TOWER_TENSORS = "visionTowerTensors";
        public static final String AUDIO_TOWER_TENSORS = "audioTowerTensors";
        public static final String VIDEO_TOWER_TENSORS = "videoTowerTensors";
        public static final String PACKED_MOE_ROUTER_TENSORS = "packedMoeRouterTensors";
        public static final String PACKED_MOE_EXPERT_TENSORS = "packedMoeExpertTensors";
        public static final String UNCLASSIFIED_TENSORS = "unclassifiedTensors";

        private TensorInventory() {
        }
    }

    /**
     * Stable fields inside the Gemma 4 header-only component readiness summary.
     */
    public static final class ComponentReadiness {
        public static final String TEXT_DECODER_READY = "textDecoderReady";
        public static final String VISION_PROJECTOR_READY = "visionProjectorReady";
        public static final String AUDIO_PROJECTOR_READY = "audioProjectorReady";
        public static final String VIDEO_PROJECTOR_READY = "videoProjectorReady";
        public static final String MULTIMODAL_PROJECTOR_READY = "multimodalProjectorReady";
        public static final String PACKED_MOE_ROUTER_READY = "packedMoeRouterReady";
        public static final String PACKED_MOE_EXPERTS_READY = "packedMoeExpertsReady";
        public static final String PACKED_MOE_HEADER_READY = "packedMoeHeaderReady";

        private ComponentReadiness() {
        }
    }

    /**
     * Stable fields inside the Gemma 4 direct SafeTensor execution-profile object.
     */
    public static final class ExecutionProfile {
        public static final String GEMMA4_UNIFIED = "gemma4Unified";
        public static final String GUARDED_TEXT_DECODER_READY = "guardedTextDecoderReady";
        public static final String MULTIMODAL_EMBEDDER_READY = "multimodalEmbedderReady";
        public static final String PACKED_MOE_ROUTER_READY = "packedMoeRouterReady";
        public static final String MOBILE_QAT_LOADER_READY = "mobileQatLoaderReady";
        public static final String PACKED_MOE_CHECKPOINT = "packedMoeCheckpoint";
        public static final String MOBILE_QAT_CHECKPOINT = "mobileQatCheckpoint";

        private ExecutionProfile() {
        }
    }

    /**
     * Stable input-mode identifiers used by route support matrices.
     */
    public static final class InputMode {
        public static final String TEXT = "text";
        public static final String IMAGE = "image";
        public static final String AUDIO = "audio";
        public static final String VIDEO = "video";

        private InputMode() {
        }
    }

    /**
     * Stable identifiers for runtime capabilities required to clear a preflight blocker.
     */
    public static final class MissingRuntimeCapability {
        public static final String GEMMA4_TEXT_SAFETENSOR_HEADERS = "gemma4_text_safetensor_headers";
        public static final String GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER = "gemma4_unified_multimodal_embedder";
        public static final String GEMMA4_PACKED_MOE_ROUTER = "gemma4_packed_moe_router";
        public static final String GEMMA4_MOBILE_QAT_LOADER = "gemma4_mobile_qat_loader";

        private MissingRuntimeCapability() {
        }
    }

    /**
     * Stable identifiers for concrete runtime capabilities exposed by direct runners.
     */
    public static final class RuntimeCapability {
        public static final String GEMMA4_GUARDED_TEXT_DECODER = "gemma4_guarded_text_decoder";

        private RuntimeCapability() {
        }
    }

    public static final class Action {
        public static final String KIND = "kind";
        public static final String REASON = "reason";
        public static final String DESCRIPTION = "description";
        public static final String ARGV = "argv";
        public static final String DETAILS = "details";

        private Action() {
        }
    }

    public static final class ProblemCode {
        public static final String MODEL_NOT_LOCAL = "model_not_local";
        public static final String PROVIDER_NOT_RESOLVED = "provider_not_resolved";
        public static final String FORMAT_NOT_RESOLVED = "format_not_resolved";
        public static final String DIRECT_ROUTE_VALIDATION_FAILED = "direct_route_validation_failed";
        public static final String GEMMA4_MULTIMODAL_RUNTIME_MISSING = "gemma4_multimodal_runtime_missing";
        public static final String GEMMA4_PACKED_MOE_RUNTIME_MISSING = "gemma4_packed_moe_runtime_missing";
        public static final String GEMMA4_TEXT_HEADER_MISMATCH = "gemma4_text_header_mismatch";
        public static final String GEMMA4_MOBILE_QAT_LOADER_MISSING = "gemma4_mobile_qat_loader_missing";

        private ProblemCode() {
        }
    }

    public static final class ActionKind {
        public static final String PULL_MODEL = "pull_model";
        public static final String ALLOW_PULL_RESOLUTION = "allow_pull_resolution";
        public static final String INSPECT_MODULES = "inspect_modules";

        private ActionKind() {
        }
    }

    public static List<String> problemFields() {
        return List.of(Problem.CODE, Problem.SEVERITY, Problem.MESSAGE, Problem.DETAILS);
    }

    public static List<String> requiredProblemFields() {
        return List.of(Problem.CODE, Problem.SEVERITY, Problem.MESSAGE);
    }

    public static List<String> problemDetailFields() {
        return List.of(
                ProblemDetail.MODEL_FAMILY,
                ProblemDetail.RUNTIME_ROUTE,
                ProblemDetail.CHECKPOINT_PROFILE,
                ProblemDetail.REQUEST_INPUT_MODE,
                ProblemDetail.MISSING_RUNTIME_CAPABILITY,
                ProblemDetail.BLOCKED_CAPABILITY,
                ProblemDetail.RECOMMENDED_RUNTIME,
                ProblemDetail.HEADER_PREFLIGHT,
                ProblemDetail.HEADER_INSPECTION,
                ProblemDetail.TENSOR_INVENTORY,
                ProblemDetail.COMPONENT_READINESS,
                ProblemDetail.DETECTED_PROJECTORS,
                ProblemDetail.DETECTED_PACKED_MOE,
                ProblemDetail.EXECUTION_PROFILE,
                ProblemDetail.SUPPORTED_INPUT_MODES,
                ProblemDetail.BLOCKED_INPUT_MODES,
                ProblemDetail.BLOCKED_CAPABILITIES,
                ProblemDetail.READY_RUNTIME_CAPABILITIES,
                ProblemDetail.MISSING_RUNTIME_CAPABILITIES);
    }

    public static List<String> headerInspectionFields() {
        return List.of(
                HeaderInspection.SAFETENSOR_FILE_COUNT,
                HeaderInspection.TENSOR_COUNT,
                HeaderInspection.HEADER_BYTES_READ,
                HeaderInspection.PAYLOAD_BYTES_LOADED);
    }

    public static List<String> tensorInventoryFields() {
        return List.of(
                TensorInventory.TEXT_DECODER_TENSORS,
                TensorInventory.EMBEDDING_TENSORS,
                TensorInventory.LOGITS_HEAD_TENSORS,
                TensorInventory.MULTIMODAL_PROJECTOR_TENSORS,
                TensorInventory.VISION_TOWER_TENSORS,
                TensorInventory.AUDIO_TOWER_TENSORS,
                TensorInventory.VIDEO_TOWER_TENSORS,
                TensorInventory.PACKED_MOE_ROUTER_TENSORS,
                TensorInventory.PACKED_MOE_EXPERT_TENSORS,
                TensorInventory.UNCLASSIFIED_TENSORS);
    }

    public static List<String> componentReadinessFields() {
        return List.of(
                ComponentReadiness.TEXT_DECODER_READY,
                ComponentReadiness.VISION_PROJECTOR_READY,
                ComponentReadiness.AUDIO_PROJECTOR_READY,
                ComponentReadiness.VIDEO_PROJECTOR_READY,
                ComponentReadiness.MULTIMODAL_PROJECTOR_READY,
                ComponentReadiness.PACKED_MOE_ROUTER_READY,
                ComponentReadiness.PACKED_MOE_EXPERTS_READY,
                ComponentReadiness.PACKED_MOE_HEADER_READY);
    }

    public static List<String> executionProfileFields() {
        return List.of(
                ExecutionProfile.GEMMA4_UNIFIED,
                ExecutionProfile.GUARDED_TEXT_DECODER_READY,
                ExecutionProfile.MULTIMODAL_EMBEDDER_READY,
                ExecutionProfile.PACKED_MOE_ROUTER_READY,
                ExecutionProfile.MOBILE_QAT_LOADER_READY,
                ExecutionProfile.PACKED_MOE_CHECKPOINT,
                ExecutionProfile.MOBILE_QAT_CHECKPOINT);
    }

    public static List<String> inputModes() {
        return List.of(
                InputMode.TEXT,
                InputMode.IMAGE,
                InputMode.AUDIO,
                InputMode.VIDEO);
    }

    public static List<String> missingRuntimeCapabilities() {
        return List.of(
                MissingRuntimeCapability.GEMMA4_TEXT_SAFETENSOR_HEADERS,
                MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER,
                MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER,
                MissingRuntimeCapability.GEMMA4_MOBILE_QAT_LOADER);
    }

    public static List<String> runtimeCapabilities() {
        return List.of(RuntimeCapability.GEMMA4_GUARDED_TEXT_DECODER);
    }

    public static List<String> actionFields() {
        return List.of(Action.KIND, Action.REASON, Action.DESCRIPTION, Action.ARGV, Action.DETAILS);
    }

    public static List<String> requiredActionFields() {
        return List.of(Action.KIND, Action.REASON, Action.DESCRIPTION, Action.ARGV);
    }

    public static List<String> actionDetailFields() {
        return problemDetailFields();
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

    public static List<String> problemCodes() {
        return List.of(
                ProblemCode.MODEL_NOT_LOCAL,
                ProblemCode.PROVIDER_NOT_RESOLVED,
                ProblemCode.FORMAT_NOT_RESOLVED,
                ProblemCode.DIRECT_ROUTE_VALIDATION_FAILED,
                ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING,
                ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING,
                ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH,
                ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING);
    }

    public static List<String> actionKinds() {
        return List.of(
                ActionKind.PULL_MODEL,
                ActionKind.ALLOW_PULL_RESOLUTION,
                ActionKind.INSPECT_MODULES);
    }

    public static String schemaFingerprint() {
        String payload = String.join("\n",
                "contractId=" + CONTRACT_ID,
                "schemaVersion=" + SCHEMA_VERSION,
                "validationRoot=" + VALIDATION_ROOT,
                "problemFields=" + String.join(",", problemFields()),
                "requiredProblemFields=" + String.join(",", requiredProblemFields()),
                "problemDetailFields=" + String.join(",", problemDetailFields()),
                "executionProfileFields=" + String.join(",", executionProfileFields()),
                "headerInspectionFields=" + String.join(",", headerInspectionFields()),
                "tensorInventoryFields=" + String.join(",", tensorInventoryFields()),
                "componentReadinessFields=" + String.join(",", componentReadinessFields()),
                "inputModes=" + String.join(",", inputModes()),
                "runtimeCapabilities=" + String.join(",", runtimeCapabilities()),
                "missingRuntimeCapabilities=" + String.join(",", missingRuntimeCapabilities()),
                "actionFields=" + String.join(",", actionFields()),
                "requiredActionFields=" + String.join(",", requiredActionFields()),
                "actionDetailFields=" + String.join(",", actionDetailFields()),
                "validationFields=" + String.join(",", validationFields()),
                "problemCodes=" + String.join(",", problemCodes()),
                "actionKinds=" + String.join(",", actionKinds()));
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
            throw new IllegalStateException("SHA-256 is required for route preflight schema fingerprints.", error);
        }
    }
}
