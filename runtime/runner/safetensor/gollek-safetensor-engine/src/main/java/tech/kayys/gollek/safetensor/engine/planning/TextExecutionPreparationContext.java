package tech.kayys.gollek.safetensor.engine.planning;

import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionState;

import java.util.Objects;

/**
 * Typed planning context for one text execution request.
 *
 * <p>This captures the backend, model family, and hardware profile the planner
 * should reason about before selecting prefill, decode, or resumability policy.
 * Keeping this explicit makes it possible to evolve Gollek toward backend- and
 * device-specific strategies instead of one generic rule set for every runtime.
 *
 * <p>The hardware-aware edge inference framing is especially relevant on Apple
 * Silicon; see Barrios (2026), <a href="https://arxiv.org/abs/2601.19139">arXiv:2601.19139</a>.
 */
public record TextExecutionPreparationContext(
        String backendId,
        String modelFamily,
        String primaryArchitecture,
        String conversationSessionId,
        ConversationExecutionState conversationExecutionState,
        HardwareProfile hardwareProfile,
        PlanningProfile planningProfile,
        boolean multimodalArchitecture,
        boolean supportsTextPrefixCaching,
        boolean supportsStatefulPreparedModels) {

    public enum HardwareProfile {
        APPLE_SILICON,
        X86_64,
        GENERIC_CPU
    }

    public enum PlanningProfile {
        GENERIC_REFERENCE,
        APPLE_SILICON_DIRECT,
        PREFIX_CACHE_READY,
        STATEFUL_PREPARED_READY
    }

    public TextExecutionPreparationContext {
        Objects.requireNonNull(backendId, "backendId");
        Objects.requireNonNull(modelFamily, "modelFamily");
        Objects.requireNonNull(primaryArchitecture, "primaryArchitecture");
        Objects.requireNonNull(conversationExecutionState, "conversationExecutionState");
        Objects.requireNonNull(hardwareProfile, "hardwareProfile");
        Objects.requireNonNull(planningProfile, "planningProfile");
    }

    public boolean hasConversationSession() {
        return conversationSessionId != null && !conversationSessionId.isBlank();
    }

    public boolean hasActiveConversationState() {
        return conversationExecutionState.active();
    }
}
