package tech.kayys.gollek.safetensor.engine.planning;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextModel;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionArtifactPlan;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionPreparationPlan;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionPlan;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionState;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

/**
 * Engine-owned policy planner for text execution preparation.
 *
 * <p>This keeps performance-sensitive policy out of backend helper records and in
 * one planner seam the engine can evolve deliberately. As Gollek grows backend-
 * specific prefill, prefix-cache, or persistent-KV strategies, this planner can
 * change policy without rewriting provider glue or backend contracts.
 *
 * <p>The split aligns with recent work on resumable edge inference and prompt
 * reuse, including Shkolnikov (2026),
 * <a href="https://arxiv.org/abs/2603.04428">arXiv:2603.04428</a> and
 * Barrios (2026), <a href="https://arxiv.org/abs/2601.19139">arXiv:2601.19139</a>.
 */
@ApplicationScoped
public class TextExecutionPreparationPlanner {

    public TextExecutionPreparationContext contextFor(
            PreparedTextModel model,
            ConversationExecutionState conversationExecutionState) {
        SafetensorEngine.LoadedModel loadedModel = model != null ? model.loadedModel() : null;
        String modelFamily = loadedModel != null && loadedModel.config() != null && loadedModel.config().modelType() != null
                ? loadedModel.config().modelType()
                : "unknown";
        String primaryArchitecture = loadedModel != null && loadedModel.config() != null
                && loadedModel.config().primaryArchitecture() != null
                ? loadedModel.config().primaryArchitecture()
                : "unknown";
        ModelRuntimeTraits runtimeTraits = loadedModel != null
                ? loadedModel.runtimeTraits()
                : ModelRuntimeTraits.EMPTY;
        boolean supportsPrefixCaching = model != null && model.capabilities().supportsTextPrefixCaching();
        boolean supportsStatefulPreparedModels = model != null && model.capabilities().supportsStatefulPreparedModels();
        boolean multimodalArchitecture = runtimeTraits.multimodalModel();
        TextExecutionPreparationContext.HardwareProfile hardwareProfile = detectHardwareProfile();
        return new TextExecutionPreparationContext(
                model != null ? model.backendId() : "unknown",
                modelFamily,
                primaryArchitecture,
                conversationSessionId(conversationExecutionState),
                conversationExecutionState != null ? conversationExecutionState : ConversationExecutionState.noneRequested(),
                hardwareProfile,
                classifyPlanningProfile(
                        model != null ? model.backendId() : "unknown",
                        hardwareProfile,
                        supportsPrefixCaching,
                        supportsStatefulPreparedModels),
                multimodalArchitecture,
                supportsPrefixCaching,
                supportsStatefulPreparedModels);
    }

    public TextExecutionPreparationPlan plan(
            PreparedPrompt prompt,
            TextExecutionPreparationContext context) {
        TextExecutionSessionPlan sessionPlan = planSession(prompt, context);
        TextExecutionArtifactPlan artifactPlan = planArtifact(prompt, sessionPlan, context);
        return new TextExecutionPreparationPlan(
                sessionPlan,
                artifactPlan,
                context,
                buildRationale(context, sessionPlan, artifactPlan));
    }

    private TextExecutionSessionPlan planSession(
            PreparedPrompt prompt,
            TextExecutionPreparationContext context) {
        if (context != null && context.hasConversationSession()) {
            return new TextExecutionSessionPlan(
                    TextExecutionSessionPlan.PrefillStrategy.FULL_PREFILL,
                    TextExecutionSessionPlan.DecodeStrategy.TOKEN_ITERATIVE,
                    TextExecutionSessionPlan.ReusePolicy.CONVERSATION_STATEFUL,
                    TextExecutionSessionPlan.Scope.CONVERSATION_STATEFUL,
                    context.conversationSessionId(),
                    conversationStateRationale(context));
        }

        if (prompt != null
                && prompt.reusePlan().cacheable()
                && context != null
                && context.supportsTextPrefixCaching()) {
            return new TextExecutionSessionPlan(
                    TextExecutionSessionPlan.PrefillStrategy.PREFIX_REUSE_CANDIDATE,
                    TextExecutionSessionPlan.DecodeStrategy.TOKEN_ITERATIVE,
                    TextExecutionSessionPlan.ReusePolicy.EXACT_PROMPT_REUSABLE,
                    TextExecutionSessionPlan.Scope.PROCESS_LOCAL_EXACT_PROMPT,
                    prompt.reusePlan().reuseKey(),
                    "Planner selected prefix-reuse candidate for exact prepared prompt on "
                            + context.planningProfile().name().toLowerCase());
        }

        if (prompt != null
                && prompt.reusePlan().cacheable()
                && context != null
                && context.supportsStatefulPreparedModels()
                && prefersStatefulPreparedReuse(context)) {
            return new TextExecutionSessionPlan(
                    TextExecutionSessionPlan.PrefillStrategy.FULL_PREFILL,
                    TextExecutionSessionPlan.DecodeStrategy.TOKEN_ITERATIVE,
                    TextExecutionSessionPlan.ReusePolicy.EXACT_PROMPT_REUSABLE,
                    TextExecutionSessionPlan.Scope.PROCESS_LOCAL_EXACT_PROMPT,
                    prompt.reusePlan().reuseKey(),
                    "Planner selected exact-prompt reusable full-prefill session for "
                            + context.planningProfile().name().toLowerCase());
        }

        return new TextExecutionSessionPlan(
                TextExecutionSessionPlan.PrefillStrategy.FULL_PREFILL,
                TextExecutionSessionPlan.DecodeStrategy.TOKEN_ITERATIVE,
                TextExecutionSessionPlan.ReusePolicy.EPHEMERAL,
                TextExecutionSessionPlan.Scope.REQUEST_LOCAL,
                prompt != null ? prompt.fingerprint() : null,
                "Planner selected full-prefill iterative decode for "
                        + (context != null ? context.planningProfile().name().toLowerCase() : "generic_reference"));
    }

    private TextExecutionArtifactPlan planArtifact(
            PreparedPrompt prompt,
            TextExecutionSessionPlan sessionPlan,
            TextExecutionPreparationContext context) {
        if (sessionPlan != null
                && sessionPlan.scope() == TextExecutionSessionPlan.Scope.CONVERSATION_STATEFUL) {
            return new TextExecutionArtifactPlan(
                    TextExecutionArtifactPlan.Strategy.KV_SNAPSHOT,
                    true,
                    sessionPlan.sessionKey(),
                    "Planner selected conversation-scoped KV artifact for "
                            + (context != null ? context.planningProfile().name().toLowerCase() : "generic_reference"));
        }

        if (sessionPlan != null
                && sessionPlan.reusable()
                && sessionPlan.prefillStrategy() == TextExecutionSessionPlan.PrefillStrategy.PREFIX_REUSE_CANDIDATE
                && context != null
                && context.supportsTextPrefixCaching()) {
            return new TextExecutionArtifactPlan(
                    TextExecutionArtifactPlan.Strategy.PREFIX_CACHE,
                    true,
                    prompt != null ? prompt.reusePlan().reuseKey() : sessionPlan.sessionKey(),
                    "Planner selected prefix-cache artifact for "
                            + context.planningProfile().name().toLowerCase());
        }

        if (sessionPlan != null
                && sessionPlan.reusable()
                && context != null
                && context.supportsStatefulPreparedModels()) {
            return new TextExecutionArtifactPlan(
                    TextExecutionArtifactPlan.Strategy.PROCESS_LOCAL_PRETOKENIZED_HANDLE,
                    true,
                    sessionPlan.sessionKey(),
                    "Planner selected process-local prepared-generation artifact for "
                            + context.planningProfile().name().toLowerCase());
        }
        return new TextExecutionArtifactPlan(
                TextExecutionArtifactPlan.Strategy.NONE,
                false,
                prompt != null ? prompt.fingerprint() : null,
                "Planner selected no reusable execution artifact for "
                        + (context != null ? context.planningProfile().name().toLowerCase() : "generic_reference"));
    }

    private TextExecutionPreparationContext.HardwareProfile detectHardwareProfile() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (osName.contains("mac") && (arch.contains("aarch64") || arch.contains("arm64"))) {
            return TextExecutionPreparationContext.HardwareProfile.APPLE_SILICON;
        }
        if (arch.contains("x86_64") || arch.contains("amd64")) {
            return TextExecutionPreparationContext.HardwareProfile.X86_64;
        }
        return TextExecutionPreparationContext.HardwareProfile.GENERIC_CPU;
    }

    private TextExecutionPreparationContext.PlanningProfile classifyPlanningProfile(
            String backendId,
            TextExecutionPreparationContext.HardwareProfile hardwareProfile,
            boolean supportsPrefixCaching,
            boolean supportsStatefulPreparedModels) {
        if ("direct".equalsIgnoreCase(backendId)
                && hardwareProfile == TextExecutionPreparationContext.HardwareProfile.APPLE_SILICON) {
            return TextExecutionPreparationContext.PlanningProfile.APPLE_SILICON_DIRECT;
        }
        if (supportsPrefixCaching) {
            return TextExecutionPreparationContext.PlanningProfile.PREFIX_CACHE_READY;
        }
        if (supportsStatefulPreparedModels) {
            return TextExecutionPreparationContext.PlanningProfile.STATEFUL_PREPARED_READY;
        }
        return TextExecutionPreparationContext.PlanningProfile.GENERIC_REFERENCE;
    }

    private boolean prefersStatefulPreparedReuse(TextExecutionPreparationContext context) {
        if (context == null || !context.supportsStatefulPreparedModels()) {
            return false;
        }
        return switch (context.planningProfile()) {
            case APPLE_SILICON_DIRECT, STATEFUL_PREPARED_READY -> true;
            default -> false;
        };
    }

    private String conversationSessionId(ConversationExecutionState conversationExecutionState) {
        if (conversationExecutionState == null || !conversationExecutionState.requested()) {
            return null;
        }
        return conversationExecutionState.sessionId();
    }

    private String conversationStateRationale(TextExecutionPreparationContext context) {
        ConversationExecutionState state = context != null ? context.conversationExecutionState() : null;
        if (state == null) {
            return "Planner selected conversation-stateful session for explicit sessionId";
        }
        return "Planner selected conversation-stateful session for explicit sessionId on "
                + context.planningProfile().name().toLowerCase()
                + " (" + state.status().name().toLowerCase() + ": " + state.rationale() + ")";
    }

    private String buildRationale(
            TextExecutionPreparationContext context,
            TextExecutionSessionPlan sessionPlan,
            TextExecutionArtifactPlan artifactPlan) {
        return "Planned by engine policy for backend="
                + context.backendId()
                + ", profile="
                + context.planningProfile().name().toLowerCase()
                + ", hardware="
                + context.hardwareProfile().name().toLowerCase()
                + ", model_family="
                + context.modelFamily()
                + ", session="
                + sessionPlan.prefillStrategy().name().toLowerCase()
                + ", artifact="
                + artifactPlan.strategy().name().toLowerCase();
    }
}
