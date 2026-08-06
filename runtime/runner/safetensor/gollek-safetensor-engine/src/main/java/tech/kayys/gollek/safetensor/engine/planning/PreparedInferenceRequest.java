package tech.kayys.gollek.safetensor.engine.planning;

import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextGeneration;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionArtifactPlan;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextModel;
import tech.kayys.gollek.safetensor.engine.backend.ResumableSessionDescriptor;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionPreparationPlan;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionPlan;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionReuseDecision;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionState;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Engine-owned prepared inference request.
 */
public record PreparedInferenceRequest(
        Path modelPath,
        boolean audioModel,
        String ttsPrompt,
        AudioConfig audioConfig,
        SafetensorEngine.LoadedModel loadedModel,
        PreparedTextModel preparedModel,
        PreparedPrompt preparedPrompt,
        ConversationExecutionState conversationExecutionState,
        TextExecutionSessionReuseDecision sessionReuseDecision,
        PreparedTextGeneration preparedGeneration) {

    public PreparedInferenceRequest {
        Objects.requireNonNull(modelPath, "modelPath");
        Objects.requireNonNull(ttsPrompt, "ttsPrompt");
        Objects.requireNonNull(audioConfig, "audioConfig");
        Objects.requireNonNull(conversationExecutionState, "conversationExecutionState");
    }

    public int estimatedPromptTokens() {
        String prompt = preparedPrompt != null ? preparedPrompt.formattedPrompt() : ttsPrompt;
        return prompt == null ? 0 : prompt.length() / 4;
    }

    public String executionBackendId() {
        if (preparedGeneration != null && preparedGeneration.backendId() != null) {
            return preparedGeneration.backendId();
        }
        if (preparedModel != null) {
            return preparedModel.backendId();
        }
        return "unknown";
    }

    public PromptReusePlan promptReusePlan() {
        return preparedPrompt != null ? preparedPrompt.reusePlan() : PromptReusePlan.none("No prepared prompt");
    }

    public TextExecutionPreparationPlan preparationPlan() {
        return preparedGeneration != null ? preparedGeneration.preparationPlan() : null;
    }

    public TextExecutionPreparationContext preparationContext() {
        return preparationPlan() != null ? preparationPlan().context() : null;
    }

    public TextExecutionSessionPlan sessionPlan() {
        return preparedGeneration != null ? preparedGeneration.sessionPlan() : null;
    }

    public TextExecutionArtifactPlan artifactPlan() {
        return preparedGeneration != null ? preparedGeneration.artifactPlan() : null;
    }

    public TextExecutionSessionReuseDecision sessionReuseDecision() {
        return sessionReuseDecision != null
                ? sessionReuseDecision
                : TextExecutionSessionReuseDecision.notReusable("No session reuse decision");
    }

    public ResumableSessionDescriptor sessionDescriptor() {
        if (preparedGeneration == null) {
            return null;
        }
        String sessionKey = sessionPlan() != null && sessionPlan().sessionKey() != null
                ? sessionPlan().sessionKey()
                : preparedGeneration.promptFingerprint();
        return new ResumableSessionDescriptor(
                executionBackendId(),
                sessionKey,
                preparedGeneration.promptFingerprint(),
                sessionPlan() != null ? sessionPlan().reusePolicy() : TextExecutionSessionPlan.ReusePolicy.EPHEMERAL,
                java.time.Instant.now(),
                sessionPlan() != null ? sessionPlan().rationale() : "No session plan");
    }
}
