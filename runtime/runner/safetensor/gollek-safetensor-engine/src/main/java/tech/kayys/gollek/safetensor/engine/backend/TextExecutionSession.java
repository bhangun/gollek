package tech.kayys.gollek.safetensor.engine.backend;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionSnapshot;
import tech.kayys.gollek.safetensor.engine.session.ConversationTurnPlan;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.Optional;

/**
 * A backend-owned execution session for one prepared generation.
 *
 * <p>Even though the current direct backend still executes synchronously,
 * making the session explicit gives Gollek a real architectural slot for
 * backend-resident prefill state, persistent KV-backed sessions, or token-step
 * schedulers later on. See Shkolnikov (2026), arXiv:2603.04428.
 */
public interface TextExecutionSession extends AutoCloseable {

    String backendId();

    PreparedTextGeneration generation();

    default TextExecutionSessionPlan plan() {
        return generation().sessionPlan();
    }

    default ResumableSessionDescriptor descriptor() {
        String sessionKey = plan().sessionKey() != null ? plan().sessionKey() : generation().promptFingerprint();
        return new ResumableSessionDescriptor(
                backendId(),
                sessionKey,
                generation().promptFingerprint(),
                plan().reusePolicy(),
                java.time.Instant.now(),
                plan().rationale());
    }

    default ResumableSessionArtifact artifact() {
        return ResumableSessionArtifact.descriptorOnly(descriptor());
    }

    default Optional<ConversationExecutionSnapshot> conversationSnapshot() {
        return Optional.empty();
    }

    default Optional<ConversationTurnPlan> conversationTurnPlan() {
        return Optional.empty();
    }

    default Optional<Boolean> conversationDeltaPrefillUsed() {
        return Optional.empty();
    }

    default Optional<String> conversationDeltaPrefillRationale() {
        return Optional.empty();
    }

    default Optional<String> conversationFastPathMode() {
        return Optional.empty();
    }

    default Optional<String> conversationFastPathRationale() {
        return Optional.empty();
    }

    default void adoptConversationSnapshot() {
    }

    Uni<InferenceResponse> generate();

    Multi<InferenceResponse> generateStream();

    @Override
    default void close() {
    }
}
