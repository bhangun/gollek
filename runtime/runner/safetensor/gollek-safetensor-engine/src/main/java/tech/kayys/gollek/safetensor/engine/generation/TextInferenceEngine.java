package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import io.quarkus.arc.Arc;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextGeneration;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionAcquisition;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSession;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionManager;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionReservation;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionCoordinator;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorFeature;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.InferenceResponse;

/**
 * CDI bean that exposes text generation as a {@link SafetensorFeature}.
 *
 * <p>Delegates all inference work to the selected {@link TextExecutionBackend},
 * acting as a thin adapter that registers the text modality with the SafeTensor
 * feature plugin system.
 *
 * @see TextExecutionBackend
 * @see SafetensorFeature
 */
@ApplicationScoped
public class TextInferenceEngine implements SafetensorFeature {

    private static final Logger log = Logger.getLogger(TextInferenceEngine.class);

    @Inject
    TextExecutionSessionManager sessionManager;

    @Inject
    ConversationExecutionCoordinator conversationCoordinator;

    /** Returns {@code "text"} — the feature identifier for text generation. */
    @Override
    public String id() {
        return "text";
    }

    /** Logs initialization; no additional setup required. */
    @Override
    public void initialize() {
        log.info("TextInferenceEngine initialized");
    }

    /**
     * Generates a complete text response for the given prompt.
     *
     * @param generation backend-prepared generation request
     * @return a {@link Uni} that resolves to the full {@link InferenceResponse}
     */
    public Uni<InferenceResponse> generate(PreparedTextGeneration generation) {
        TextExecutionSessionAcquisition acquisition = manager().acquire(generation);
        TextExecutionSessionReservation reservation = acquisition.reservation();
        TextExecutionSession session = acquisition.session();
        log.debugf("TextInferenceEngine: session acquisition mode=%s backend=%s rationale=%s",
                acquisition.mode().name().toLowerCase(), session.backendId(), acquisition.rationale());
        manager().publish(session, reservation);
        return session.generate()
                .map(response -> {
                    session.conversationSnapshot().ifPresent(snapshot -> {
                        coordinator().recordSnapshot(snapshot);
                        session.adoptConversationSnapshot();
                    });
                    InferenceResponse.Builder builder = response.toBuilder()
                            .metadata("session_acquisition_mode", acquisition.mode().name().toLowerCase())
                            .metadata("session_acquisition_rationale", acquisition.rationale())
                            .metadata("kv_cache_quantization",
                                    generation.generationConfig().kvCacheQuant().name().toLowerCase());
                    tech.kayys.gollek.safetensor.engine.backend.ResumableSessionArtifact artifact = session.artifact();
                    if (artifact != null) {
                        builder.metadata("session_artifact_kind", artifact.kind().name().toLowerCase());
                        builder.metadata("session_artifact_resume_capable", artifact.resumeCapable());
                    }
                    session.conversationTurnPlan().ifPresent(plan -> {
                        builder.metadata("conversation_turn_mode", plan.mode().name().toLowerCase());
                        builder.metadata("conversation_turn_cached_tokens", plan.cachedTokens());
                        builder.metadata("conversation_turn_prompt_tokens", plan.promptTokens());
                        builder.metadata("conversation_turn_shared_prefix_tokens", plan.sharedPrefixTokens());
                        builder.metadata("conversation_turn_delta_tokens", plan.deltaPromptTokens());
                        builder.metadata("conversation_turn_rationale", plan.rationale());
                    });
                    session.conversationDeltaPrefillUsed()
                            .ifPresent(used -> builder.metadata("conversation_delta_prefill_used", used));
                    session.conversationDeltaPrefillRationale()
                            .ifPresent(rationale -> builder.metadata("conversation_delta_prefill_rationale", rationale));
                    session.conversationFastPathMode()
                            .ifPresent(mode -> builder.metadata("conversation_fast_path_mode", mode));
                    session.conversationFastPathRationale()
                            .ifPresent(rationale -> builder.metadata("conversation_fast_path_rationale", rationale));
                    builder.metadata("conversation_session_snapshot_recorded", session.conversationSnapshot().isPresent());
                    builder.metadata("conversation_session_kv_retained", session.conversationSnapshot()
                            .map(tech.kayys.gollek.safetensor.engine.session.ConversationExecutionSnapshot::hasRetainedKvCache)
                            .orElse(false));
                    return builder.build();
                })
                .onTermination().invoke(() -> {
                    try {
                        session.close();
                    } finally {
                        manager().release(session, reservation);
                    }
                });
    }

    /**
     * Streams generated tokens incrementally for the given prompt.
     *
     * @param generation backend-prepared generation request
     * @return a {@link Multi} emitting one {@link InferenceResponse} per generated token
     */
    public Multi<InferenceResponse> generateStream(PreparedTextGeneration generation) {
        TextExecutionSessionAcquisition acquisition = manager().acquire(generation);
        TextExecutionSessionReservation reservation = acquisition.reservation();
        TextExecutionSession session = acquisition.session();
        log.debugf("TextInferenceEngine: streaming session acquisition mode=%s backend=%s rationale=%s",
                acquisition.mode().name().toLowerCase(), session.backendId(), acquisition.rationale());
        manager().publish(session, reservation);
        return session.generateStream()
                .map(response -> {
                    InferenceResponse.Builder builder = response.toBuilder()
                            .metadata("session_acquisition_mode", acquisition.mode().name().toLowerCase())
                            .metadata("session_acquisition_rationale", acquisition.rationale())
                            .metadata("kv_cache_quantization",
                                    generation.generationConfig().kvCacheQuant().name().toLowerCase());
                    tech.kayys.gollek.safetensor.engine.backend.ResumableSessionArtifact artifact = session.artifact();
                    if (artifact != null) {
                        builder.metadata("session_artifact_kind", artifact.kind().name().toLowerCase());
                        builder.metadata("session_artifact_resume_capable", artifact.resumeCapable());
                    }
                    session.conversationTurnPlan().ifPresent(plan -> {
                        builder.metadata("conversation_turn_mode", plan.mode().name().toLowerCase());
                        builder.metadata("conversation_turn_cached_tokens", plan.cachedTokens());
                        builder.metadata("conversation_turn_prompt_tokens", plan.promptTokens());
                        builder.metadata("conversation_turn_shared_prefix_tokens", plan.sharedPrefixTokens());
                        builder.metadata("conversation_turn_delta_tokens", plan.deltaPromptTokens());
                        builder.metadata("conversation_turn_rationale", plan.rationale());
                    });
                    session.conversationDeltaPrefillUsed()
                            .ifPresent(used -> builder.metadata("conversation_delta_prefill_used", used));
                    session.conversationDeltaPrefillRationale()
                            .ifPresent(rationale -> builder.metadata("conversation_delta_prefill_rationale", rationale));
                    session.conversationFastPathMode()
                            .ifPresent(mode -> builder.metadata("conversation_fast_path_mode", mode));
                    session.conversationFastPathRationale()
                            .ifPresent(rationale -> builder.metadata("conversation_fast_path_rationale", rationale));
                    builder.metadata("conversation_session_snapshot_recorded", session.conversationSnapshot().isPresent());
                    builder.metadata("conversation_session_kv_retained", session.conversationSnapshot()
                            .map(tech.kayys.gollek.safetensor.engine.session.ConversationExecutionSnapshot::hasRetainedKvCache)
                            .orElse(false));
                    return builder.build();
                })
                .onCompletion().invoke(() -> session.conversationSnapshot().ifPresent(snapshot -> {
                    coordinator().recordSnapshot(snapshot);
                    session.adoptConversationSnapshot();
                }))
                .onTermination().invoke(() -> {
                    try {
                        session.close();
                    } finally {
                        manager().release(session, reservation);
                    }
                });
    }

    private TextExecutionSessionManager manager() {
        if (sessionManager != null) {
            return sessionManager;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(TextExecutionSessionManager.class);
                if (instance.isAvailable()) {
                    sessionManager = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        return sessionManager != null ? sessionManager : new TextExecutionSessionManager();
    }

    private ConversationExecutionCoordinator coordinator() {
        if (conversationCoordinator != null) {
            return conversationCoordinator;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(ConversationExecutionCoordinator.class);
                if (instance.isAvailable()) {
                    conversationCoordinator = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        return conversationCoordinator != null ? conversationCoordinator : new ConversationExecutionCoordinator();
    }
}
