package tech.kayys.gollek.safetensor.engine.session;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import io.quarkus.arc.Arc;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextModel;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;

/**
 * Bridges legacy conversation/KV session state into the new execution planner.
 *
 * <p>The planner should reason about typed conversation state, not raw access to
 * the old session manager implementation.
 */
@ApplicationScoped
public class ConversationExecutionStateResolver {

    @Inject
    ConversationSessionManager conversationSessions;

    public ConversationExecutionStateResolver() {
    }

    public ConversationExecutionStateResolver(ConversationSessionManager conversationSessions) {
        this.conversationSessions = conversationSessions;
    }

    public ConversationExecutionState resolve(PreparedTextModel model, String requestedSessionId) {
        String normalizedSessionId = normalizeSessionId(requestedSessionId);
        if (normalizedSessionId == null) {
            return ConversationExecutionState.noneRequested();
        }
        String modelKey = resolveModelKey(model);
        return sessions().probe(normalizedSessionId, modelKey);
    }

    public java.util.Optional<ConversationSessionManager.ConversationSession> findActiveSession(
            PreparedTextModel model,
            String requestedSessionId) {
        String normalizedSessionId = normalizeSessionId(requestedSessionId);
        if (normalizedSessionId == null) {
            return java.util.Optional.empty();
        }
        return sessions().find(normalizedSessionId, resolveModelKey(model));
    }

    private String resolveModelKey(PreparedTextModel model) {
        if (model == null || model.loadedModel() == null) {
            return "unknown";
        }
        SafetensorEngine.LoadedModel loadedModel = model.loadedModel();
        return loadedModel.key() != null ? loadedModel.key() : "unknown";
    }

    private String normalizeSessionId(String sessionId) {
        if (sessionId == null) {
            return null;
        }
        String normalized = sessionId.trim();
        return normalized.isEmpty() ? null : normalized;
    }

    private ConversationSessionManager sessions() {
        if (conversationSessions != null) {
            return conversationSessions;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(ConversationSessionManager.class);
                if (instance.isAvailable()) {
                    conversationSessions = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (conversationSessions == null) {
            throw new IllegalStateException("ConversationSessionManager is not available");
        }
        return conversationSessions;
    }
}
