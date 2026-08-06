package tech.kayys.gollek.safetensor.engine.session;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import io.quarkus.arc.Arc;

/**
 * Applies engine-owned conversation execution snapshots to the legacy
 * conversation session manager.
 *
 * <p>This is a transition seam: today it records token snapshots first, and a
 * future KV-aware backend can publish real KV-backed artifacts through the same
 * coordinator path.
 */
@ApplicationScoped
public class ConversationExecutionCoordinator {

    @Inject
    ConversationSessionManager conversationSessions;

    public ConversationExecutionCoordinator() {
    }

    public ConversationExecutionCoordinator(ConversationSessionManager conversationSessions) {
        this.conversationSessions = conversationSessions;
    }

    public void recordSnapshot(ConversationExecutionSnapshot snapshot) {
        if (snapshot == null || snapshot.tokenIds().length == 0) {
            return;
        }
        sessions().recordSnapshot(
                snapshot.sessionId(),
                snapshot.modelKey(),
                toIntArray(snapshot.tokenIds()),
                snapshot.kvCacheSession(),
                snapshot.pendingReplayTokenId());
    }

    private int[] toIntArray(long[] tokenIds) {
        int[] out = new int[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            long tokenId = tokenIds[i];
            if (tokenId < Integer.MIN_VALUE || tokenId > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Token id out of int range at index " + i + ": " + tokenId);
            }
            out[i] = (int) tokenId;
        }
        return out;
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
