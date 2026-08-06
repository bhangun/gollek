package tech.kayys.gollek.safetensor.engine.session;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.util.Objects;

/**
 * Full token snapshot for one conversation-scoped execution result.
 *
 * <p>This gives the new execution architecture a typed way to publish
 * conversation state back into the legacy conversation session store.
 */
public record ConversationExecutionSnapshot(
        String sessionId,
        String modelKey,
        long[] tokenIds,
        KVCacheManager.KVCacheSession kvCacheSession,
        boolean kvStateAvailable,
        Integer pendingReplayTokenId,
        String rationale) {

    public ConversationExecutionSnapshot {
        Objects.requireNonNull(sessionId, "sessionId");
        Objects.requireNonNull(modelKey, "modelKey");
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(rationale, "rationale");
        tokenIds = tokenIds.clone();
    }

    public int tokenCount() {
        return tokenIds.length;
    }

    public boolean hasRetainedKvCache() {
        return kvCacheSession != null;
    }

    public boolean hasPendingReplayToken() {
        return pendingReplayTokenId != null;
    }
}
