package tech.kayys.gollek.safetensor.engine.session;

import java.util.Objects;

/**
 * Typed inspection result for conversation-scoped execution state.
 *
 * <p>This lets the new planner reason about existing conversation/KV state
 * without depending on the old session manager's internal value types.
 */
public record ConversationExecutionState(
        Status status,
        String sessionId,
        String requestedModelKey,
        String existingModelKey,
        int cachedTokens,
        long[] cachedTokenIds,
        Integer pendingReplayTokenId,
        boolean kvStateAvailable,
        String rationale) {

    public enum Status {
        NONE_REQUESTED,
        MISS,
        HIT,
        MODEL_MISMATCH,
        EXPIRED
    }

    public ConversationExecutionState {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(rationale, "rationale");
        cachedTokenIds = cachedTokenIds != null ? cachedTokenIds.clone() : new long[0];
    }

    public static ConversationExecutionState noneRequested() {
        return new ConversationExecutionState(
                Status.NONE_REQUESTED,
                null,
                null,
                null,
                0,
                new long[0],
                null,
                false,
                "No conversation session was requested");
    }

    public static ConversationExecutionState miss(String sessionId, String requestedModelKey, String rationale) {
        return new ConversationExecutionState(
                Status.MISS,
                sessionId,
                requestedModelKey,
                null,
                0,
                new long[0],
                null,
                false,
                rationale);
    }

    public static ConversationExecutionState hit(
            String sessionId,
            String requestedModelKey,
            int cachedTokens,
            long[] cachedTokenIds,
            Integer pendingReplayTokenId,
            boolean kvStateAvailable,
            String rationale) {
        return new ConversationExecutionState(
                Status.HIT,
                sessionId,
                requestedModelKey,
                requestedModelKey,
                cachedTokens,
                cachedTokenIds,
                pendingReplayTokenId,
                kvStateAvailable,
                rationale);
    }

    public static ConversationExecutionState modelMismatch(
            String sessionId,
            String requestedModelKey,
            String existingModelKey,
            int cachedTokens,
            long[] cachedTokenIds,
            String rationale) {
        return new ConversationExecutionState(
                Status.MODEL_MISMATCH,
                sessionId,
                requestedModelKey,
                existingModelKey,
                cachedTokens,
                cachedTokenIds,
                null,
                false,
                rationale);
    }

    public static ConversationExecutionState expired(
            String sessionId,
            String requestedModelKey,
            int cachedTokens,
            long[] cachedTokenIds,
            String rationale) {
        return new ConversationExecutionState(
                Status.EXPIRED,
                sessionId,
                requestedModelKey,
                requestedModelKey,
                cachedTokens,
                cachedTokenIds,
                null,
                false,
                rationale);
    }

    public boolean requested() {
        return status != Status.NONE_REQUESTED;
    }

    public boolean active() {
        return status == Status.HIT;
    }

    public boolean hasCachedTokenIds() {
        return cachedTokenIds != null && cachedTokenIds.length > 0;
    }
}
