package tech.kayys.gollek.safetensor.engine.session;

import java.util.Objects;

/**
 * Token-level relationship between cached conversation state and the current
 * prompt for one request.
 *
 * <p>This is the seam future backends can use to decide whether a request is an
 * exact replay, a pure prefix extension, or a divergence that requires a full
 * rebuild. That split is directly relevant to prefix reuse and resumable edge
 * inference on Apple Silicon; see Barrios (2026),
 * <a href="https://arxiv.org/abs/2601.19139">arXiv:2601.19139</a>.
 */
public record ConversationTurnPlan(
        Mode mode,
        int cachedTokens,
        int promptTokens,
        int sharedPrefixTokens,
        int deltaPromptTokens,
        String rationale) {

    public enum Mode {
        NONE_REQUESTED,
        NO_PRIOR_STATE,
        EXACT_REPLAY,
        PREFIX_EXTENSION,
        DIVERGED
    }

    public ConversationTurnPlan {
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(rationale, "rationale");
    }

    public static ConversationTurnPlan noneRequested() {
        return new ConversationTurnPlan(
                Mode.NONE_REQUESTED,
                0,
                0,
                0,
                0,
                "No conversation-scoped request was active");
    }

    public static ConversationTurnPlan noPriorState(int promptTokens) {
        return new ConversationTurnPlan(
                Mode.NO_PRIOR_STATE,
                0,
                promptTokens,
                0,
                promptTokens,
                "Conversation-scoped request has no prior cached token state");
    }

    public static ConversationTurnPlan exactReplay(int cachedTokens) {
        return new ConversationTurnPlan(
                Mode.EXACT_REPLAY,
                cachedTokens,
                cachedTokens,
                cachedTokens,
                0,
                "Current prompt exactly matches cached conversation tokens");
    }

    public static ConversationTurnPlan prefixExtension(int cachedTokens, int promptTokens) {
        return new ConversationTurnPlan(
                Mode.PREFIX_EXTENSION,
                cachedTokens,
                promptTokens,
                cachedTokens,
                Math.max(0, promptTokens - cachedTokens),
                "Current prompt extends the cached conversation prefix");
    }

    public static ConversationTurnPlan diverged(
            int cachedTokens,
            int promptTokens,
            int sharedPrefixTokens) {
        return new ConversationTurnPlan(
                Mode.DIVERGED,
                cachedTokens,
                promptTokens,
                sharedPrefixTokens,
                Math.max(0, promptTokens - sharedPrefixTokens),
                "Current prompt diverges from cached conversation tokens");
    }
}
