package tech.kayys.gollek.safetensor.engine.backend;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionSnapshot;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionState;
import tech.kayys.gollek.safetensor.engine.session.ConversationTurnPlan;
import tech.kayys.gollek.safetensor.engine.session.ConversationSessionManager;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.Objects;
import java.util.Optional;

/**
 * Direct-engine execution session.
 *
 * <p>Today this is a thin wrapper over {@link DirectInferenceEngine}, but it
 * creates a stable session seam for future direct-backend prefill reuse or
 * cached decode state.
 */
public final class DirectTextExecutionSession implements TextExecutionSession {

    private final DirectInferenceEngine engine;
    private final PreparedTextGeneration generation;
    private final ResumableSessionArtifact artifact;
    private final long[] inputIds;
    private final ConversationSessionManager.ConversationSession conversationSession;
    private volatile ConversationExecutionSnapshot conversationSnapshot;
    private volatile ConversationTurnPlan conversationTurnPlan;
    private volatile Boolean conversationDeltaPrefillUsed;
    private volatile String conversationDeltaPrefillRationale;
    private volatile String conversationFastPathMode;
    private volatile String conversationFastPathRationale;
    private volatile boolean conversationSnapshotAdopted;

    public DirectTextExecutionSession(
            DirectInferenceEngine engine,
            PreparedTextGeneration generation,
            ResumableSessionArtifact artifact,
            long[] inputIds,
            ConversationSessionManager.ConversationSession conversationSession) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.generation = Objects.requireNonNull(generation, "generation");
        this.artifact = Objects.requireNonNull(artifact, "artifact");
        this.inputIds = inputIds != null ? inputIds.clone() : new long[0];
        this.conversationSession = conversationSession;
    }

    @Override
    public String backendId() {
        return generation.backendId();
    }

    @Override
    public PreparedTextGeneration generation() {
        return generation;
    }

    @Override
    public ResumableSessionArtifact artifact() {
        return artifact;
    }

    @Override
    public Optional<ConversationExecutionSnapshot> conversationSnapshot() {
        return Optional.ofNullable(conversationSnapshot);
    }

    @Override
    public Optional<ConversationTurnPlan> conversationTurnPlan() {
        return Optional.ofNullable(conversationTurnPlan);
    }

    @Override
    public void adoptConversationSnapshot() {
        this.conversationSnapshotAdopted = true;
    }

    @Override
    public Optional<Boolean> conversationDeltaPrefillUsed() {
        return Optional.ofNullable(conversationDeltaPrefillUsed);
    }

    @Override
    public Optional<String> conversationDeltaPrefillRationale() {
        return Optional.ofNullable(conversationDeltaPrefillRationale);
    }

    @Override
    public Optional<String> conversationFastPathMode() {
        return Optional.ofNullable(conversationFastPathMode);
    }

    @Override
    public Optional<String> conversationFastPathRationale() {
        return Optional.ofNullable(conversationFastPathRationale);
    }

    @Override
    public Uni<InferenceResponse> generate() {
        long[] effectiveInputIds = inputIds.length > 0
                ? inputIds
                : engine.encodePrompt(generation.prompt().formattedPrompt(), generation.model().modelPath());
        conversationTurnPlan = buildTurnPlan(effectiveInputIds);
        evaluateConversationFastPathDecision();
        if (canUseExactReplay()) {
            Integer replayTokenId = resolvedReplayTokenId();
            return engine.generateContinuationWithConversationTrace(
                    effectiveInputIds,
                    conversationSession.totalTokens(),
                    conversationSession.kvCache(),
                    generation.model().modelPath(),
                    generation.generationConfig(),
                    replayTokenId)
                    .map(trace -> {
                        conversationSnapshot = buildConversationSnapshot(
                                trace.inputIds(),
                                trace.generatedTokenIds(),
                                trace.kvCacheSession());
                        return trace.response();
                    });
        }
        if (canUseDeltaPrefill()) {
            return engine.generateContinuationWithConversationTrace(
                    effectiveInputIds,
                    conversationSession.totalTokens(),
                    conversationSession.kvCache(),
                    generation.model().modelPath(),
                    generation.generationConfig())
                    .map(trace -> {
                        conversationSnapshot = buildConversationSnapshot(
                                trace.inputIds(),
                                trace.generatedTokenIds(),
                                trace.kvCacheSession());
                        return trace.response();
                    });
        }
        if (generation.sessionPlan().conversationStateful()) {
            return engine.generateWithConversationTrace(
                    effectiveInputIds,
                    generation.model().modelPath(),
                    generation.generationConfig())
                    .map(trace -> {
                        conversationSnapshot = buildConversationSnapshot(
                                trace.inputIds(),
                                trace.generatedTokenIds(),
                                trace.kvCacheSession());
                        return trace.response();
                    });
        }
        return engine.generateWithTrace(
                effectiveInputIds,
                generation.model().modelPath(),
                generation.generationConfig())
                .map(trace -> {
                    conversationSnapshot = buildConversationSnapshot(trace.inputIds(), trace.generatedTokenIds(), null);
                    return trace.response();
                });
    }

    @Override
    public Multi<InferenceResponse> generateStream() {
        long[] effectiveInputIds = inputIds.length > 0
                ? inputIds
                : engine.encodePrompt(generation.prompt().formattedPrompt(), generation.model().modelPath());
        conversationTurnPlan = buildTurnPlan(effectiveInputIds);
        evaluateConversationFastPathDecision();
        if (canUseExactReplay()) {
            Integer replayTokenId = resolvedReplayTokenId();
            return engine.generateContinuationStreamWithConversationTrace(
                    effectiveInputIds,
                    conversationSession.totalTokens(),
                    conversationSession.kvCache(),
                    generation.model().modelPath(),
                    generation.generationConfig(),
                    trace -> conversationSnapshot = buildConversationSnapshot(
                            trace.inputIds(),
                            trace.generatedTokenIds(),
                            trace.kvCacheSession()),
                    replayTokenId);
        }
        if (canUseDeltaPrefill()) {
            return engine.generateContinuationStreamWithConversationTrace(
                    effectiveInputIds,
                    conversationSession.totalTokens(),
                    conversationSession.kvCache(),
                    generation.model().modelPath(),
                    generation.generationConfig(),
                    trace -> conversationSnapshot = buildConversationSnapshot(
                            trace.inputIds(),
                            trace.generatedTokenIds(),
                            trace.kvCacheSession()));
        }
        if (generation.sessionPlan().conversationStateful()) {
            return engine.generateStreamWithConversationTrace(
                    effectiveInputIds,
                    generation.model().modelPath(),
                    generation.generationConfig(),
                    trace -> conversationSnapshot = buildConversationSnapshot(
                            trace.inputIds(),
                            trace.generatedTokenIds(),
                            trace.kvCacheSession()));
        }
        return engine.generateStream(
                effectiveInputIds,
                generation.model().modelPath(),
                generation.generationConfig());
    }

    @Override
    public void close() {
        ConversationExecutionSnapshot snapshot = conversationSnapshot;
        if (snapshot != null && snapshot.kvCacheSession() != null && !conversationSnapshotAdopted) {
            try {
                snapshot.kvCacheSession().close();
            } catch (Exception ignored) {
            }
        }
    }

    private ConversationExecutionSnapshot buildConversationSnapshot(
            long[] promptTokenIds,
            long[] generatedTokenIds,
            tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager.KVCacheSession kvCacheSession) {
        if (!generation.sessionPlan().conversationStateful() || generation.sessionPlan().sessionKey() == null) {
            return null;
        }
        String modelKey = generation.model().loadedModel() != null
                ? generation.model().loadedModel().key()
                : generation.model().modelPath().getFileName().toString();
        long[] combined = kvResidentTokenIds(promptTokenIds, generatedTokenIds, kvCacheSession != null);
        return new ConversationExecutionSnapshot(
                generation.sessionPlan().sessionKey(),
                modelKey,
                combined,
                kvCacheSession,
                kvCacheSession != null,
                deterministicReplayTokenId(generatedTokenIds, kvCacheSession != null),
                kvCacheSession != null
                        ? "Direct backend recorded KV-resident conversation prefix and retained KV state"
                        : "Direct backend recorded token snapshot for conversation-scoped execution");
    }

    private ConversationTurnPlan buildTurnPlan(long[] promptTokenIds) {
        if (!generation.sessionPlan().conversationStateful()) {
            return ConversationTurnPlan.noneRequested();
        }
        ConversationExecutionState state = generation.preparationPlan().context().conversationExecutionState();
        if (state == null || !state.requested() || !state.hasCachedTokenIds()) {
            return ConversationTurnPlan.noPriorState(promptTokenIds.length);
        }
        long[] cachedTokenIds = state.cachedTokenIds();
        int sharedPrefix = sharedPrefixLength(cachedTokenIds, promptTokenIds);
        if (sharedPrefix == cachedTokenIds.length && sharedPrefix == promptTokenIds.length) {
            return ConversationTurnPlan.exactReplay(cachedTokenIds.length);
        }
        if (sharedPrefix == cachedTokenIds.length && promptTokenIds.length > cachedTokenIds.length) {
            return ConversationTurnPlan.prefixExtension(cachedTokenIds.length, promptTokenIds.length);
        }
        return ConversationTurnPlan.diverged(cachedTokenIds.length, promptTokenIds.length, sharedPrefix);
    }

    private boolean canUseDeltaPrefill() {
        if (conversationTurnPlan == null || conversationTurnPlan.mode() != ConversationTurnPlan.Mode.PREFIX_EXTENSION) {
            return false;
        }
        if (conversationSession == null || conversationSession.kvCache() == null || conversationSession.isClosed()) {
            return false;
        }
        int cachedTokens = conversationSession.totalTokens();
        return conversationTurnPlan.cachedTokens() == cachedTokens
                && conversationSession.kvCache().currentPos() == cachedTokens;
    }

    private boolean canUseExactReplay() {
        if (conversationTurnPlan == null || conversationTurnPlan.mode() != ConversationTurnPlan.Mode.EXACT_REPLAY) {
            return false;
        }
        if (conversationSession == null || conversationSession.kvCache() == null || conversationSession.isClosed()) {
            return false;
        }
        int cachedTokens = conversationSession.totalTokens();
        return conversationTurnPlan.cachedTokens() == cachedTokens
                && conversationSession.kvCache().currentPos() == cachedTokens
                && supportsDeterministicReplay()
                && resolvedReplayTokenId() != null;
    }

    private void evaluateConversationFastPathDecision() {
        if (conversationTurnPlan == null) {
            conversationFastPathMode = "full_prefill";
            conversationFastPathRationale = "No conversation turn plan was computed";
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "No conversation turn plan was computed";
            return;
        }
        if (conversationTurnPlan.mode() == ConversationTurnPlan.Mode.EXACT_REPLAY) {
            if (conversationSession == null) {
                conversationFastPathMode = "full_prefill";
                conversationFastPathRationale = "Exact replay was requested, but no active conversation execution session is attached";
                conversationDeltaPrefillUsed = false;
                conversationDeltaPrefillRationale = "Exact replay fast path is unavailable without an active conversation session";
                return;
            }
            if (conversationSession.kvCache() == null) {
                conversationFastPathMode = "full_prefill";
                conversationFastPathRationale = "Exact replay was requested, but the conversation session does not have retained KV state";
                conversationDeltaPrefillUsed = false;
                conversationDeltaPrefillRationale = "Exact replay fast path is unavailable without retained KV state";
                return;
            }
            if (conversationSession.isClosed()) {
                conversationFastPathMode = "full_prefill";
                conversationFastPathRationale = "Exact replay was requested, but the conversation session is already closed";
                conversationDeltaPrefillUsed = false;
                conversationDeltaPrefillRationale = "Exact replay fast path is unavailable because the conversation session is closed";
                return;
            }
            int cachedTokens = conversationSession.totalTokens();
            if (conversationTurnPlan.cachedTokens() != cachedTokens) {
                conversationFastPathMode = "full_prefill";
                conversationFastPathRationale = "Exact replay token snapshot mismatch: planned cached tokens="
                        + conversationTurnPlan.cachedTokens()
                        + ", live session tokens="
                        + cachedTokens;
                conversationDeltaPrefillUsed = false;
                conversationDeltaPrefillRationale = "Exact replay fast path is unavailable because cached token counts diverged";
                return;
            }
            int currentPos = conversationSession.kvCache().currentPos();
            if (currentPos != cachedTokens) {
                conversationFastPathMode = "full_prefill";
                conversationFastPathRationale = "Exact replay KV cache position mismatch: currentPos="
                        + currentPos
                        + ", cached tokens="
                        + cachedTokens;
                conversationDeltaPrefillUsed = false;
                conversationDeltaPrefillRationale = "Exact replay fast path is unavailable because retained KV position diverged";
                return;
            }
            if (!supportsDeterministicReplay()) {
                conversationFastPathMode = "full_prefill";
                conversationFastPathRationale = "Exact replay matched retained KV state, but the current generation config is not deterministic";
                conversationDeltaPrefillUsed = false;
                conversationDeltaPrefillRationale = "Exact replay fast path is only enabled for deterministic direct-greedy generation";
                return;
            }
            if (resolvedReplayTokenId() == null) {
                conversationFastPathMode = "full_prefill";
                conversationFastPathRationale = "Exact replay matched retained KV state, but no cached deterministic replay token is available in either the live conversation session or the prepared execution state";
                conversationDeltaPrefillUsed = false;
                conversationDeltaPrefillRationale = "Exact replay fast path requires a cached deterministic next token";
                return;
            }
            conversationFastPathMode = "zero_prefill_replay";
            conversationFastPathRationale = "Exact replay matches retained KV state and a deterministic replay token is cached; zero-prefill replay is active";
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "Zero-prefill exact replay is active instead of delta prefill";
            return;
        }
        if (conversationTurnPlan.mode() != ConversationTurnPlan.Mode.PREFIX_EXTENSION) {
            conversationFastPathMode = "full_prefill";
            conversationFastPathRationale = "Conversation turn mode is "
                    + conversationTurnPlan.mode().name().toLowerCase()
                    + ", so no retained-session prefill shortcut is applicable";
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "Conversation turn mode is "
                    + conversationTurnPlan.mode().name().toLowerCase()
                    + ", so delta prefill is not applicable";
            return;
        }
        if (conversationSession == null) {
            conversationFastPathMode = "full_prefill";
            conversationFastPathRationale = "No active conversation execution session is attached";
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "No active conversation execution session is attached";
            return;
        }
        if (conversationSession.kvCache() == null) {
            conversationFastPathMode = "full_prefill";
            conversationFastPathRationale = "Conversation session does not have retained KV state";
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "Conversation session does not have retained KV state";
            return;
        }
        if (conversationSession.isClosed()) {
            conversationFastPathMode = "full_prefill";
            conversationFastPathRationale = "Conversation session is already closed";
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "Conversation session is already closed";
            return;
        }
        int cachedTokens = conversationSession.totalTokens();
        if (conversationTurnPlan.cachedTokens() != cachedTokens) {
            conversationFastPathMode = "full_prefill";
            conversationFastPathRationale = "Conversation token snapshot mismatch: planned cached tokens="
                    + conversationTurnPlan.cachedTokens()
                    + ", live session tokens="
                    + cachedTokens;
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "Conversation token snapshot mismatch: planned cached tokens="
                    + conversationTurnPlan.cachedTokens()
                    + ", live session tokens="
                    + cachedTokens;
            return;
        }
        int currentPos = conversationSession.kvCache().currentPos();
        if (currentPos != cachedTokens) {
            conversationFastPathMode = "full_prefill";
            conversationFastPathRationale = "KV cache position mismatch: currentPos="
                    + currentPos
                    + ", cached tokens="
                    + cachedTokens;
            conversationDeltaPrefillUsed = false;
            conversationDeltaPrefillRationale = "KV cache position mismatch: currentPos="
                    + currentPos
                    + ", cached tokens="
                    + cachedTokens;
            return;
        }
        conversationFastPathMode = "delta_prefill";
        conversationFastPathRationale = "Conversation prefix extension matches retained KV state; delta prefill is active";
        conversationDeltaPrefillUsed = true;
        conversationDeltaPrefillRationale = "Conversation prefix extension matches retained KV state; delta prefill is active";
    }

    private boolean supportsDeterministicReplay() {
        return generation.generationConfig().isPenaltyFreeGreedy();
    }

    private Integer resolvedReplayTokenId() {
        if (conversationSession != null && conversationSession.pendingReplayTokenId() != null) {
            return conversationSession.pendingReplayTokenId();
        }
        ConversationExecutionState state = generation.preparationPlan() != null
                && generation.preparationPlan().context() != null
                        ? generation.preparationPlan().context().conversationExecutionState()
                        : null;
        return state != null ? state.pendingReplayTokenId() : null;
    }

    private Integer deterministicReplayTokenId(long[] generatedTokenIds, boolean kvRetained) {
        if (!kvRetained || !supportsDeterministicReplay() || generatedTokenIds.length == 0) {
            return null;
        }
        long tokenId = generatedTokenIds[generatedTokenIds.length - 1];
        if (tokenId < Integer.MIN_VALUE || tokenId > Integer.MAX_VALUE) {
            return null;
        }
        return (int) tokenId;
    }

    private int sharedPrefixLength(long[] cachedTokenIds, long[] promptTokenIds) {
        int limit = Math.min(cachedTokenIds.length, promptTokenIds.length);
        int i = 0;
        while (i < limit && cachedTokenIds[i] == promptTokenIds[i]) {
            i++;
        }
        return i;
    }

    private long[] kvResidentTokenIds(
            long[] promptTokenIds,
            long[] generatedTokenIds,
            boolean kvRetained) {
        if (!kvRetained || generatedTokenIds.length == 0) {
            long[] combined = new long[promptTokenIds.length + generatedTokenIds.length];
            System.arraycopy(promptTokenIds, 0, combined, 0, promptTokenIds.length);
            System.arraycopy(generatedTokenIds, 0, combined, promptTokenIds.length, generatedTokenIds.length);
            return combined;
        }
        int generatedResident = Math.max(0, generatedTokenIds.length - 1);
        long[] combined = new long[promptTokenIds.length + generatedResident];
        System.arraycopy(promptTokenIds, 0, combined, 0, promptTokenIds.length);
        if (generatedResident > 0) {
            System.arraycopy(generatedTokenIds, 0, combined, promptTokenIds.length, generatedResident);
        }
        return combined;
    }
}
