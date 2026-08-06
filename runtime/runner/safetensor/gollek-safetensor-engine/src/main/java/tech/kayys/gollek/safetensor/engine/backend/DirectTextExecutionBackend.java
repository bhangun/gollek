package tech.kayys.gollek.safetensor.engine.backend;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.planning.PreparedPrompt;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionStateResolver;
import tech.kayys.gollek.safetensor.engine.session.ConversationSessionManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;

/**
 * Adapter that exposes the existing DirectInferenceEngine through the new
 * backend execution contract.
 */
public class DirectTextExecutionBackend implements TextExecutionBackend {

    private final DirectInferenceEngine engine;
    @Inject
    DirectPreparedGenerationRegistry handleRegistry;
    @Inject
    ConversationExecutionStateResolver conversationStateResolver;

    public DirectTextExecutionBackend(DirectInferenceEngine engine) {
        this.engine = engine;
    }

    public DirectTextExecutionBackend(
            DirectInferenceEngine engine,
            DirectPreparedGenerationRegistry handleRegistry) {
        this.engine = engine;
        this.handleRegistry = handleRegistry;
    }

    @Override
    public String id() {
        return "direct";
    }

    @Override
    public TextExecutionBackendCapabilities capabilities() {
        return TextExecutionBackendCapabilities.directReference();
    }

    @Override
    public PreparedTextModel prepareModel(Path modelPath, Path adapterPath, QuantizationEngine.QuantStrategy quantStrategy) {
        engine.loadModel(modelPath, adapterPath, quantStrategy);
        return new PreparedTextModel(id(), modelPath, engine.getLoadedModel(modelPath), capabilities());
    }

    @Override
    public PreparedTextGeneration prepareGeneration(
            PreparedPrompt prompt,
            PreparedTextModel model,
            TextExecutionPreparationPlan preparationPlan,
            GenerationConfig cfg) {
        return TextExecutionBackend.super.prepareGeneration(prompt, model, preparationPlan, cfg);
    }

    @Override
    public TextExecutionSession openSession(PreparedTextGeneration generation) {
        ConversationSessionManager.ConversationSession conversationSession =
                resolveConversationSession(generation).orElse(null);
        if (generation.artifactPlan().strategy() == TextExecutionArtifactPlan.Strategy.KV_SNAPSHOT
                && generation.sessionPlan().conversationStateful()) {
            return new DirectTextExecutionSession(
                    engine,
                    generation,
                    conversationArtifactFor(generation),
                    new long[0],
                    conversationSession);
        }
        if (generation.artifactPlan().strategy() != TextExecutionArtifactPlan.Strategy.PROCESS_LOCAL_PRETOKENIZED_HANDLE) {
            return new DirectTextExecutionSession(
                    engine,
                    generation,
                    ResumableSessionArtifact.descriptorOnly(descriptorFor(generation)),
                    new long[0],
                    conversationSession);
        }
        long[] inputIds = engine.encodePrompt(generation.prompt().formattedPrompt(), generation.model().modelPath());
        DirectPreparedGenerationHandle handle = handles().publish(generation, inputIds);
        ResumableSessionArtifact artifact = ResumableSessionArtifact.backendSessionHandle(
                descriptorFor(generation),
                handle.sessionKey(),
                handle.capturedAt(),
                Map.of(
                        "scope", "process_local",
                        "backend", id(),
                        "prepared_generation", true,
                        "pretokenized_prompt", true,
                        "artifact_strategy", generation.artifactPlan().strategy().name().toLowerCase()));
        return new DirectTextExecutionSession(engine, generation, artifact, inputIds, conversationSession);
    }

    @Override
    public Optional<TextExecutionSession> resumeSession(
            PreparedTextGeneration generation,
            ResumableSessionArtifact artifact) {
        if (artifact == null || !artifact.resumeCapable()) {
            return Optional.empty();
        }
        if (artifact.kind() == ResumableSessionArtifact.Kind.KV_SNAPSHOT) {
            Optional<ConversationSessionManager.ConversationSession> conversationSession =
                    resolveConversationSession(generation, artifact.artifactKey());
            if (conversationSession.isEmpty()) {
                return Optional.empty();
            }
            long[] inputIds = engine.encodePrompt(generation.prompt().formattedPrompt(), generation.model().modelPath());
            return Optional.of(new DirectTextExecutionSession(
                    engine,
                    generation,
                    conversationArtifactFor(generation),
                    inputIds,
                    conversationSession.get()));
        }
        if (artifact.kind() != ResumableSessionArtifact.Kind.BACKEND_SESSION_HANDLE) {
            return Optional.empty();
        }
        Optional<DirectPreparedGenerationHandle> handle = handles().find(artifact.artifactKey());
        if (handle.isEmpty()) {
            return Optional.empty();
        }
        DirectPreparedGenerationHandle stored = handle.get();
        if (!stored.promptFingerprint().equals(generation.promptFingerprint())) {
            return Optional.empty();
        }
        PreparedTextGeneration rebound = new PreparedTextGeneration(
                id(),
                this,
                generation.model(),
                generation.prompt(),
                generation.preparationPlan(),
                generation.generationConfig());
        ResumableSessionArtifact refreshedArtifact = ResumableSessionArtifact.backendSessionHandle(
                descriptorFor(rebound),
                stored.sessionKey(),
                stored.capturedAt(),
                Map.of(
                        "scope", "process_local",
                        "backend", id(),
                        "prepared_generation", true,
                        "pretokenized_prompt", true,
                        "resumed", true));
        return Optional.of(new DirectTextExecutionSession(
                engine,
                rebound,
                refreshedArtifact,
                stored.inputIds(),
                resolveConversationSession(rebound).orElse(null)));
    }

    @Override
    public Uni<InferenceResponse> generate(PreparedTextGeneration generation) {
        return openSession(generation).generate();
    }

    @Override
    public Multi<InferenceResponse> generateStream(PreparedTextGeneration generation) {
        return openSession(generation).generateStream();
    }

    @Override
    public void unloadModel(Path modelPath) {
        engine.unloadModel(modelPath);
    }

    @Override
    public void releaseArtifact(ResumableSessionArtifact artifact) {
        if (artifact != null && artifact.kind() == ResumableSessionArtifact.Kind.BACKEND_SESSION_HANDLE) {
            handles().evict(artifact.artifactKey());
        }
    }

    private DirectPreparedGenerationRegistry handles() {
        return handleRegistry != null ? handleRegistry : new DirectPreparedGenerationRegistry();
    }

    private Optional<ConversationSessionManager.ConversationSession> resolveConversationSession(
            PreparedTextGeneration generation) {
        return resolveConversationSession(generation, null);
    }

    private Optional<ConversationSessionManager.ConversationSession> resolveConversationSession(
            PreparedTextGeneration generation,
            String sessionIdOverride) {
        if (generation == null || generation.preparationPlan() == null || generation.preparationPlan().context() == null) {
            return Optional.empty();
        }
        String sessionId = sessionIdOverride != null && !sessionIdOverride.isBlank()
                ? sessionIdOverride
                : generation.preparationPlan().context().conversationSessionId();
        if (sessionId == null || sessionId.isBlank()) {
            return Optional.empty();
        }
        ConversationExecutionStateResolver resolver = conversationResolver();
        return resolver.findActiveSession(generation.model(), sessionId);
    }

    private ConversationExecutionStateResolver conversationResolver() {
        return conversationStateResolver != null
                ? conversationStateResolver
                : new ConversationExecutionStateResolver();
    }

    private ResumableSessionDescriptor descriptorFor(PreparedTextGeneration generation) {
        String sessionKey = generation.sessionPlan().sessionKey() != null
                ? generation.sessionPlan().sessionKey()
                : generation.promptFingerprint();
        return new ResumableSessionDescriptor(
                id(),
                sessionKey,
                generation.promptFingerprint(),
                generation.sessionPlan().reusePolicy(),
                java.time.Instant.now(),
                generation.sessionPlan().rationale());
    }

    private ResumableSessionArtifact conversationArtifactFor(PreparedTextGeneration generation) {
        ResumableSessionDescriptor descriptor = descriptorFor(generation);
        String sessionKey = descriptor.sessionKey();
        return ResumableSessionArtifact.kvSnapshot(
                descriptor,
                sessionKey,
                java.time.Instant.now(),
                Map.of(
                        "scope", "conversation_stateful",
                        "backend", id(),
                        "conversation_session", true,
                        "kv_snapshot", true,
                        "artifact_strategy", generation.artifactPlan().strategy().name().toLowerCase()));
    }
}
