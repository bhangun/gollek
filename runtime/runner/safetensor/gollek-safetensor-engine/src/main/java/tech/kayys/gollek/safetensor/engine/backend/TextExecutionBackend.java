package tech.kayys.gollek.safetensor.engine.backend;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.engine.planning.PreparedPrompt;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Execution contract for text generation backends.
 *
 * <p>This is the hot-path boundary Gollek can route across when a platform-specific
 * backend exists. The current direct safetensor engine is the first implementation,
 * but Apple/MLX or other backend-native implementations can adopt the same contract
 * without leaking their details into CLI/provider orchestration code.
 *
 * <p>Design note: the explicit model-preparation and generation-preparation split is
 * intentional. It leaves room for backend-resident prompt state, prefill reuse, or
 * persistent KV-backed execution sessions later on. See Shkolnikov (2026),
 * <a href="https://arxiv.org/abs/2603.04428">arXiv:2603.04428</a>.
 */
public interface TextExecutionBackend {

    /**
     * Stable backend id, e.g. {@code direct}, {@code mlx}, {@code metal-native}.
     */
    String id();

    /**
     * Whether this backend can be used in the current runtime.
     */
    default boolean isAvailable() {
        return true;
    }

    /**
     * Backend capabilities for orchestration and validation.
     */
    TextExecutionBackendCapabilities capabilities();

    /**
     * Load or prepare a model for execution on this backend.
     */
    PreparedTextModel prepareModel(Path modelPath, Path adapterPath, QuantizationEngine.QuantStrategy quantStrategy);

    /**
     * Prepare a concrete text generation request for this backend.
     *
     * <p>The engine computes the execution policy first and passes it here as an
     * explicit preparation plan. Backends can still attach backend-native prompt
     * handles, prefill state, or resumable decode state, but they no longer need
     * to derive session or artifact policy for themselves.
     */
    default PreparedTextGeneration prepareGeneration(
            PreparedPrompt prompt,
            PreparedTextModel model,
            TextExecutionPreparationPlan preparationPlan,
            GenerationConfig cfg) {
        return new PreparedTextGeneration(id(), this, model, prompt, preparationPlan, cfg);
    }

    /**
     * Open a backend-owned execution session for the prepared generation.
     */
    default TextExecutionSession openSession(PreparedTextGeneration generation) {
        return new TextExecutionSession() {
            @Override
            public String backendId() {
                return generation.backendId();
            }

            @Override
            public PreparedTextGeneration generation() {
                return generation;
            }

            @Override
            public Uni<InferenceResponse> generate() {
                return TextExecutionBackend.this.generate(generation);
            }

            @Override
            public Multi<InferenceResponse> generateStream() {
                return TextExecutionBackend.this.generateStream(generation);
            }

            @Override
            public void close() {
                TextExecutionBackend.this.releaseGeneration(generation);
            }
        };
    }

    /**
     * Attempt to resume or restore a previously reusable session.
     *
     * <p>Backends that support backend-resident prompt state, KV snapshots, or
     * equivalent resumable execution artifacts can override this to reattach a
     * prepared generation to existing state. The default implementation returns
     * empty, causing the engine to fall back to a fresh session. This explicit
     * seam is part of Gollek's transition toward persistent prefill/KV-backed
     * execution sessions; see Shkolnikov (2026),
     * <a href="https://arxiv.org/abs/2603.04428">arXiv:2603.04428</a>.
     */
    default Optional<TextExecutionSession> resumeSession(
            PreparedTextGeneration generation,
            ResumableSessionArtifact artifact) {
        return Optional.empty();
    }

    /**
     * Release any backend-owned state behind a resumable artifact.
     */
    default void releaseArtifact(ResumableSessionArtifact artifact) {
    }

    /**
     * Generate a full response.
     */
    Uni<InferenceResponse> generate(PreparedTextGeneration generation);

    /**
     * Stream response chunks.
     */
    Multi<InferenceResponse> generateStream(PreparedTextGeneration generation);

    /** Release per-generation resources if the backend allocated any. */
    default void releaseGeneration(PreparedTextGeneration generation) {
    }

    /**
     * Release model resources.
     */
    default void unloadModel(PreparedTextModel model) {
        unloadModel(model.modelPath());
    }

    /**
     * Release model resources by path for compatibility with the current direct backend.
     */
    void unloadModel(Path modelPath);
}
