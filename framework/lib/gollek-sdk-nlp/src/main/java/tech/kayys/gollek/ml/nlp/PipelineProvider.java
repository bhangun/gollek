package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.spi.model.ModalityType;

import java.util.Set;

/**
 * Service Provider Interface for contributing custom pipeline tasks.
 * <p>
 * Plugins implement this interface and register via
 * {@code META-INF/services/tech.kayys.gollek.ml.nlp.PipelineProvider}
 * to auto-register new pipeline tasks in the {@link PipelineFactory}.
 *
 * <h3>Example Implementation (in a plugin)</h3>
 * <pre>{@code
 * public class RagPipelineProvider implements PipelineProvider {
 *     public String task() { return "rag"; }
 *     public Pipeline<?, ?> create(PipelineConfig config) {
 *         return new RagPipeline(config);
 *     }
 *     public Set<ModalityType> inputModalities() {
 *         return Set.of(ModalityType.TEXT);
 *     }
 *     public Set<ModalityType> outputModalities() {
 *         return Set.of(ModalityType.TEXT);
 *     }
 * }
 * }</pre>
 *
 * <p>Once registered, the pipeline becomes available via:
 * <pre>{@code
 * var rag = Gollek.pipeline("rag", "my-rag-model");
 * }</pre>
 */
public interface PipelineProvider {

    /**
     * Returns the task identifier this provider contributes
     * (e.g., "rag", "image-captioning", "speech-to-text").
     *
     * @return task name
     */
    String task();

    /**
     * Creates a pipeline instance for the given configuration.
     *
     * @param config pipeline configuration with model, parameters, etc.
     * @return a new pipeline instance
     */
    Pipeline<?, ?> create(PipelineConfig config);

    /**
     * Returns the modality types this pipeline accepts as input.
     *
     * @return set of input modalities (default: TEXT only)
     */
    default Set<ModalityType> inputModalities() {
        return Set.of(ModalityType.TEXT);
    }

    /**
     * Returns the modality types this pipeline produces as output.
     *
     * @return set of output modalities (default: TEXT only)
     */
    default Set<ModalityType> outputModalities() {
        return Set.of(ModalityType.TEXT);
    }

    /**
     * Returns a human-readable description of this pipeline task.
     *
     * @return description (default: empty)
     */
    default String description() {
        return "";
    }
}
