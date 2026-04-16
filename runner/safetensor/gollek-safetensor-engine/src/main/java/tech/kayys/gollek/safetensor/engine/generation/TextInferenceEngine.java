package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.spi.SafetensorFeature;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;

/**
 * CDI bean that exposes text generation as a {@link SafetensorFeature}.
 *
 * <p>Delegates all inference work to the injected {@link DirectInferenceEngine},
 * acting as a thin adapter that registers the text modality with the SafeTensor
 * feature plugin system.
 *
 * @see DirectInferenceEngine
 * @see SafetensorFeature
 */
@ApplicationScoped
public class TextInferenceEngine implements SafetensorFeature {

    private static final Logger log = Logger.getLogger(TextInferenceEngine.class);

    @Inject
    DirectInferenceEngine engine;

    /** Returns {@code "text"} — the feature identifier for text generation. */
    @Override
    public String id() {
        return "text";
    }

    /** Logs initialization; no additional setup required. */
    @Override
    public void initialize() {
        log.info("TextInferenceEngine initialized and registered with SafeTensor engine");
    }

    /**
     * Generates a complete text response for the given prompt.
     *
     * @param prompt    the input prompt string
     * @param modelPath path to the SafeTensors model directory or file
     * @param cfg       generation configuration (sampling strategy, token limits, etc.)
     * @return a {@link Uni} that resolves to the full {@link InferenceResponse}
     */
    public Uni<InferenceResponse> generate(String prompt, Path modelPath, GenerationConfig cfg) {
        return engine.generate(prompt, modelPath, cfg);
    }

    /**
     * Streams generated tokens incrementally for the given prompt.
     *
     * @param prompt    the input prompt string
     * @param modelPath path to the SafeTensors model directory or file
     * @param cfg       generation configuration
     * @return a {@link Multi} emitting one {@link InferenceResponse} per generated token
     */
    public Multi<InferenceResponse> generateStream(String prompt, Path modelPath, GenerationConfig cfg) {
        return engine.generateStream(prompt, modelPath, cfg);
    }
}
