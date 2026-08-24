package tech.kayys.gollek.spi.image;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.exception.InferenceException;

import java.util.Set;

/**
 * Service Provider Interface for image generation pipelines (Stable Diffusion, FLUX, etc.).
 */
public interface ImageGenerationPipeline {

    /**
     * Unique identifier for this pipeline instance / model.
     */
    String pipelineId();

    /**
     * Set of capabilities supported by this pipeline (e.g. TEXT_TO_IMAGE, INPAINTING).
     */
    Set<PipelineCapability> pipelineCapabilities();

    /**
     * Returns true if this pipeline supports the specified capability.
     */
    default boolean supports(PipelineCapability capability) {
        Set<PipelineCapability> caps = pipelineCapabilities();
        return caps != null && caps.contains(capability);
    }

    /**
     * Generate an image synchronously.
     *
     * @param request the generation parameters
     * @return the generated image output
     * @throws InferenceException if generation fails
     */
    GeneratedImage generate(ImageGenRequest request) throws InferenceException;

    /**
     * Generate an image asynchronously with streaming progress updates.
     *
     * @param request the generation parameters
     * @return a Multi stream emitting step progress events
     */
    default Multi<ImageGenProgress> generateStreaming(ImageGenRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                emitter.emit(ImageGenProgress.step(request.requestId(), 0, request.steps(), "Starting generation..."));
                GeneratedImage img = generate(request);
                emitter.emit(ImageGenProgress.step(request.requestId(), request.steps(), request.steps(), "Generation complete."));
                emitter.complete();
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    /**
     * Return default/recommended parameters for this pipeline.
     */
    default ImageGenRequest defaultRequest(String prompt) {
        return ImageGenRequest.builder()
                .prompt(prompt)
                .width(512)
                .height(512)
                .steps(20)
                .guidanceScale(7.5f)
                .build();
    }
}
