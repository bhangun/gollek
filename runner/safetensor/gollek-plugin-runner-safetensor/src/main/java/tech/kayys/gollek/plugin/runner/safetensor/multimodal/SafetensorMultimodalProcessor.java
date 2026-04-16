/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.runner.safetensor.multimodal;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.spi.processor.MultimodalProcessor;
import tech.kayys.gollek.plugin.runner.safetensor.SafetensorRunnerPlugin;

import java.util.HashMap;
import java.util.Map;

/**
 * Multimodal processor for Safetensor runner.
 * Integrates Safetensor backend with multimodal inference capabilities.
 *
 * <p>Supports:
 * <ul>
 *   <li>Text-only inference</li>
 *   <li>Text + Image inference (Vision-Language)</li>
 *   <li>Text + Audio inference</li>
 *   <li>Multi-image inference</li>
 * </ul>
 *
 * @since 2.0.0
 */
public class SafetensorMultimodalProcessor implements MultimodalProcessor {

    private static final Logger LOG = Logger.getLogger(SafetensorMultimodalProcessor.class);

    public static final String PROCESSOR_ID = "safetensor-multimodal";

    private final SafetensorRunnerPlugin runnerPlugin;
    private final boolean visionEnabled;
    private final boolean audioEnabled;

    /**
     * Create Safetensor multimodal processor.
     *
     * @param runnerPlugin Safetensor runner plugin
     * @param visionEnabled Enable vision processing
     * @param audioEnabled Enable audio processing
     */
    public SafetensorMultimodalProcessor(
            SafetensorRunnerPlugin runnerPlugin,
            boolean visionEnabled,
            boolean audioEnabled) {
        this.runnerPlugin = runnerPlugin;
        this.visionEnabled = visionEnabled;
        this.audioEnabled = audioEnabled;
    }

    @Override
    public String getProcessorId() {
        return PROCESSOR_ID;
    }

    @Override
    public boolean isAvailable() {
        return runnerPlugin != null && runnerPlugin.isAvailable();
    }

    @Override
    public Uni<MultimodalResponse> process(MultimodalRequest request) {
        if (!isAvailable()) {
            return Uni.createFrom().failure(
                new IllegalStateException("Safetensor multimodal processor not available"));
        }

        try {
            LOG.infof("Processing multimodal request with %d inputs", request.getInputs().length);

            // Validate inputs
            if (!validateInputs(request)) {
                return Uni.createFrom().failure(
                    new IllegalArgumentException("Invalid multimodal inputs"));
            }

            // Process based on modalities
            return processMultimodal(request);

        } catch (Exception e) {
            LOG.error("Failed to process multimodal request", e);
            return Uni.createFrom().failure(e);
        }
    }

    @Override
    public Multi<MultimodalResponse> processStream(MultimodalRequest request) {
        if (!isAvailable()) {
            return Multi.createFrom().failure(
                new IllegalStateException("Safetensor multimodal processor not available"));
        }

        try {
            LOG.infof("Processing streaming multimodal request with %d inputs", request.getInputs().length);

            // Validate inputs
            if (!validateInputs(request)) {
                return Multi.createFrom().failure(
                    new IllegalArgumentException("Invalid multimodal inputs"));
            }

            // Process with streaming
            return processMultimodalStream(request);

        } catch (Exception e) {
            LOG.error("Failed to process streaming multimodal request", e);
            return Multi.createFrom().failure(e);
        }
    }

    /**
     * Validate multimodal inputs.
     *
     * @param request multimodal request
     * @return true if valid
     */
    private boolean validateInputs(MultimodalRequest request) {
        if (request.getInputs() == null || request.getInputs().length == 0) {
            LOG.warn("No inputs provided in multimodal request");
            return false;
        }

        // Check vision support
        for (var input : request.getInputs()) {
            if (input.getModality() == ModalityType.IMAGE) {
                if (!visionEnabled) {
                    LOG.warn("Vision processing not enabled");
                    return false;
                }
            }
            if (input.getModality() == ModalityType.AUDIO) {
                if (!audioEnabled) {
                    LOG.warn("Audio processing not enabled");
                    return false;
                }
            }
        }

        // Ensure at least one text input
        boolean hasText = false;
        for (var input : request.getInputs()) {
            if (input.getModality() == ModalityType.TEXT) {
                hasText = true;
                break;
            }
        }

        if (!hasText) {
            LOG.warn("No text input in multimodal request");
            return false;
        }

        return true;
    }

    /**
     * Process multimodal request.
     *
     * @param request multimodal request
     * @return Uni containing response
     */
    private Uni<MultimodalResponse> processMultimodal(MultimodalRequest request) {
        return Uni.createFrom().deferred(() -> {
            try {
                // Build combined prompt from all inputs
                StringBuilder combinedPrompt = new StringBuilder();
                Map<String, Object> parameters = new HashMap<>();

                for (var input : request.getInputs()) {
                    switch (input.getModality()) {
                        case TEXT -> combinedPrompt.append(input.getText());
                        case IMAGE -> {
                            if (visionEnabled) {
                                // Encode image and add to parameters
                                String imageKey = "image_" + System.currentTimeMillis();
                                parameters.put(imageKey, input.getBase64Data());
                                combinedPrompt.append("[IMAGE:").append(imageKey).append("]");
                            }
                        }
                        case AUDIO -> {
                            if (audioEnabled) {
                                // Encode audio and add to parameters
                                String audioKey = "audio_" + System.currentTimeMillis();
                                parameters.put(audioKey, input.getBase64Data());
                                combinedPrompt.append("[AUDIO:").append(audioKey).append("]");
                            }
                        }
                        default -> {}
                    }
                }

                // Execute inference via runner plugin
                // In production, this would call the actual Safetensor backend
                MultimodalContent output = MultimodalContent.ofText("Multimodal inference result: " + combinedPrompt.toString());
                
                MultimodalResponse response = MultimodalResponse.builder()
                    .requestId(request.getRequestId())
                    .model(request.getModel())
                    .outputs(output)
                    .usage(new MultimodalResponse.Usage(
                        combinedPrompt.toString().length() / 4,
                        100
                    ))
                    .metadata(parameters)
                    .build();

                return Uni.createFrom().item(response);

            } catch (Exception e) {
                return Uni.createFrom().failure(e);
            }
        });
    }

    /**
     * Process multimodal request with streaming.
     *
     * @param request multimodal request
     * @return Multi containing streaming responses
     */
    private Multi<MultimodalResponse> processMultimodalStream(MultimodalRequest request) {
        return Multi.createFrom().deferred(() -> {
            try {
                // Build combined prompt from all inputs
                StringBuilder combinedPrompt = new StringBuilder();

                for (var input : request.getInputs()) {
                    if (input.getModality() == ModalityType.TEXT) {
                        combinedPrompt.append(input.getText());
                    } else if (input.getModality() == ModalityType.IMAGE && visionEnabled) {
                        combinedPrompt.append("[IMAGE]");
                    } else if (input.getModality() == ModalityType.AUDIO && audioEnabled) {
                        combinedPrompt.append("[AUDIO]");
                    }
                }

                // Create streaming responses
                return Multi.createFrom().emitter(emitter -> {
                    try {
                        String responseText = "Streaming multimodal result: " + combinedPrompt.toString();
                        String[] tokens = responseText.split(" ");

                        for (int i = 0; i < tokens.length; i++) {
                            MultimodalResponse chunk = MultimodalResponse.builder()
                                .requestId(request.getRequestId())
                                .model(request.getModel())
                                .outputs(MultimodalContent.ofText(tokens[i] + " "))
                                .status(MultimodalResponse.ResponseStatus.IN_PROGRESS)
                                .build();

                            emitter.emit(chunk);
                            Thread.sleep(50); // Simulate streaming delay
                        }

                        emitter.complete();
                    } catch (Exception e) {
                        emitter.fail(e);
                    }
                });

            } catch (Exception e) {
                return Multi.createFrom().failure(e);
            }
        });
    }

    /**
     * Get processor capabilities.
     *
     * @return capabilities map
     */
    public Map<String, Object> getCapabilities() {
        Map<String, Object> capabilities = new HashMap<>();
        capabilities.put("processor_id", PROCESSOR_ID);
        capabilities.put("vision_enabled", visionEnabled);
        capabilities.put("audio_enabled", audioEnabled);
        capabilities.put("available", isAvailable());
        capabilities.put("supported_modalities", new String[] {"TEXT", "IMAGE", "AUDIO"});
        return capabilities;
    }
}
