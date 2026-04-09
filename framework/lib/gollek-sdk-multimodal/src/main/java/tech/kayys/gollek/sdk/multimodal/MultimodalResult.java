package tech.kayys.gollek.sdk.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalResponse;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Unified result from a multimodal inference operation.
 * <p>
 * Wraps {@link MultimodalResponse} with convenient accessors for common
 * use cases like extracting text, getting usage statistics, and inspecting
 * output modalities.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * MultimodalResult result = Gollek.vision("gemini-2.0-flash")
 *     .image(Path.of("cat.jpg"))
 *     .prompt("What animal is this?")
 *     .generate();
 *
 * String answer = result.text();           // "This is a cat."
 * int tokens = result.totalTokens();       // 47
 * long ms = result.durationMs();           // 1200
 * }</pre>
 */
public final class MultimodalResult {

    private final MultimodalResponse response;

    /**
     * Creates a result from the raw SPI response.
     *
     * @param response the multimodal response from the processor
     */
    public MultimodalResult(MultimodalResponse response) {
        this.response = response;
    }

    /**
     * Returns the underlying SPI response.
     */
    public MultimodalResponse raw() {
        return response;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Text extraction
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Extracts the first text output from the response.
     *
     * @return the generated text, or an empty string if no text output exists
     */
    public String text() {
        return textOutput().orElse("");
    }

    /**
     * Returns the first text output as an Optional.
     *
     * @return text content, or empty if the response contains no text
     */
    public Optional<String> textOutput() {
        if (response.getOutputs() == null) return Optional.empty();
        return Arrays.stream(response.getOutputs())
                .filter(c -> c.getModality() == ModalityType.TEXT)
                .map(MultimodalContent::getText)
                .findFirst();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Binary output extraction
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Returns all output content parts.
     *
     * @return list of output content parts
     */
    public List<MultimodalContent> outputs() {
        if (response.getOutputs() == null) return List.of();
        return Arrays.asList(response.getOutputs());
    }

    /**
     * Returns the first output of the given modality type.
     *
     * @param type the modality to find
     * @return the first matching content, or empty
     */
    public Optional<MultimodalContent> output(ModalityType type) {
        return outputs().stream()
                .filter(c -> c.getModality() == type)
                .findFirst();
    }

    /**
     * Returns the first image output as raw bytes (if present).
     *
     * @return image bytes, or empty
     */
    public Optional<byte[]> imageBytes() {
        return output(ModalityType.IMAGE)
                .map(MultimodalContent::getRawBytes);
    }

    /**
     * Returns the first audio output as raw bytes (if present).
     *
     * @return audio bytes, or empty
     */
    public Optional<byte[]> audioBytes() {
        return output(ModalityType.AUDIO)
                .map(MultimodalContent::getRawBytes);
    }

    /**
     * Returns whether the response contains any binary (non-text) output.
     *
     * @return true if IMAGE, AUDIO, or VIDEO output is present
     */
    public boolean hasBinary() {
        return outputs().stream()
                .anyMatch(c -> c.getModality() != ModalityType.TEXT);
    }

    /**
     * Returns the raw bytes of the first binary (non-text) output.
     *
     * @return the raw bytes, or null if no binary output exists
     */
    public byte[] binaryData() {
        return outputs().stream()
                .filter(c -> c.getModality() != ModalityType.TEXT)
                .map(MultimodalContent::getRawBytes)
                .findFirst()
                .orElse(null);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Metadata
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Returns the model used for generation.
     */
    public String model() {
        return response.getModel();
    }

    /**
     * Returns the request ID.
     */
    public String requestId() {
        return response.getRequestId();
    }

    /**
     * Returns the inference duration in milliseconds.
     */
    public long durationMs() {
        return response.getDurationMs();
    }

    /**
     * Returns the response status.
     */
    public MultimodalResponse.ResponseStatus status() {
        return response.getStatus();
    }

    /**
     * Returns whether the request was successful.
     */
    public boolean isSuccess() {
        return response.getStatus() == MultimodalResponse.ResponseStatus.SUCCESS;
    }

    /**
     * Returns the total tokens used (input + output).
     *
     * @return total token count, or -1 if usage info is unavailable
     */
    public int totalTokens() {
        var usage = response.getUsage();
        return usage != null ? usage.getTotalTokens() : -1;
    }

    /**
     * Returns the number of input tokens.
     *
     * @return input token count, or -1 if usage info is unavailable
     */
    public int inputTokens() {
        var usage = response.getUsage();
        return usage != null ? usage.getInputTokens() : -1;
    }

    /**
     * Returns the number of output tokens.
     *
     * @return output token count, or -1 if usage info is unavailable
     */
    public int outputTokens() {
        var usage = response.getUsage();
        return usage != null ? usage.getOutputTokens() : -1;
    }

    /**
     * Returns the response metadata.
     */
    public Map<String, Object> metadata() {
        return response.getMetadata() != null ? response.getMetadata() : Map.of();
    }

    @Override
    public String toString() {
        return "MultimodalResult{" +
                "status=" + status() +
                ", model=" + model() +
                ", text=" + (text().length() > 50 ? text().substring(0, 50) + "..." : text()) +
                ", tokens=" + totalTokens() +
                ", duration=" + durationMs() + "ms" +
                '}';
    }
}
