package tech.kayys.gollek.sdk.multimodal;

import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * Fluent builder for mixed-modality inference (text + images + audio + video + documents).
 * <p>
 * This is the most flexible builder, allowing arbitrary combinations of content
 * parts from different modalities in a single request.
 *
 * <h3>Example — Image Comparison</h3>
 * <pre>{@code
 * var result = Gollek.multimodal("gpt-4o")
 *     .text("Compare these two images and describe the differences")
 *     .image(Path.of("before.png"))
 *     .image(Path.of("after.png"))
 *     .maxTokens(1000)
 *     .generate();
 * }</pre>
 *
 * <h3>Example — Document + Image Analysis</h3>
 * <pre>{@code
 * var result = Gollek.multimodal("gemini-2.0-flash")
 *     .text("Extract the table data from this document")
 *     .document(Path.of("report.pdf"), "pdf")
 *     .generate();
 * }</pre>
 *
 * <h3>Example — Auto-detect from files</h3>
 * <pre>{@code
 * var result = Gollek.multimodal("gemini-2.0-pro")
 *     .text("Analyze all these inputs")
 *     .file(Path.of("photo.jpg"))
 *     .file(Path.of("recording.wav"))
 *     .generate();
 * }</pre>
 */
public final class MultimodalBuilder {

    private final String model;
    private final GollekSdk sdk;
    private final List<MultimodalContent> parts = new ArrayList<>();
    private int maxTokens = 2048;
    private double temperature = 0.7;
    private double topP = 0.9;
    private boolean stream = false;
    private ModalityType[] outputModalities = {ModalityType.TEXT};
    private long timeoutMs = 60_000;
    private final Map<String, Object> parameters = new LinkedHashMap<>();

    public MultimodalBuilder(String model, GollekSdk sdk) {
        this.model = Objects.requireNonNull(model, "model");
        this.sdk = Objects.requireNonNull(sdk, "sdk");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Content Parts
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Adds a text part.
     *
     * @param text the text content
     * @return this builder
     */
    public MultimodalBuilder text(String text) {
        parts.add(MultimodalContent.ofText(text));
        return this;
    }

    /**
     * Adds an image from a file.
     *
     * @param path image file path
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public MultimodalBuilder image(Path path) throws IOException {
        parts.add(ContentPart.image(path).toSpiContent());
        return this;
    }

    /**
     * Adds an image from raw bytes.
     *
     * @param bytes    image bytes
     * @param mimeType MIME type
     * @return this builder
     */
    public MultimodalBuilder image(byte[] bytes, String mimeType) {
        parts.add(ContentPart.image(bytes, mimeType).toSpiContent());
        return this;
    }

    /**
     * Adds an image from a URL.
     *
     * @param url image URL
     * @return this builder
     */
    public MultimodalBuilder imageUrl(String url) {
        parts.add(ContentPart.imageUrl(url).toSpiContent());
        return this;
    }

    /**
     * Adds an audio file.
     *
     * @param path audio file path
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public MultimodalBuilder audio(Path path) throws IOException {
        parts.add(ContentPart.audio(path).toSpiContent());
        return this;
    }

    /**
     * Adds audio from raw bytes.
     *
     * @param bytes    audio bytes
     * @param mimeType MIME type
     * @return this builder
     */
    public MultimodalBuilder audio(byte[] bytes, String mimeType) {
        parts.add(ContentPart.audio(bytes, mimeType).toSpiContent());
        return this;
    }

    /**
     * Adds a video file.
     *
     * @param path video file path
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public MultimodalBuilder video(Path path) throws IOException {
        parts.add(ContentPart.video(path).toSpiContent());
        return this;
    }

    /**
     * Adds a document file (PDF, DOCX, HTML).
     *
     * @param path   document file path
     * @param format document format (e.g., "pdf", "docx")
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public MultimodalBuilder document(Path path, String format) throws IOException {
        parts.add(ContentPart.document(path, format).toSpiContent());
        return this;
    }

    /**
     * Adds any file, auto-detecting its modality from the file extension.
     *
     * @param path file path
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public MultimodalBuilder file(Path path) throws IOException {
        parts.add(ContentPart.fromFile(path).toSpiContent());
        return this;
    }

    /**
     * Adds a pre-built content part.
     *
     * @param part the content part
     * @return this builder
     */
    public MultimodalBuilder part(ContentPart part) {
        parts.add(part.toSpiContent());
        return this;
    }

    /**
     * Adds a raw SPI content object.
     *
     * @param content the SPI content
     * @return this builder
     */
    public MultimodalBuilder content(MultimodalContent content) {
        parts.add(content);
        return this;
    }

    /**
     * Adds multiple pre-built content parts.
     *
     * @param contentParts the content parts to add
     * @return this builder
     */
    public MultimodalBuilder parts(Collection<ContentPart> contentParts) {
        if (contentParts != null) {
            contentParts.forEach(this::part);
        }
        return this;
    }

    /**
     * Adds multiple raw SPI content objects.
     *
     * @param multimodalContents the SPI contents to add
     * @return this builder
     */
    public MultimodalBuilder contents(Collection<MultimodalContent> multimodalContents) {
        if (multimodalContents != null) {
            multimodalContents.forEach(this::content);
        }
        return this;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════════

    /** Sets the desired output modality types. */
    public MultimodalBuilder outputModalities(ModalityType... modalities) {
        this.outputModalities = modalities;
        return this;
    }

    /** Sets max tokens. */
    public MultimodalBuilder maxTokens(int maxTokens) {
        this.maxTokens = maxTokens;
        return this;
    }

    /** Sets temperature. */
    public MultimodalBuilder temperature(double temperature) {
        this.temperature = temperature;
        return this;
    }

    /** Sets top-p. */
    public MultimodalBuilder topP(double topP) {
        this.topP = topP;
        return this;
    }

    /** Enables/disables streaming. */
    public MultimodalBuilder stream(boolean stream) {
        this.stream = stream;
        return this;
    }

    /** Sets timeout in milliseconds. */
    public MultimodalBuilder timeout(long timeoutMs) {
        this.timeoutMs = timeoutMs;
        return this;
    }

    /** Adds a custom parameter. */
    public MultimodalBuilder parameter(String key, Object value) {
        this.parameters.put(key, value);
        return this;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Execution
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Executes the multimodal request and returns the result.
     *
     * @return the multimodal result
     * @throws IllegalStateException if no content parts were added
     */
    public MultimodalResult generate() {
        if (parts.isEmpty()) {
            throw new IllegalStateException(
                    "At least one content part is required. Use .text(), .image(), .audio(), etc.");
        }

        var outputConfig = MultimodalRequest.OutputConfig.builder()
                .outputModalities(outputModalities)
                .maxTokens(maxTokens)
                .temperature(temperature)
                .topP(topP)
                .stream(stream)
                .build();

        var request = MultimodalRequest.builder()
                .model(model)
                .inputs(parts.toArray(new MultimodalContent[0]))
                .outputConfig(outputConfig)
                .parameters(parameters.isEmpty() ? null : parameters)
                .timeoutMs(timeoutMs)
                .build();

        return new MultimodalResult(sdk.processMultimodal(request));
    }
}
