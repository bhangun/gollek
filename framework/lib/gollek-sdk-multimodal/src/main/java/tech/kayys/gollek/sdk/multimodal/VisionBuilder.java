package tech.kayys.gollek.sdk.multimodal;

import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * Fluent builder for vision (image understanding) tasks.
 * <p>
 * Provides a natural API for sending images to vision-capable models
 * and getting text descriptions, analysis, or structured outputs.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var result = Gollek.vision("gemini-2.0-flash")
 *     .image(Path.of("photo.jpg"))
 *     .prompt("What's in this image?")
 *     .maxTokens(500)
 *     .generate();
 *
 * System.out.println(result.text());
 * }</pre>
 *
 * <h3>Multiple Images</h3>
 * <pre>{@code
 * var result = Gollek.vision("gpt-4o")
 *     .image(Path.of("before.png"))
 *     .image(Path.of("after.png"))
 *     .prompt("What changed between these two images?")
 *     .generate();
 * }</pre>
 */
public final class VisionBuilder {

    private final String model;
    private final GollekSdk sdk;
    private final List<MultimodalContent> parts = new ArrayList<>();
    private String prompt;
    private int maxTokens = 2048;
    private double temperature = 0.7;
    private double topP = 0.9;
    private boolean stream = false;
    private long timeoutMs = 30_000;
    private final Map<String, Object> parameters = new LinkedHashMap<>();

    /**
     * Creates a vision builder for the given model using the provided SDK.
     *
     * @param model model identifier (e.g., "gemini-2.0-flash", "gpt-4o")
     * @param sdk   the SDK instance to execute against
     */
    public VisionBuilder(String model, GollekSdk sdk) {
        this.model = Objects.requireNonNull(model, "model");
        this.sdk = Objects.requireNonNull(sdk, "sdk");
    }

    /**
     * Adds an image from a file path.
     *
     * @param path path to the image file
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public VisionBuilder image(Path path) throws IOException {
        parts.add(ContentPart.image(path).toSpiContent());
        return this;
    }

    /**
     * Adds an image from raw bytes.
     *
     * @param bytes    image bytes
     * @param mimeType MIME type (e.g., "image/jpeg")
     * @return this builder
     */
    public VisionBuilder image(byte[] bytes, String mimeType) {
        parts.add(ContentPart.image(bytes, mimeType).toSpiContent());
        return this;
    }

    /**
     * Adds an image from a URL.
     *
     * @param url URL to the image
     * @return this builder
     */
    public VisionBuilder imageUrl(String url) {
        parts.add(ContentPart.imageUrl(url).toSpiContent());
        return this;
    }

    /**
     * Adds an image from a URL with explicit MIME type.
     *
     * @param url      URL to the image
     * @param mimeType MIME type
     * @return this builder
     */
    public VisionBuilder imageUrl(String url, String mimeType) {
        parts.add(ContentPart.imageUrl(url, mimeType).toSpiContent());
        return this;
    }

    /**
     * Sets the text prompt / instruction for the vision model.
     *
     * @param prompt the text instruction
     * @return this builder
     */
    public VisionBuilder prompt(String prompt) {
        this.prompt = prompt;
        return this;
    }

    /**
     * Sets the maximum number of tokens to generate.
     *
     * @param maxTokens token limit (default: 2048)
     * @return this builder
     */
    public VisionBuilder maxTokens(int maxTokens) {
        this.maxTokens = maxTokens;
        return this;
    }

    /**
     * Sets the sampling temperature.
     *
     * @param temperature value in [0.0, 2.0] (default: 0.7)
     * @return this builder
     */
    public VisionBuilder temperature(double temperature) {
        this.temperature = temperature;
        return this;
    }

    /**
     * Sets the nucleus sampling probability.
     *
     * @param topP value in (0.0, 1.0] (default: 0.9)
     * @return this builder
     */
    public VisionBuilder topP(double topP) {
        this.topP = topP;
        return this;
    }

    /**
     * Enables or disables streaming output.
     *
     * @param stream {@code true} to enable streaming
     * @return this builder
     */
    public VisionBuilder stream(boolean stream) {
        this.stream = stream;
        return this;
    }

    /**
     * Sets the request timeout.
     *
     * @param timeoutMs timeout in milliseconds (default: 30000)
     * @return this builder
     */
    public VisionBuilder timeout(long timeoutMs) {
        this.timeoutMs = timeoutMs;
        return this;
    }

    /**
     * Adds a custom generation parameter.
     *
     * @param key   parameter key
     * @param value parameter value
     * @return this builder
     */
    public VisionBuilder parameter(String key, Object value) {
        this.parameters.put(key, value);
        return this;
    }

    /**
     * Executes the vision request and returns the result.
     *
     * @return the multimodal result
     * @throws IllegalStateException if no image was added
     */
    public MultimodalResult generate() {
        if (parts.isEmpty()) {
            throw new IllegalStateException("At least one image is required. Call .image() first.");
        }

        // Add prompt as text part at the beginning
        List<MultimodalContent> allParts = new ArrayList<>();
        if (prompt != null && !prompt.isBlank()) {
            allParts.add(MultimodalContent.ofText(prompt));
        }
        allParts.addAll(parts);

        var outputConfig = MultimodalRequest.OutputConfig.builder()
                .outputModalities(ModalityType.TEXT)
                .maxTokens(maxTokens)
                .temperature(temperature)
                .topP(topP)
                .stream(stream)
                .build();

        var request = MultimodalRequest.builder()
                .model(model)
                .inputs(allParts.toArray(new MultimodalContent[0]))
                .outputConfig(outputConfig)
                .parameters(parameters.isEmpty() ? null : parameters)
                .timeoutMs(timeoutMs)
                .build();

        return new MultimodalResult(sdk.processMultimodal(request));
    }
}
