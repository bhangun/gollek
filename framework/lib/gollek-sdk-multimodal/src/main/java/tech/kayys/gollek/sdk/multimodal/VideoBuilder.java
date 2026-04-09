package tech.kayys.gollek.sdk.multimodal;

import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * Fluent builder for video understanding tasks.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var result = Gollek.video("gemini-2.0-pro")
 *     .videoFile(Path.of("clip.mp4"))
 *     .prompt("Summarize the key events in this video")
 *     .generate();
 *
 * System.out.println(result.text());
 * }</pre>
 */
public final class VideoBuilder {

    private final String model;
    private final GollekSdk sdk;
    private final List<MultimodalContent> parts = new ArrayList<>();
    private String prompt;
    private int maxTokens = 4096;
    private double temperature = 0.7;
    private double topP = 0.9;
    private long timeoutMs = 120_000;
    private final Map<String, Object> parameters = new LinkedHashMap<>();

    public VideoBuilder(String model, GollekSdk sdk) {
        this.model = Objects.requireNonNull(model, "model");
        this.sdk = Objects.requireNonNull(sdk, "sdk");
    }

    /**
     * Adds a video file.
     *
     * @param path path to the video (MP4, AVI, MOV, etc.)
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public VideoBuilder videoFile(Path path) throws IOException {
        parts.add(ContentPart.video(path).toSpiContent());
        return this;
    }

    /**
     * Adds video from raw bytes.
     *
     * @param bytes    video bytes
     * @param mimeType MIME type (e.g., "video/mp4")
     * @return this builder
     */
    public VideoBuilder video(byte[] bytes, String mimeType) {
        parts.add(ContentPart.video(bytes, mimeType).toSpiContent());
        return this;
    }

    /**
     * Adds a video from a URL.
     *
     * @param url URL to the video
     * @return this builder
     */
    public VideoBuilder videoUrl(String url) {
        parts.add(ContentPart.videoUrl(url).toSpiContent());
        return this;
    }

    /**
     * Sets the text prompt / instruction.
     *
     * @param prompt instruction text
     * @return this builder
     */
    public VideoBuilder prompt(String prompt) {
        this.prompt = prompt;
        return this;
    }

    /** Sets max tokens. */
    public VideoBuilder maxTokens(int maxTokens) {
        this.maxTokens = maxTokens;
        return this;
    }

    /** Sets temperature. */
    public VideoBuilder temperature(double temperature) {
        this.temperature = temperature;
        return this;
    }

    /** Sets top-p. */
    public VideoBuilder topP(double topP) {
        this.topP = topP;
        return this;
    }

    /** Sets timeout in milliseconds. */
    public VideoBuilder timeout(long timeoutMs) {
        this.timeoutMs = timeoutMs;
        return this;
    }

    /** Adds a custom parameter. */
    public VideoBuilder parameter(String key, Object value) {
        this.parameters.put(key, value);
        return this;
    }

    /**
     * Executes the video understanding request.
     *
     * @return the multimodal result
     */
    public MultimodalResult generate() {
        if (parts.isEmpty()) {
            throw new IllegalStateException("At least one video is required. Call .videoFile() first.");
        }

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
