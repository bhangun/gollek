package tech.kayys.gollek.sdk.multimodal;

import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * Fluent builder for audio processing tasks (transcription, translation, understanding).
 *
 * <h3>Example — Transcription</h3>
 * <pre>{@code
 * var result = Gollek.audio("whisper-large-v3")
 *     .audioFile(Path.of("meeting.wav"))
 *     .task("transcription")
 *     .language("en")
 *     .generate();
 *
 * System.out.println(result.text());
 * }</pre>
 *
 * <h3>Example — Audio Understanding</h3>
 * <pre>{@code
 * var result = Gollek.audio("gemini-2.0-flash")
 *     .audioFile(Path.of("song.mp3"))
 *     .prompt("Describe the mood and instruments in this music")
 *     .generate();
 * }</pre>
 */
public final class AudioBuilder {

    private final String model;
    private final GollekSdk sdk;
    private final List<MultimodalContent> parts = new ArrayList<>();
    private String prompt;
    private String task = "transcription";
    private String language;
    private int maxTokens = 4096;
    private double temperature = 0.0;
    private long timeoutMs = 60_000;
    private final Map<String, Object> parameters = new LinkedHashMap<>();

    public AudioBuilder(String model, GollekSdk sdk) {
        this.model = Objects.requireNonNull(model, "model");
        this.sdk = Objects.requireNonNull(sdk, "sdk");
    }

    /**
     * Adds an audio file.
     *
     * @param path path to the audio file (WAV, MP3, FLAC, etc.)
     * @return this builder
     * @throws IOException if the file cannot be read
     */
    public AudioBuilder audioFile(Path path) throws IOException {
        parts.add(ContentPart.audio(path).toSpiContent());
        return this;
    }

    /**
     * Adds audio from raw bytes.
     *
     * @param bytes    audio bytes
     * @param mimeType MIME type (e.g., "audio/wav")
     * @return this builder
     */
    public AudioBuilder audio(byte[] bytes, String mimeType) {
        parts.add(ContentPart.audio(bytes, mimeType).toSpiContent());
        return this;
    }

    /**
     * Adds audio from a URL.
     *
     * @param url URL to the audio resource
     * @return this builder
     */
    public AudioBuilder audioUrl(String url) {
        parts.add(ContentPart.audioUrl(url).toSpiContent());
        return this;
    }

    /**
     * Sets the processing task type.
     *
     * @param task one of "transcription", "translation", "understanding" (default: "transcription")
     * @return this builder
     */
    public AudioBuilder task(String task) {
        this.task = task;
        return this;
    }

    /**
     * Sets a text prompt for audio understanding tasks.
     *
     * @param prompt instruction text
     * @return this builder
     */
    public AudioBuilder prompt(String prompt) {
        this.prompt = prompt;
        return this;
    }

    /**
     * Sets the target language for transcription/translation.
     *
     * @param language ISO 639-1 language code (e.g., "en", "id", "ja")
     * @return this builder
     */
    public AudioBuilder language(String language) {
        this.language = language;
        return this;
    }

    /** Sets max tokens. */
    public AudioBuilder maxTokens(int maxTokens) {
        this.maxTokens = maxTokens;
        return this;
    }

    /** Sets temperature. */
    public AudioBuilder temperature(double temperature) {
        this.temperature = temperature;
        return this;
    }

    /** Sets timeout in milliseconds. */
    public AudioBuilder timeout(long timeoutMs) {
        this.timeoutMs = timeoutMs;
        return this;
    }

    /** Adds a custom parameter. */
    public AudioBuilder parameter(String key, Object value) {
        this.parameters.put(key, value);
        return this;
    }

    /**
     * Executes the audio processing request.
     *
     * @return the multimodal result
     */
    public MultimodalResult generate() {
        if (parts.isEmpty()) {
            throw new IllegalStateException("At least one audio input is required. Call .audioFile() first.");
        }

        List<MultimodalContent> allParts = new ArrayList<>();
        // Build instruction text
        String instruction = buildInstruction();
        if (instruction != null && !instruction.isBlank()) {
            allParts.add(MultimodalContent.ofText(instruction));
        }
        allParts.addAll(parts);

        // Set task-specific parameters
        Map<String, Object> params = new LinkedHashMap<>(parameters);
        params.put("task", task);
        if (language != null) params.put("language", language);

        var outputConfig = MultimodalRequest.OutputConfig.builder()
                .outputModalities(ModalityType.TEXT)
                .maxTokens(maxTokens)
                .temperature(temperature)
                .build();

        var request = MultimodalRequest.builder()
                .model(model)
                .inputs(allParts.toArray(new MultimodalContent[0]))
                .outputConfig(outputConfig)
                .parameters(params)
                .timeoutMs(timeoutMs)
                .build();

        return new MultimodalResult(sdk.processMultimodal(request));
    }

    private String buildInstruction() {
        if (prompt != null && !prompt.isBlank()) return prompt;
        return switch (task) {
            case "transcription" -> language != null
                    ? "Transcribe this audio in " + language + "."
                    : "Transcribe this audio.";
            case "translation" -> language != null
                    ? "Translate this audio to " + language + "."
                    : "Translate this audio to English.";
            default -> null;
        };
    }
}
