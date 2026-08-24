package tech.kayys.gollek.spi.audio;

import java.util.List;
import java.util.Map;

/**
 * Request payload for Speech-To-Text (STT) transcription (Whisper batch or realtime stream chunk).
 */
public record SttRequest(
    byte[] audioData,
    String mimeType,
    String language,
    String initialPrompt,
    float temperature,
    boolean wordTimestamps,
    Map<String, Object> options
) {
    public SttRequest {
        audioData = audioData == null ? new byte[0] : audioData.clone();
        if (mimeType == null || mimeType.isBlank()) mimeType = "audio/wav";
        if (language == null) language = "auto";
        if (initialPrompt == null) initialPrompt = "";
        options = options == null ? Map.of() : Map.copyOf(options);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private byte[] audioData;
        private String mimeType = "audio/wav";
        private String language = "auto";
        private String initialPrompt = "";
        private float temperature = 0.0f;
        private boolean wordTimestamps = false;
        private Map<String, Object> options = Map.of();

        public Builder audioData(byte[] audioData) { this.audioData = audioData; return this; }
        public Builder mimeType(String mimeType) { this.mimeType = mimeType; return this; }
        public Builder language(String language) { this.language = language; return this; }
        public Builder initialPrompt(String prompt) { this.initialPrompt = prompt; return this; }
        public Builder temperature(float temperature) { this.temperature = temperature; return this; }
        public Builder wordTimestamps(boolean wordTimestamps) { this.wordTimestamps = wordTimestamps; return this; }
        public Builder options(Map<String, Object> options) { this.options = options; return this; }

        public SttRequest build() {
            return new SttRequest(audioData, mimeType, language, initialPrompt, temperature, wordTimestamps, options);
        }
    }
}
