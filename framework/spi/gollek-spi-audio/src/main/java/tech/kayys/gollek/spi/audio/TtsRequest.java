package tech.kayys.gollek.spi.audio;

import java.util.Map;

/**
 * Request payload for Text-To-Speech (TTS) synthesis.
 */
public record TtsRequest(
    String text,
    String voice,
    String language,
    float speed,
    float pitch,
    String outputFormat,
    Map<String, Object> options
) {
    public TtsRequest {
        if (text == null || text.isBlank()) throw new IllegalArgumentException("text required");
        if (voice == null || voice.isBlank()) voice = "default";
        if (language == null || language.isBlank()) language = "en";
        if (speed <= 0) speed = 1.0f;
        if (pitch <= 0) pitch = 1.0f;
        if (outputFormat == null || outputFormat.isBlank()) outputFormat = "audio/wav";
        options = options == null ? Map.of() : Map.copyOf(options);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String text;
        private String voice = "default";
        private String language = "en";
        private float speed = 1.0f;
        private float pitch = 1.0f;
        private String outputFormat = "audio/wav";
        private Map<String, Object> options = Map.of();

        public Builder text(String text) { this.text = text; return this; }
        public Builder voice(String voice) { this.voice = voice; return this; }
        public Builder language(String language) { this.language = language; return this; }
        public Builder speed(float speed) { this.speed = speed; return this; }
        public Builder pitch(float pitch) { this.pitch = pitch; return this; }
        public Builder outputFormat(String format) { this.outputFormat = format; return this; }
        public Builder options(Map<String, Object> options) { this.options = options; return this; }

        public TtsRequest build() {
            return new TtsRequest(text, voice, language, speed, pitch, outputFormat, options);
        }
    }
}
