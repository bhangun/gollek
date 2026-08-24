package tech.kayys.gollek.spi.audio;

import java.util.Map;

/**
 * Result payload containing synthesized audio from TTS.
 */
public record TtsResult(
    byte[] audioData,
    String mimeType,
    long durationMs,
    String modelId,
    Map<String, Object> metadata
) {
    public TtsResult {
        audioData = audioData == null ? new byte[0] : audioData.clone();
        if (mimeType == null) mimeType = "audio/wav";
        if (modelId == null) modelId = "unknown";
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
