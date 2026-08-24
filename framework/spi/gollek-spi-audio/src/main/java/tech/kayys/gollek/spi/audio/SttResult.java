package tech.kayys.gollek.spi.audio;

import java.util.List;
import java.util.Map;

/**
 * Result payload from Speech-To-Text transcription.
 */
public record SttResult(
    String text,
    String detectedLanguage,
    double confidence,
    List<Segment> segments,
    long processingTimeMs,
    String modelId,
    Map<String, Object> metadata
) {
    public SttResult {
        if (text == null) text = "";
        if (detectedLanguage == null) detectedLanguage = "unknown";
        segments = segments == null ? List.of() : List.copyOf(segments);
        if (modelId == null) modelId = "unknown";
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public record Segment(
        long startMs,
        long endMs,
        String text,
        double confidence
    ) {}
}
