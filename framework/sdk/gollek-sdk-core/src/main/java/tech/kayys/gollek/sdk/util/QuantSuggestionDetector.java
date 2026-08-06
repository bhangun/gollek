package tech.kayys.gollek.sdk.util;

import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Smart detector that analyzes model size from name patterns and file size.
 */
public final class QuantSuggestionDetector {

    private QuantSuggestionDetector() {}

    // Matches patterns like "7B", "13b", "70B", "0.5B", "1.5b", "72b"
    private static final Pattern PARAM_PATTERN = Pattern.compile(
            "(?:^|[-_./])([0-9]+(?:\\.[0-9]+)?)[Bb](?:[-_./]|$)");

    // FP16 size thresholds (bytes)
    private static final long THRESHOLD_STRONG_BYTES = 12L * 1024 * 1024 * 1024;  // ~12 GB → likely 7B+
    private static final long THRESHOLD_MILD_BYTES = 5L * 1024 * 1024 * 1024;     // ~5 GB → likely 3B+

    // Parameter count thresholds
    private static final double THRESHOLD_STRONG_PARAMS = 7.0;  // 7B
    private static final double THRESHOLD_MILD_PARAMS = 3.0;    // 3B

    public record QuantSuggestion(
        double estimatedParams,
        boolean stronglyRecommended,
        String recommendedStrategy,
        int recommendedBits
    ) {
        public boolean isSuggested() {
            return estimatedParams >= THRESHOLD_MILD_PARAMS;
        }
    }

    /**
     * Detect model size and return a quantization suggestion if appropriate.
     */
    public static Optional<QuantSuggestion> detect(String modelId, Long sizeBytes) {
        double estimatedParams = parseParamCount(modelId);

        if (estimatedParams <= 0 && sizeBytes != null && sizeBytes > 0) {
            if (sizeBytes >= THRESHOLD_STRONG_BYTES) {
                estimatedParams = THRESHOLD_STRONG_PARAMS;
            } else if (sizeBytes >= THRESHOLD_MILD_BYTES) {
                estimatedParams = THRESHOLD_MILD_PARAMS;
            }
        }

        if (estimatedParams >= THRESHOLD_MILD_PARAMS) {
            boolean strong = estimatedParams >= THRESHOLD_STRONG_PARAMS;
            String strategy = estimatedParams >= 13 ? "bnb" : "turbo";
            int bits = 4;
            return Optional.of(new QuantSuggestion(estimatedParams, strong, strategy, bits));
        }

        return Optional.empty();
    }

    public static double parseParamCount(String modelId) {
        if (modelId == null) return -1;
        Matcher m = PARAM_PATTERN.matcher(modelId);
        double largest = -1;
        while (m.find()) {
            try {
                double val = Double.parseDouble(m.group(1));
                if (val > largest) largest = val;
            } catch (NumberFormatException ignored) {}
        }
        return largest;
    }
}
