package tech.kayys.gollek.cli.util;

import tech.kayys.gollek.cli.chat.ChatUIRenderer;

import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Smart detector that analyzes model size from name patterns and file size,
 * then suggests quantization when appropriate.
 * <p>
 * Detection sources (in priority order):
 * <ol>
 *   <li>Model name heuristics: "7B", "13B", "70B", "0.5B", "1.5B" etc.</li>
 *   <li>File/directory size on disk</li>
 * </ol>
 *
 * <p>
 * Thresholds:
 * <ul>
 *   <li>&ge; 7B params (~14 GB FP16): strong suggestion to quantize</li>
 *   <li>&ge; 3B params (~6 GB FP16): mild suggestion for memory-constrained devices</li>
 *   <li>&lt; 3B params: no suggestion</li>
 * </ul>
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

    /**
     * Detect model size and print a quantization suggestion if appropriate.
     *
     * @param modelId       the model name or HuggingFace ID
     * @param modelPath     local path to the model (may be null)
     * @param quantizeFlag  current --quantize value (null if not set)
     * @param quiet         suppress output
     * @return true if a strong suggestion was made (7B+)
     */
    public static boolean suggestIfNeeded(String modelId, String modelPath,
                                          String quantizeFlag, boolean quiet) {
        if (quiet) return false;

        // Skip if user already enabled quantization
        if (quantizeFlag != null && !quantizeFlag.isBlank()) return false;

        // 1. Try to detect from model name
        double estimatedParams = parseParamCount(modelId);

        // 2. If name parsing fails, try file size
        if (estimatedParams <= 0 && modelPath != null) {
            long sizeBytes = calculateSize(modelPath);
            if (sizeBytes >= THRESHOLD_STRONG_BYTES) {
                estimatedParams = THRESHOLD_STRONG_PARAMS;
            } else if (sizeBytes >= THRESHOLD_MILD_BYTES) {
                estimatedParams = THRESHOLD_MILD_PARAMS;
            }
        }

        // 3. Decide suggestion level
        if (estimatedParams >= THRESHOLD_STRONG_PARAMS) {
            printStrongSuggestion(modelId, estimatedParams);
            return true;
        } else if (estimatedParams >= THRESHOLD_MILD_PARAMS) {
            printMildSuggestion(modelId, estimatedParams);
            return false;
        }

        return false;
    }

    /**
     * Parse parameter count (in billions) from a model name.
     * Examples: "Qwen2.5-7B-Instruct" → 7.0, "Llama-3.1-70B" → 70.0,
     *           "Phi-3.5-mini-3.8B"  → 3.8, "gemma-2b" → 2.0
     */
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

    private static void printStrongSuggestion(String modelId, double params) {
        String Y = ChatUIRenderer.YELLOW;
        String B = ChatUIRenderer.BOLD;
        String C = ChatUIRenderer.CYAN;
        String D = ChatUIRenderer.DIM;
        String R = ChatUIRenderer.RESET;

        long estimatedMemGB = Math.round(params * 2); // FP16: ~2 GB per 1B params
        String strategy = params >= 13 ? "bnb" : "turbo";
        int bits = params >= 30 ? 4 : 4;

        System.out.println();
        System.out.println(Y + B + "⚡ Large model detected: " + R + C + String.format("%.1fB parameters", params) + R);
        System.out.println(Y + "   Estimated FP16 memory: ~" + estimatedMemGB + " GB" + R);
        System.out.println(Y + "   Quantization is strongly recommended to reduce memory usage." + R);
        System.out.println();
        System.out.println(D + "   Add " + R + C + "--quantize " + strategy + R + D + " to enable " + bits + "-bit quantization:" + R);
        System.out.println(D + "   Example: " + R + C + "gollek run --model " + shortenModelId(modelId) + " --quantize " + strategy + R);
        System.out.println(D + "   Or pre-quantize: " + R + C + "gollek quantize --model " + shortenModelId(modelId) + " --strategy " + strategy + R);
        System.out.println();
    }

    private static void printMildSuggestion(String modelId, double params) {
        String D = ChatUIRenderer.DIM;
        String C = ChatUIRenderer.CYAN;
        String R = ChatUIRenderer.RESET;

        System.out.println(D + "💡 Tip: Model is ~" + String.format("%.1fB", params) +
                " params. Consider " + C + "--quantize bnb" + D +
                " for lower memory usage." + R);
    }

    private static long calculateSize(String path) {
        Path p = Path.of(path);
        if (!Files.exists(p)) return 0;

        if (Files.isRegularFile(p)) {
            try { return Files.size(p); } catch (IOException e) { return 0; }
        }

        AtomicLong size = new AtomicLong(0);
        try {
            Files.walkFileTree(p, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                    String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
                    if (name.endsWith(".safetensors") || name.endsWith(".bin") ||
                        name.endsWith(".gguf") || name.endsWith(".pt")) {
                        size.addAndGet(attrs.size());
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException ignored) {}
        return size.get();
    }

    private static String shortenModelId(String modelId) {
        // Truncate long paths for display
        if (modelId != null && modelId.length() > 40) {
            return "..." + modelId.substring(modelId.length() - 37);
        }
        return modelId;
    }
}
