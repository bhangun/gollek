package tech.kayys.gollek.cli.audit;

import tech.kayys.gollek.cli.chat.ChatUIRenderer;

import java.time.Duration;

/**
 * Terminal renderer for quantization audit trail tables.
 * Uses ANSI colors consistent with {@link ChatUIRenderer}.
 */
public final class QuantAuditRenderer {

    private QuantAuditRenderer() {}

    /**
     * Print a rich audit summary table to the terminal.
     */
    public static void printAuditTable(QuantAuditRecord record, String auditFilePath) {
        String B = ChatUIRenderer.BOLD;
        String C = ChatUIRenderer.CYAN;
        String G = ChatUIRenderer.GREEN;
        String Y = ChatUIRenderer.YELLOW;
        String D = ChatUIRenderer.DIM;
        String R = ChatUIRenderer.RESET;

        String border = D + "─".repeat(66) + R;

        System.out.println();
        System.out.println(B + C + "┌" + "─".repeat(66) + "┐" + R);
        System.out.println(B + C + "│" + R + B + center("Quantization Audit Trail", 66) + B + C + "│" + R);
        System.out.println(B + C + "├" + "─".repeat(16) + "┬" + "─".repeat(49) + "┤" + R);

        row("Model", record.modelId(), C);
        row("Strategy", strategyDisplayName(record.strategy()), Y);
        row("Bits", String.valueOf(record.bits()), G);
        row("Group Size", String.valueOf(record.groupSize()), G);
        row("Accelerator", record.accelerator(), C);
        row("Java Version", record.javaVersion() + (record.simdEnabled() ? " + Vector API" : ""), D);
        row("Mode", record.mode().equals("offline") ? "Offline (gollek quantize)" : "JIT (--quantize)", Y);

        System.out.println(B + C + "├" + "─".repeat(16) + "┼" + "─".repeat(49) + "┤" + R);

        row("Original", formatSize(record.originalSizeBytes()) + params(record.parameterCount()), G);
        row("Quantized", formatSize(record.quantizedSizeBytes()), G);
        row("Compression", String.format("%.2fx", record.compressionRatio()), Y + B);
        row("Duration", formatDuration(record.quantizationDuration()), G);
        row("Throughput", formatThroughput(record.paramsPerSecond()), G);

        if (record.tokensPerSecond() > 0) {
            row("Inference", String.format("%.2f t/s", record.tokensPerSecond()), C);
        }

        if (record.registeredInManifest()) {
            System.out.println(B + C + "├" + "─".repeat(16) + "┼" + "─".repeat(49) + "┤" + R);
            row("Manifest", "✓ Registered as new model", G);
            if (record.outputPath() != null) {
                row("Output", shortenPath(record.outputPath()), D);
            }
        }

        System.out.println(B + C + "├" + "─".repeat(16) + "┼" + "─".repeat(49) + "┤" + R);
        row("Audit File", shortenPath(auditFilePath), D);
        row("CSV Export", shortenPath(QuantAuditTrail.csvFile().toString()), D);

        System.out.println(B + C + "└" + "─".repeat(16) + "┴" + "─".repeat(49) + "┘" + R);
        System.out.println();
    }

    /**
     * Print a compact one-line summary (for --quantize on run/chat).
     */
    public static void printCompactSummary(QuantAuditRecord record) {
        String D = ChatUIRenderer.DIM;
        String Y = ChatUIRenderer.YELLOW;
        String G = ChatUIRenderer.GREEN;
        String R = ChatUIRenderer.RESET;

        System.out.printf(D + "[Quant: %s%s %d-bit%s | %s → %s | %s%.2fx%s | %s%.1fs%s]" + R + "%n",
                Y, record.strategy().toUpperCase(), record.bits(), D,
                formatSize(record.originalSizeBytes()),
                formatSize(record.quantizedSizeBytes()),
                G, record.compressionRatio(), D,
                G, record.quantizationDuration().toMillis() / 1000.0, D);
    }

    // ── Formatting helpers ────────────────────────────────────────────

    private static void row(String label, String value, String valueColor) {
        String B = ChatUIRenderer.BOLD;
        String C = ChatUIRenderer.CYAN;
        String R = ChatUIRenderer.RESET;

        System.out.printf(B + C + "│" + R + " " + B + "%-14s" + R +
                        B + C + "│" + R + " " + valueColor + "%-47s" + R +
                        B + C + "│" + R + "%n",
                label, truncate(value, 47));
    }

    private static String center(String text, int width) {
        int padding = (width - text.length()) / 2;
        return " ".repeat(Math.max(0, padding)) + text +
                " ".repeat(Math.max(0, width - text.length() - padding));
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max - 3) + "...";
    }

    private static String shortenPath(String path) {
        if (path == null) return "";
        String home = System.getProperty("user.home");
        if (home != null && path.startsWith(home)) {
            return "~" + path.substring(home.length());
        }
        return path;
    }

    static String formatSize(long bytes) {
        if (bytes <= 0) return "N/A";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private static String params(long count) {
        if (count <= 0) return "";
        if (count < 1_000_000) return String.format(" (%dK params)", count / 1000);
        if (count < 1_000_000_000) return String.format(" (%dM params)", count / 1_000_000);
        return String.format(" (%.1fB params)", count / 1_000_000_000.0);
    }

    private static String formatDuration(java.time.Duration d) {
        long ms = d.toMillis();
        if (ms < 1000) return ms + "ms";
        if (ms < 60_000) return String.format("%.1fs", ms / 1000.0);
        return String.format("%dm %ds", ms / 60_000, (ms % 60_000) / 1000);
    }

    private static String formatThroughput(double paramsPerSec) {
        if (paramsPerSec <= 0) return "N/A";
        if (paramsPerSec < 1_000_000) return String.format("%.1fK params/s", paramsPerSec / 1000);
        return String.format("%.1fM params/s", paramsPerSec / 1_000_000);
    }

    private static String strategyDisplayName(String strategy) {
        if (strategy == null) return "Unknown";
        return switch (strategy.toLowerCase()) {
            case "bnb" -> "BitsAndBytes NF4";
            case "turbo" -> "TurboQuant (SIMD VQ)";
            case "awq" -> "AWQ (Activation-Aware)";
            case "gptq" -> "GPTQ (Hessian)";
            case "autoround" -> "AutoRound (SignSGD)";
            default -> strategy.toUpperCase();
        };
    }
}
