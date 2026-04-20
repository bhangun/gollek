package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import java.util.List;
import java.util.Map;

/**
 * Handles terminal output rendering and colors.
 * Respects the NO_COLOR env var (https://no-color.org) and --no-color flag.
 */
@Dependent
public class ChatUIRenderer {

    // Populated at init time; empty strings when color is disabled.
    public static String RESET  = "\u001B[0m";
    public static String BOLD   = "\u001B[1m";
    public static String YELLOW = "\u001B[33m";
    public static String CYAN   = "\u001B[36m";
    public static String GREEN  = "\u001B[32m";
    public static String RED    = "\u001B[31m";
    public static String DIM    = "\u001B[2m";

    static {
        if (System.getenv("NO_COLOR") != null) disableColor();
    }

    public static void disableColor() {
        RESET = BOLD = YELLOW = CYAN = GREEN = RED = DIM = "";
    }

    private boolean jsonMode = false;
    
    public void setJsonMode(boolean jsonMode) {
        this.jsonMode = jsonMode;
    }

    public void printBanner() {
        if (jsonMode) return;
        System.out.println(BOLD + YELLOW + "  _____       _  _      _    " + RESET);
        System.out.println(BOLD + YELLOW + " / ____|     | || |    | |   " + RESET);
        System.out.println(BOLD + YELLOW + "| |  __  ___ | || | ___| | __" + RESET);
        System.out.println(BOLD + YELLOW + "| | |_ |/ _ \\| || |/ _ \\ |/ /" + RESET);
        System.out.println(BOLD + YELLOW + "| |__| | (_) | || |  __/   < " + RESET);
        System.out.println(BOLD + YELLOW + " \\_____|\\___/|_||_|\\___|_|\\_\\" + RESET);
        System.out.println();
    }

    public void printModelInfo(String modelId, String providerId, String format, String outputFile, boolean isInteractive) {
        if (jsonMode) return;
        System.out.printf(BOLD + "Model: " + RESET + CYAN + "%s" + RESET + "%n", modelId);
        String provStr = providerId != null ? providerId : "auto-select";
        if (format != null && !format.isBlank()) {
            System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s, format=%s" + RESET + "%n", provStr, format.toLowerCase());
        } else {
            System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s" + RESET + "%n", provStr);
        }
        if (outputFile != null) {
            System.out.printf(BOLD + "Output: " + RESET + YELLOW + "%s" + RESET + "%n", outputFile);
        }
        if (isInteractive) {
            System.out.println(DIM + "Commands: 'exit' to quit, '/reset' to clear history." + RESET);
        }
        System.out.println(DIM + "-".repeat(50) + RESET);
    }

    public void printAssistantPrefix(boolean quiet, boolean streaming) {
        if (jsonMode) return;
        if (!quiet) {
            if (!streaming) System.out.print("\n");
            System.out.print(BOLD + GREEN + "Assistant: " + RESET);
            System.out.flush();
        }
    }

    public void printStats(int tokens, double duration, double tps, boolean quiet) {
        if (jsonMode) return;
        if (!quiet) {
            System.out.printf(DIM + "\n[Chunks: %d, Duration: %.2fs, Speed: %.2f t/s]" + RESET + "%n",
                    tokens, duration, tps);
        }
    }

    public void printBenchmarks(Map<String, Object> metadata, boolean quiet) {
        if (jsonMode || quiet || metadata == null || metadata.isEmpty()) return;
        
        System.out.println(DIM + "Performance Metrics (llama.cpp style):" + RESET);
        
        Double loadMs = (Double) metadata.get("bench.load_ms");
        Double prefillTps = (Double) metadata.get("bench.prefill_tps");
        Double genTps = (Double) metadata.get("bench.generation_tps");
        Double ttftMs = (Double) metadata.get("bench.ttft_ms");
        Integer inputTokens = (Integer) metadata.get("tokens.input");
        Integer outputTokens = (Integer) metadata.get("tokens.output");

        if (loadMs != null) System.out.printf(DIM + "  load time      = %9.2f ms" + RESET + "%n", loadMs);
        if (prefillTps != null && inputTokens != null) {
            System.out.printf(DIM + "  prompt eval    = %9.2f t/s (%d tokens)" + RESET + "%n", prefillTps, inputTokens);
        }
        if (genTps != null && outputTokens != null) {
            System.out.printf(DIM + "  generation     = %9.2f t/s (%d tokens)" + RESET + "%n", genTps, outputTokens);
        }
        if (ttftMs != null) System.out.printf(DIM + "  latency (ttft) = %9.2f ms" + RESET + "%n", ttftMs);
        System.out.println();
    }

    public String getPrompt(boolean quiet) {
        return quiet ? "\n>>> " : "\n" + BOLD + CYAN + ">>> " + RESET;
    }

    public String getSecondaryPrompt(boolean quiet) {
        return quiet ? "... " : DIM + "... " + RESET;
    }

    public void printGoodbye(boolean quiet) {
        if (!quiet) System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
    }

    public void printError(String message, boolean quiet) {
        System.err.println("\n" + RED + "Error: " + RESET + message);
    }

    public void printWarning(String message, boolean quiet) {
        if (!quiet) System.out.println(YELLOW + "Warning: " + RESET + message);
    }
    
    public void printInfo(String message, boolean quiet) {
        if (!quiet) System.out.println(DIM + message + RESET);
    }

    public void printSuccess(String message, boolean quiet) {
        if (!quiet) System.out.println(GREEN + message + RESET);
    }

    public void printHardwareInfo(String hardware, boolean quiet) {
        if (!quiet) {
            System.out.printf(DIM + "[Running on %s]" + RESET + "%n", hardware);
        }
    }
}
