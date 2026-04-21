package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import tech.kayys.gollek.cli.commands.*;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelInfo;

import java.util.List;

/**
 * Handles slash commands within the chat session.
 */
@Dependent
public class ChatCommandHandler {

    @Inject
    ListCommand listCommand;
    @Inject
    ProvidersCommand providersCommand;
    @Inject
    InfoCommand infoCommand;
    @Inject
    ExtensionsCommand extensionsCommand;
    @Inject
    GollekSdk sdk;
    @Inject
    tech.kayys.gollek.log.service.LogParsingService logParsingService;

    @org.eclipse.microprofile.config.inject.ConfigProperty(name = "quarkus.log.file.path")
    String logFilePath;

    public boolean handleCommand(String input, ChatSessionManager session, ChatUIRenderer ui) {
        String cmd = input.toLowerCase().trim();

        if (cmd.equals("/reset")) {
            session.reset();
            System.out.println(ChatUIRenderer.YELLOW + "[Conversation reset]" + ChatUIRenderer.RESET);
            return true;
        }

        if (cmd.equals("/help")) {
            printHelp();
            return true;
        }

        if (cmd.equals("/list")) {
            listCommand.run();
            return true;
        }

        if (cmd.equals("/providers")) {
            providersCommand.run();
            return true;
        }

        if (cmd.startsWith("/provider ")) {
            handleProviderSwitch(cmd.substring(10).trim(), session, ui);
            return true;
        }

        if (cmd.equals("/info")) {
            infoCommand.run();
            return true;
        }

        if (cmd.equals("/extensions")) {
            extensionsCommand.run();
            return true;
        }

        if (cmd.equals("/log")) {
            printLogs();
            return true;
        }

        if (cmd.equals("/models")) {
            handleListModels();
            return true;
        }

        if (cmd.startsWith("/model ")) {
            handleModelSwitch(cmd.substring(7).trim(), session, ui);
            return true;
        }

        if (cmd.equals("/model")) {
            System.out.println(ChatUIRenderer.YELLOW + "Usage: /model <model-id>" + ChatUIRenderer.RESET);
            System.out.println(ChatUIRenderer.DIM + "Use /models to see available models" + ChatUIRenderer.RESET);
            return true;
        }

        if (cmd.equals("/stats") || cmd.equals("/audit") || cmd.equals("/statistic") || cmd.equals("/statistics")) {
            handleStats(session);
            return true;
        }

        return false;
    }

    private void printHelp() {
        System.out.println(ChatUIRenderer.DIM + "Available commands:" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /reset        - Clear conversation history" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /quit         - Exit the chat session" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /log          - Show last 100 lines of log" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /list         - List available models" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /models       - List models for the current provider" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /model <id>   - Switch to a different model" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /providers    - List available LLM providers" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /provider <id>- Switch to a different provider" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /info         - Display system info" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /stats        - Show session usage statistics" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /extensions   - Show packaged extension modules" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /help         - Show this help message" + ChatUIRenderer.RESET);
    }

    private void handleProviderSwitch(String newProviderId, ChatSessionManager session, ChatUIRenderer ui) {
        if (newProviderId.isEmpty()) {
            System.out.println(ChatUIRenderer.YELLOW + "Usage: /provider <provider-id>" + ChatUIRenderer.RESET);
            return;
        }
        try {
            session.switchProvider(newProviderId);
            System.out.println(ChatUIRenderer.GREEN + "Switched to provider: " + ChatUIRenderer.RESET + ChatUIRenderer.CYAN + newProviderId + ChatUIRenderer.RESET);
        } catch (Exception e) {
            ui.printError("Failed to switch provider: " + e.getMessage(), false);
        }
    }

    private void handleListModels() {
        try {
            List<ModelInfo> models = sdk.listModels(0, 50);
            if (models.isEmpty()) {
                System.out.println(ChatUIRenderer.YELLOW + "No models found." + ChatUIRenderer.RESET);
                return;
            }
            System.out.println(ChatUIRenderer.DIM + "Available models:" + ChatUIRenderer.RESET);
            System.out.printf(ChatUIRenderer.DIM + "  %-30s %-12s %-10s" + ChatUIRenderer.RESET + "%n", "MODEL", "SIZE", "FORMAT");
            System.out.println(ChatUIRenderer.DIM + "  " + "-".repeat(55) + ChatUIRenderer.RESET);
            for (ModelInfo m : models) {
                System.out.printf("  " + ChatUIRenderer.CYAN + "%-30s" + ChatUIRenderer.RESET + " %-12s %-10s%n",
                        truncate(m.getModelId(), 30),
                        m.getSizeFormatted(),
                        m.getFormat() != null ? m.getFormat() : "N/A");
            }
            System.out.printf(ChatUIRenderer.DIM + "  %d model(s) found" + ChatUIRenderer.RESET + "%n", models.size());
        } catch (Exception e) {
            System.err.println(ChatUIRenderer.YELLOW + "Failed to list models: " + e.getMessage() + ChatUIRenderer.RESET);
        }
    }

    private void handleModelSwitch(String newModelId, ChatSessionManager session, ChatUIRenderer ui) {
        if (newModelId.isEmpty()) {
            System.out.println(ChatUIRenderer.YELLOW + "Usage: /model <model-id>" + ChatUIRenderer.RESET);
            return;
        }
        try {
            session.switchModel(newModelId);
            System.out.println(ChatUIRenderer.GREEN + "Switched to model: " + ChatUIRenderer.RESET + ChatUIRenderer.CYAN + newModelId + ChatUIRenderer.RESET);
        } catch (Exception e) {
            ui.printError("Failed to switch model: " + e.getMessage(), false);
        }
    }

    private String truncate(String str, int maxLen) {
        if (str == null) return "";
        return str.length() > maxLen ? str.substring(0, maxLen - 3) + "..." : str;
    }

    private void printLogs() {
        try {
            java.nio.file.Path path = java.nio.file.Path.of(logFilePath);
            if (java.nio.file.Files.exists(path)) {
                System.out.println(ChatUIRenderer.DIM + "--- Processing logs from: " + logFilePath + " ---" + ChatUIRenderer.RESET);
                
                // Use the service to parse the file - taking last 100 entries
                java.util.List<tech.kayys.gollek.log.SimplifiedLog> logs = logParsingService.parseLogFile(path, true, false)
                        .await().indefinitely();
                
                int start = Math.max(0, logs.size() - 50);
                java.util.List<tech.kayys.gollek.log.SimplifiedLog> recent = logs.subList(start, logs.size());

                if (!recent.isEmpty()) {
                    for (tech.kayys.gollek.log.SimplifiedLog logEntry : recent) {
                        System.out.println(logEntry.toString());
                    }
                    System.out.println(ChatUIRenderer.DIM + "--------------------------" + ChatUIRenderer.RESET);
                } else {
                    System.out.println(ChatUIRenderer.YELLOW + "No readable JSON logs found yet. Perform some actions first." + ChatUIRenderer.RESET);
                }
            } else {
                System.out.println(ChatUIRenderer.YELLOW + "Log file not found at: " + logFilePath + ChatUIRenderer.RESET);
            }
        } catch (Exception e) {
             System.err.println(ChatUIRenderer.YELLOW + "Failed to retrieve logs: " + e.getMessage() + ChatUIRenderer.RESET);
             // Fallback to raw logs if parsing fails completely
             try {
                 java.util.List<String> rawLines = sdk.getRecentLogs(20);
                 if (!rawLines.isEmpty()) {
                     System.out.println(ChatUIRenderer.DIM + "--- Raw Log Fallback ---" + ChatUIRenderer.RESET);
                     rawLines.forEach(System.out::println);
                 }
             } catch (Exception ignored) {}
        }
    }

    private void handleStats(ChatSessionManager session) {
        var stats = session.getSessionStats();

        System.out.println();
        System.out.println(ChatUIRenderer.BOLD + "┌──────────────────── Session Statistics ────────────────────┐" + ChatUIRenderer.RESET);
        System.out.printf("│ %-22s │ %-33s │%n", "Session started", stats.sessionStart().toString().substring(0, 19));
        System.out.printf("│ %-22s │ %-33s │%n", "Duration", formatDuration(stats.sessionDurationSeconds()));
        System.out.printf("│ %-22s │ %-33d │%n", "Total requests", stats.totalRequests());
        System.out.printf("│ %-22s │ %-33d │%n", "Total tokens", stats.totalTokens());
        System.out.printf("│ %-22s │ %-33d │%n", "Total errors", stats.totalErrors());
        System.out.printf("│ %-22s │ %-33.1f │%n", "Avg tokens/request", stats.avgTokensPerRequest());
        System.out.printf("│ %-22s │ %-30.2f t/s │%n", "Avg speed", stats.avgTokensPerSecond());
        System.out.printf("│ %-22s │ %-30.2f s   │%n", "Total inference time", stats.totalDurationMs() / 1000.0);
        System.out.printf("│ %-22s │ %-33s │%n", "Avg TTFT", stats.avgTtftMs() + " ms");
        System.out.printf("│ %-22s │ %-33s │%n", "Avg TPOT", stats.avgTpotMs() + " ms");
        System.out.printf("│ %-22s │ %-33s │%n", "Avg ITL", stats.avgItlMs() + " ms");

        if (!stats.perModelStats().isEmpty()) {
            System.out.println("├──────────────────── Per Model ─────────────────────────────┤");
            System.out.printf("│ %-22s │ %6s %8s %6s          │%n", "Model", "Reqs", "Tokens", "Errs");
            System.out.println("│──────────────────────────────────────────────────────────── │");
            stats.perModelStats().forEach((model, s) ->
                    System.out.printf("│ %-22s │ %6d %8d %6d          │%n",
                            truncate(model, 22), s[0], s[1], s[2]));
        }

        if (!stats.perProviderStats().isEmpty()) {
            System.out.println("├──────────────────── Per Provider ──────────────────────────┤");
            System.out.printf("│ %-22s │ %6s %8s %6s          │%n", "Provider", "Reqs", "Tokens", "Errs");
            System.out.println("│──────────────────────────────────────────────────────────── │");
            stats.perProviderStats().forEach((provider, s) ->
                    System.out.printf("│ %-22s │ %6d %8d %6d          │%n",
                            truncate(provider, 22), s[0], s[1], s[2]));
        }

        System.out.println("└────────────────────────────────────────────────────────────┘");
        System.out.println();
    }

    private String formatDuration(long seconds) {
        if (seconds < 60) return seconds + "s";
        if (seconds < 3600) return (seconds / 60) + "m " + (seconds % 60) + "s";
        return (seconds / 3600) + "h " + ((seconds % 3600) / 60) + "m";
    }
}
