package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import tech.kayys.gollek.cli.commands.*;
import tech.kayys.gollek.sdk.core.GollekSdk;

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

        return false;
    }

    private void printHelp() {
        System.out.println(ChatUIRenderer.DIM + "Available commands:" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /reset      - Clear conversation history" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /quit       - Exit the chat session" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /log        - Show last 100 lines of log" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /list       - List available models" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /providers  - List available LLM providers" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /provider <id> - Switch to a different provider" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /info       - Display system info" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /extensions - Show packaged extension modules" + ChatUIRenderer.RESET);
        System.out.println(ChatUIRenderer.DIM + "  /help       - Show this help message" + ChatUIRenderer.RESET);
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

    private void printLogs() {
        try {
            java.util.List<String> lines = sdk.getRecentLogs(100);
            if (!lines.isEmpty()) {
                System.out.println(ChatUIRenderer.DIM + "--- Last 100 log lines ---" + ChatUIRenderer.RESET);
                for (String line : lines) {
                    System.out.println(ChatUIRenderer.DIM + line + ChatUIRenderer.RESET);
                }
                System.out.println(ChatUIRenderer.DIM + "--------------------------" + ChatUIRenderer.RESET);
            } else {
                System.out.println(ChatUIRenderer.YELLOW + "No recent logs found." + ChatUIRenderer.RESET);
            }
        } catch (Exception e) {
             System.err.println(ChatUIRenderer.YELLOW + "Failed to retrieve logs: " + e.getMessage() + ChatUIRenderer.RESET);
        }
    }
}
