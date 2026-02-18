package tech.kayys.gollek.mcp.prompt;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.mcp.client.MCPClient;
import tech.kayys.gollek.mcp.dto.MCPConnection;
import tech.kayys.gollek.mcp.dto.MCPPrompt;
import tech.kayys.gollek.mcp.dto.MCPPromptMessage;
import tech.kayys.gollek.mcp.dto.MCPPromptResult;
import tech.kayys.gollek.mcp.dto.MCPResponse;
import tech.kayys.gollek.mcp.exception.PromptExecutionException;
import tech.kayys.gollek.mcp.exception.PromptNotFoundException;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Provider for executing MCP prompts.
 */
@ApplicationScoped
public class MCPPromptProvider {

    private static final Logger LOG = Logger.getLogger(MCPPromptProvider.class);

    @Inject
    MCPClient mcpClient;

    // promptName -> connectionName
    private final Map<String, String> promptToConnection = new ConcurrentHashMap<>();

    // connectionName -> prompts
    private final Map<String, Map<String, MCPPrompt>> promptsByConnection = new ConcurrentHashMap<>();

    /**
     * Register prompts from a connection
     */
    public void registerConnection(MCPConnection connection) {
        String connectionName = connection.getConfig().getName();
        Map<String, MCPPrompt> prompts = connection.getPrompts();

        promptsByConnection.put(connectionName, new ConcurrentHashMap<>(prompts));

        prompts.keySet().forEach(promptName -> promptToConnection.put(promptName, connectionName));

        LOG.infof("Registered %d prompts from connection: %s", prompts.size(), connectionName);
    }

    /**
     * Execute a prompt
     */
    public Uni<MCPPromptResult> executePrompt(
            String promptName,
            Map<String, String> arguments) {
        LOG.debugf("Executing prompt: %s with args: %s", promptName, arguments);

        // Get connection
        String connectionName = promptToConnection.get(promptName);
        if (connectionName == null) {
            return Uni.createFrom().failure(
                    new PromptNotFoundException("Prompt not found: " + promptName));
        }

        Optional<MCPConnection> connectionOpt = mcpClient.getConnection(connectionName);
        if (connectionOpt.isEmpty()) {
            return Uni.createFrom().failure(
                    new IllegalStateException("Connection not found: " + connectionName));
        }

        MCPConnection connection = connectionOpt.get();

        // Execute prompt
        return connection.getPrompt(promptName, arguments)
                .onItem().transform(response -> convertResponse(promptName, response))
                .onFailure().invoke(error -> LOG.errorf(error, "Failed to execute prompt: %s", promptName));
    }

    /**
     * Convert MCP response to prompt result
     */
    private MCPPromptResult convertResponse(String promptName, MCPResponse response) {
        if (!response.isSuccess()) {
            throw new PromptExecutionException(
                    "Failed to execute prompt: " + response.getError());
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) response.getResult();
        String description = (String) result.get("description");

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> messageList = (List<Map<String, Object>>) result.get("messages");

        var builder = MCPPromptResult.builder()
                .promptName(promptName)
                .description(description);

        if (messageList != null) {
            messageList.forEach(msgData -> {
                String roleStr = (String) msgData.get("role");
                MCPPromptMessage.Role role = MCPPromptMessage.Role.valueOf(roleStr.toUpperCase());

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) msgData.get("content");

                List<MCPPromptMessage.Content> contents = new ArrayList<>();
                if (contentList != null) {
                    contentList.forEach(contentData -> {
                        String type = (String) contentData.get("type");
                        String text = (String) contentData.get("text");
                        String data = (String) contentData.get("data");
                        String mimeType = (String) contentData.get("mimeType");

                        contents.add(new MCPPromptMessage.Content(type, text, data, mimeType));
                    });
                }

                builder.message(new MCPPromptMessage(role, contents));
            });
        }

        return builder.build();
    }

    /**
     * Get prompt by name
     */
    public Optional<MCPPrompt> getPrompt(String promptName) {
        String connectionName = promptToConnection.get(promptName);
        if (connectionName == null) {
            return Optional.empty();
        }

        Map<String, MCPPrompt> prompts = promptsByConnection.get(connectionName);
        return Optional.ofNullable(prompts != null ? prompts.get(promptName) : null);
    }

    /**
     * Get all prompts
     */
    public List<MCPPrompt> getAllPrompts() {
        return promptsByConnection.values().stream()
                .flatMap(prompts -> prompts.values().stream())
                .toList();
    }
}