package com.kmpchatbot.server.service;

import com.kmpchatbot.server.domain.Message;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.ws.rs.client.Client;
import jakarta.ws.rs.client.ClientBuilder;
import jakarta.ws.rs.client.Entity;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@ApplicationScoped
public class AIService {

    private static final Logger LOG = Logger.getLogger(AIService.class);

    @ConfigProperty(name = "ai.api.base-url")
    String apiBaseUrl;

    @ConfigProperty(name = "ai.api.key")
    String apiKey;

    @ConfigProperty(name = "ai.api.model")
    String model;

    @ConfigProperty(name = "ai.api.max-tokens")
    int maxTokens;

    private final Client client;

    public AIService() {
        this.client = ClientBuilder.newClient();
    }

    public String sendMessage(String userMessage, List<Message> conversationHistory) {
        try {
            // Build messages array from conversation history
            List<Map<String, String>> messages = conversationHistory.stream()
                    .filter(m -> m.getRole() != Message.MessageRole.SYSTEM)
                    .map(m -> Map.of(
                            "role", m.getRole() == Message.MessageRole.USER ? "user" : "assistant",
                            "content", m.getContent()
                    ))
                    .collect(Collectors.toList());

            // Add current user message
            messages.add(Map.of("role", "user", "content", userMessage));

            // Build request payload
            Map<String, Object> requestBody = Map.of(
                    "model", model,
                    "messages", messages,
                    "max_tokens", maxTokens
            );

            LOG.debugf("Sending request to AI API: %s", apiBaseUrl);

            // Make API call
            Response response = client
                    .target(apiBaseUrl + "/messages")
                    .request(MediaType.APPLICATION_JSON)
                    .header("x-api-key", apiKey)
                    .header("anthropic-version", "2023-06-01")
                    .post(Entity.json(requestBody));

            if (response.getStatus() == 200) {
                Map<String, Object> responseData = response.readEntity(Map.class);
                List<Map<String, Object>> content = (List<Map<String, Object>>) responseData.get("content");
                
                if (content != null && !content.isEmpty()) {
                    return (String) content.get(0).get("text");
                }
            } else {
                String error = response.readEntity(String.class);
                LOG.errorf("AI API error (status %d): %s", response.getStatus(), error);
                throw new RuntimeException("AI API returned error: " + error);
            }

        } catch (Exception e) {
            LOG.error("Error calling AI API", e);
            throw new RuntimeException("Failed to get AI response: " + e.getMessage(), e);
        }

        return "Sorry, I couldn't process your request at this time.";
    }

    public String sendSimpleMessage(String message) {
        return sendMessage(message, List.of());
    }
}
