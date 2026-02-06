package com.kmpchatbot.server.service;

import com.kmpchatbot.server.client.AnthropicClient;
import com.kmpchatbot.server.client.AnthropicDTO;
import com.kmpchatbot.server.domain.Conversation;
import com.kmpchatbot.server.domain.Message;
import com.kmpchatbot.server.dto.ChatRequest;
import com.kmpchatbot.server.dto.ChatResponse;
import com.kmpchatbot.server.dto.ConversationDTO;
import com.kmpchatbot.server.repository.ConversationRepository;
import com.kmpchatbot.server.repository.MessageRepository;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@ApplicationScoped
public class ChatService {
    
    private static final Logger LOG = Logger.getLogger(ChatService.class);
    
    @Inject
    MessageRepository messageRepository;
    
    @Inject
    ConversationRepository conversationRepository;
    
    @Inject
    @RestClient
    AnthropicClient anthropicClient;
    
    @ConfigProperty(name = "anthropic.api.key")
    String apiKey;
    
    @ConfigProperty(name = "anthropic.api.version")
    String apiVersion;
    
    @ConfigProperty(name = "anthropic.api.model")
    String model;
    
    @ConfigProperty(name = "anthropic.api.max-tokens")
    Integer maxTokens;
    
    @Transactional
    public ChatResponse sendMessage(ChatRequest request) {
        String sessionId = request.getSessionId();
        if (sessionId == null || sessionId.isEmpty()) {
            sessionId = UUID.randomUUID().toString();
        }
        
        LOG.infof("Processing message for session: %s", sessionId);
        
        // Find or create conversation
        Conversation conversation = conversationRepository.findOrCreate(sessionId);
        
        // Save user message
        Message userMessage = new Message();
        userMessage.setSessionId(sessionId);
        userMessage.setContent(request.getMessage());
        userMessage.setRole(Message.MessageRole.USER);
        conversation.addMessage(userMessage);
        messageRepository.persist(userMessage);
        
        // Get conversation history
        List<Message> history = messageRepository.findBySessionId(sessionId);
        
        // Build AI request
        List<AnthropicDTO.AnthropicMessage> messages = history.stream()
                .filter(m -> m.getRole() != Message.MessageRole.SYSTEM)
                .map(m -> AnthropicDTO.AnthropicMessage.builder()
                        .role(m.getRole() == Message.MessageRole.USER ? "user" : "assistant")
                        .content(m.getContent())
                        .build())
                .collect(Collectors.toList());
        
        AnthropicDTO.AnthropicRequest aiRequest = AnthropicDTO.AnthropicRequest.builder()
                .model(model)
                .maxTokens(maxTokens)
                .messages(messages)
                .build();
        
        // Call AI API
        AnthropicDTO.AnthropicResponse aiResponse;
        try {
            aiResponse = anthropicClient.sendMessage(apiKey, apiVersion, aiRequest);
        } catch (Exception e) {
            LOG.errorf(e, "Error calling Anthropic API");
            throw new RuntimeException("Failed to get AI response: " + e.getMessage());
        }
        
        // Extract response text
        String responseText = aiResponse.getContent().stream()
                .filter(c -> "text".equals(c.getType()))
                .findFirst()
                .map(AnthropicDTO.ContentBlock::getText)
                .orElse("No response");
        
        // Save assistant message
        Message assistantMessage = new Message();
        assistantMessage.setSessionId(sessionId);
        assistantMessage.setContent(responseText);
        assistantMessage.setRole(Message.MessageRole.ASSISTANT);
        conversation.addMessage(assistantMessage);
        messageRepository.persist(assistantMessage);
        
        conversationRepository.persist(conversation);
        
        return ChatResponse.builder()
                .id(assistantMessage.getId())
                .message(responseText)
                .role("assistant")
                .sessionId(sessionId)
                .timestamp(assistantMessage.getTimestamp())
                .build();
    }
    
    public List<ChatResponse> getHistory(String sessionId) {
        return messageRepository.findBySessionId(sessionId).stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }
    
    public List<ConversationDTO> getAllConversations() {
        return conversationRepository.listAll().stream()
                .map(this::toConversationDTO)
                .collect(Collectors.toList());
    }
    
    public ConversationDTO getConversation(String sessionId) {
        return conversationRepository.findBySessionId(sessionId)
                .map(this::toConversationDTO)
                .orElse(null);
    }
    
    @Transactional
    public void deleteConversation(String sessionId) {
        conversationRepository.findBySessionId(sessionId)
                .ifPresent(conversation -> {
                    messageRepository.deleteBySessionId(sessionId);
                    conversationRepository.delete(conversation);
                });
    }
    
    private ChatResponse toDTO(Message message) {
        return ChatResponse.builder()
                .id(message.getId())
                .message(message.getContent())
                .role(message.getRole().name().toLowerCase())
                .sessionId(message.getSessionId())
                .timestamp(message.getTimestamp())
                .build();
    }
    
    private ConversationDTO toConversationDTO(Conversation conversation) {
        List<ChatResponse> messages = conversation.getMessages().stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
        
        return ConversationDTO.builder()
                .id(conversation.getId())
                .sessionId(conversation.getSessionId())
                .title(conversation.getTitle())
                .createdAt(conversation.getCreatedAt())
                .updatedAt(conversation.getUpdatedAt())
                .messages(messages)
                .messageCount(messages.size())
                .build();
    }
}
