package com.kmpchatbot.server.service;

import com.kmpchatbot.server.api.dto.ChatResponse;
import com.kmpchatbot.server.api.dto.MessageResponse;
import com.kmpchatbot.server.domain.Conversation;
import com.kmpchatbot.server.domain.Message;
import com.kmpchatbot.server.domain.User;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.jboss.logging.Logger;

import java.time.LocalDateTime;
import java.util.List;

@ApplicationScoped
public class ChatService {

    private static final Logger LOG = Logger.getLogger(ChatService.class);

    @Inject
    AIService aiService;

    @Transactional
    public ChatResponse processMessage(String userMessage, Long conversationId, Long userId) {
        LOG.debugf("Processing message for user %d, conversation %d", userId, conversationId);

        User user = User.findById(userId);
        if (user == null) {
            throw new RuntimeException("User not found");
        }

        Conversation conversation;
        if (conversationId == null) {
            // Create new conversation
            conversation = new Conversation();
            conversation.setTitle(generateConversationTitle(userMessage));
            conversation.setUser(user);
            conversation.persist();
            LOG.debugf("Created new conversation %d", conversation.getId());
        } else {
            conversation = Conversation.findByIdAndUser(conversationId, userId);
            if (conversation == null) {
                throw new RuntimeException("Conversation not found or access denied");
            }
        }

        // Save user message
        Message userMsg = new Message();
        userMsg.setConversation(conversation);
        userMsg.setContent(userMessage);
        userMsg.setRole(Message.MessageRole.USER);
        userMsg.setTimestamp(LocalDateTime.now());
        userMsg.persist();

        // Get conversation history
        List<Message> history = Message.findByConversation(conversation.getId());

        // Get AI response
        String aiResponse;
        try {
            aiResponse = aiService.sendMessage(userMessage, history);
        } catch (Exception e) {
            LOG.error("Error getting AI response", e);
            aiResponse = "I'm sorry, I encountered an error processing your request. Please try again.";
        }

        // Save assistant message
        Message assistantMsg = new Message();
        assistantMsg.setConversation(conversation);
        assistantMsg.setContent(aiResponse);
        assistantMsg.setRole(Message.MessageRole.ASSISTANT);
        assistantMsg.setTimestamp(LocalDateTime.now());
        assistantMsg.setModelUsed("claude-sonnet-4");
        assistantMsg.persist();

        // Update conversation timestamp
        conversation.setUpdatedAt(LocalDateTime.now());

        // Build response
        return new ChatResponse(
                toMessageResponse(userMsg),
                toMessageResponse(assistantMsg),
                conversation.getId()
        );
    }

    private String generateConversationTitle(String firstMessage) {
        if (firstMessage.length() > 50) {
            return firstMessage.substring(0, 47) + "...";
        }
        return firstMessage;
    }

    private MessageResponse toMessageResponse(Message message) {
        return new MessageResponse(
                message.getId(),
                message.getContent(),
                message.getRole().toString(),
                message.getTimestamp(),
                message.getModelUsed()
        );
    }

    public List<MessageResponse> getConversationMessages(Long conversationId, Long userId) {
        Conversation conversation = Conversation.findByIdAndUser(conversationId, userId);
        if (conversation == null) {
            throw new RuntimeException("Conversation not found or access denied");
        }

        return Message.findByConversation(conversationId).stream()
                .map(this::toMessageResponse)
                .toList();
    }
}
