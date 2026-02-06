package com.kmpchatbot.server.api;

import com.kmpchatbot.server.api.dto.ChatResponse;
import com.kmpchatbot.server.api.dto.MessageRequest;
import com.kmpchatbot.server.api.dto.MessageResponse;
import com.kmpchatbot.server.service.ChatService;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.SecurityContext;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;

import java.util.List;
import java.util.Map;

@Path("/api/chat")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Chat", description = "Chat and messaging endpoints")
public class ChatResource {

    private static final Logger LOG = Logger.getLogger(ChatResource.class);

    @Inject
    ChatService chatService;

    @POST
    @Path("/message")
    @Operation(summary = "Send a chat message", description = "Send a message and get AI response")
    @APIResponse(responseCode = "200", description = "Message sent successfully")
    @APIResponse(responseCode = "400", description = "Invalid request")
    @APIResponse(responseCode = "500", description = "Internal server error")
    public Response sendMessage(
            @Valid MessageRequest request,
            @Context SecurityContext securityContext) {
        
        try {
            // For now, using a default user ID (1)
            // In production, extract from JWT token
            Long userId = 1L;
            
            LOG.debugf("Received message: %s", request.getContent());
            
            ChatResponse response = chatService.processMessage(
                    request.getContent(),
                    request.getConversationId(),
                    userId
            );
            
            return Response.ok(response).build();
            
        } catch (Exception e) {
            LOG.error("Error processing message", e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }

    @GET
    @Path("/conversations/{conversationId}/messages")
    @Operation(summary = "Get conversation messages", description = "Retrieve all messages in a conversation")
    @APIResponse(responseCode = "200", description = "Messages retrieved successfully")
    @APIResponse(responseCode = "404", description = "Conversation not found")
    public Response getConversationMessages(
            @PathParam("conversationId") Long conversationId,
            @Context SecurityContext securityContext) {
        
        try {
            Long userId = 1L; // Extract from JWT in production
            
            List<MessageResponse> messages = chatService.getConversationMessages(conversationId, userId);
            return Response.ok(messages).build();
            
        } catch (Exception e) {
            LOG.error("Error retrieving messages", e);
            return Response.status(Response.Status.NOT_FOUND)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }

    @GET
    @Path("/health")
    @Operation(summary = "Health check", description = "Check if chat service is available")
    @APIResponse(responseCode = "200", description = "Service is healthy")
    public Response health() {
        return Response.ok(Map.of(
                "status", "UP",
                "service", "chat",
                "timestamp", System.currentTimeMillis()
        )).build();
    }
}
