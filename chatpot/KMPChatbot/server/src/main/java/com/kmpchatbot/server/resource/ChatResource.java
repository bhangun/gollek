package com.kmpchatbot.server.resource;

import com.kmpchatbot.server.dto.ChatRequest;
import com.kmpchatbot.server.dto.ChatResponse;
import com.kmpchatbot.server.dto.ConversationDTO;
import com.kmpchatbot.server.service.ChatService;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.media.Content;
import org.eclipse.microprofile.openapi.annotations.media.Schema;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;

import java.util.List;

@Path("/api/chat")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Chat", description = "Chat endpoints")
public class ChatResource {
    
    private static final Logger LOG = Logger.getLogger(ChatResource.class);
    
    @Inject
    ChatService chatService;
    
    @POST
    @Path("/message")
    @Operation(summary = "Send a message", description = "Send a message and get AI response")
    @APIResponse(
            responseCode = "200",
            description = "Message sent successfully",
            content = @Content(schema = @Schema(implementation = ChatResponse.class))
    )
    @APIResponse(responseCode = "400", description = "Invalid request")
    @APIResponse(responseCode = "500", description = "Server error")
    public Response sendMessage(@Valid ChatRequest request) {
        LOG.infof("Received message: %s", request.getMessage());
        
        try {
            ChatResponse response = chatService.sendMessage(request);
            return Response.ok(response).build();
        } catch (Exception e) {
            LOG.errorf(e, "Error processing message");
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(new ErrorResponse(e.getMessage()))
                    .build();
        }
    }
    
    @GET
    @Path("/history/{sessionId}")
    @Operation(summary = "Get chat history", description = "Get all messages for a session")
    @APIResponse(
            responseCode = "200",
            description = "History retrieved successfully",
            content = @Content(schema = @Schema(implementation = ChatResponse.class))
    )
    public Response getHistory(@PathParam("sessionId") String sessionId) {
        List<ChatResponse> history = chatService.getHistory(sessionId);
        return Response.ok(history).build();
    }
    
    @GET
    @Path("/conversations")
    @Operation(summary = "Get all conversations", description = "Get all chat conversations")
    @APIResponse(
            responseCode = "200",
            description = "Conversations retrieved successfully",
            content = @Content(schema = @Schema(implementation = ConversationDTO.class))
    )
    public Response getAllConversations() {
        List<ConversationDTO> conversations = chatService.getAllConversations();
        return Response.ok(conversations).build();
    }
    
    @GET
    @Path("/conversation/{sessionId}")
    @Operation(summary = "Get conversation", description = "Get a specific conversation with messages")
    @APIResponse(
            responseCode = "200",
            description = "Conversation retrieved successfully",
            content = @Content(schema = @Schema(implementation = ConversationDTO.class))
    )
    @APIResponse(responseCode = "404", description = "Conversation not found")
    public Response getConversation(@PathParam("sessionId") String sessionId) {
        ConversationDTO conversation = chatService.getConversation(sessionId);
        if (conversation == null) {
            return Response.status(Response.Status.NOT_FOUND)
                    .entity(new ErrorResponse("Conversation not found"))
                    .build();
        }
        return Response.ok(conversation).build();
    }
    
    @DELETE
    @Path("/conversation/{sessionId}")
    @Operation(summary = "Delete conversation", description = "Delete a conversation and all its messages")
    @APIResponse(responseCode = "204", description = "Conversation deleted successfully")
    public Response deleteConversation(@PathParam("sessionId") String sessionId) {
        chatService.deleteConversation(sessionId);
        return Response.noContent().build();
    }
    
    @GET
    @Path("/health")
    @Operation(summary = "Health check", description = "Check if the chat service is running")
    public Response health() {
        return Response.ok(new HealthResponse("Chat service is running")).build();
    }
    
    // Helper classes
    public static class ErrorResponse {
        public String error;
        
        public ErrorResponse(String error) {
            this.error = error;
        }
    }
    
    public static class HealthResponse {
        public String status;
        
        public HealthResponse(String status) {
            this.status = status;
        }
    }
}
