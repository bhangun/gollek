package com.kmpchatbot.server.api;

import com.kmpchatbot.server.domain.Conversation;
import com.kmpchatbot.server.domain.User;
import jakarta.transaction.Transactional;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.SecurityContext;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Path("/api/conversations")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Conversations", description = "Conversation management endpoints")
public class ConversationResource {

    private static final Logger LOG = Logger.getLogger(ConversationResource.class);

    @GET
    @Operation(summary = "List conversations", description = "Get all conversations for current user")
    public Response listConversations(@Context SecurityContext securityContext) {
        try {
            Long userId = 1L; // Extract from JWT in production
            
            List<Conversation> conversations = Conversation.findByUser(userId);
            
            List<Map<String, Object>> response = conversations.stream()
                    .map(conv -> Map.of(
                            "id", conv.getId(),
                            "title", conv.getTitle(),
                            "createdAt", conv.getCreatedAt().toString(),
                            "updatedAt", conv.getUpdatedAt().toString(),
                            "messageCount", conv.getMessages().size(),
                            "archived", conv.isArchived()
                    ))
                    .collect(Collectors.toList());
            
            return Response.ok(response).build();
            
        } catch (Exception e) {
            LOG.error("Error listing conversations", e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }

    @POST
    @Transactional
    @Operation(summary = "Create conversation", description = "Create a new conversation")
    public Response createConversation(
            Map<String, String> request,
            @Context SecurityContext securityContext) {
        
        try {
            Long userId = 1L; // Extract from JWT in production
            User user = User.findById(userId);
            
            if (user == null) {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity(Map.of("error", "User not found"))
                        .build();
            }
            
            Conversation conversation = new Conversation();
            conversation.setTitle(request.getOrDefault("title", "New Conversation"));
            conversation.setUser(user);
            conversation.persist();
            
            return Response.status(Response.Status.CREATED)
                    .entity(Map.of(
                            "id", conversation.getId(),
                            "title", conversation.getTitle(),
                            "createdAt", conversation.getCreatedAt().toString()
                    ))
                    .build();
                    
        } catch (Exception e) {
            LOG.error("Error creating conversation", e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }

    @GET
    @Path("/{id}")
    @Operation(summary = "Get conversation", description = "Get conversation by ID")
    public Response getConversation(
            @PathParam("id") Long id,
            @Context SecurityContext securityContext) {
        
        try {
            Long userId = 1L; // Extract from JWT in production
            Conversation conversation = Conversation.findByIdAndUser(id, userId);
            
            if (conversation == null) {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity(Map.of("error", "Conversation not found"))
                        .build();
            }
            
            return Response.ok(Map.of(
                    "id", conversation.getId(),
                    "title", conversation.getTitle(),
                    "createdAt", conversation.getCreatedAt().toString(),
                    "updatedAt", conversation.getUpdatedAt().toString(),
                    "messageCount", conversation.getMessages().size()
            )).build();
            
        } catch (Exception e) {
            LOG.error("Error getting conversation", e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }

    @DELETE
    @Path("/{id}")
    @Transactional
    @Operation(summary = "Delete conversation", description = "Delete a conversation")
    public Response deleteConversation(
            @PathParam("id") Long id,
            @Context SecurityContext securityContext) {
        
        try {
            Long userId = 1L; // Extract from JWT in production
            Conversation conversation = Conversation.findByIdAndUser(id, userId);
            
            if (conversation == null) {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity(Map.of("error", "Conversation not found"))
                        .build();
            }
            
            conversation.delete();
            
            return Response.ok(Map.of("message", "Conversation deleted successfully")).build();
            
        } catch (Exception e) {
            LOG.error("Error deleting conversation", e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }

    @PUT
    @Path("/{id}/archive")
    @Transactional
    @Operation(summary = "Archive conversation", description = "Archive/unarchive a conversation")
    public Response archiveConversation(
            @PathParam("id") Long id,
            Map<String, Boolean> request,
            @Context SecurityContext securityContext) {
        
        try {
            Long userId = 1L; // Extract from JWT in production
            Conversation conversation = Conversation.findByIdAndUser(id, userId);
            
            if (conversation == null) {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity(Map.of("error", "Conversation not found"))
                        .build();
            }
            
            boolean archived = request.getOrDefault("archived", true);
            conversation.setArchived(archived);
            
            return Response.ok(Map.of(
                    "message", archived ? "Conversation archived" : "Conversation unarchived",
                    "archived", conversation.isArchived()
            )).build();
            
        } catch (Exception e) {
            LOG.error("Error archiving conversation", e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }
}
