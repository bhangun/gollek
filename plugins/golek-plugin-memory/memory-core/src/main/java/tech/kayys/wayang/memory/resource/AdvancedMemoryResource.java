package tech.kayys.wayang.memory.resource;

import tech.kayys.wayang.memory.service.*;
import tech.kayys.wayang.memory.model.*;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.parameters.Parameter;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import tech.kayys.wayang.error.ErrorCode;
import tech.kayys.wayang.error.ErrorResponse;
import tech.kayys.wayang.error.WayangException;

import java.time.Duration;
import java.util.List;
import java.util.Map;

@Path("/api/v1/memory/advanced")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Advanced Memory Operations", description = "Advanced memory management features")
public class AdvancedMemoryResource {
    
    @Inject
    MemoryAnalyticsService analyticsService;
    
    @Inject
    MemoryBackupService backupService;
    
    @Inject
    MemoryOptimizationService optimizationService;
    
    @Inject
    MemorySecurityService securityService;
    
    @Inject
    MemoryService memoryService;

    @GET
    @Path("/analytics/{userId}")
    @Operation(summary = "Get memory analytics for user")
    public Uni<Response> getUserAnalytics(
            @Parameter(description = "User ID") @PathParam("userId") String userId,
            @Parameter(description = "Time window in hours") @QueryParam("hours") @DefaultValue("24") int hours) {
        
        return analyticsService.analyzeMemoryUsage(userId, Duration.ofHours(hours))
            .onItem().transform(analytics -> Response.ok(analytics).build())
            .onFailure().recoverWithItem(throwable -> 
                Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(ErrorResponse.from(new WayangException(
                            ErrorCode.INTERNAL_ERROR,
                            throwable.getMessage(),
                            throwable)))
                    .build());
    }

    @GET
    @Path("/insights/{sessionId}")
    @Operation(summary = "Get memory insights for session")
    public Uni<Response> getSessionInsights(
            @Parameter(description = "Session ID") @PathParam("sessionId") String sessionId) {
        
        return analyticsService.generateMemoryInsights(sessionId)
            .onItem().transform(insights -> Response.ok(insights).build())
            .onFailure().recoverWithItem(throwable -> 
                Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(ErrorResponse.from(new WayangException(
                            ErrorCode.INTERNAL_ERROR,
                            throwable.getMessage(),
                            throwable)))
                    .build());
    }

    @POST
    @Path("/backup")
    @Operation(summary = "Create memory backup")
    public Uni<Response> createBackup(BackupRequest request) {
        return backupService.createBackup(request.getUserId(), request.getSessionIds())
            .onItem().transform(backup -> Response.ok(backup).build())
            .onFailure().recoverWithItem(throwable -> 
                Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(ErrorResponse.from(new WayangException(
                            ErrorCode.INTERNAL_ERROR,
                            throwable.getMessage(),
                            throwable)))
                    .build());
    }

    @POST
    @Path("/backup/{backupId}/restore")
    @Operation(summary = "Restore memory backup")
    public Uni<Response> restoreBackup(
            @Parameter(description = "Backup ID") @PathParam("backupId") String backupId) {
        
        return backupService.restoreBackup(backupId)
            .onItem().transform(contexts -> Response.ok(Map.of(
                "restored", contexts.size(),
                "contexts", contexts
            )).build())
            .onFailure().recoverWithItem(throwable -> 
                Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(ErrorResponse.from(new WayangException(
                            ErrorCode.INTERNAL_ERROR,
                            throwable.getMessage(),
                            throwable)))
                    .build());
    }

    @POST
    @Path("/optimize/{sessionId}")
    @Operation(summary = "Optimize memory for session")
    public Uni<Response> optimizeMemory(
            @Parameter(description = "Session ID") @PathParam("sessionId") String sessionId) {
        
        return optimizationService.optimizeMemory(sessionId)
            .onItem().transform(result -> Response.ok(result).build())
            .onFailure().recoverWithItem(throwable -> 
                Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(ErrorResponse.from(new WayangException(
                            ErrorCode.INTERNAL_ERROR,
                            throwable.getMessage(),
                            throwable)))
                    .build());
    }

    @POST
    @Path("/security/scan/{sessionId}")
    @Operation(summary = "Scan memory for PII and security issues")
    public Uni<Response> scanSecurity(
            @Parameter(description = "Session ID") @PathParam("sessionId") String sessionId) {
        
        return memoryService.getContext(sessionId, null)
            .onItem().transformToUni(context -> 
                securityService.scanMemoryForPII(context))
            .onItem().transform(result -> Response.ok(result).build())
            .onFailure().recoverWithItem(throwable -> 
                Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(ErrorResponse.from(new WayangException(
                            ErrorCode.INTERNAL_ERROR,
                            throwable.getMessage(),
                            throwable)))
                    .build());
    }

    @POST
    @Path("/security/sanitize/{sessionId}")
    @Operation(summary = "Sanitize memory by removing PII")
    public Uni<Response> sanitizeMemory(
            @Parameter(description = "Session ID") @PathParam("sessionId") String sessionId) {
        
        return memoryService.getContext(sessionId, null)
            .onItem().transformToUni(context -> 
                securityService.sanitizeMemory(context))
            .onItem().transformToUni(sanitized -> 
                memoryService.storeContext(sanitized)
                    .replaceWith(Response.ok(Map.of("status", "sanitized")).build()))
            .onFailure().recoverWithItem(throwable -> 
                Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(ErrorResponse.from(new WayangException(
                            ErrorCode.INTERNAL_ERROR,
                            throwable.getMessage(),
                            throwable)))
                    .build());
    }

    public static class BackupRequest {
        private String userId;
        private List<String> sessionIds;

        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public List<String> getSessionIds() { return sessionIds; }
        public void setSessionIds(List<String> sessionIds) { this.sessionIds = sessionIds; }
    }

 
}
