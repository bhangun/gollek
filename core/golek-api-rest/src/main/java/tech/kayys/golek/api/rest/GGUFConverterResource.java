package tech.kayys.golek.api.rest;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import lombok.extern.slf4j.Slf4j;
import tech.kayys.golek.converter.ConversionStorageService;
import tech.kayys.golek.converter.GGUFConverter;
import tech.kayys.golek.converter.GGUFException;
import tech.kayys.golek.converter.dto.ConversionRequest;
import tech.kayys.golek.converter.dto.ConversionResponse;
import tech.kayys.golek.converter.dto.ModelInfoResponse;
import tech.kayys.golek.converter.dto.ProgressUpdate;
import tech.kayys.golek.converter.model.ConversionProgress;
import tech.kayys.golek.converter.model.ConversionResult;
import tech.kayys.golek.converter.model.GGUFConversionParams;
import tech.kayys.golek.converter.model.ModelInfo;
import tech.kayys.golek.converter.model.QuantizationType;
import tech.kayys.golek.spi.auth.ApiKeyConstants;
import tech.kayys.golek.spi.context.RequestContext;
import tech.kayys.golek.spi.model.ModelFormat;

import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.parameters.Parameter;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.resteasy.reactive.RestStreamElementType;

import java.util.List;
import java.util.Map;

/**
 * REST API for GGUF model conversion.
 * 
 * <p>
 * Provides endpoints for:
 * <ul>
 * <li>Converting models to GGUF format</li>
 * <li>Detecting model formats</li>
 * <li>Extracting model information</li>
 * <li>Verifying GGUF files</li>
 * <li>Listing available quantizations</li>
 * </ul>
 * 
 * @author Bhangun
 * @version 1.0.0
 */
@Path("/v1/converter/gguf")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "GGUF Converter", description = "Model conversion to GGUF format")
@Slf4j
public class GGUFConverterResource {

    @Inject
    GGUFConverter converter;

    @Inject
    RequestContext requestContext;

    @Inject
    ConversionStorageService storageService;

    /**
     * Convert model to GGUF format (async).
     */
    @POST
    @Path("/convert")
    @Operation(summary = "Convert model to GGUF", description = "Converts a model from PyTorch, SafeTensors, TensorFlow, or Flax to GGUF format")
    public Uni<ConversionResponse> convertModel(@Valid @NotNull ConversionRequest request) {
        String requestId = resolveRequestId();
        log.info("Tenant {} requested conversion: {} -> {}",
                requestId, request.getInputPath(), request.getQuantization());

        // Build conversion parameters
        GGUFConversionParams params = GGUFConversionParams.builder()
                .inputPath(getTenantPath(requestId, request.getInputPath()))
                .outputPath(getTenantPath(requestId, request.getOutputPath()))
                .modelType(request.getModelType())
                .quantization(request.getQuantization())
                .vocabOnly(request.isVocabOnly())
                .numThreads(request.getNumThreads())
                .vocabType(request.getVocabType())
                .build();

        return converter.convertAsync(params)
                .map(result -> ConversionResponse.fromResult(result, requestId))
                .onFailure().transform(e -> {
                    log.error("Conversion failed for tenant {}", requestId, e);
                    return new GGUFException("Conversion failed: " + e.getMessage(), e);
                });
    }

    /**
     * Convert model with progress updates (SSE stream).
     */
    @POST
    @Path("/convert/stream")
    @Produces(MediaType.SERVER_SENT_EVENTS)
    @RestStreamElementType(MediaType.APPLICATION_JSON)
    @Operation(summary = "Convert model with progress updates", description = "Converts a model and streams progress updates via Server-Sent Events")
    public Multi<Object> convertModelWithProgress(@Valid @NotNull ConversionRequest request) {
        String requestId = resolveRequestId();
        log.info("Tenant {} requested streaming conversion: {}", requestId, request.getInputPath());

        GGUFConversionParams params = GGUFConversionParams.builder()
                .inputPath(getTenantPath(requestId, request.getInputPath()))
                .outputPath(getTenantPath(requestId, request.getOutputPath()))
                .modelType(request.getModelType())
                .quantization(request.getQuantization())
                .vocabOnly(request.isVocabOnly())
                .numThreads(request.getNumThreads())
                .vocabType(request.getVocabType())
                .build();

        return converter.convertWithProgress(params)
                .map(obj -> {
                    if (obj instanceof ConversionProgress progress) {
                        return ProgressUpdate.fromProgress(progress);
                    } else if (obj instanceof ConversionResult result) {
                        return ConversionResponse.fromResult(result, requestId);
                    }
                    return obj;
                });
    }

    /**
     * Cancel an active conversion.
     */
    @POST
    @Path("/convert/{conversionId}/cancel")
    @Operation(summary = "Cancel conversion", description = "Cancels an active conversion")
    public Response cancelConversion(
            @Parameter(description = "Conversion ID") @PathParam("conversionId") long conversionId) {

        String requestId = resolveRequestId();
        log.info("Tenant {} requested cancellation of conversion {}", requestId, conversionId);

        boolean cancelled = converter.cancelConversion(conversionId);

        if (cancelled) {
            return Response.ok(Map.of("status", "cancelled", "conversionId", conversionId)).build();
        } else {
            return Response.status(Response.Status.NOT_FOUND)
                    .entity(Map.of("error", "Conversion not found or already completed"))
                    .build();
        }
    }

    /**
     * Detect model format.
     */
    @GET
    @Path("/detect-format")
    @Operation(summary = "Detect model format", description = "Detects the format of a model file or directory")
    public Response detectFormat(@QueryParam("path") @NotNull String path) {
        String requestId = resolveRequestId();
        java.nio.file.Path fullPath = getTenantPath(requestId, path);

        ModelFormat format = converter.detectFormat(fullPath);

        return Response.ok(Map.of(
                "path", path,
                "format", format.getId(),
                "displayName", format.getDisplayName(),
                "convertible", format.isConvertible())).build();
    }

    /**
     * Get model information.
     */
    @GET
    @Path("/model-info")
    @Operation(summary = "Get model information", description = "Extracts metadata from a model without converting")
    public Response getModelInfo(@QueryParam("path") @NotNull String path) {
        String requestId = resolveRequestId();
        java.nio.file.Path fullPath = getTenantPath(requestId, path);

        try {
            ModelInfo info = converter.getModelInfo(fullPath);
            return Response.ok(ModelInfoResponse.fromModelInfo(info)).build();
        } catch (GGUFException e) {
            log.error("Failed to get model info for tenant {}, path {}", requestId, path, e);
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity(Map.of("error", e.getMessage()))
                    .build();
        }
    }

    /**
     * Verify GGUF file integrity.
     */
    @GET
    @Path("/verify")
    @Operation(summary = "Verify GGUF file", description = "Verifies the integrity of a GGUF file")
    public Response verifyGGUF(@QueryParam("path") @NotNull String path) {
        String requestId = resolveRequestId();
        java.nio.file.Path fullPath = getTenantPath(requestId, path);

        try {
            ModelInfo info = converter.verifyGGUF(fullPath);
            return Response.ok(Map.of(
                    "valid", true,
                    "info", ModelInfoResponse.fromModelInfo(info))).build();
        } catch (GGUFException e) {
            return Response.ok(Map.of(
                    "valid", false,
                    "error", e.getMessage())).build();
        }
    }

    /**
     * List available quantization types.
     */
    @GET
    @Path("/quantizations")
    @Operation(summary = "List quantization types", description = "Returns all available quantization types with metadata")
    public Response getQuantizations() {
        QuantizationType[] types = converter.getAvailableQuantizations();

        List<Map<String, Object>> quantizations = java.util.Arrays.stream(types)
                .map(type -> Map.<String, Object>of(
                        "id", type.getNativeName(),
                        "name", type.name(),
                        "description", type.getDescription(),
                        "qualityLevel", type.getQualityLevel().name(),
                        "compressionRatio", type.getCompressionRatio(),
                        "useCase", type.getUseCase()))
                .toList();

        return Response.ok(Map.of("quantizations", quantizations)).build();
    }

    /**
     * Get recommended quantization for a model.
     */
    @GET
    @Path("/recommend-quantization")
    @Operation(summary = "Get recommended quantization", description = "Recommends optimal quantization based on model size and requirements")
    public Response recommendQuantization(
            @QueryParam("modelSizeGb") double modelSizeGb,
            @QueryParam("prioritizeQuality") @DefaultValue("false") boolean prioritizeQuality) {

        QuantizationType recommended = QuantizationType.recommend(modelSizeGb, prioritizeQuality);

        return Response.ok(Map.of(
                "recommended", recommended.getNativeName(),
                "name", recommended.name(),
                "description", recommended.getDescription(),
                "qualityLevel", recommended.getQualityLevel().name(),
                "compressionRatio", recommended.getCompressionRatio(),
                "reason", "Recommended for " + modelSizeGb + " GB model, " +
                        (prioritizeQuality ? "prioritizing quality" : "balanced approach")))
                .build();
    }

    private String resolveRequestId() {
        if (requestContext == null || requestContext.getRequestId() == null) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return requestContext.getRequestId();
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Get tenant-specific path with proper isolation.
     */
    private java.nio.file.Path getTenantPath(String requestId, String relativePath) {
        // In production, this would use proper storage service
        // For now, simple path construction with tenant isolation
        java.nio.file.Path basePath = storageService.getTenantBasePath(requestId);
        return basePath.resolve(relativePath).normalize();
    }
}
