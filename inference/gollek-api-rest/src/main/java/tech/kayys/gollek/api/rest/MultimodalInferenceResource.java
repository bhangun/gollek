package tech.kayys.gollek.api.rest;

import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.media.Content;
import org.eclipse.microprofile.openapi.annotations.media.Schema;
import org.eclipse.microprofile.openapi.annotations.parameters.RequestBody;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;
import org.jboss.resteasy.reactive.multipart.FileUpload;
import org.jboss.resteasy.reactive.RestForm;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.spi.model.MultimodalRequest.OutputConfig;
import tech.kayys.gollek.multimodal.MultimodalRouter;
import tech.kayys.gollek.multimodal.MultimodalCapabilityRegistry;

import java.util.*;

/**
 * REST API for multimodal inference in the Gollek server.
 *
 * <h3>Endpoints</h3>
 * <ul>
 * <li>{@code POST /v1/multimodal/chat} — JSON multimodal request</li>
 * <li>{@code POST /v1/multimodal/upload} — multipart form (file upload)</li>
 * <li>{@code GET /v1/multimodal/capabilities} — list all provider
 * capabilities</li>
 * <li>{@code GET /v1/multimodal/health} — health of all providers</li>
 * </ul>
 *
 * <h3>JSON chat example</h3>
 * <pre>{@code
 * POST /v1/multimodal/chat
 * Content-Type: application/json
 *
 * {
 * "model": "gemini-pro-vision",
 * "parts": [
 * { "modality": "TEXT", "text": "What is in this image?" },
 * { "modality": "IMAGE", "mimeType": "image/jpeg",
 * "base64Data": "/9j/4AAQSkZJRgAB..." }
 * ],
 * "parameters": { "max_tokens": 256, "temperature": 0.3 }
 * }
 * }</pre>
 */
@Path("/v1/multimodal")
@Produces(MediaType.APPLICATION_JSON)
@Tag(name = "Multimodal Inference", description = "Vision, audio, video, and document inference endpoints")
public class MultimodalInferenceResource {

    private static final Logger LOG = Logger.getLogger(MultimodalInferenceResource.class);

    @Inject
    MultimodalRouter router;

    @Inject
    MultimodalCapabilityRegistry registry;

    // -------------------------------------------------------------------------
    // JSON chat endpoint
    // -------------------------------------------------------------------------

    @POST
    @Path("/chat")
    @Consumes(MediaType.APPLICATION_JSON)
    @Operation(summary = "Multimodal inference (JSON)", description = "Send a multimodal request with text, image (base64 or URL), audio, video, or document parts.")
    @APIResponse(responseCode = "200", description = "Successful inference", content = @Content(schema = @Schema(implementation = MultimodalResponse.class)))
    @APIResponse(responseCode = "400", description = "Validation error or unsupported modality")
    @APIResponse(responseCode = "503", description = "No provider available")
    public Uni<MultimodalResponse> chat(
            @RequestBody(required = true) MultimodalChatRequest body) {

        if (body == null || body.model() == null || body.parts() == null || body.parts().isEmpty()) {
            return Uni.createFrom().item(
                    MultimodalResponse.error("unknown", "unknown",
                            "MM-API-400", "Request must include 'model' and at least one 'parts' entry."));
        }

        MultimodalRequest.Builder reqBuilder = MultimodalRequest.builder()
                .model(body.model())
                .inputs(body.parts().toArray(new MultimodalContent[0]))
                .timeoutMs(body.timeoutMs() > 0 ? body.timeoutMs() : 30_000L);

        if (body.parameters() != null) {
            reqBuilder.parameters(body.parameters());
        }
        if (body.expectedOutputModalities() != null) {
            List<ModalityType> modalities = body.expectedOutputModalities().stream()
                    .map(s -> {
                        try {
                            return ModalityType.valueOf(s.toUpperCase());
                        } catch (Exception e) {
                            return null;
                        }
                    })
                    .filter(Objects::nonNull)
                    .toList();
            reqBuilder.outputConfig(OutputConfig.builder().outputModalities(modalities.toArray(new ModalityType[0])).build());
        }

        MultimodalRequest request = reqBuilder.build();
        LOG.debugf("[%s] Received multimodal chat request for model=%s, parts=%d",
                request.getRequestId(), request.getModel(), request.getInputs().length);

        return router.route(request);
    }

    // -------------------------------------------------------------------------
    // Multipart file upload endpoint
    // -------------------------------------------------------------------------

    @POST
    @Path("/upload")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Operation(summary = "Multimodal inference via file upload", description = "Upload binary files (images, audio, video, documents) alongside an optional text prompt.")
    public Uni<MultimodalResponse> upload(
            @RestForm("model") String model,
            @RestForm("prompt") String prompt,
            @RestForm("file") FileUpload file,
            @RestForm("mimeType") String mimeType,
            @RestForm("timeoutMs") long timeoutMs) {

        if (model == null || model.isBlank()) {
            return Uni.createFrom().item(
                    MultimodalResponse.error("unknown", "unknown",
                            "MM-API-400", "'model' form field is required."));
        }

        List<MultimodalContent> parts = new ArrayList<>();

        if (prompt != null && !prompt.isBlank()) {
            parts.add(MultimodalContent.ofText(prompt));
        }

        if (file != null && file.filePath() != null) {
            try {
                byte[] bytes = java.nio.file.Files.readAllBytes(file.filePath());
                String resolvedMime = mimeType != null ? mimeType : file.contentType();
                ModalityType modality = detectModality(resolvedMime);

                MultimodalContent filePart = switch (modality) {
                    case IMAGE -> MultimodalContent.ofBase64Image(bytes, resolvedMime);
                    case AUDIO -> MultimodalContent.ofAudio(bytes, resolvedMime);
                    case VIDEO -> MultimodalContent.ofVideo(bytes, resolvedMime);
                    case DOCUMENT -> MultimodalContent.ofDocument(bytes,
                            resolvedMime.contains("pdf") ? "pdf" : "binary", resolvedMime);
                    default -> MultimodalContent.ofText(new String(bytes));
                };
                parts.add(filePart);
            } catch (Exception e) {
                return Uni.createFrom().item(
                        MultimodalResponse.error("unknown", model,
                                "MM-API-IO", "Failed to read uploaded file: " + e.getMessage()));
            }
        }

        MultimodalRequest request = MultimodalRequest.builder()
                .model(model)
                .inputs(parts.toArray(new MultimodalContent[0]))
                .timeoutMs(timeoutMs > 0 ? timeoutMs : 60_000L)
                .build();

        return router.route(request);
    }

    // -------------------------------------------------------------------------
    // Capabilities endpoint
    // -------------------------------------------------------------------------

    @GET
    @Path("/capabilities")
    @Operation(summary = "List multimodal provider capabilities", description = "Returns the registered input/output modalities and limits for every active provider.")
    public Response capabilities() {
        Map<String, Object> response = new LinkedHashMap<>();
        Map<String, Object> providers = new LinkedHashMap<>();

        registry.allCapabilities().forEach((id, cap) -> {
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("modelId", cap.getModelId());
            info.put("inputModalities", cap.getInputModalities());
            info.put("outputModalities", cap.getOutputModalities());
            info.put("maxImageSizeBytes", cap.getMaxImageSizeBytes());
            info.put("maxAudioSizeBytes", cap.getMaxAudioSizeBytes());
            info.put("maxVideoSizeBytes", cap.getMaxVideoSizeBytes());
            info.put("maxAudioDurationSecs", cap.getMaxAudioDurationSecs());
            info.put("maxVideoDurationSecs", cap.getMaxVideoDurationSecs());
            info.put("maxPartsPerRequest", cap.getMaxPartsPerRequest());
            info.put("supportsStreaming", cap.isSupportsStreaming());
            info.put("supportsDocuments", cap.isSupportsDocuments());
            info.put("supportedMimeTypes", cap.getSupportedMimeTypes());
            providers.put(id, info);
        });

        response.put("providers", providers);
        response.put("totalProviders", providers.size());
        return Response.ok(response).build();
    }

    // -------------------------------------------------------------------------
    // Health endpoint
    // -------------------------------------------------------------------------

    @GET
    @Path("/health")
    @Operation(summary = "Health check for all multimodal providers")
    public Response health() {
        Map<String, Object> result = new LinkedHashMap<>();
        Map<String, Object> statuses = new LinkedHashMap<>();
        boolean allHealthy = true;

        for (String id : registry.registeredIds()) {
            boolean available = registry.providerFor(id)
                    .map(p -> p.isAvailable())
                    .orElse(false);
            statuses.put(id, available ? "UP" : "DOWN");
            if (!available)
                allHealthy = false;
        }

        result.put("status", allHealthy ? "UP" : "DEGRADED");
        result.put("providers", statuses);

        int httpStatus = allHealthy ? 200 : 207;
        return Response.status(httpStatus).entity(result).build();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private ModalityType detectModality(String mimeType) {
        if (mimeType == null)
            return ModalityType.TEXT;
        String m = mimeType.toLowerCase();
        if (m.startsWith("image/"))
            return ModalityType.IMAGE;
        if (m.startsWith("audio/"))
            return ModalityType.AUDIO;
        if (m.startsWith("video/"))
            return ModalityType.VIDEO;
        if (m.contains("pdf") || m.contains("word")
                || m.contains("document"))
            return ModalityType.DOCUMENT;
        return ModalityType.TEXT;
    }

    // -------------------------------------------------------------------------
    // Request DTO
    // -------------------------------------------------------------------------

    /**
     * JSON body for {@code POST /v1/multimodal/chat}.
     */
    public record MultimodalChatRequest(
            String model,
            List<MultimodalContent> parts,
            Map<String, Object> parameters,
            List<String> expectedOutputModalities,
            long timeoutMs) {
    }
}
