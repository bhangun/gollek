/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizationResource.java
 * ───────────────────────
 * REST API for quantization operations.
 */
package tech.kayys.gollek.safetensor.api;

import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.media.Content;
import org.eclipse.microprofile.openapi.annotations.media.Schema;
import org.eclipse.microprofile.openapi.annotations.parameters.RequestBody;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
// import tech.kayys.gollek.quantizer.gptq.GPTQConfig;
// import tech.kayys.gollek.quantizer.gptq.GPTQQuantizerService;
// import tech.kayys.gollek.quantizer.gptq.GPTQSafetensorConverter;
import tech.kayys.gollek.safetensor.api.dto.QuantizationResponse;
import tech.kayys.gollek.safetensor.api.dto.QuantizationRequest;

import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * REST resource for model quantization operations.
 * Uses reflection/Object to avoid circular dependencies with Quantization
 * module.
 */
@jakarta.ws.rs.Path("/api/v1/quantization")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class QuantizationResource {

    private static final org.jboss.logging.Logger log = org.jboss.logging.Logger.getLogger(QuantizationResource.class);

    @Inject
    jakarta.enterprise.inject.Instance<Object> engineInstance;

    // @Inject
    // jakarta.enterprise.inject.Instance<GPTQQuantizerService> quantizerServiceInstance;

    private Object getEngine() {
        try {
            return engineInstance.get();
        } catch (Exception e) {
            return null;
        }
    }

    /*
    private GPTQQuantizerService getQuantizerService() {
        try {
            return quantizerServiceInstance.get();
        } catch (Exception e) {
            return new GPTQQuantizerService();
        }
    }
    */

    @POST
    @jakarta.ws.rs.Path("/quantize")
    @Operation(summary = "Quantize a model", description = "Quantize a model using the specified strategy and configuration")
    @APIResponse(responseCode = "200", description = "Quantization completed successfully", content = @Content(schema = @Schema(implementation = QuantizationResponse.class)))
    @APIResponse(responseCode = "400", description = "Invalid request parameters")
    @APIResponse(responseCode = "500", description = "Quantization failed")
    public Response quantize(@RequestBody(description = "Quantization request") QuantizationRequest request) {
        log.infof("Received quantization request: input=%s, output=%s, strategy=%s",
                request.getInputPath(), request.getOutputPath(), request.getStrategy());

        Object engine = getEngine();
        if (engine == null)
            return Response.status(503).entity(QuantizationResponse.error("Quantization engine not available")).build();

        try {
            validateRequest(request);

            java.nio.file.Path inputPath = Paths.get(request.getInputPath());
            java.nio.file.Path outputPath = Paths.get(request.getOutputPath());

            Class<?> strategyEnum = Class
                    .forName("tech.kayys.gollek.safetensor.quantization.QuantizationEngine$QuantStrategy");
            Object strategy = Enum.valueOf((Class<Enum>) strategyEnum, request.getStrategy().toUpperCase());

            Class<?> configClass = Class.forName("tech.kayys.gollek.safetensor.quantization.QuantConfig");
            Object config = buildConfig(request, configClass, strategyEnum);

            var result = engine.getClass()
                    .getMethod("quantize", java.nio.file.Path.class, java.nio.file.Path.class, strategyEnum,
                            configClass)
                    .invoke(engine, inputPath, outputPath, strategy, config);

            var response = QuantizationResponse.fromReflection(result);

            return Response.ok(response).build();

        } catch (BadRequestException e) {
            log.warnf("Invalid quantization request: %s", e.getMessage());
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity(QuantizationResponse.error(e.getMessage()))
                    .build();
        } catch (Exception e) {
            log.errorf(e, "Quantization failed");
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(QuantizationResponse.error("Quantization failed: " + e.getMessage()))
                    .build();
        }
    }

    /*
    @POST
    @jakarta.ws.rs.Path("/quantize/gptq")
    @Operation(summary = "Quantize a model using GPTQ", description = "Quantize a model using GPTQ algorithm with advanced options")
    @APIResponse(responseCode = "202", description = "Quantization started, returns task ID for async tracking", content = @Content(schema = @Schema(implementation = Map.class)))
    @APIResponse(responseCode = "400", description = "Invalid request parameters")
    @APIResponse(responseCode = "500", description = "Quantization failed")
    public Response quantizeGPTQ(@RequestBody(description = "GPTQ Quantization request") QuantizationRequest request) {
        // Implementation disabled to unblock build
        return Response.status(501).entity(Map.of("error", "GPTQ quantization not currently available")).build();
    }
    */

    /*
    @GET
    @jakarta.ws.rs.Path("/inspect/{modelPath}")
    @Operation(summary = "Inspect a quantized model", description = "Get detailed information about a quantized model")
    @APIResponse(responseCode = "200", description = "Model inspection successful", content = @Content(schema = @Schema(implementation = Map.class)))
    @APIResponse(responseCode = "404", description = "Model not found")
    @APIResponse(responseCode = "500", description = "Inspection failed")
    public Response inspectModel(@PathParam("modelPath") String modelPath) {
        return Response.status(501).entity(Map.of("error", "Model inspection not currently available")).build();
    }
    */

    private void validateRequest(QuantizationRequest request) {
        if (request.getInputPath() == null || request.getInputPath().isBlank()) {
            throw new BadRequestException("input_path is required");
        }
        if (request.getOutputPath() == null || request.getOutputPath().isBlank()) {
            throw new BadRequestException("output_path is required");
        }
        if (request.getStrategy() == null || request.getStrategy().isBlank()) {
            throw new BadRequestException("strategy is required");
        }
    }

    private Object buildConfig(QuantizationRequest request, Class<?> configClass, Class<?> strategyEnum)
            throws Exception {
        Object builder = configClass.getMethod("builder").invoke(null);
        Object strategy = Enum.valueOf((Class<Enum>) strategyEnum, request.getStrategy().toUpperCase());

        builder.getClass().getMethod("strategy", strategyEnum).invoke(builder, strategy);
        builder.getClass().getMethod("bits", int.class).invoke(builder, request.getBits());
        builder.getClass().getMethod("groupSize", int.class).invoke(builder, request.getGroupSize());
        builder.getClass().getMethod("symmetric", boolean.class).invoke(builder, request.isSymmetric());
        builder.getClass().getMethod("perChannel", boolean.class).invoke(builder, request.isPerChannel());
        builder.getClass().getMethod("actOrder", boolean.class).invoke(builder, request.isActOrder());
        builder.getClass().getMethod("dampPercent", double.class).invoke(builder, request.getDampPercent());
        builder.getClass().getMethod("numSamples", int.class).invoke(builder, request.getNumSamples());
        builder.getClass().getMethod("seqLen", int.class).invoke(builder, request.getSeqLen());

        return builder.getClass().getMethod("build").invoke(builder);
    }
}
