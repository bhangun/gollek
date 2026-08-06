/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticContract;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Action;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ActionKind;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ComponentReadiness;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ExecutionProfile;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.HeaderInspection;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.InputMode;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.MissingRuntimeCapability;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.Problem;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemDetail;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemCode;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.RuntimeCapability;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.TensorInventory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectSafetensorRoutePolicyTest {
    static { tech.kayys.gollek.safetensor.engine.route.DirectSafetensorRoutePolicy.resolver = new CliLocalModelResolver(); }
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    private String previousGemma4MobileQatLiteRtCacheDir;
    private String previousGemma4MobileQatLiteRtCacheEnabled;
    private String previousGemma4TextGgufCacheDir;
    private String previousGemma4TextGgufCacheEnabled;
    private String previousCommunityTextGgufCacheDir;
    private String previousCommunityTextGgufCacheEnabled;

    @BeforeEach
    void isolateRouteArtifactCaches() throws IOException {
        previousGemma4MobileQatLiteRtCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY);
        previousGemma4MobileQatLiteRtCacheEnabled = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_ENABLED_PROPERTY);
        previousGemma4TextGgufCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY);
        previousGemma4TextGgufCacheEnabled = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY);
        previousCommunityTextGgufCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY);
        previousCommunityTextGgufCacheEnabled = System.getProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY);

        Path cacheRoot = Files.createDirectories(tempDir.resolve("route-artifact-cache"));
        System.setProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                cacheRoot.resolve("gemma4-mobile-qat-litert").toString());
        System.setProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_ENABLED_PROPERTY,
                "true");
        System.setProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY,
                cacheRoot.resolve("gemma4-text-gguf").toString());
        System.setProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                "true");
        System.setProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY,
                cacheRoot.resolve("community-text-gguf").toString());
        System.setProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                "true");
    }

    @AfterEach
    void restoreRouteArtifactCaches() {
        restoreProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                previousGemma4MobileQatLiteRtCacheDir);
        restoreProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_ENABLED_PROPERTY,
                previousGemma4MobileQatLiteRtCacheEnabled);
        restoreProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY,
                previousGemma4TextGgufCacheDir);
        restoreProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                previousGemma4TextGgufCacheEnabled);
        restoreProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY,
                previousCommunityTextGgufCacheDir);
        restoreProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                previousCommunityTextGgufCacheEnabled);
    }

    @Test
    void gemma3AlternateRuntimePrefersAvailableLiteRtArtifact() throws Exception {
        Path modelDir = modelDir("gemma3_text");
        Path litert = Files.writeString(modelDir.resolve("functiongemma-270m-it.litertlm"), "");
        Files.writeString(modelDir.resolve("functiongemma-270m-it.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma3AlternateRuntime(
                        "safetensor",
                        "google/functiongemma-270m-it",
                        modelDir.toString(),
                        false,
                        true,
                        "litert"::equals);

        assertTrue(selection.selected());
        assertEquals("litert", selection.provider());
        assertEquals(litert.toAbsolutePath().normalize().toString(), selection.localPath());
        assertTrue(selection.notice().contains("switching to LiteRT runtime artifact"));
    }

    @Test
    void gemma3AlternateRuntimeHonorsExplicitProviderUnlessForced() throws Exception {
        Path modelDir = modelDir("gemma3_text");
        Files.writeString(modelDir.resolve("gemma-3-demo.litertlm"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma3AlternateRuntime(
                        "safetensor",
                        "google/gemma-3-demo",
                        modelDir.toString(),
                        true,
                        false,
                        "litert"::equals);

        assertFalse(selection.selected());
        assertFalse(selection.hasNotice());
    }

    @Test
    void gemma3DirectSafetensorRouteIsBlockedWithoutAlternateArtifact() throws Exception {
        Path modelDir = modelDir("gemma3_text");

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma3ExecutionRoute(
                        "safetensor",
                        "google/gemma-3-demo",
                        modelDir.toString());

        assertFalse(validation.allowed());
        assertTrue(validation.messages().get(0).contains("Gemma3 safetensor direct path is disabled"));
    }

    @Test
    void gemma3DirectSafetensorRouteAllowsCompatibleAlternateArtifact() throws Exception {
        Path modelDir = modelDir("gemma3_text");
        Files.writeString(modelDir.resolve("gemma-3-demo.gguf"), "");

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma3ExecutionRoute(
                        "safetensor",
                        "google/gemma-3-demo",
                        modelDir.toString());

        assertTrue(validation.allowed());
    }

    @Test
    void directRunRejectsGemma4Profile() throws Exception {
        Path modelDir = modelDir("gemma4_text");

        boolean directRun = DirectSafetensorRoutePolicy.shouldUseDirectRun(
                "safetensor",
                modelDir,
                true,
                null,
                null,
                "where is jakarta",
                true);

        assertFalse(directRun);
    }

    @Test
    void directRunRejectsGemma4UnifiedProfile() throws Exception {
        Path modelDir = modelDir("gemma4_unified");

        boolean directRun = DirectSafetensorRoutePolicy.shouldUseDirectRun(
                "safetensor",
                modelDir,
                true,
                null,
                null,
                "describe this audio",
                true);

        assertFalse(directRun);
    }

    @Test
    void directRunAllowsGuardedGemma4UnifiedTextProfile() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        writeSafetensorsHeader(modelDir, minimalGemma4UnifiedTextTensors());

        boolean directRun = DirectSafetensorRoutePolicy.shouldUseDirectRun(
                "safetensor",
                modelDir,
                true,
                null,
                null,
                "where is jakarta",
                true);

        assertTrue(directRun);
    }

    @Test
    void gemma4UnifiedTextSafetensorRoutePassesGuard() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        writeSafetensorsHeader(modelDir, minimalGemma4UnifiedTextTensors());

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString());

        assertTrue(validation.allowed(), () -> String.join("\n", validation.messages()));
    }

    @Test
    void gemma4UnifiedTwelveBTextSafetensorRoutePassesRealHeaderShape() throws Exception {
        Path modelDir = gemma4UnifiedTwelveBTextModelDir();
        writeSafetensorsHeader(modelDir, gemma4UnifiedTwelveBTextTensors());

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString());

        assertTrue(validation.allowed(), () -> String.join("\n", validation.messages()));
    }

    @Test
    void gemma4UnifiedTextSafetensorRouteRejectsMissingDecoderTensor() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        tensors.remove("model.language_model.layers.1.mlp.down_proj.weight");
        writeSafetensorsHeader(modelDir, tensors);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString());

        assertFalse(validation.allowed());
        assertTrue(validation.messages().get(0).contains("Gemma 4 unified safetensor text preflight failed"));
        assertTrue(validation.messages().stream().anyMatch(message -> message.contains("mlp.down_proj.weight")));
        assertTrue(validation.messages().stream().anyMatch(message -> message.contains("SafeTensors headers")));
    }

    @Test
    void gemma4UnifiedTextSafetensorRouteRejectsWrongFullAttentionRows() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.language_model.layers.1.self_attn.q_proj.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.1.self_attn.o_proj.weight", 8, 8);
        writeSafetensorsHeader(modelDir, tensors);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString());

        assertFalse(validation.allowed());
        assertTrue(validation.messages().stream().anyMatch(message ->
                message.contains("layers.1.self_attn.q_proj.weight")
                        && message.contains("expected 16")));
        assertTrue(validation.messages().stream().anyMatch(message ->
                message.contains("layers.1.self_attn.o_proj.weight")
                        && message.contains("expected 16")));
    }

    @Test
    void gemma4UnifiedTextSafetensorRouteRejectsWrongSlidingKvRows() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.language_model.layers.0.self_attn.k_proj.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.0.self_attn.v_proj.weight", 8, 8);
        writeSafetensorsHeader(modelDir, tensors);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString());

        assertFalse(validation.allowed());
        assertTrue(validation.messages().stream().anyMatch(message ->
                message.contains("layers.0.self_attn.k_proj.weight")
                        && message.contains("expected 4")));
        assertTrue(validation.messages().stream().anyMatch(message ->
                message.contains("layers.0.self_attn.v_proj.weight")
                        && message.contains("expected 4")));
    }

    @Test
    void gemma4UnifiedTextSafetensorRouteRejectsMissingLayerScalar() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        tensors.remove("model.language_model.layers.0.layer_scalar");
        writeSafetensorsHeader(modelDir, tensors);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString());

        assertFalse(validation.allowed());
        assertTrue(validation.messages().stream().anyMatch(message -> message.contains("layer_scalar")));
    }

    @Test
    void gemma4UnifiedPackedMoeRouteRejectsBeforeSafetensorHeaderInspection() throws Exception {
        Path modelDir = gemma4UnifiedPackedMoeModelDir();

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it-moe",
                        modelDir.toString());

        assertFalse(validation.allowed());
        assertTrue(validation.messages().get(0).contains("packed MoE safetensor runtime is not enabled"));
        assertTrue(validation.messages().stream().anyMatch(message -> message.contains("packed expert routing")));
        assertTrue(validation.messages().stream().noneMatch(message -> message.contains("no .safetensors file")));
    }

    @Test
    void gemma4UnifiedImageRequestRejectsBeforeSafetensorHeaderInspection() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedMultimodalExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        true,
                        false);

        assertFalse(validation.allowed());
        assertTrue(validation.messages().get(0).contains("multimodal safetensor runtime is not enabled"));
        assertTrue(validation.messages().stream().anyMatch(message -> message.contains("--image")));
        assertTrue(validation.messages().stream().anyMatch(message -> message.contains("guarded dense text decoder")));
        assertTrue(validation.messages().stream().noneMatch(message -> message.contains("no .safetensors file")));
    }

    @Test
    void gemma4UnifiedImageRequestReportsDetectedProjectorTensors() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.embed_vision.embedding_projection.weight", 8, 8);
        putTensor(tensors, "model.embed_audio.embedding_projection.weight", 8, 8);
        writeSafetensorsHeader(modelDir, tensors);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedMultimodalExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        true,
                        false);

        assertFalse(validation.allowed());
        assertTrue(validation.messages().stream()
                .anyMatch(message -> message.contains("vision/audio projector tensors detected")));
        assertTrue(validation.messages().stream()
                .anyMatch(message -> message.contains("no 12B weight payload was loaded")));
    }

    @Test
    void gemma4UnifiedImageRequestReportsVideoProjectorAndPackedMoeHeaderSignals() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.embed_vision.embedding_projection.weight", 8, 8);
        putTensor(tensors, "model.embed_audio.embedding_projection.weight", 8, 8);
        putTensor(tensors, "model.embed_video.embedding_projection.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.router.weight", 4, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", 16, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.up_proj.weight", 16, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.down_proj.weight", 8, 16);
        writeSafetensorsHeader(modelDir, tensors);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedMultimodalExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        true,
                        false);

        assertFalse(validation.allowed());
        assertTrue(validation.messages().stream()
                .anyMatch(message -> message.contains("vision/audio/video projector tensors detected")));
        assertTrue(validation.messages().stream()
                .anyMatch(message -> message.contains("packed MoE router and expert tensors detected")));
    }

    @Test
    void gemma4UnifiedOcrRequestRejectsBeforeSafetensorHeaderInspection() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedMultimodalExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        false,
                        true);

        assertFalse(validation.allowed());
        assertTrue(validation.messages().stream().anyMatch(message -> message.contains("--ocr")));
        assertTrue(validation.messages().stream().noneMatch(message -> message.contains("no .safetensors file")));
    }

    @Test
    void gemma4UnifiedMultimodalGuardAllowsTextOnlyRequest() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedMultimodalExecutionRoute(
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        false,
                        false);

        assertTrue(validation.allowed());
    }

    @Test
    void routePreflightDiagnosticContractValidatesRuntimeCapabilityLists() {
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.READY_RUNTIME_CAPABILITIES,
                List.of(RuntimeCapability.GEMMA4_GUARDED_TEXT_DECODER));
        details.put(ProblemDetail.MISSING_RUNTIME_CAPABILITIES,
                List.of(MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER));
        Map<String, Object> problem = new LinkedHashMap<>();
        problem.put(Problem.CODE, ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING);
        problem.put(Problem.SEVERITY, "error");
        problem.put(Problem.MESSAGE, "packed MoE runtime is missing");
        problem.put(Problem.DETAILS, details);

        assertEquals(List.of(), RoutePreflightDiagnosticContract.validateProblems(List.of(problem)));

        details.put(ProblemDetail.READY_RUNTIME_CAPABILITIES, List.of("gemma4_fake_decoder"));

        assertTrue(RoutePreflightDiagnosticContract.validateProblems(List.of(problem)).stream().anyMatch(message ->
                message.contains(ProblemDetail.READY_RUNTIME_CAPABILITIES)
                        && message.contains("unknown value")));
    }

    @Test
    void routePreflightDiagnosticContractValidatesInputModeSupportMatrix() {
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.SUPPORTED_INPUT_MODES, List.of(InputMode.TEXT));
        details.put(ProblemDetail.BLOCKED_INPUT_MODES,
                List.of(InputMode.IMAGE, InputMode.AUDIO, InputMode.VIDEO));
        details.put(ProblemDetail.BLOCKED_CAPABILITIES,
                List.of(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER));
        Map<String, Object> problem = new LinkedHashMap<>();
        problem.put(Problem.CODE, ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING);
        problem.put(Problem.SEVERITY, "error");
        problem.put(Problem.MESSAGE, "Gemma 4 multimodal runtime is missing");
        problem.put(Problem.DETAILS, details);

        assertEquals(List.of(), RoutePreflightDiagnosticContract.validateProblems(List.of(problem)));

        details.put(ProblemDetail.BLOCKED_INPUT_MODES, List.of("telepathy"));

        assertTrue(RoutePreflightDiagnosticContract.validateProblems(List.of(problem)).stream().anyMatch(message ->
                message.contains(ProblemDetail.BLOCKED_INPUT_MODES)
                        && message.contains("unknown value")));
    }

    @Test
    void routePreflightDiagnosticContractValidatesHeaderInspectionMetrics() {
        Map<String, Object> headerInspection = new LinkedHashMap<>();
        headerInspection.put(HeaderInspection.SAFETENSOR_FILE_COUNT, 1);
        headerInspection.put(HeaderInspection.TENSOR_COUNT, 33);
        headerInspection.put(HeaderInspection.HEADER_BYTES_READ, 4096L);
        headerInspection.put(HeaderInspection.PAYLOAD_BYTES_LOADED, 0L);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.HEADER_INSPECTION, headerInspection);
        Map<String, Object> problem = new LinkedHashMap<>();
        problem.put(Problem.CODE, ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH);
        problem.put(Problem.SEVERITY, "error");
        problem.put(Problem.MESSAGE, "Gemma 4 header contract mismatch");
        problem.put(Problem.DETAILS, details);

        assertEquals(List.of(), RoutePreflightDiagnosticContract.validateProblems(List.of(problem)));

        headerInspection.put(HeaderInspection.PAYLOAD_BYTES_LOADED, -1L);

        assertTrue(RoutePreflightDiagnosticContract.validateProblems(List.of(problem)).stream().anyMatch(message ->
                message.contains(HeaderInspection.PAYLOAD_BYTES_LOADED)
                        && message.contains("non-negative integer")));
    }

    @Test
    void routePreflightDiagnosticContractValidatesTensorInventoryMetrics() {
        Map<String, Object> tensorInventory = new LinkedHashMap<>();
        tensorInventory.put(TensorInventory.TEXT_DECODER_TENSORS, 28);
        tensorInventory.put(TensorInventory.EMBEDDING_TENSORS, 1);
        tensorInventory.put(TensorInventory.LOGITS_HEAD_TENSORS, 0);
        tensorInventory.put(TensorInventory.MULTIMODAL_PROJECTOR_TENSORS, 1);
        tensorInventory.put(TensorInventory.VISION_TOWER_TENSORS, 0);
        tensorInventory.put(TensorInventory.AUDIO_TOWER_TENSORS, 0);
        tensorInventory.put(TensorInventory.VIDEO_TOWER_TENSORS, 0);
        tensorInventory.put(TensorInventory.PACKED_MOE_ROUTER_TENSORS, 1);
        tensorInventory.put(TensorInventory.PACKED_MOE_EXPERT_TENSORS, 1);
        tensorInventory.put(TensorInventory.UNCLASSIFIED_TENSORS, 0);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.TENSOR_INVENTORY, tensorInventory);
        Map<String, Object> problem = new LinkedHashMap<>();
        problem.put(Problem.CODE, ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING);
        problem.put(Problem.SEVERITY, "error");
        problem.put(Problem.MESSAGE, "Gemma 4 packed MoE runtime is missing");
        problem.put(Problem.DETAILS, details);

        assertEquals(List.of(), RoutePreflightDiagnosticContract.validateProblems(List.of(problem)));

        tensorInventory.put(TensorInventory.PACKED_MOE_EXPERT_TENSORS, -1);

        assertTrue(RoutePreflightDiagnosticContract.validateProblems(List.of(problem)).stream().anyMatch(message ->
                message.contains(TensorInventory.PACKED_MOE_EXPERT_TENSORS)
                        && message.contains("non-negative integer")));
    }

    @Test
    void routePreflightDiagnosticContractValidatesComponentReadinessBooleans() {
        Map<String, Object> readiness = new LinkedHashMap<>();
        readiness.put(ComponentReadiness.TEXT_DECODER_READY, true);
        readiness.put(ComponentReadiness.VISION_PROJECTOR_READY, true);
        readiness.put(ComponentReadiness.AUDIO_PROJECTOR_READY, false);
        readiness.put(ComponentReadiness.VIDEO_PROJECTOR_READY, false);
        readiness.put(ComponentReadiness.MULTIMODAL_PROJECTOR_READY, true);
        readiness.put(ComponentReadiness.PACKED_MOE_ROUTER_READY, true);
        readiness.put(ComponentReadiness.PACKED_MOE_EXPERTS_READY, true);
        readiness.put(ComponentReadiness.PACKED_MOE_HEADER_READY, true);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.COMPONENT_READINESS, readiness);
        Map<String, Object> problem = new LinkedHashMap<>();
        problem.put(Problem.CODE, ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING);
        problem.put(Problem.SEVERITY, "error");
        problem.put(Problem.MESSAGE, "Gemma 4 multimodal runtime is missing");
        problem.put(Problem.DETAILS, details);

        assertEquals(List.of(), RoutePreflightDiagnosticContract.validateProblems(List.of(problem)));

        readiness.put(ComponentReadiness.TEXT_DECODER_READY, "true");

        assertTrue(RoutePreflightDiagnosticContract.validateProblems(List.of(problem)).stream().anyMatch(message ->
                message.contains(ComponentReadiness.TEXT_DECODER_READY)
                        && message.contains("must be a boolean")));
    }

    @Test
    void routePreflightDiagnosticContractValidatesExecutionProfileBooleans() {
        Map<String, Object> executionProfile = new LinkedHashMap<>();
        executionProfile.put(ExecutionProfile.GEMMA4_UNIFIED, true);
        executionProfile.put(ExecutionProfile.GUARDED_TEXT_DECODER_READY, true);
        executionProfile.put(ExecutionProfile.MULTIMODAL_EMBEDDER_READY, false);
        executionProfile.put(ExecutionProfile.PACKED_MOE_ROUTER_READY, false);
        executionProfile.put(ExecutionProfile.MOBILE_QAT_LOADER_READY, false);
        executionProfile.put(ExecutionProfile.PACKED_MOE_CHECKPOINT, false);
        executionProfile.put(ExecutionProfile.MOBILE_QAT_CHECKPOINT, false);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(ProblemDetail.EXECUTION_PROFILE, executionProfile);
        Map<String, Object> problem = new LinkedHashMap<>();
        problem.put(Problem.CODE, ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING);
        problem.put(Problem.SEVERITY, "error");
        problem.put(Problem.MESSAGE, "Gemma 4 multimodal runtime is missing");
        problem.put(Problem.DETAILS, details);

        assertEquals(List.of(), RoutePreflightDiagnosticContract.validateProblems(List.of(problem)));

        executionProfile.put(ExecutionProfile.GUARDED_TEXT_DECODER_READY, "true");

        assertTrue(RoutePreflightDiagnosticContract.validateProblems(List.of(problem)).stream().anyMatch(message ->
                message.contains(ExecutionProfile.GUARDED_TEXT_DECODER_READY)
                        && message.contains("must be a boolean")));
    }

    @Test
    void routeReportPreflightIncludesGemma4UnifiedSafetensorHeaderFailure() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString());

        assertFalse(preflight.passed());
        assertEquals(RoutePreflightReport.NOT_READY_EXIT_CODE, preflight.exitCode());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH.equals(problem.get("code"))
                        && problem.get("message").toString().contains("no .safetensors file")
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_TEXT_SAFETENSOR_HEADERS)
                        && details(problem).get(ProblemDetail.CHECKPOINT_PROFILE).equals("gemma4_text")
                        && details(problem).get(ProblemDetail.SUPPORTED_INPUT_MODES).equals(List.of(InputMode.TEXT))
                        && details(problem).get(ProblemDetail.BLOCKED_INPUT_MODES)
                                .equals(List.of(InputMode.IMAGE, InputMode.AUDIO, InputMode.VIDEO))
                        && details(problem).get(ProblemDetail.READY_RUNTIME_CAPABILITIES)
                                .equals(List.of(RuntimeCapability.GEMMA4_GUARDED_TEXT_DECODER))
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITIES)
                                .equals(List.of(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER))
                        && headerInspection(problem).get("safetensorFileCount").equals(0)
                        && headerInspection(problem).get("tensorCount").equals(0)
                        && headerInspection(problem).get("payloadBytesLoaded").equals(0L)
                        && tensorInventory(problem).get(TensorInventory.TEXT_DECODER_TENSORS).equals(0)
                        && componentReadiness(problem).get(ComponentReadiness.TEXT_DECODER_READY).equals(false)
                        && Boolean.TRUE.equals(executionProfile(problem)
                                .get(ExecutionProfile.GUARDED_TEXT_DECODER_READY))));
        assertTrue(preflight.nextActionMaps().stream().anyMatch(action ->
                ActionKind.INSPECT_MODULES.equals(action.get("kind"))
                        && ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH.equals(action.get("reason"))
                        && action.get("description").toString().contains("direct text-adapter tensor coverage")
                        && List.of("gollek", "modules", "--json").equals(action.get("argv"))
                        && actionDetails(action).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_TEXT_SAFETENSOR_HEADERS)));
    }

    @Test
    void routeReportPreflightIncludesGemma4UnifiedMultimodalFailureBeforeHeaderFailure() throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        true,
                        false);

        assertFalse(preflight.passed());
        assertEquals(RoutePreflightReport.NOT_READY_EXIT_CODE, preflight.exitCode());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING.equals(problem.get("code"))
                        && problem.get("message").toString().contains("multimodal safetensor runtime")
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER)
                        && details(problem).get(ProblemDetail.BLOCKED_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER)
                        && details(problem).get(ProblemDetail.RECOMMENDED_RUNTIME)
                                .equals("gemma4_unified_multimodal_embedder_runtime")
                        && details(problem).get(ProblemDetail.HEADER_PREFLIGHT).toString()
                                .contains("not detected")
                        && details(problem).get(ProblemDetail.DETECTED_PROJECTORS).equals(List.of())
                        && details(problem).get(ProblemDetail.DETECTED_PACKED_MOE).equals(List.of())
                        && details(problem).get(ProblemDetail.SUPPORTED_INPUT_MODES).equals(List.of(InputMode.TEXT))
                        && details(problem).get(ProblemDetail.BLOCKED_INPUT_MODES)
                                .equals(List.of(InputMode.IMAGE, InputMode.AUDIO, InputMode.VIDEO))
                        && details(problem).get(ProblemDetail.BLOCKED_CAPABILITIES)
                                .equals(List.of(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER))
                        && details(problem).get(ProblemDetail.READY_RUNTIME_CAPABILITIES)
                                .equals(List.of(RuntimeCapability.GEMMA4_GUARDED_TEXT_DECODER))
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITIES)
                                .equals(List.of(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER))
                        && Boolean.FALSE.equals(executionProfile(problem)
                                .get(ExecutionProfile.MULTIMODAL_EMBEDDER_READY))
                        && details(problem).get(ProblemDetail.REQUEST_INPUT_MODE).equals("image")));
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                problem.get("message").toString().contains("--image")));
        assertTrue(preflight.problemMaps().stream().noneMatch(problem ->
                problem.get("message").toString().contains("no .safetensors file")));
        assertTrue(preflight.nextActionMaps().stream().anyMatch(action ->
                ActionKind.INSPECT_MODULES.equals(action.get("kind"))
                        && ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING.equals(action.get("reason"))
                        && action.get("description").toString().contains("multimodal embedder")
                        && List.of("gollek", "modules", "--json").equals(action.get("argv"))
                        && actionDetails(action).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER)
                        && actionDetails(action).get(ProblemDetail.REQUEST_INPUT_MODE).equals("image")));
    }

    @Test
    void routeReportPreflightKeepsImageRequestClassifiedAsMultimodalWhenHeadersContainMoeSignals()
            throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.embed_vision.embedding_projection.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.router.weight", 4, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", 16, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.up_proj.weight", 16, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.down_proj.weight", 8, 16);
        writeSafetensorsHeader(modelDir, tensors);
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        true,
                        false);

        assertFalse(preflight.passed());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING.equals(problem.get("code"))
                        && problem.get("message").toString().contains("packed MoE router and expert tensors")
                        && details(problem).get(ProblemDetail.DETECTED_PROJECTORS).equals(List.of("vision"))
                        && details(problem).get(ProblemDetail.DETECTED_PACKED_MOE)
                                .equals(List.of("router", "experts"))
                        && details(problem).get(ProblemDetail.BLOCKED_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER)
                        && details(problem).get(ProblemDetail.RECOMMENDED_RUNTIME)
                                .equals("gemma4_unified_multimodal_embedder_runtime")
                        && details(problem).get(ProblemDetail.READY_RUNTIME_CAPABILITIES)
                                .equals(List.of(RuntimeCapability.GEMMA4_GUARDED_TEXT_DECODER))
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITIES)
                                .equals(List.of(MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER))
                        && headerInspection(problem).get("safetensorFileCount").equals(1)
                        && headerInspection(problem).get("tensorCount").equals(tensors.size())
                        && headerInspection(problem).get("payloadBytesLoaded").equals(0L)
                        && ((Number) tensorInventory(problem)
                                .get(TensorInventory.TEXT_DECODER_TENSORS)).intValue() > 0
                        && tensorInventory(problem).get(TensorInventory.MULTIMODAL_PROJECTOR_TENSORS)
                                .equals(1)
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_ROUTER_TENSORS)
                                .equals(1)
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_EXPERT_TENSORS)
                                .equals(3)
                        && componentReadiness(problem).get(ComponentReadiness.TEXT_DECODER_READY).equals(true)
                        && componentReadiness(problem).get(ComponentReadiness.VISION_PROJECTOR_READY).equals(true)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_HEADER_READY).equals(true)
                        && Boolean.TRUE.equals(executionProfile(problem)
                                .get(ExecutionProfile.GUARDED_TEXT_DECODER_READY))));
        assertTrue(preflight.problemMaps().stream().noneMatch(problem ->
                ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problem.get("code"))));
    }

    @Test
    void routeReportPreflightDetectsProjectorButKeepsReadinessFalseForWrongHiddenShape()
            throws Exception {
        Path modelDir = gemma4UnifiedTextModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.embed_vision.embedding_projection.weight", 7, 9);
        writeSafetensorsHeader(modelDir, tensors);
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it",
                        modelDir.toString(),
                        true,
                        false);

        assertFalse(preflight.passed());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING.equals(problem.get("code"))
                        && problem.get("message").toString().contains("incompatible with text hidden size")
                        && details(problem).get(ProblemDetail.DETECTED_PROJECTORS).equals(List.of("vision"))
                        && tensorInventory(problem).get(TensorInventory.MULTIMODAL_PROJECTOR_TENSORS)
                                .equals(1)
                        && componentReadiness(problem).get(ComponentReadiness.VISION_PROJECTOR_READY)
                                .equals(false)
                        && componentReadiness(problem).get(ComponentReadiness.MULTIMODAL_PROJECTOR_READY)
                                .equals(false)));
    }

    @Test
    void routeReportPreflightIncludesGemma4PackedMoeRemediationAction() throws Exception {
        Path modelDir = gemma4UnifiedPackedMoeModelDir();
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it-moe",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it-moe",
                        modelDir.toString());

        assertFalse(preflight.passed());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problem.get("code"))
                        && problem.get("message").toString().contains("packed MoE safetensor runtime")
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER)
                        && details(problem).get(ProblemDetail.BLOCKED_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER)
                        && details(problem).get(ProblemDetail.RECOMMENDED_RUNTIME)
                                .equals("gemma4_packed_moe_router_runtime")
                        && details(problem).get(ProblemDetail.CHECKPOINT_PROFILE).equals("gemma4_packed_moe")
                        && details(problem).get(ProblemDetail.SUPPORTED_INPUT_MODES).equals(List.of())
                        && details(problem).get(ProblemDetail.BLOCKED_INPUT_MODES)
                                .equals(List.of(InputMode.TEXT, InputMode.IMAGE, InputMode.AUDIO, InputMode.VIDEO))
                        && details(problem).get(ProblemDetail.BLOCKED_CAPABILITIES)
                                .equals(List.of(MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER))
                        && details(problem).get(ProblemDetail.READY_RUNTIME_CAPABILITIES).equals(List.of())
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITIES)
                                .equals(List.of(MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER))
                        && Boolean.TRUE.equals(executionProfile(problem)
                                .get(ExecutionProfile.PACKED_MOE_CHECKPOINT))));
        assertTrue(preflight.nextActionMaps().stream().anyMatch(action ->
                ActionKind.INSPECT_MODULES.equals(action.get("kind"))
                        && ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(action.get("reason"))
                        && action.get("description").toString().contains("packed-expert routing")
                        && List.of("gollek", "modules", "--json").equals(action.get("argv"))
                        && actionDetails(action).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER)));
    }

    @Test
    void routeReportPreflightIncludesGemma4PackedMoeHeaderInventory() throws Exception {
        Path modelDir = gemma4UnifiedPackedMoeModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.language_model.layers.0.mlp.router.weight", 128, 8);
        putGemma4MoeExpertTriplets(tensors, 0, 128, 704, 8);
        writeSafetensorsHeader(modelDir, tensors);
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it-moe",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it-moe",
                        modelDir.toString());

        assertFalse(preflight.passed());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problem.get("code"))
                        && headerInspection(problem).get(HeaderInspection.SAFETENSOR_FILE_COUNT).equals(1)
                        && headerInspection(problem).get(HeaderInspection.TENSOR_COUNT).equals(tensors.size())
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_ROUTER_TENSORS)
                                .equals(1)
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_EXPERT_TENSORS)
                                .equals(384)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_ROUTER_READY)
                                .equals(true)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_EXPERTS_READY)
                                .equals(true)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_HEADER_READY)
                                .equals(true)
                        && details(problem).get(ProblemDetail.DETECTED_PACKED_MOE)
                                .equals(List.of("router", "experts"))));
    }

    @Test
    void routeReportPreflightDetectsPackedMoeButKeepsReadinessFalseForWrongShapes() throws Exception {
        Path modelDir = gemma4UnifiedPackedMoeModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.language_model.layers.0.mlp.router.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", 16, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.up_proj.weight", 16, 8);
        putTensor(tensors, "model.language_model.layers.0.mlp.experts.0.down_proj.weight", 8, 16);
        writeSafetensorsHeader(modelDir, tensors);
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it-moe",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it-moe",
                        modelDir.toString());

        assertFalse(preflight.passed());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problem.get("code"))
                        && details(problem).get(ProblemDetail.HEADER_PREFLIGHT).toString()
                                .contains("incomplete or incompatible with config dimensions")
                        && details(problem).get(ProblemDetail.DETECTED_PACKED_MOE)
                                .equals(List.of("router", "experts"))
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_ROUTER_TENSORS)
                                .equals(1)
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_EXPERT_TENSORS)
                                .equals(3)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_ROUTER_READY)
                                .equals(false)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_EXPERTS_READY)
                                .equals(false)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_HEADER_READY)
                                .equals(false)));
    }

    @Test
    void routeReportPreflightDetectsPackedMoeButRejectsIncompleteExpertCoverage() throws Exception {
        Path modelDir = gemma4UnifiedPackedMoeModelDir();
        Map<String, TensorSpec> tensors = minimalGemma4UnifiedTextTensors();
        putTensor(tensors, "model.language_model.layers.0.mlp.router.weight", 128, 8);
        putGemma4MoeExpertTriplets(tensors, 0, 1, 704, 8);
        writeSafetensorsHeader(modelDir, tensors);
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-12B-it-moe",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-12B-it-moe",
                        modelDir.toString());

        assertFalse(preflight.passed());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING.equals(problem.get("code"))
                        && details(problem).get(ProblemDetail.HEADER_PREFLIGHT).toString()
                                .contains("incomplete or incompatible with config dimensions")
                        && details(problem).get(ProblemDetail.DETECTED_PACKED_MOE)
                                .equals(List.of("router", "experts"))
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_ROUTER_TENSORS)
                                .equals(1)
                        && tensorInventory(problem).get(TensorInventory.PACKED_MOE_EXPERT_TENSORS)
                                .equals(3)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_ROUTER_READY)
                                .equals(true)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_EXPERTS_READY)
                                .equals(false)
                        && componentReadiness(problem).get(ComponentReadiness.PACKED_MOE_HEADER_READY)
                                .equals(false)));
    }

    @Test
    void nativeBf16MatvecSafetensorRoutePassesGemma4UnifiedGuard() throws Exception {
        Path modelDir = modelDir("gemma4_text", "Gemma4ForCausalLM");

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-text-it",
                        modelDir.toString());

        assertTrue(validation.allowed());
    }

    @Test
    void nativeBf16MatvecSafetensorRouteSwitchesToLocalGgufEquivalentWhenAuto() throws Exception {
        Path modelDir = modelDir("gemma4", "Gemma4ForConditionalGeneration");
        Path gguf = Files.writeString(tempDir.resolve("gemma-4-E2B-it-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma4TextAlternateRuntime(
                        "safetensor",
                        "97cbf2",
                        modelDir.toString(),
                        false,
                        false,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertTrue(selection.selected());
        assertEquals("gguf", selection.provider());
        assertEquals(gguf.toAbsolutePath().normalize().toString(), selection.localPath());
        assertTrue(selection.notice().contains("switching to GGUF"));
        assertFalse(selection.cacheHit());
    }

    @Test
    void nativeBf16MatvecSafetensorRouteHonorsExplicitSafetensorProvider() throws Exception {
        Path modelDir = modelDir("gemma4", "Gemma4ForConditionalGeneration");
        Path gguf = Files.writeString(tempDir.resolve("gemma-4-E2B-it-explicit-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma4TextAlternateRuntime(
                        "safetensor",
                        "google/gemma-4-E2B-it",
                        modelDir.toString(),
                        true,
                        false,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertFalse(selection.selected());
        assertFalse(selection.hasNotice());
    }

    @Test
    void nativeBf16MatvecSafetensorRouteCanBeForcedToGgufWithAlternatePreference() throws Exception {
        Path modelDir = modelDir("gemma4", "Gemma4ForConditionalGeneration");
        Path gguf = Files.writeString(tempDir.resolve("gemma-4-E2B-it-forced-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma4TextAlternateRuntime(
                        "safetensor",
                        "google/gemma-4-E2B-it",
                        modelDir.toString(),
                        true,
                        true,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertTrue(selection.selected());
        assertEquals("gguf", selection.provider());
        assertEquals(gguf.toAbsolutePath().normalize().toString(), selection.localPath());
    }

    @Test
    void nativeBf16MatvecSafetensorRouteReusesCachedGgufEquivalentBeforeResolver() throws Exception {
        Path modelDir = modelDir("gemma4", "Gemma4ForConditionalGeneration");
        Path gguf = Files.writeString(tempDir.resolve("gemma-4-E2B-it-cache-Q4_K_M.gguf"), "");
        Path cacheDir = Files.createDirectory(tempDir.resolve("gemma4-text-gguf-route-cache"));
        AtomicBoolean resolverCalled = new AtomicBoolean(false);

        String previousCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY);
        String previousCacheEnabled = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY);
        try {
            System.setProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY,
                    cacheDir.toString());
            System.setProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                    "true");

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection first =
                    DirectSafetensorRoutePolicy.selectGemma4TextAlternateRuntime(
                            "safetensor",
                            "google/gemma-4-E2B-it",
                            modelDir.toString(),
                            false,
                            false,
                            "gguf"::equals,
                            (requested, localPath) -> Optional.of(gguf));
            assertTrue(first.selected());
            assertFalse(first.cacheHit());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection second =
                    DirectSafetensorRoutePolicy.selectGemma4TextAlternateRuntime(
                            "safetensor",
                            "google/gemma-4-E2B-it",
                            modelDir.toString(),
                            false,
                            false,
                            "gguf"::equals,
                            (requested, localPath) -> {
                                resolverCalled.set(true);
                                return Optional.empty();
                            });

            assertTrue(second.selected());
            assertEquals("gguf", second.provider());
            assertEquals(gguf.toAbsolutePath().normalize().toString(), second.localPath());
            assertFalse(second.hasNotice());
            assertTrue(second.cacheHit());
            assertEquals(DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_KIND, second.cacheKind());
            assertFalse(resolverCalled.get());
        } finally {
            restoreProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY,
                    previousCacheDir);
            restoreProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                    previousCacheEnabled);
        }
    }

    @Test
    void communityTextSafetensorRouteSwitchesQwenToLocalGgufEquivalentWhenAuto() throws Exception {
        Path modelDir = modelDir("qwen3", "Qwen3ForCausalLM");
        Path gguf = Files.writeString(tempDir.resolve("Qwen3-4B-Instruct-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                        "safetensor",
                        "Qwen/Qwen3-4B-Instruct",
                        modelDir.toString(),
                        false,
                        false,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertTrue(selection.selected());
        assertEquals("gguf", selection.provider());
        assertEquals("gguf", selection.format());
        assertEquals(gguf.toAbsolutePath().normalize().toString(), selection.localPath());
        assertTrue(selection.notice().contains("Detected Qwen safetensor checkpoint"));
        assertTrue(selection.notice().contains("switching to GGUF"));
        assertFalse(selection.cacheHit());
    }

    @Test
    void communityTextSafetensorRouteReusesCachedGgufEquivalentBeforeResolver() throws Exception {
        Path modelDir = modelDir("qwen3", "Qwen3ForCausalLM");
        Path gguf = Files.writeString(tempDir.resolve("Qwen3-4B-Instruct-cache-Q4_K_M.gguf"), "");
        Path cacheDir = Files.createDirectory(tempDir.resolve("community-text-gguf-route-cache"));
        AtomicBoolean resolverCalled = new AtomicBoolean(false);

        String previousCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY);
        String previousCacheEnabled = System.getProperty(
                DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY);
        try {
            System.setProperty(
                    DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY,
                    cacheDir.toString());
            System.setProperty(
                    DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                    "true");

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection first =
                    DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                            "safetensor",
                            "Qwen/Qwen3-4B-Instruct",
                            modelDir.toString(),
                            false,
                            false,
                            "gguf"::equals,
                            (requested, localPath) -> Optional.of(gguf));
            assertTrue(first.selected());
            assertFalse(first.cacheHit());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection second =
                    DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                            "safetensor",
                            "Qwen/Qwen3-4B-Instruct",
                            modelDir.toString(),
                            false,
                            false,
                            "gguf"::equals,
                            (requested, localPath) -> {
                                resolverCalled.set(true);
                                return Optional.empty();
                            });

            assertTrue(second.selected());
            assertEquals("gguf", second.provider());
            assertEquals(gguf.toAbsolutePath().normalize().toString(), second.localPath());
            assertFalse(second.hasNotice());
            assertTrue(second.cacheHit());
            assertEquals(DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_KIND, second.cacheKind());
            assertFalse(resolverCalled.get());
        } finally {
            restoreProperty(
                    DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY,
                    previousCacheDir);
            restoreProperty(
                    DirectSafetensorRoutePolicy.COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                    previousCacheEnabled);
        }
    }

    @Test
    void communityTextSafetensorRouteFindsLlamaGgufSiblingInModelDirectory() throws Exception {
        Path modelDir = modelDir("llama", "LlamaForCausalLM");
        Path gguf = Files.writeString(modelDir.resolve("Llama-3-8B-Instruct-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                        "safetensor",
                        "meta-llama/Llama-3-8B-Instruct",
                        modelDir.toString(),
                        false,
                        false,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.empty());

        assertTrue(selection.selected());
        assertEquals("gguf", selection.provider());
        assertEquals("gguf", selection.format());
        assertEquals(gguf.toAbsolutePath().normalize().toString(), selection.localPath());
        assertTrue(selection.notice().contains("Detected Llama safetensor checkpoint"));
    }

    @Test
    void communityTextSafetensorRouteHonorsExplicitSafetensorProvider() throws Exception {
        Path modelDir = modelDir("phi3", "Phi3ForCausalLM");
        Path gguf = Files.writeString(tempDir.resolve("Phi-3-mini-4k-instruct-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                        "safetensor",
                        "microsoft/Phi-3-mini-4k-instruct",
                        modelDir.toString(),
                        true,
                        false,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertFalse(selection.selected());
        assertFalse(selection.hasNotice());
    }

    @Test
    void communityTextSafetensorRouteCanBeForcedForExplicitProvider() throws Exception {
        Path modelDir = modelDir("falcon", "FalconForCausalLM");
        Path gguf = Files.writeString(tempDir.resolve("falcon-7b-instruct-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                        "safetensor",
                        "tiiuae/falcon-7b-instruct",
                        modelDir.toString(),
                        true,
                        true,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertTrue(selection.selected());
        assertEquals("gguf", selection.provider());
        assertEquals("gguf", selection.format());
        assertEquals(gguf.toAbsolutePath().normalize().toString(), selection.localPath());
    }

    @Test
    void communityTextSafetensorRouteSkipsQwenVlMultimodalModels() throws Exception {
        Path modelDir = modelDir("qwen2_vl", "Qwen2VLForConditionalGeneration");
        Path gguf = Files.writeString(tempDir.resolve("Qwen2-VL-7B-Instruct-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                        "safetensor",
                        "Qwen/Qwen2-VL-7B-Instruct",
                        modelDir.toString(),
                        false,
                        false,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertFalse(selection.selected());
        assertFalse(selection.hasNotice());
    }

    @Test
    void communityTextSafetensorRouteSkipsBertEncoderModels() throws Exception {
        Path modelDir = modelDir("bert", "BertForMaskedLM");
        Path gguf = Files.writeString(tempDir.resolve("bert-base-uncased-Q4_K_M.gguf"), "");

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectCommunityTextGgufAlternateRuntime(
                        "safetensor",
                        "google-bert/bert-base-uncased",
                        modelDir.toString(),
                        false,
                        false,
                        "gguf"::equals,
                        (requested, localPath) -> Optional.of(gguf));

        assertFalse(selection.selected());
        assertFalse(selection.hasNotice());
    }

    @Test
    void gemma4MobileQatSafetensorRouteIsBlockedBeforeDirectEngine() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-mobile-qat"));
        writeGemma4MobileQatConfig(modelDir);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-E2B-it-qat-mobile-transformers",
                        modelDir.toString());

        assertFalse(validation.allowed());
        assertTrue(validation.messages().get(0).contains("Gemma 4 mobile QAT safetensor runtime is not enabled"));
        assertTrue(validation.messages().get(1).contains("mobile audio/vision/text towers"));
    }

    @Test
    void routeReportPreflightIncludesGemma4MobileQatLoaderFailure() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-mobile-qat-route-report"));
        writeGemma4MobileQatConfig(modelDir);
        RoutePreflightReport base = RoutePreflightReport.evaluate(
                "google/gemma-4-E2B-it-qat-mobile-transformers",
                modelDir.toString(),
                modelDir.toString(),
                "safetensor",
                "safetensors",
                false,
                false);

        RoutePreflightReport preflight =
                DirectSafetensorRoutePreflight.applyGemma4UnifiedValidation(
                        base,
                        "safetensor",
                        "google/gemma-4-E2B-it-qat-mobile-transformers",
                        modelDir.toString());

        assertFalse(preflight.passed());
        assertTrue(preflight.problemMaps().stream().anyMatch(problem ->
                ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING.equals(problem.get("code"))
                        && problem.get("message").toString().contains("mobile QAT safetensor runtime")
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_MOBILE_QAT_LOADER)
                        && details(problem).get(ProblemDetail.RUNTIME_ROUTE).equals("safetensor_direct")
                        && details(problem).get(ProblemDetail.SUPPORTED_INPUT_MODES).equals(List.of())
                        && details(problem).get(ProblemDetail.BLOCKED_INPUT_MODES)
                                .equals(List.of(InputMode.TEXT, InputMode.IMAGE, InputMode.AUDIO, InputMode.VIDEO))
                        && details(problem).get(ProblemDetail.READY_RUNTIME_CAPABILITIES).equals(List.of())
                        && details(problem).get(ProblemDetail.MISSING_RUNTIME_CAPABILITIES)
                                .equals(List.of(MissingRuntimeCapability.GEMMA4_MOBILE_QAT_LOADER))
                        && Boolean.TRUE.equals(executionProfile(problem)
                                .get(ExecutionProfile.MOBILE_QAT_CHECKPOINT))));
        assertTrue(preflight.nextActionMaps().stream().anyMatch(action ->
                ActionKind.INSPECT_MODULES.equals(action.get("kind"))
                        && ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING.equals(action.get("reason"))
                        && action.get("description").toString().contains("mobile QAT loader")
                        && List.of("gollek", "modules", "--json").equals(action.get("argv"))
                        && actionDetails(action).get(ProblemDetail.MISSING_RUNTIME_CAPABILITY)
                                .equals(MissingRuntimeCapability.GEMMA4_MOBILE_QAT_LOADER)));
    }

    @Test
    void gemma4MobileQatSafetensorRouteSwitchesToLocalLiteRtEquivalent() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-mobile-qat-auto"));
        Path litert = Files.writeString(tempDir.resolve("gemma-4-E2B-it.litertlm"), "");
        writeGemma4MobileQatConfig(modelDir);

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                        "safetensor",
                        "0576e9",
                        modelDir.toString(),
                        false,
                        false,
                        "litert"::equals,
                        ref -> "0576e9".equals(ref) ? java.util.Optional.of(litert) : java.util.Optional.empty());

        assertTrue(selection.selected());
        assertEquals("litert", selection.provider());
        assertEquals(litert.toAbsolutePath().normalize().toString(), selection.localPath());
        assertTrue(selection.notice().contains("switching to local LiteRT-LM artifact"));
    }

    @Test
    void gemma4MobileQatSafetensorRouteHonorsExplicitSafetensorProvider() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-mobile-qat-explicit"));
        Path litert = Files.writeString(tempDir.resolve("gemma-4-E2B-it-explicit.litertlm"), "");
        writeGemma4MobileQatConfig(modelDir);

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                        "safetensor",
                        "0576e9",
                        modelDir.toString(),
                        true,
                        false,
                        "litert"::equals,
                        ref -> java.util.Optional.of(litert));

        assertFalse(selection.selected());
        assertFalse(selection.hasNotice());
    }

    @Test
    void gemma4MobileQatSafetensorRouteSelectsLiteRtFallbackWhenProviderIsInactive() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-mobile-qat-no-provider"));
        Path litert = Files.writeString(tempDir.resolve("gemma-4-E2B-it-unavailable.litertlm"), "");
        writeGemma4MobileQatConfig(modelDir);

        DirectSafetensorRoutePolicy.AlternateRuntimeSelection selection =
                DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                        "safetensor",
                        "0576e9",
                        modelDir.toString(),
                        false,
                        false,
                        provider -> false,
                        ref -> java.util.Optional.of(litert));

        assertTrue(selection.selected());
        assertEquals("litert", selection.provider());
        assertEquals(litert.toAbsolutePath().normalize().toString(), selection.localPath());
        assertTrue(selection.hasNotice());
        assertTrue(selection.notice().contains("trying the local LiteRT-LM fallback"));
    }

    @Test
    void gemma4MobileQatSafetensorRouteReusesCachedLiteRtEquivalent() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma-4-E2B-it-qat-mobile-transformers"));
        Path litert = Files.writeString(tempDir.resolve("gemma-4-E2B-it.litertlm"), "");
        Path cacheDir = Files.createDirectory(tempDir.resolve("route-cache"));
        writeGemma4MobileQatConfig(modelDir);

        String previousCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY);
        try {
            System.setProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                    cacheDir.toString());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection first =
                    DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                            "safetensor",
                            "0576e9",
                            modelDir.toString(),
                            false,
                            false,
                            "litert"::equals,
                            ref -> "0576e9".equals(ref) ? Optional.of(litert) : Optional.empty());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection second =
                    DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                            "safetensor",
                            "0576e9",
                            modelDir.toString(),
                            false,
                            false,
                            "litert"::equals,
                            ref -> Optional.empty());

            assertTrue(first.selected());
            assertTrue(second.selected());
            assertFalse(first.cacheHit());
            assertTrue(second.cacheHit());
            assertEquals(DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_KIND, second.cacheKind());
            assertEquals(litert.toAbsolutePath().normalize().toString(), second.localPath());
            assertFalse(second.hasNotice());
        } finally {
            restoreProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                    previousCacheDir);
        }
    }

    @Test
    void gemma4MobileQatSafetensorRouteUsesCacheBeforeResolver() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma-4-E2B-it-qat-mobile-transformers-cache-first"));
        Path litert = Files.writeString(tempDir.resolve("gemma-4-E2B-it.litertlm"), "");
        Path cacheDir = Files.createDirectory(tempDir.resolve("route-cache-first"));
        writeGemma4MobileQatConfig(modelDir);
        AtomicBoolean resolverCalled = new AtomicBoolean(false);

        String previousCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY);
        try {
            System.setProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                    cacheDir.toString());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection first =
                    DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                            "safetensor",
                            "0576e9",
                            modelDir.toString(),
                            false,
                            false,
                            "litert"::equals,
                            ref -> Optional.of(litert));
            assertTrue(first.selected());
            assertFalse(first.cacheHit());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection second =
                    DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                            "safetensor",
                            "0576e9",
                            modelDir.toString(),
                            false,
                            false,
                            "litert"::equals,
                            ref -> {
                                resolverCalled.set(true);
                                return Optional.empty();
                            });

            assertTrue(second.selected());
            assertEquals(litert.toAbsolutePath().normalize().toString(), second.localPath());
            assertFalse(second.hasNotice());
            assertTrue(second.cacheHit());
            assertEquals(DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_KIND, second.cacheKind());
            assertFalse(resolverCalled.get());
        } finally {
            restoreProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                    previousCacheDir);
        }
    }

    @Test
    void cachedGemma4MobileQatLiteRtRouteSelectsWithoutResolverFallback() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma-4-E2B-it-qat-mobile-transformers-cache-only"));
        Path litert = Files.writeString(tempDir.resolve("gemma-4-E2B-it.litertlm"), "");
        Path cacheDir = Files.createDirectory(tempDir.resolve("route-cache-only"));
        writeGemma4MobileQatConfig(modelDir);

        String previousCacheDir = System.getProperty(
                DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY);
        try {
            System.setProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                    cacheDir.toString());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection first =
                    DirectSafetensorRoutePolicy.selectGemma4MobileQatAlternateRuntime(
                            "safetensor",
                            "0576e9",
                            modelDir.toString(),
                            false,
                            false,
                            "litert"::equals,
                            ref -> Optional.of(litert));
            assertTrue(first.selected());

            DirectSafetensorRoutePolicy.AlternateRuntimeSelection cached =
                    DirectSafetensorRoutePolicy.selectCachedGemma4MobileQatLiteRtAlternateRuntime(
                            "safetensor",
                            "0576e9",
                            modelDir.toString(),
                            false,
                            false);

            assertTrue(cached.selected());
            assertEquals("litert", cached.provider());
            assertEquals(litert.toAbsolutePath().normalize().toString(), cached.localPath());
            assertFalse(cached.hasNotice());
            assertTrue(cached.cacheHit());
            assertEquals(DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_KIND, cached.cacheKind());
        } finally {
            restoreProperty(
                    DirectSafetensorRoutePolicy.GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                    previousCacheDir);
        }
    }

    @Test
    void gemma4QatTextOnlySafetensorRoutePassesMobileQatGuard() throws Exception {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-qat-text-only"));
        Files.writeString(modelDir.resolve("config.json"), """
                {
                  "model_type": "gemma4",
                  "quantization_config": {
                    "quant_method": "gemma",
                    "num_bits": 4
                  },
                  "text_config": {
                    "model_type": "gemma4_text"
                  }
                }
                """);

        DirectSafetensorRoutePolicy.RouteValidation validation =
                DirectSafetensorRoutePolicy.validateGemma4UnifiedExecutionRoute(
                        "safetensor",
                        "google/gemma-4-text-qat",
                        modelDir.toString());

        assertTrue(validation.allowed());
    }

    @Test
    void directLiteRtStreamRequiresPropertyAndArtifact() throws Exception {
        Path litert = Files.writeString(tempDir.resolve("demo.litertlm"), "");

        String property = "gollek.cli.enable_direct_litert_stream";
        String previous = System.getProperty(property);
        try {
            System.clearProperty(property);
            assertFalse(DirectSafetensorRoutePolicy.shouldUseDirectLiteRtStreamPath("litert", litert.toString()));

            System.setProperty(property, "true");
            assertTrue(DirectSafetensorRoutePolicy.shouldUseDirectLiteRtStreamPath("litert", litert.toString()));
        } finally {
            if (previous == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, previous);
            }
        }
    }

    private Path modelDir(String modelType) throws IOException {
        return modelDir(modelType, "GemmaForCausalLM");
    }

    private Path modelDir(String modelType, String architectureClassName) throws IOException {
        Path modelDir = Files.createDirectory(tempDir.resolve(modelType + "-" + System.nanoTime()));
        Files.writeString(modelDir.resolve("config.json"), """
                {
                  "model_type": "%s",
                  "architectures": ["%s"]
                }
                """.formatted(modelType, architectureClassName));
        return modelDir;
    }

    private Path gemma4UnifiedTextModelDir() throws IOException {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-unified-text-" + System.nanoTime()));
        Files.writeString(modelDir.resolve("config.json"), """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "tie_word_embeddings": true,
                  "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "intermediate_size": 16,
                    "vocab_size": 32,
                    "head_dim": 4,
                    "global_head_dim": 8,
                    "attention_k_eq_v": true,
                    "tie_word_embeddings": true,
                    "layer_types": ["sliding_attention", "full_attention"]
                  },
                  "vision_config": {"model_type": "gemma4_vision"},
                  "audio_config": {"model_type": "gemma4_audio"}
                }
                """);
        return modelDir;
    }

    private Path gemma4UnifiedTwelveBTextModelDir() throws IOException {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-unified-12b-text-" + System.nanoTime()));
        Files.writeString(modelDir.resolve("config.json"), """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "tie_word_embeddings": true,
                  "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 3840,
                    "num_hidden_layers": 48,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 1,
                    "intermediate_size": 15360,
                    "vocab_size": 262144,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "attention_k_eq_v": true,
                    "tie_word_embeddings": true,
                    "sliding_window": 1024,
                    "max_position_embeddings": 131072,
                    "layer_types": [%s]
                  },
                  "vision_config": {"model_type": "gemma4_unified_vision"},
                  "audio_config": {"model_type": "gemma4_unified_audio"}
                }
                """.formatted(gemma4TwelveBLayerTypesJson()));
        return modelDir;
    }

    private Path gemma4UnifiedPackedMoeModelDir() throws IOException {
        Path modelDir = Files.createDirectory(tempDir.resolve("gemma4-unified-moe-" + System.nanoTime()));
        Files.writeString(modelDir.resolve("config.json"), """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "intermediate_size": 16,
                    "vocab_size": 32,
                    "enable_moe_block": true,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 704
                  },
                  "vision_config": {"model_type": "gemma4_vision"},
                  "audio_config": {"model_type": "gemma4_audio"}
                }
                """);
        return modelDir;
    }

    private static Map<String, TensorSpec> minimalGemma4UnifiedTextTensors() {
        Map<String, TensorSpec> tensors = new LinkedHashMap<>();
        putTensor(tensors, "model.language_model.embed_tokens.weight", 32, 8);
        putTensor(tensors, "model.language_model.norm.weight", 8);

        putLayerCommonTensors(tensors, 0);
        putTensor(tensors, "model.language_model.layers.0.self_attn.q_proj.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.0.self_attn.k_proj.weight", 4, 8);
        putTensor(tensors, "model.language_model.layers.0.self_attn.v_proj.weight", 4, 8);
        putTensor(tensors, "model.language_model.layers.0.self_attn.o_proj.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.0.self_attn.q_norm.weight", 4);
        putTensor(tensors, "model.language_model.layers.0.self_attn.k_norm.weight", 4);
        putLayerMlpTensors(tensors, 0);

        putLayerCommonTensors(tensors, 1);
        putTensor(tensors, "model.language_model.layers.1.self_attn.q_proj.weight", 16, 8);
        putTensor(tensors, "model.language_model.layers.1.self_attn.k_proj.weight", 8, 8);
        putTensor(tensors, "model.language_model.layers.1.self_attn.o_proj.weight", 8, 16);
        putTensor(tensors, "model.language_model.layers.1.self_attn.q_norm.weight", 8);
        putTensor(tensors, "model.language_model.layers.1.self_attn.k_norm.weight", 8);
        putLayerMlpTensors(tensors, 1);
        return tensors;
    }

    private static Map<String, TensorSpec> gemma4UnifiedTwelveBTextTensors() {
        Map<String, TensorSpec> tensors = new LinkedHashMap<>();
        putTensor(tensors, "model.language_model.embed_tokens.weight", 262144, 3840);
        putTensor(tensors, "model.language_model.norm.weight", 3840);
        for (int layer = 0; layer < 48; layer++) {
            putTwelveBLayerTensors(tensors, layer);
        }
        return tensors;
    }

    private static void putTwelveBLayerTensors(Map<String, TensorSpec> tensors, int layer) {
        boolean fullAttention = isGemma4TwelveBFullAttentionLayer(layer);
        int headDim = fullAttention ? 512 : 256;
        int queryRows = 16 * headDim;
        int kvRows = (fullAttention ? 1 : 8) * headDim;
        String prefix = "model.language_model.layers.%d.".formatted(layer);

        putTensor(tensors, prefix + "layer_scalar", 1);
        putTensor(tensors, prefix + "input_layernorm.weight", 3840);
        putTensor(tensors, prefix + "post_attention_layernorm.weight", 3840);
        putTensor(tensors, prefix + "pre_feedforward_layernorm.weight", 3840);
        putTensor(tensors, prefix + "post_feedforward_layernorm.weight", 3840);
        putTensor(tensors, prefix + "self_attn.q_proj.weight", queryRows, 3840);
        putTensor(tensors, prefix + "self_attn.k_proj.weight", kvRows, 3840);
        if (!fullAttention) {
            putTensor(tensors, prefix + "self_attn.v_proj.weight", kvRows, 3840);
        }
        putTensor(tensors, prefix + "self_attn.o_proj.weight", 3840, queryRows);
        putTensor(tensors, prefix + "self_attn.q_norm.weight", headDim);
        putTensor(tensors, prefix + "self_attn.k_norm.weight", headDim);
        putTensor(tensors, prefix + "mlp.gate_proj.weight", 15360, 3840);
        putTensor(tensors, prefix + "mlp.up_proj.weight", 15360, 3840);
        putTensor(tensors, prefix + "mlp.down_proj.weight", 3840, 15360);
    }

    private static String gemma4TwelveBLayerTypesJson() {
        StringBuilder json = new StringBuilder();
        for (int layer = 0; layer < 48; layer++) {
            if (layer > 0) {
                json.append(", ");
            }
            json.append('"')
                    .append(isGemma4TwelveBFullAttentionLayer(layer)
                            ? "full_attention"
                            : "sliding_attention")
                    .append('"');
        }
        return json.toString();
    }

    private static boolean isGemma4TwelveBFullAttentionLayer(int layer) {
        return layer > 0 && (layer + 1) % 6 == 0;
    }

    private static void putLayerCommonTensors(Map<String, TensorSpec> tensors, int layer) {
        putTensor(tensors, "model.language_model.layers.%d.layer_scalar".formatted(layer), 1);
        putTensor(tensors, "model.language_model.layers.%d.input_layernorm.weight".formatted(layer), 8);
        putTensor(tensors, "model.language_model.layers.%d.post_attention_layernorm.weight".formatted(layer), 8);
        putTensor(tensors, "model.language_model.layers.%d.pre_feedforward_layernorm.weight".formatted(layer), 8);
        putTensor(tensors, "model.language_model.layers.%d.post_feedforward_layernorm.weight".formatted(layer), 8);
    }

    private static void putLayerMlpTensors(Map<String, TensorSpec> tensors, int layer) {
        putTensor(tensors, "model.language_model.layers.%d.mlp.gate_proj.weight".formatted(layer), 16, 8);
        putTensor(tensors, "model.language_model.layers.%d.mlp.up_proj.weight".formatted(layer), 16, 8);
        putTensor(tensors, "model.language_model.layers.%d.mlp.down_proj.weight".formatted(layer), 8, 16);
    }

    private static void putGemma4MoeExpertTriplets(
            Map<String, TensorSpec> tensors,
            int layer,
            int experts,
            int intermediateSize,
            int hiddenSize) {
        for (int expert = 0; expert < experts; expert++) {
            String prefix = "model.language_model.layers.%d.mlp.experts.%d".formatted(layer, expert);
            putTensor(tensors, prefix + ".gate_proj.weight", intermediateSize, hiddenSize);
            putTensor(tensors, prefix + ".up_proj.weight", intermediateSize, hiddenSize);
            putTensor(tensors, prefix + ".down_proj.weight", hiddenSize, intermediateSize);
        }
    }

    private static void putTensor(Map<String, TensorSpec> tensors, String name, long... shape) {
        tensors.put(name, new TensorSpec("BF16", shape));
    }

    private static void writeSafetensorsHeader(Path modelDir, Map<String, TensorSpec> tensors) throws IOException {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        for (Map.Entry<String, TensorSpec> entry : tensors.entrySet()) {
            ObjectNode tensor = root.putObject(entry.getKey());
            tensor.put("dtype", entry.getValue().dtype());
            var shape = tensor.putArray("shape");
            for (long dim : entry.getValue().shape()) {
                shape.add(dim);
            }
            var offsets = tensor.putArray("data_offsets");
            offsets.add(0);
            offsets.add(0);
        }

        byte[] header = OBJECT_MAPPER.writeValueAsBytes(root);
        ByteBuffer length = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        length.putLong(header.length);
        byte[] file = new byte[Long.BYTES + header.length];
        System.arraycopy(length.array(), 0, file, 0, Long.BYTES);
        System.arraycopy(header, 0, file, Long.BYTES, header.length);
        Files.write(modelDir.resolve("model.safetensors"), file);
    }

    /**
     * Minimal SafeTensors header entry used by route-policy tests without allocating model payload bytes.
     */
    private record TensorSpec(String dtype, long[] shape) {
    }

    private static void writeGemma4MobileQatConfig(Path modelDir) throws IOException {
        Files.writeString(modelDir.resolve("config.json"), """
                {
                  "model_type": "gemma4",
                  "quantization_config": {
                    "quant_method": "gemma",
                    "num_bits": 4
                  },
                  "text_config": {
                    "model_type": "gemma4_text"
                  },
                  "vision_config": {
                    "model_type": "gemma4_vision"
                  },
                  "audio_config": {
                    "model_type": "gemma4_audio"
                  }
                }
                """);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> details(Map<String, Object> problem) {
        return (Map<String, Object>) problem.get(Problem.DETAILS);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> executionProfile(Map<String, Object> problem) {
        return (Map<String, Object>) details(problem).get(ProblemDetail.EXECUTION_PROFILE);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> headerInspection(Map<String, Object> problem) {
        return (Map<String, Object>) details(problem).get(ProblemDetail.HEADER_INSPECTION);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> tensorInventory(Map<String, Object> problem) {
        return (Map<String, Object>) details(problem).get(ProblemDetail.TENSOR_INVENTORY);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> componentReadiness(Map<String, Object> problem) {
        return (Map<String, Object>) details(problem).get(ProblemDetail.COMPONENT_READINESS);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> actionDetails(Map<String, Object> action) {
        return (Map<String, Object>) action.get(Action.DETAILS);
    }

    private static void restoreProperty(String property, String previous) {
        if (previous == null) {
            System.clearProperty(property);
        } else {
            System.setProperty(property, previous);
        }
    }
}
