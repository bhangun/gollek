package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

class OnnxInferenceProfileTest {

    @Test
    void disabledProfileProducesNoMetadata() {
        OnnxInferenceProfile profile = OnnxInferenceProfile.start(request(false));

        assertFalse(profile.enabled());
        assertTrue(profile.metadata("cpu", 2, 1, 10L).isEmpty());
    }

    @Test
    void requestParameterEnablesBenchMetadata() {
        OnnxInferenceProfile profile = OnnxInferenceProfile.start(request(true));

        assertTrue(profile.enabled());
        profile.recordTokenize(profile.mark());
        profile.recordInputPrepare(profile.mark());
        profile.recordOrtRun(profile.mark(), true);
        profile.recordOrtRun(profile.mark(), false);
        profile.recordLogitsSelect(profile.mark());
        profile.recordSampling(profile.mark());
        profile.recordFinalDecode(profile.mark());
        profile.markFirstToken();
        profile.recordWorkspaceLease(true, 1, 8, 16);
        profile.recordInputTensorCache(new OnnxInputTensorCacheStats(2, 1, 3, 1, 4, 2, 6, 1, 5, 0, 3));

        Map<String, Object> metadata = profile.metadata("CoreMLExecutionProvider", 4, 2, 123L);

        assertEquals("onnx", metadata.get("profile_mode"));
        assertEquals("CoreMLExecutionProvider", metadata.get("profile_backend"));
        assertEquals(4, metadata.get("tokens.input"));
        assertEquals(2, metadata.get("tokens.output"));
        assertEquals(1, metadata.get("tokens.decode"));
        assertEquals(123L, metadata.get("bench.latency_ms"));
        assertEquals(1, metadata.get("profile_onnx_prefill_steps"));
        assertEquals(1, metadata.get("profile_onnx_decode_steps"));
        assertEquals(true, metadata.get("profile_onnx_workspace_reused"));
        assertEquals(1, metadata.get("profile_onnx_workspace_evictions"));
        assertEquals(8, metadata.get("profile_onnx_workspace_requested_capacity"));
        assertEquals(16, metadata.get("profile_onnx_workspace_capacity"));
        assertEquals(5, metadata.get("profile_onnx_scalar_tensor_cache_hits"));
        assertEquals(2, metadata.get("profile_onnx_scalar_tensor_cache_misses"));
        assertEquals(2, metadata.get("profile_onnx_input_ids_cache_hits"));
        assertEquals(1, metadata.get("profile_onnx_input_ids_cache_misses"));
        assertEquals(3, metadata.get("profile_onnx_position_ids_cache_hits"));
        assertEquals(1, metadata.get("profile_onnx_position_ids_cache_misses"));
        assertEquals(4, metadata.get("profile_onnx_attention_mask_cache_hits"));
        assertEquals(2, metadata.get("profile_onnx_attention_mask_cache_misses"));
        assertEquals(6, metadata.get("profile_onnx_prefix_input_ids_cache_hits"));
        assertEquals(1, metadata.get("profile_onnx_prefix_input_ids_cache_misses"));
        assertEquals(5, metadata.get("profile_onnx_range_position_ids_cache_hits"));
        assertEquals(0, metadata.get("profile_onnx_range_position_ids_cache_misses"));
        assertEquals(20, metadata.get("profile_onnx_input_tensor_cache_hits"));
        assertEquals(5, metadata.get("profile_onnx_input_tensor_cache_misses"));
        assertEquals(3, metadata.get("profile_onnx_input_tensor_cache_evictions"));
        assertTrue(metadata.containsKey("profile_onnx_logits_select_ms"));
        assertTrue(metadata.containsKey("profile_onnx_final_decode_ms"));
        assertTrue(metadata.containsKey("profile_onnx_generation_ms"));
        assertTrue(metadata.containsKey("profile_onnx_profiled_ms"));
        assertTrue(metadata.containsKey("profile_onnx_unprofiled_ms"));
        assertTrue(metadata.containsKey("profile_onnx_primary_stage"));
        assertTrue(metadata.containsKey("profile_onnx_primary_value_ms"));
        assertTrue(metadata.containsKey("profile_onnx_primary_share_percent"));
        assertTrue(metadata.containsKey("profile_onnx_summary"));
        assertEquals("reused requested=8 capacity=16 evictions=1", metadata.get("profile_onnx_workspace"));
        assertEquals("hits=5 misses=2 hit_rate=71.4%", metadata.get("profile_onnx_scalar_tensor_cache"));
        assertEquals(
                "hits=20 misses=5 hit_rate=80.0% evictions=3 scalar=5/2 attention=4/2 prefix_ids=6/1 position_ranges=5/0",
                metadata.get("profile_onnx_input_tensor_cache"));
        assertEquals(5 * 100.0 / 7.0, (Double) metadata.get("profile_onnx_scalar_tensor_cache_hit_rate_percent"),
                0.01);
        assertEquals(20 * 100.0 / 25.0,
                (Double) metadata.get("profile_onnx_input_tensor_cache_hit_rate_percent"),
                0.01);
    }

    @Test
    void metadataIdentifiesPrimaryOnnxStage() {
        OnnxInferenceProfile profile = OnnxInferenceProfile.start(request(true));

        profile.recordInputPrepare(System.nanoTime() - 2_000_000L);
        profile.recordOrtRun(System.nanoTime() - 20_000_000L, false);
        profile.recordSampling(System.nanoTime() - 1_000_000L);

        Map<String, Object> metadata = profile.metadata("CPUExecutionProvider", 4, 1, 123L);

        assertEquals("ort_run", metadata.get("profile_onnx_primary_stage"));
        assertTrue((Double) metadata.get("profile_onnx_primary_value_ms") >= 20.0);
        assertTrue((Double) metadata.get("profile_onnx_primary_share_percent") > 80.0);
    }

    @Test
    void metadataCanIdentifyFinalDecodeAsPrimaryOnnxStage() {
        OnnxInferenceProfile profile = OnnxInferenceProfile.start(request(true));

        profile.recordInputPrepare(System.nanoTime() - 2_000_000L);
        profile.recordOrtRun(System.nanoTime() - 1_000_000L, false);
        profile.recordFinalDecode(System.nanoTime() - 20_000_000L);

        Map<String, Object> metadata = profile.metadata("CPUExecutionProvider", 4, 1, 123L);

        assertEquals("final_decode", metadata.get("profile_onnx_primary_stage"));
        assertTrue((Double) metadata.get("profile_onnx_primary_value_ms") >= 20.0);
        assertTrue((Double) metadata.get("profile_onnx_primary_share_percent") > 80.0);
    }

    @Test
    void requestMetadataCanEnableProfiling() {
        InferenceRequest request = InferenceRequest.builder()
                .model("onnx-test")
                .message(Message.user("hello"))
                .metadata("onnx_profile", "yes")
                .build();

        assertTrue(OnnxInferenceProfile.start(request).enabled());
    }

    private static InferenceRequest request(boolean profile) {
        return InferenceRequest.builder()
                .model("onnx-test")
                .message(Message.user("hello"))
                .parameter("onnx_profile", profile)
                .build();
    }
}
