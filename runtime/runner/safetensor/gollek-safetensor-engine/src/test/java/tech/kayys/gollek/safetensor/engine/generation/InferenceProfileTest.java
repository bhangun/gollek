/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InferenceProfileTest {

    @Test
    void exposesLogitsPathSeparatelyFromLinearPathEvidence() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            DirectInferenceProfiler.recordLinearPath("attn_q_proj", "matvec");
            DirectInferenceProfiler.recordLinearPath("logits", "metal_matmul_f16");
            DirectInferenceProfiler.recordLinearPath("logits", "metal_matmul_f16");

            String summary = profile.summary("metal");
            assertTrue(summary.contains("linear_paths={logits:metal_matmul_f16=2, attn_q_proj:matvec=1}"));
            assertTrue(summary.contains("logits_paths={metal_matmul_f16=2}"));

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals(2, metadata.get("profile_linear_path_logits_metal_matmul_f16_count"));
            assertEquals(2, metadata.get("profile_logits_path_metal_matmul_f16_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void exposesCorePathStatusForMetalAndFallbackEvidence() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            DirectInferenceProfiler.recordLinearPath("attn_q_proj", "accelerate_linear");
            DirectInferenceProfiler.recordLinearPath("ffn_down", "metal_matmul_f16");
            DirectInferenceProfiler.recordLinearPath("logits", "metal_matmul_f16");
            DirectInferenceProfiler.recordFfnPath("metal_geglu");
            DirectInferenceProfiler.recordAttentionPath("paged_java");
            DirectInferenceProfiler.recordGreedyArgmaxPath("java_memory_segment");

            String summary = profile.summary("metal");
            assertTrue(summary.contains(
                    "path_status={core=mixed, linear=mixed, logits=metal, ffn=metal, attention=fallback, argmax=fallback}"));
            assertTrue(summary.contains(
                    "path_coverage={core=4/6, linear=2/3, logits=1/1, ffn=1/1, attention=0/1, argmax=0/1}"));

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals("mixed", metadata.get("profile_core_path_status"));
            assertEquals("mixed", metadata.get("profile_linear_path_status"));
            assertEquals("metal", metadata.get("profile_logits_path_status"));
            assertEquals("metal", metadata.get("profile_ffn_path_status"));
            assertEquals("fallback", metadata.get("profile_attention_path_status"));
            assertEquals("fallback", metadata.get("profile_argmax_path_status"));
            assertEquals(4, metadata.get("profile_core_metal_path_count"));
            assertEquals(2, metadata.get("profile_core_fallback_path_count"));
            assertEquals(0, metadata.get("profile_core_unknown_path_count"));
            assertEquals(6, metadata.get("profile_core_total_path_count"));
            assertEquals(4.0 / 6.0, (Double) metadata.get("profile_core_metal_path_ratio"), 1.0e-9);
            assertEquals(0, metadata.get("profile_attention_metal_path_count"));
            assertEquals(1, metadata.get("profile_attention_fallback_path_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void classifiesGemma4UnifiedPrefillPathEvidenceAsMetal() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            DirectInferenceProfiler.recordLinearPath("attn_qkv_proj_triple", "mixed_triple_matmul_bf16");
            DirectInferenceProfiler.recordLinearPath("attn_qk_proj_pair", "mixed_pair_matmul_bf16");
            DirectInferenceProfiler.recordLinearPath("logits", "bf16_matvec");
            DirectInferenceProfiler.recordFfnPath("matvec-gated-ffn:reject:not_single_token_rows:12");
            DirectInferenceProfiler.recordFfnPath("fused-gated-ffn:accept:geglu:native_bf16=true");
            DirectInferenceProfiler.recordAttentionPath("fa4_gathered");

            String summary = profile.summary("metal");
            assertTrue(summary.contains(
                    "path_status={core=metal, linear=metal, logits=metal, ffn=metal, attention=metal, argmax=missing}"));
            assertTrue(summary.contains(
                    "path_coverage={core=6/6, linear=3/3, logits=1/1, ffn=1/1, attention=1/1, argmax=0/0}"));
            assertTrue(summary.contains("ffn_strategy=fused_geglu_prefill_active"));

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals("metal", metadata.get("profile_core_path_status"));
            assertEquals("metal", metadata.get("profile_linear_path_status"));
            assertEquals("metal", metadata.get("profile_ffn_path_status"));
            assertEquals("metal", metadata.get("profile_attention_path_status"));
            assertEquals(3, metadata.get("profile_linear_metal_path_count"));
            assertEquals(1, metadata.get("profile_ffn_metal_path_count"));
            assertEquals(0, metadata.get("profile_ffn_fallback_path_count"));
            assertEquals(1, metadata.get("profile_attention_metal_path_count"));
            assertEquals(1.0, (Double) metadata.get("profile_core_metal_path_ratio"), 1.0e-9);
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void reportsFfnBottleneckAdviceForGemma4FusedPrefill() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");
            profile.firstTokenNanos = 20_000_000_000L;
            profile.prefillNanos = 19_000_000_000L;
            profile.ffnNanos = 12_000_000_000L;
            profile.attentionNanos = 5_000_000_000L;
            profile.logitsProjectionNanos = 1_000_000_000L;

            DirectInferenceProfiler.recordFfnPath("fused-gated-ffn:accept:geglu:native_bf16=true");

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals("ffn", metadata.get("profile_bottleneck_stage"));
            assertEquals(12_000.0, (Double) metadata.get("profile_bottleneck_value_ms"), 1.0e-9);
            assertEquals(60.0, (Double) metadata.get("profile_bottleneck_share_percent"), 1.0e-9);
            assertTrue(((String) metadata.get("profile_bottleneck_advice"))
                    .contains("batched native FFN kernel"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void reportsFfnStrategyAdviceWhenFusedPrefillWinsOverRowMatvec() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");
            profile.firstTokenNanos = 20_000_000_000L;
            profile.prefillNanos = 19_000_000_000L;
            profile.ffnNanos = 12_000_000_000L;
            profile.attentionNanos = 5_000_000_000L;
            profile.logitsProjectionNanos = 1_000_000_000L;

            DirectInferenceProfiler.recordFfnPath(
                    "matvec-gated-ffn-prefill-rows:skip:strategy_prefers_fused_geglu_prefill");
            DirectInferenceProfiler.recordFfnPath("fused-gated-ffn:accept:geglu:native_bf16=true");

            String summary = profile.summary("metal");
            assertTrue(summary.contains("ffn_strategy=fused_geglu_prefill_over_row_prefill"));

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals("ffn", metadata.get("profile_bottleneck_stage"));
            assertEquals("fused_geglu_prefill_over_row_prefill", metadata.get("profile_ffn_strategy"));
            assertEquals(false, metadata.get("profile_ffn_strategy_row_prefill_active"));
            assertEquals(true, metadata.get("profile_ffn_strategy_fused_prefill_active"));
            assertEquals(
                    "gollek.safetensor.prefer_metal_matvec_ffn_prefill_rows",
                    metadata.get("profile_ffn_strategy_ab_test_property"));
            assertTrue(((String) metadata.get("profile_bottleneck_advice"))
                    .contains("prefer_metal_matvec_ffn_prefill_rows=true"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void exposesRowPrefillStrategyVariantMetadata() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            DirectInferenceProfiler.recordFfnPath(
                    "matvec-gated-ffn-prefill-rows:accept:geglu:native_bf16=true:native_rows=12:variant=x4");

            String summary = profile.summary("metal");
            assertTrue(summary.contains("ffn_strategy=row_prefill_matvec_active"));

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals("row_prefill_matvec_active", metadata.get("profile_ffn_strategy"));
            assertEquals(true, metadata.get("profile_ffn_strategy_row_prefill_active"));
            assertEquals(false, metadata.get("profile_ffn_strategy_fused_prefill_active"));
            assertEquals(12, metadata.get("profile_ffn_strategy_row_prefill_native_rows"));
            assertEquals("x4", metadata.get("profile_ffn_strategy_row_prefill_variant"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void ignoresStrategySkipsWhenClassifyingFfnCoverage() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            DirectInferenceProfiler.recordFfnPath(
                    "matvec-gated-ffn-prefill-rows:skip:strategy_prefers_fused_geglu_prefill");
            DirectInferenceProfiler.recordFfnPath("fused-gated-ffn:accept:geglu:native_bf16=true");

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals("metal", metadata.get("profile_ffn_path_status"));
            assertEquals(1, metadata.get("profile_ffn_metal_path_count"));
            assertEquals(0, metadata.get("profile_ffn_fallback_path_count"));
            assertEquals(0, metadata.get("profile_ffn_unknown_path_count"));
            assertEquals(1, metadata.get("profile_ffn_total_path_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void classifiesNativeGreedyArgmaxAsMetalEvidence() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            DirectInferenceProfiler.recordGreedyArgmaxPath("native_argmax_f32");

            String summary = profile.summary("metal");
            assertTrue(summary.contains(
                    "path_status={core=missing, linear=missing, logits=missing, ffn=missing, attention=missing, argmax=metal}"));
            assertTrue(summary.contains(
                    "path_coverage={core=0/0, linear=0/0, logits=0/0, ffn=0/0, attention=0/0, argmax=1/1}"));

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals("metal", metadata.get("profile_argmax_path_status"));
            assertEquals(1, metadata.get("profile_argmax_metal_path_count"));
            assertEquals(0, metadata.get("profile_argmax_fallback_path_count"));
            assertEquals(1, metadata.get("profile_argmax_total_path_count"));
            assertEquals(1.0, (Double) metadata.get("profile_argmax_metal_path_ratio"), 1.0e-9);
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void greedyArgmaxTimingSkipsClockedRecordWhenProfilingIsDisabled() {
        String previousProfile = System.getProperty("gollek.profile");
        System.clearProperty("gollek.profile");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            long started = DirectInferenceProfiler.startGreedyArgmaxTiming();
            DirectInferenceProfiler.recordGreedyArgmaxTiming(started, "native_argmax_f32");

            assertTrue(started < 0L);
            assertEquals(0L, profile.greedyArgmaxNanos);
            assertTrue(profile.greedyArgmaxPathCounts.isEmpty());
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    @Test
    void greedyArgmaxTimingRecordsDurationAndPathWhenProfilingIsEnabled() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            long started = DirectInferenceProfiler.startGreedyArgmaxTiming();
            DirectInferenceProfiler.recordGreedyArgmaxTiming(started, "native_argmax_f32");

            assertTrue(started >= 0L);
            assertTrue(profile.greedyArgmaxNanos >= 0L);
            assertEquals(1, profile.greedyArgmaxPathCounts.get("native_argmax_f32"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProfileProperty(previousProfile);
        }
    }

    private static void restoreProfileProperty(String previousProfile) {
        if (previousProfile == null) {
            System.clearProperty("gollek.profile");
        } else {
            System.setProperty("gollek.profile", previousProfile);
        }
    }
}
