/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.audio;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;

import java.util.List;

public final class AudioModelFamilies {

    private AudioModelFamilies() {}

    /**
     * Whisper architecture.
     * Note: Whisper is an encoder-decoder model. This ModelArchitecture mapping
     * provides the bare minimum compatibility for loading the tensors into the
     * engine's weight registry.
     */
    @ApplicationScoped
    public static final class WhisperFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "whisper";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("WhisperForConditionalGeneration");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("whisper");
        }

        @Override
        public String embedTokensWeight() {
            return "model.decoder.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "model.decoder.layer_norm.weight";
        }

        @Override
        public String lmHeadWeight() {
            return "proj_out.weight"; // typically whisper doesn't have an explicit lm_head in the same way
        }

        @Override
        public String layerQueryWeight(int i) {
            return "model.decoder.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "model.decoder.layers.%d.self_attn.k_proj.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "model.decoder.layers.%d.self_attn.v_proj.weight".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "model.decoder.layers.%d.self_attn.out_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.decoder.layers.%d.self_attn_layer_norm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "model.decoder.layers.%d.fc1.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "model.decoder.layers.%d.fc1.weight".formatted(i); // fused typically
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.decoder.layers.%d.fc2.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.decoder.layers.%d.final_layer_norm.weight".formatted(i);
        }
        
        @Override
        public boolean usesRmsNorm() {
            return false; // Whisper uses LayerNorm
        }
    }
}
