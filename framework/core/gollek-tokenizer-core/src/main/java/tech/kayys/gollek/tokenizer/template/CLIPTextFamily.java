package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import java.util.List;

/**
 * CLIP (Contrastive Language-Image Pre-training) Text Model family.
 * Used as the text encoder in Stable Diffusion v1.4, v1.5, and v2.1.
 * 
 * Typically based on OpenAI's Vit-L/14 or Vit-H architectures.
 */
@ApplicationScoped
public class CLIPTextFamily implements ModelArchitecture {
    @Override
    public String id() {
        return "clip";
    }

    @Override
    public List<String> supportedArchClassNames() {
        // CLIPTextModel covers SD v1.4, v1.5, v2.1
        return List.of("CLIPTextModel", "CLIPTextModelWithProjection");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("clip_text_model", "clip");
    }

    @Override
    public String embedTokensWeight() {
        return "text_model.embeddings.token_embedding.weight";
    }

    @Override
    public String finalNormWeight() {
        return "text_model.final_layer_norm.weight";
    }

    @Override
    public String lmHeadWeight() {
        // CLIP doesn't have a traditional LM head for generation in the SD pipeline
        return null; 
    }

    @Override
    public String layerQueryWeight(int i) {
        return "text_model.encoder.layers.%d.self_attn.q_proj.weight".formatted(i);
    }

    @Override
    public String layerKeyWeight(int i) {
        return "text_model.encoder.layers.%d.self_attn.k_proj.weight".formatted(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return "text_model.encoder.layers.%d.self_attn.v_proj.weight".formatted(i);
    }

    @Override
    public String layerOutputWeight(int i) {
        return "text_model.encoder.layers.%d.self_attn.out_proj.weight".formatted(i);
    }

    @Override
    public String layerAttentionNormWeight(int i) {
        return "text_model.encoder.layers.%d.layer_norm1.weight".formatted(i);
    }

    @Override
    public String layerFfnGateWeight(int i) {
        // CLIP uses standard MLP (no gate projection)
        return null;
    }

    @Override
    public String layerFfnUpWeight(int i) {
        // In the CLIP transformer, fc1 is the up-projection
        return "text_model.encoder.layers.%d.mlp.fc1.weight".formatted(i);
    }

    @Override
    public String layerFfnDownWeight(int i) {
        // fc2 is the down-projection
        return "text_model.encoder.layers.%d.mlp.fc2.weight".formatted(i);
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "text_model.encoder.layers.%d.layer_norm2.weight".formatted(i);
    }

    /** 
     * CLIP specific: position embeddings are mandatory. 
     */
    public String positionEmbeddingsWeight() {
        return "text_model.embeddings.position_embedding.weight";
    }
}
