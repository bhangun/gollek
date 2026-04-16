package tech.kayys.gollek.safetensor.engine.fusion;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.safetensor.spi.EncodedInput;
import tech.kayys.gollek.safetensor.spi.FusionEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModalityType;

import java.util.ArrayList;
import java.util.List;

/**
 * Baseline multimodal fusion strategy.
 * 
 * <p>Implements early fusion (concatenation) by merging text and vision tokens 
 * into a single sequence, with modality-aware position offsets.
 */
@ApplicationScoped
public class EarlyFusionEngine implements FusionEngine {

    @Override
    public String id() {
        return "early";
    }

    @Override
    public FusionResult fuse(List<EncodedInput> inputs, ModelConfig config, InferenceRequest request) {
        int totalTokens = inputs.stream().mapToInt(EncodedInput::tokenCount).sum();
        float[][] fusedEmbeds = new float[totalTokens][config.hiddenSize()];
        int[] positionIds = new int[totalTokens];

        int currentToken = 0;
        int textOffset = 0;
        int imageOffset = 512; // Example offset for vision tokens to avoid overlap in some models

        for (EncodedInput input : inputs) {
            int offset = input.modality() == ModalityType.IMAGE ? imageOffset : textOffset;
            
            for (int i = 0; i < input.tokenCount(); i++) {
                // Dim validation
                if (input.hiddenSize() != config.hiddenSize()) {
                    // In a production scenario, we'd add an alignment layer (linear projection)
                    // For now, assume encoders match LLM hidden size.
                }
                
                fusedEmbeds[currentToken] = input.embeddings()[i];
                positionIds[currentToken] = offset + i;
                currentToken++;
            }
            
            // Advance offsets
            if (input.modality() == ModalityType.IMAGE) {
                imageOffset += input.tokenCount();
            } else {
                textOffset += input.tokenCount();
            }
        }

        return new FusionResult(
            fusedEmbeds,
            positionIds,
            totalTokens,
            false // Early fusion typically uses a standard causal mask
        );
    }
}
