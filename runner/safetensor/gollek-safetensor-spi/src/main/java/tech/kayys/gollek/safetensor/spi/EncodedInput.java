package tech.kayys.gollek.safetensor.spi;

import tech.kayys.gollek.spi.model.ModalityType;

/**
 * Container for modality-specific embeddings and metadata.
 * 
 * <p>Used by encoder plugins to return pre-processed hidden states 
 * before they are merged by a {@link FusionEngine}.
 */
public record EncodedInput(
    ModalityType modality,
    float[][] embeddings, 
    int tokenCount,
    int hiddenSize
) {
    public static EncodedInput of(ModalityType modality, float[][] embeddings) {
        if (embeddings == null || embeddings.length == 0) {
            return new EncodedInput(modality, new float[0][0], 0, 0);
        }
        return new EncodedInput(modality, embeddings, embeddings.length, embeddings[0].length);
    }
}
