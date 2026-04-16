package tech.kayys.gollek.safetensor.spi;

import tech.kayys.gollek.spi.model.ModalityType;

/**
 * Represents a single unit of fused input.
 * 
 * <p>Contains the embedding vector, its original modality, and its 
 * assigned sequence position.
 */
public record FusedToken(
    float[] embedding,
    ModalityType modality,
    int position
) {}
