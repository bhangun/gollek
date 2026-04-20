package tech.kayys.gollek.safetensor.engine.warmup;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

/**
 * A pair of LoRA matrices (A, B) for a single module.
 *
 * @param a the down-projection matrix (r × d)
 * @param b the up-projection matrix (k × r)
 */
public record LoraPair(AccelTensor a, AccelTensor b) {
}
