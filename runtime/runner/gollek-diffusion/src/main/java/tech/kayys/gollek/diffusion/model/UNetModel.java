package tech.kayys.gollek.diffusion.model;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

public interface UNetModel {
    Tensor predict(Tensor latents, Tensor embedding, int timestep);

    default Tensor predictBatched(Tensor latents, Tensor cond, Tensor uncond, int timestep) {
        throw new UnsupportedOperationException("predictBatched not implemented");
    }
}