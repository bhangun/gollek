package tech.kayys.gollek.diffusion.scheduler;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

/**
 * Diffusion scheduler contract used by the Java graph and runner layers.
 */
public interface Scheduler {
    Tensor step(Tensor x_t, Tensor eps, int tIndex);

    int[] timesteps();
}
