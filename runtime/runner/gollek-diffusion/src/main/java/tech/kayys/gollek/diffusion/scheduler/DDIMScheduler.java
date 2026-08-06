package tech.kayys.gollek.diffusion.scheduler;

import tech.kayys.alkhawarizm.core.backend.ComputeBackend;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class DDIMScheduler implements Scheduler {
    private final float[] alphasCumprod;
    private final int[] timesteps;
    private final ComputeBackend backend;

    public DDIMScheduler(float[] alphasCumprod, int steps, ComputeBackend backend) {
        this.alphasCumprod = alphasCumprod;
        this.backend = backend;
        this.timesteps = new int[steps];
        int stride = alphasCumprod.length / steps;
        for (int i = 0; i < steps; i++) {
            timesteps[i] = alphasCumprod.length - 1 - i * stride;
        }
    }

    @Override
    public int[] timesteps() {
        return timesteps;
    }

    @Override
    public Tensor step(Tensor x_t, Tensor eps, int tIndex) {
        int t = timesteps[tIndex];
        int prevT = (tIndex == timesteps.length - 1)
                ? 0
                : timesteps[tIndex + 1];
        float alphaT = alphasCumprod[t];
        float alphaPrev = alphasCumprod[prevT];
        float sqrtAlphaT = (float) Math.sqrt(alphaT);
        float sqrtOneMinusT = (float) Math.sqrt(1 - alphaT);
        float sqrtAlphaPrev = (float) Math.sqrt(alphaPrev);
        float sqrtOneMinusPrev = (float) Math.sqrt(1 - alphaPrev);
        // x0 = (x_t - sqrt(1-a_t)*eps) / sqrt(a_t)
        Tensor x0 = x_t.sub(eps.mul(sqrtOneMinusT))
                .div(sqrtAlphaT);
        // x_{t-1}
        return x0.mul(sqrtAlphaPrev)
                .add(eps.mul(sqrtOneMinusPrev));
    }
}
