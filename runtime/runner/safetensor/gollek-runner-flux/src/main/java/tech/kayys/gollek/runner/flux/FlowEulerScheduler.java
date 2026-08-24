package tech.kayys.gollek.runner.flux;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

/**
 * Flow Matching Euler scheduler for FLUX inference.
 */
public final class FlowEulerScheduler {

    private final int numSteps;
    private final float[] timesteps;

    public FlowEulerScheduler(int numSteps, float maxShift, float baseShift, int imageSeqLen) {
        this.numSteps = numSteps;
        this.timesteps = computeTimesteps(numSteps);
    }

    public FlowEulerScheduler(int numSteps) {
        this(numSteps, 1.15f, 0.5f, 256);
    }

    public AccelTensor step(AccelTensor latents, AccelTensor velocity, int stepIndex) {
        float t = timesteps[stepIndex];
        float tPrev = (stepIndex + 1 < numSteps) ? timesteps[stepIndex + 1] : 0.0f;
        float dt = tPrev - t;
        return AccelOps.add(latents, AccelOps.mulScalar(velocity, dt));
    }

    public float[] timesteps() {
        return timesteps.clone();
    }

    public int numSteps() {
        return numSteps;
    }

    private static float[] computeTimesteps(int numSteps) {
        float[] t = new float[numSteps];
        for (int i = 0; i < numSteps; i++) {
            t[i] = 1.0f - (float) i / numSteps;
        }
        return t;
    }
}
