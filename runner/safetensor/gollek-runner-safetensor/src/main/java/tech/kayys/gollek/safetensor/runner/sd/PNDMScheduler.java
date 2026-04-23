package tech.kayys.gollek.safetensor.runner.sd;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import java.util.ArrayList;
import java.util.List;

/**
 * PNDM Scheduler implementation in pure Java.
 */
public class PNDMScheduler {
    private final int numTrainTimesteps = 1000;
    private final float betaStart = 0.00085f;
    private final float betaEnd = 0.012f;
    
    private float[] alphas;
    private float[] alphasCumprod;
    private List<Long> timesteps;
    
    private List<AccelTensor> ets = new ArrayList<>();

    public PNDMScheduler(int steps) {
        // Initialize alphas and timesteps
        alphas = new float[numTrainTimesteps];
        alphasCumprod = new float[numTrainTimesteps];
        
        float betaStep = (betaEnd - betaStart) / (numTrainTimesteps - 1);
        float currentCumprod = 1.0f;
        for (int i = 0; i < numTrainTimesteps; i++) {
            float beta = betaStart + i * betaStep;
            alphas[i] = 1.0f - beta;
            currentCumprod *= alphas[i];
            alphasCumprod[i] = currentCumprod;
        }
        
        // Linear spacing for inference timesteps
        timesteps = new ArrayList<>();
        float stepRatio = (float) numTrainTimesteps / steps;
        for (int i = steps - 1; i >= 0; i--) {
            timesteps.add((long) (i * stepRatio));
        }
    }

    public List<Long> getTimesteps() {
        return timesteps;
    }

    public AccelTensor step(AccelTensor modelOutput, long timestep, AccelTensor sample) {
        // PNDM logic (Simplified for 1st step / DDIM fallback)
        // For production, needs the multi-step ETs logic
        
        int t = (int) timestep;
        int prevT = t - (1000 / 15); // Approximate previous timestep
        if (prevT < 0) prevT = 0;
        
        float alphaProdT = alphasCumprod[t];
        float alphaProdTPrev = (prevT >= 0) ? alphasCumprod[prevT] : 1.0f;
        float betaProdT = 1 - alphaProdT;
        
        // DDIM step: 
        // pred_sample = (sample - sqrt(beta) * modelOutput) / sqrt(alpha)
        // sample_prev = sqrt(alpha_prev) * pred_sample + sqrt(1 - alpha_prev) * modelOutput
        
        AccelTensor predSample = AccelOps.div(
            AccelOps.sub(sample, AccelOps.mulScalar(modelOutput, (float) Math.sqrt(betaProdT))),
            AccelTensor.fromFloatArray(new float[]{(float) Math.sqrt(alphaProdT)}, 1)
        );
        
        return AccelOps.add(
            AccelOps.mulScalar(predSample, (float) Math.sqrt(alphaProdTPrev)),
            AccelOps.mulScalar(modelOutput, (float) Math.sqrt(1 - alphaProdTPrev))
        );
    }
}
