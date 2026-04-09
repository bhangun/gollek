package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Denoising Diffusion Probabilistic Model (DDPM) — learns to reverse a
 * Gaussian noise process to generate samples.
 *
 * <p>Based on <em>"Denoising Diffusion Probabilistic Models"</em>
 * (Ho et al., 2020).
 *
 * <p>Forward process (fixed): {@code q(xₜ|x₀) = N(√ᾱₜ·x₀, (1-ᾱₜ)·I)}
 * <p>Reverse process (learned): {@code p_θ(xₜ₋₁|xₜ) = N(μ_θ(xₜ,t), σₜ²·I)}
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var ddpm = new DDPM(inputDim=784, hiddenDim=256, timesteps=1000);
 *
 * // Training step
 * GradTensor loss = ddpm.trainingLoss(x0);
 *
 * // Sampling (inference)
 * GradTensor sample = ddpm.sample(batchSize=4);
 * }</pre>
 */
public final class DDPM extends NNModule {

    private final int timesteps;
    private final float[] alphas;       // αₜ = 1 - βₜ
    private final float[] alphasCumprod; // ᾱₜ = ∏αᵢ
    private final float[] betas;        // noise schedule

    /** U-Net-like denoising network: predicts noise ε from (xₜ, t). */
    private final NNModule denoiser;

    /**
     * Creates a DDPM with a linear noise schedule.
     *
     * @param inputDim  data dimension (e.g. 784 for MNIST)
     * @param hiddenDim denoiser hidden dimension
     * @param timesteps number of diffusion steps T (default 1000)
     */
    public DDPM(int inputDim, int hiddenDim, int timesteps) {
        this.timesteps = timesteps;
        this.betas     = linearSchedule(1e-4f, 0.02f, timesteps);
        this.alphas    = new float[timesteps];
        this.alphasCumprod = new float[timesteps];

        float cumprod = 1f;
        for (int t = 0; t < timesteps; t++) {
            alphas[t]       = 1f - betas[t];
            cumprod        *= alphas[t];
            alphasCumprod[t] = cumprod;
        }

        // Simple MLP denoiser: [inputDim + 1 (time embedding)] → hiddenDim → inputDim
        this.denoiser = register("denoiser", new Sequential(
            new Linear(inputDim + 1, hiddenDim), new SiLU(),
            new Linear(hiddenDim, hiddenDim),    new SiLU(),
            new Linear(hiddenDim, inputDim)
        ));
    }

    /**
     * Computes the simplified DDPM training loss: MSE between predicted and actual noise.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Sample random timestep t ~ Uniform(0, T)</li>
     *   <li>Sample noise ε ~ N(0, I)</li>
     *   <li>Compute noisy sample: xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε</li>
     *   <li>Predict noise: ε_θ = denoiser(xₜ, t)</li>
     *   <li>Loss: ||ε - ε_θ||²</li>
     * </ol>
     *
     * @param x0 clean data samples {@code [N, inputDim]}
     * @return scalar MSE loss
     */
    public GradTensor trainingLoss(GradTensor x0) {
        int N = (int) x0.shape()[0];
        int D = (int) x0.shape()[1];

        // Random timesteps
        int[] ts = new int[N];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < N; i++) ts[i] = rng.nextInt(timesteps);

        // Sample noise and compute xₜ
        GradTensor noise = GradTensor.randn(N, D);
        float[] x0d = x0.data(), nd = noise.data();
        float[] xt = new float[N * D];
        for (int n = 0; n < N; n++) {
            float sqrtAlpha = (float) Math.sqrt(alphasCumprod[ts[n]]);
            float sqrtOneMinusAlpha = (float) Math.sqrt(1f - alphasCumprod[ts[n]]);
            for (int d = 0; d < D; d++)
                xt[n * D + d] = sqrtAlpha * x0d[n * D + d] + sqrtOneMinusAlpha * nd[n * D + d];
        }

        // Concatenate xₜ with normalized time embedding
        float[] xtWithT = new float[N * (D + 1)];
        for (int n = 0; n < N; n++) {
            System.arraycopy(xt, n * D, xtWithT, n * (D + 1), D);
            xtWithT[n * (D + 1) + D] = ts[n] / (float) timesteps; // normalized t
        }

        GradTensor predNoise = denoiser.forward(GradTensor.of(xtWithT, N, D + 1));
        return predNoise.sub(noise).pow(2f).mean();
    }

    /**
     * Generates samples by running the reverse diffusion process.
     *
     * <p>Starts from pure Gaussian noise and iteratively denoises for T steps.
     *
     * @param batchSize number of samples to generate
     * @return generated samples {@code [batchSize, inputDim]}
     */
    public GradTensor sample(int batchSize) {
        int D = (int) denoiser.parameters().get(0).data().shape()[1] - 1;
        float[] x = new float[batchSize * D];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < x.length; i++) x[i] = (float) rng.nextGaussian();

        for (int t = timesteps - 1; t >= 0; t--) {
            float[] xtWithT = new float[batchSize * (D + 1)];
            for (int n = 0; n < batchSize; n++) {
                System.arraycopy(x, n * D, xtWithT, n * (D + 1), D);
                xtWithT[n * (D + 1) + D] = t / (float) timesteps;
            }
            float[] predNoise = denoiser.forward(GradTensor.of(xtWithT, batchSize, D + 1)).data();

            float alpha = alphas[t], alphaCum = alphasCumprod[t];
            float beta  = betas[t];
            float coeff = (1f - alpha) / (float) Math.sqrt(1f - alphaCum);

            for (int i = 0; i < x.length; i++) {
                x[i] = (1f / (float) Math.sqrt(alpha)) * (x[i] - coeff * predNoise[i]);
                if (t > 0) x[i] += (float) Math.sqrt(beta) * (float) rng.nextGaussian();
            }
        }
        return GradTensor.of(x, batchSize, D);
    }

    @Override
    public GradTensor forward(GradTensor x) { return trainingLoss(x); }

    private static float[] linearSchedule(float start, float end, int T) {
        float[] b = new float[T];
        for (int t = 0; t < T; t++) b[t] = start + (end - start) * t / (T - 1);
        return b;
    }
}
