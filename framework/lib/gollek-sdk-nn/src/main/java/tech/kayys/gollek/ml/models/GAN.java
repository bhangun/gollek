package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;

/**
 * Generative Adversarial Network (GAN) — trains a generator and discriminator
 * in a minimax game to produce realistic samples.
 *
 * <p>Based on <em>"Generative Adversarial Nets"</em> (Goodfellow et al., 2014).
 *
 * <p>Training objective:
 * <pre>
 *   D loss = -[log D(x) + log(1 - D(G(z)))]   (maximize)
 *   G loss = -log D(G(z))                       (non-saturating)
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var gan = new GAN(latentDim=100, hiddenDim=256, outputDim=784);
 * GradTensor fake = gan.generate(batchSize=32);
 * GradTensor dLoss = gan.discriminatorLoss(real, fake.detach());
 * GradTensor gLoss = gan.generatorLoss(fake);
 * }</pre>
 */
public final class GAN extends NNModule {

    private final int latentDim;
    private final NNModule generator;
    private final NNModule discriminator;

    /**
     * Creates a GAN with MLP generator and discriminator.
     *
     * @param latentDim  noise vector dimension
     * @param hiddenDim  hidden layer size for both networks
     * @param outputDim  generator output / discriminator input dimension
     */
    public GAN(int latentDim, int hiddenDim, int outputDim) {
        this.latentDim     = latentDim;
        this.generator     = register("generator",     buildGenerator(latentDim, hiddenDim, outputDim));
        this.discriminator = register("discriminator", buildDiscriminator(outputDim, hiddenDim));
    }

    /**
     * Generates fake samples from random noise.
     *
     * @param batchSize number of samples to generate
     * @return generated samples {@code [batchSize, outputDim]}
     */
    public GradTensor generate(int batchSize) {
        GradTensor z = GradTensor.randn(batchSize, latentDim);
        return generator.forward(z);
    }

    /**
     * Computes the discriminator loss (binary cross-entropy on real=1, fake=0).
     *
     * @param real real samples {@code [N, outputDim]}
     * @param fake fake samples from generator (detached) {@code [N, outputDim]}
     * @return scalar discriminator loss
     */
    public GradTensor discriminatorLoss(GradTensor real, GradTensor fake) {
        GradTensor dReal = discriminator.forward(real);
        GradTensor dFake = discriminator.forward(fake);
        // -[log(D(x)) + log(1 - D(G(z)))]
        float loss = 0f;
        float[] dr = dReal.data(), df = dFake.data();
        for (int i = 0; i < dr.length; i++) {
            float r = Math.max(1e-7f, Math.min(1-1e-7f, dr[i]));
            float f = Math.max(1e-7f, Math.min(1-1e-7f, df[i]));
            loss -= (float)(Math.log(r) + Math.log(1 - f));
        }
        return GradTensor.scalar(loss / dr.length);
    }

    /**
     * Computes the generator loss (non-saturating: -log D(G(z))).
     *
     * @param fake fake samples from generator {@code [N, outputDim]}
     * @return scalar generator loss
     */
    public GradTensor generatorLoss(GradTensor fake) {
        GradTensor dFake = discriminator.forward(fake);
        float loss = 0f;
        float[] df = dFake.data();
        for (int i = 0; i < df.length; i++) {
            float f = Math.max(1e-7f, Math.min(1-1e-7f, df[i]));
            loss -= (float) Math.log(f);
        }
        return GradTensor.scalar(loss / df.length);
    }

    @Override
    public GradTensor forward(GradTensor z) { return generator.forward(z); }

    /** @return the generator module */
    public NNModule getGenerator()     { return generator; }

    /** @return the discriminator module */
    public NNModule getDiscriminator() { return discriminator; }

    // ── Network builders ──────────────────────────────────────────────────

    private static NNModule buildGenerator(int latentDim, int hiddenDim, int outputDim) {
        return new Sequential(
            new Linear(latentDim, hiddenDim), new ReLU(),
            new Linear(hiddenDim, hiddenDim), new ReLU(),
            new Linear(hiddenDim, outputDim)  // tanh applied externally
        );
    }

    private static NNModule buildDiscriminator(int inputDim, int hiddenDim) {
        return new Sequential(
            new Linear(inputDim,  hiddenDim), new LeakyReLU(0.2f),
            new Linear(hiddenDim, hiddenDim), new LeakyReLU(0.2f),
            new Linear(hiddenDim, 1)          // sigmoid applied in loss
        );
    }
}
