package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;

import java.util.Map;
import java.util.Random;

/**
 * Node for sampling Gaussian latent noise (16 channels for FLUX).
 */
public final class LatentNoiseGeneratorNode implements VisualGraphNode {

    private final int width;
    private final int height;
    private final long seed;
    private final int channels;

    public LatentNoiseGeneratorNode(int width, int height, long seed, int channels) {
        this.width = width;
        this.height = height;
        this.seed = seed;
        this.channels = channels;
    }

    public LatentNoiseGeneratorNode(int width, int height, long seed) {
        this(width, height, seed, 16);
    }

    @Override
    public String id() {
        return "latent_noise_generator";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        int latentH = height / 8;
        int latentW = width / 8;
        long[] shape = new long[]{1L, (long) channels, (long) latentH, (long) latentW};
        float[] noise = new float[1 * channels * latentH * latentW];
        Random rng = new Random(seed == 0L ? System.nanoTime() : seed);
        for (int i = 0; i < noise.length; i++) {
            noise[i] = (float) rng.nextGaussian();
        }
        return TensorFactory.of(noise, shape);
    }
}
