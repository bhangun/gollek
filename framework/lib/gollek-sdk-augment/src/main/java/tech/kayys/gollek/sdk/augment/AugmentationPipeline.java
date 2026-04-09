package tech.kayys.gollek.sdk.augment;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Pipeline for composing multiple augmentations.
 *
 * <p>Applies a sequence of augmentations to input data, similar to torchvision's
 * Compose transform.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * AugmentationPipeline pipeline = new AugmentationPipeline(
 *     Augmentation.RandomHorizontalFlip.create(),
 *     Augmentation.RandomCrop.of(224),
 *     Augmentation.ColorJitter.of(0.2, 0.2, 0.2)
 * );
 *
 * GradTensor augmented = pipeline.apply(input);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class AugmentationPipeline {

    private final List<Augmentation> augmentations;
    private final Random rng;

    /**
     * Create augmentation pipeline.
     *
     * @param augmentations list of augmentations to apply
     */
    public AugmentationPipeline(Augmentation... augmentations) {
        this(Arrays.asList(augmentations), new Random(42));
    }

    /**
     * Create augmentation pipeline with custom RNG.
     *
     * @param augmentations list of augmentations
     * @param rng           random number generator
     */
    public AugmentationPipeline(List<Augmentation> augmentations, Random rng) {
        this.augmentations = new ArrayList<>(augmentations);
        this.rng = rng;
    }

    /**
     * Apply all augmentations to input.
     *
     * @param input input tensor
     * @return augmented tensor
     */
    public GradTensor apply(GradTensor input) {
        GradTensor result = input;
        for (Augmentation aug : augmentations) {
            result = aug.apply(result, rng);
        }
        return result;
    }

    /**
     * Get list of augmentations.
     */
    public List<Augmentation> getAugmentations() {
        return List.copyOf(augmentations);
    }

    /**
     * Get number of augmentations.
     */
    public int size() {
        return augmentations.size();
    }

    @Override
    public String toString() {
        return String.format("AugmentationPipeline(%d transforms)", augmentations.size());
    }
}
