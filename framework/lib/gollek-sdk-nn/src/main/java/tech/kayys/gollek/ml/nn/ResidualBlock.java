package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Residual Block (ResidualBlock) for skip connections.
 * <p>
 * A residual block applies a transformation to the input and adds the original input back.
 * Skip connections enable training of very deep networks by providing direct gradient paths.
 * <p>
 * {@code output = x + f(x)}
 * <p>
 * Where f(x) is typically a sequence of linear/convolutional layers with activation.
 * The skip connection must ensure input and output shapes match (or use projection).
 *
 * <h3>Architecture</h3>
 * <pre>
 * Input x
 *   ├─→ [transformation] → f(x) ──────┐
 *   │                                  │ (Add)
 *   └──────────────────────────────────┤
 *                                      ↓
 *                                   Output
 * </pre>
 *
 * <h3>Example: Simple Residual Block</h3>
 * <pre>{@code
 * var block = new ResidualBlock(
 *     new Sequential(
 *         new Linear(256, 256),
 *         new ReLU(),
 *         new Linear(256, 256)
 *     )
 * );
 * var output = block.forward(input);  // input + transformation(input)
 * }</example>
 *
 * <h3>Typical Usage in ResNet</h3>
 * <pre>{@code
 * // Build a simple residual network
 * var model = new Sequential(
 *     new Linear(784, 256),
 *     new ResidualBlock(
 *         new Sequential(
 *             new Linear(256, 256),
 *             new ReLU(),
 *             new Linear(256, 256)
 *         )
 *     ),
 *     new ReLU(),
 *     new ResidualBlock(
 *         new Sequential(
 *             new Linear(256, 256),
 *             new ReLU(),
 *             new Linear(256, 256)
 *         )
 *     ),
 *     new ReLU(),
 *     new Linear(256, 10)
 * );
 * }</pre>
 *
 * <h3>Key Advantages</h3>
 * <ul>
 *   <li>Enables training of very deep networks (100+ layers)</li>
 *   <li>Better gradient flow during backpropagation</li>
 *   <li>Identity mapping: if f(x) ≈ 0, output ≈ x</li>
 *   <li>Helps with vanishing gradient problem</li>
 * </ul>
 *
 * <h3>Important Requirements</h3>
 * <ul>
 *   <li>Input and output shapes must match for direct addition</li>
 *   <li>If shapes differ, use projection shortcut (separate layer)</li>
 *   <li>Typical pattern: LinearLayer → Activation → LinearLayer (bottleneck)</li>
 * </ul>
 *
 * <h3>Common Variants</h3>
 * <ul>
 *   <li><b>Basic Block:</b> Conv → ReLU → Conv (like above)</li>
 *   <li><b>Bottleneck Block:</b> Conv(reduce) → Conv → Conv(expand)</li>
 *   <li><b>Pre-activation:</b> BN → ReLU → Conv (reverse order)</li>
 * </ul>
 */
public class ResidualBlock extends NNModule {

    private final NNModule transformation;

    /**
     * Create a residual block with the given transformation.
     *
     * @param transformation the module to apply inside the block (usually Sequential)
     *
     * @throws IllegalArgumentException if transformation is null
     */
    public ResidualBlock(NNModule transformation) {
        if (transformation == null) {
            throw new IllegalArgumentException("transformation cannot be null");
        }
        this.transformation = register("transformation", transformation);
    }

    /**
     * Apply residual block: output = input + transformation(input).
     *
     * @param input tensor of arbitrary shape
     * @return output with same shape as input (identity mapping when transformation ≈ 0)
     */
    @Override
    public GradTensor forward(GradTensor input) {
        GradTensor transformed = transformation.forward(input);
        return input.add(transformed);
    }

    /**
     * Get the transformation module.
     *
     * @return the transformation applied inside this block
     */
    public NNModule getTransformation() {
        return transformation;
    }

    @Override
    public String toString() {
        return "ResidualBlock(\n  " + transformation + "\n)";
    }
}
