package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A sequential container that applies modules in order.
 * <p>
 * Sequential passes the output of each module as input to the next module,
 * making it useful for building simple sequential networks. More complex
 * architectures should extend NNModule directly for custom forward() logic.
 * <p>
 * Equivalent to {@code torch.nn.Sequential} in PyTorch.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var model = new Sequential(
 *     new Linear(784, 256),
 *     new ReLU(),
 *     new Dropout(0.2),
 *     new Linear(256, 10)
 * );
 * var output = model.forward(input);  // [batch, 784] → [batch, 10]
 * }</pre>
 *
 * <h3>Dynamic Addition</h3>
 * <pre>{@code
 * var model = new Sequential();
 * model.add(new Linear(100, 50))
 *      .add(new ReLU())
 *      .add(new Linear(50, 10));
 * }</pre>
 */
public class Sequential extends NNModule {

    private final List<NNModule> layers = new ArrayList<>();

    /**
     * Create an empty Sequential container.
     */
    public Sequential() {
    }

    /**
     * Create a Sequential container with initial modules.
     *
     * @param modules the modules to add in order
     *
     * @throws IllegalArgumentException if any module is null
     */
    public Sequential(NNModule... modules) {
        for (int i = 0; i < modules.length; i++) {
            if (modules[i] == null) {
                throw new IllegalArgumentException("NNModule at index " + i + " is null");
            }
            layers.add(register(String.valueOf(i), modules[i]));
        }
    }

    /**
     * Add a module to the end of the sequence.
     *
     * @param module the module to add
     * @return this Sequential instance (for method chaining)
     *
     * @throws IllegalArgumentException if module is null
     */
    public Sequential add(NNModule module) {
        if (module == null) {
            throw new IllegalArgumentException("Cannot add null module");
        }
        layers.add(register(String.valueOf(layers.size()), module));
        return this;
    }

    /**
     * Apply all modules in sequence to the input.
     *
     * @param input the input tensor
     * @return the output after all modules have been applied
     */
    @Override
    public GradTensor forward(GradTensor input) {
        GradTensor x = input;
        for (NNModule layer : layers) {
            x = layer.forward(x);
        }
        return x;
    }

    /**
     * Get the number of layers in this Sequential.
     *
     * @return the number of modules
     */
    public int size() {
        return layers.size();
    }

    /**
     * Get a layer by index.
     *
     * @param index the layer index (0-based)
     * @return the module at the given index
     *
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public NNModule get(int index) {
        return layers.get(index);
    }

    /**
     * Get all layers as a list.
     *
     * @return unmodifiable list of modules in order
     */
    public List<NNModule> getLayers() {
        return java.util.Collections.unmodifiableList(layers);
    }
}
