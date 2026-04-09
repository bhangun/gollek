package tech.kayys.gollek.sdk.core;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Sequential container for modules.
 *
 * <p>Similar to PyTorch's `nn.Sequential`, this container applies modules
 * in sequence, passing the output of each as input to the next.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * Sequential model = new Sequential()
 *     .add(new Conv2d(3, 64, 3, 1, 1))
 *     .add(new BatchNorm2d(64))
 *     .add(new ReLU())
 *     .add(new MaxPool2d(2))
 *     .add(new Flatten())
 *     .add(new Linear(64 * 112 * 112, 10));
 *
 * Tensor output = model.forward(input);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class Sequential {

    private final List<Module> modules = new ArrayList<>();

    /**
     * Add a module to the sequence.
     *
     * @param module module to add
     * @return this sequential container
     */
    public Sequential add(Module module) {
        modules.add(module);
        return this;
    }

    /**
     * Forward pass through all modules.
     *
     * @param input input tensor
     * @return output tensor
     */
    public Tensor forward(Tensor input) {
        Tensor x = input;
        for (Module module : modules) {
            x = module.forward(x);
        }
        return x;
    }

    /**
     * Get list of modules.
     */
    public List<Module> getModules() {
        return List.copyOf(modules);
    }

    /**
     * Get number of modules.
     */
    public int size() {
        return modules.size();
    }

    /**
     * Module interface for sequential containers.
     */
    @FunctionalInterface
    public interface Module {
        /**
         * Forward pass.
         *
         * @param input input tensor
         * @return output tensor
         */
        Tensor forward(Tensor input);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Sequential(");
        for (int i = 0; i < modules.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(modules.get(i).getClass().getSimpleName());
        }
        sb.append(")");
        return sb.toString();
    }
}
