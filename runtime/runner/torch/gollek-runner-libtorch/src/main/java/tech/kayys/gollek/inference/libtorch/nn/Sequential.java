package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Sequential container that chains modules together.
 * <p>
 * Modules are executed in the order they are added, with each module's output
 * serving as the input to the next. This is the most common way to build
 * feed-forward neural networks.
 * </p>
 * <p>
 * <h2>Usage Example</h2>
 * <pre>{@code
 * Sequential model = new Sequential(
 *     new Linear(784, 128),
 *     new ReLU(),
 *     new Linear(128, 64),
 *     new ReLU(),
 *     new Linear(64, 10)
 * );
 * 
 * TorchTensor input = TorchTensor.randn(32, 784);  // batch of 32
 * TorchTensor output = model.forward(input);
 * }</pre>
 *
 * @see Module
 * @see Linear
 * @since 1.0
 */
public class Sequential extends Module {

    private final List<Module> moduleList = new ArrayList<>();

    /**
     * Creates an empty sequential container.
     */
    public Sequential() {
    }

    /**
     * Creates a sequential container with initial modules.
     *
     * @param modules the modules to add in order
     */
    public Sequential(Module... modules) {
        for (int i = 0; i < modules.length; i++) {
            add(String.valueOf(i), modules[i]);
        }
    }

    /**
     * Adds a module with a name.
     *
     * @param name the module name
     * @param module the module to add
     * @return this container for method chaining
     */
    public Sequential add(String name, Module module) {
        registerModule(name, module);
        moduleList.add(module);
        return this;
    }

    /**
     * Adds a module with an auto-generated name.
     *
     * @param module the module to add
     * @return this container for method chaining
     */
    public Sequential add(Module module) {
        return add(String.valueOf(moduleList.size()), module);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        TorchTensor output = input;
        for (Module module : moduleList) {
            output = module.forward(output);
        }
        return output;
    }

    /**
     * Returns the module at the specified index.
     *
     * @param index the module index
     * @return the module at that index
     */
    public Module get(int index) {
        return moduleList.get(index);
    }

    /**
     * Returns the number of modules in this container.
     *
     * @return the module count
     */
    public int size() {
        return moduleList.size();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Sequential(\n");
        for (int i = 0; i < moduleList.size(); i++) {
            sb.append("  (").append(i).append("): ")
                    .append(moduleList.get(i))
                    .append("\n");
        }
        sb.append(")");
        return sb.toString();
    }
}
