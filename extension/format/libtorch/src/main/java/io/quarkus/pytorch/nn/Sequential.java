package io.quarkus.pytorch.nn;

import io.quarkus.pytorch.core.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Sequential container that chains modules together.
 * Modules are executed in the order they are added.
 */
public class Sequential extends Module {
    
    private final List<Module> moduleList = new ArrayList<>();
    
    /**
     * Create an empty sequential container.
     */
    public Sequential() {
    }
    
    /**
     * Create a sequential container with initial modules.
     */
    public Sequential(Module... modules) {
        for (int i = 0; i < modules.length; i++) {
            add(String.valueOf(i), modules[i]);
        }
    }
    
    /**
     * Add a module to the sequence.
     */
    public Sequential add(String name, Module module) {
        registerModule(name, module);
        moduleList.add(module);
        return this;
    }
    
    /**
     * Add a module with auto-generated name.
     */
    public Sequential add(Module module) {
        return add(String.valueOf(moduleList.size()), module);
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Module module : moduleList) {
            output = module.forward(output);
        }
        return output;
    }
    
    /**
     * Get module at index.
     */
    public Module get(int index) {
        return moduleList.get(index);
    }
    
    /**
     * Get number of modules.
     */
    public int size() {
        return moduleList.size();
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Sequential(\n");
        for (int i = 0; i < moduleList.size(); i++) {
            sb.append("  (").append(i).append("): ")
              .append(moduleList.get(i).toString())
              .append("\n");
        }
        sb.append(")");
        return sb.toString();
    }
}
