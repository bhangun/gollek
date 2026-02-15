package io.github.pytorch.nn;

import io.github.pytorch.core.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Sequential container for modules
 */
public class Sequential extends Module {
    
    private final List<Module> modules;
    
    public Sequential(Module... modules) {
        super();
        this.modules = new ArrayList<>(Arrays.asList(modules));
        for (int i = 0; i < modules.length; i++) {
            registerModule(String.valueOf(i), modules[i]);
        }
    }
    
    public Sequential(List<Module> modules) {
        super();
        this.modules = new ArrayList<>(modules);
        for (int i = 0; i < modules.size(); i++) {
            registerModule(String.valueOf(i), modules.get(i));
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Module module : modules) {
            output = module.forward(output);
        }
        return output;
    }
    
    public Sequential add(Module module) {
        modules.add(module);
        registerModule(String.valueOf(modules.size() - 1), module);
        return this;
    }
    
    public Module get(int index) {
        return modules.get(index);
    }
    
    public int size() {
        return modules.size();
    }
}
