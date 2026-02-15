package io.quarkus.pytorch.nn;

import io.quarkus.pytorch.core.Tensor;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base class for all neural network modules, mirroring torch::nn::Module.
 * Manages parameters and sub-modules with automatic registration.
 */
public abstract class Module implements AutoCloseable {
    
    private final Map<String, Tensor> parameters = new ConcurrentHashMap<>();
    private final Map<String, Module> modules = new LinkedHashMap<>();
    private boolean trainingMode = true;
    
    /**
     * Forward pass - must be implemented by subclasses.
     */
    public abstract Tensor forward(Tensor input);
    
    /**
     * Register a parameter tensor.
     */
    protected void registerParameter(String name, Tensor parameter) {
        if (parameters.containsKey(name)) {
            throw new IllegalArgumentException("Parameter already registered: " + name);
        }
        parameters.put(name, parameter);
    }
    
    /**
     * Register a sub-module.
     */
    protected <T extends Module> T registerModule(String name, T module) {
        if (modules.containsKey(name)) {
            throw new IllegalArgumentException("Module already registered: " + name);
        }
        modules.put(name, module);
        return module;
    }
    
    /**
     * Get all parameters including sub-modules.
     */
    public List<Tensor> parameters() {
        List<Tensor> allParams = new ArrayList<>(parameters.values());
        for (Module module : modules.values()) {
            allParams.addAll(module.parameters());
        }
        return Collections.unmodifiableList(allParams);
    }
    
    /**
     * Get all named parameters including sub-modules.
     */
    public Map<String, Tensor> namedParameters() {
        Map<String, Tensor> allParams = new LinkedHashMap<>(parameters);
        for (Map.Entry<String, Module> entry : modules.entrySet()) {
            String prefix = entry.getKey();
            Module module = entry.getValue();
            for (Map.Entry<String, Tensor> param : module.namedParameters().entrySet()) {
                allParams.put(prefix + "." + param.getKey(), param.getValue());
            }
        }
        return Collections.unmodifiableMap(allParams);
    }
    
    /**
     * Set training mode.
     */
    public Module train(boolean mode) {
        this.trainingMode = mode;
        for (Module module : modules.values()) {
            module.train(mode);
        }
        return this;
    }
    
    /**
     * Set to training mode.
     */
    public Module train() {
        return train(true);
    }
    
    /**
     * Set to evaluation mode.
     */
    public Module eval() {
        return train(false);
    }
    
    /**
     * Check if in training mode.
     */
    public boolean isTraining() {
        return trainingMode;
    }
    
    /**
     * Zero all gradients.
     */
    public void zeroGrad() {
        for (Tensor param : parameters()) {
            Tensor grad = param.grad();
            if (grad != null) {
                grad.close();
            }
        }
    }
    
    /**
     * Move all parameters to specified device.
     */
    public Module to(Tensor.Device device) {
        // Create new parameters on target device
        Map<String, Tensor> newParams = new HashMap<>();
        for (Map.Entry<String, Tensor> entry : parameters.entrySet()) {
            Tensor oldParam = entry.getValue();
            Tensor newParam = oldParam.to(device);
            newParams.put(entry.getKey(), newParam);
            oldParam.close();
        }
        parameters.clear();
        parameters.putAll(newParams);
        
        // Move sub-modules
        for (Module module : modules.values()) {
            module.to(device);
        }
        
        return this;
    }
    
    @Override
    public void close() {
        for (Tensor param : parameters.values()) {
            param.close();
        }
        for (Module module : modules.values()) {
            module.close();
        }
        parameters.clear();
        modules.clear();
    }
}
