package io.github.pytorch.nn;

import io.github.pytorch.core.Device;
import io.github.pytorch.core.ScalarType;
import io.github.pytorch.core.Tensor;
import io.github.pytorch.ffm.LibTorchFFM;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.*;

/**
 * Base class for all neural network modules
 * Equivalent to torch::nn::Module in C++
 */
public abstract class Module implements AutoCloseable {
    
    protected final Map<String, Tensor> parameters = new LinkedHashMap<>();
    protected final Map<String, Tensor> buffers = new LinkedHashMap<>();
    protected final Map<String, Module> submodules = new LinkedHashMap<>();
    protected boolean training = true;
    protected MemorySegment nativeHandle;
    protected Arena arena;
    
    public Module() {
        this.arena = Arena.ofConfined();
        // Initialize native module if needed
    }
    
    /**
     * Forward pass - must be implemented by subclasses
     */
    public abstract Tensor forward(Tensor input);
    
    /**
     * Register a parameter
     */
    protected void registerParameter(String name, Tensor parameter) {
        if (parameter == null) {
            parameters.remove(name);
        } else {
            parameters.put(name, parameter);
        }
    }
    
    /**
     * Register a buffer (non-trainable tensor)
     */
    protected void registerBuffer(String name, Tensor buffer) {
        if (buffer == null) {
            buffers.remove(name);
        } else {
            buffers.put(name, buffer);
        }
    }
    
    /**
     * Register a submodule
     */
    protected void registerModule(String name, Module module) {
        if (module == null) {
            submodules.remove(name);
        } else {
            submodules.put(name, module);
        }
    }
    
    /**
     * Get all parameters recursively
     */
    public List<Tensor> parameters() {
        List<Tensor> allParams = new ArrayList<>(parameters.values());
        for (Module submodule : submodules.values()) {
            allParams.addAll(submodule.parameters());
        }
        return allParams;
    }
    
    /**
     * Get named parameters recursively
     */
    public Map<String, Tensor> namedParameters() {
        Map<String, Tensor> allParams = new LinkedHashMap<>(parameters);
        for (Map.Entry<String, Module> entry : submodules.entrySet()) {
            String prefix = entry.getKey() + ".";
            for (Map.Entry<String, Tensor> param : entry.getValue().namedParameters().entrySet()) {
                allParams.put(prefix + param.getKey(), param.getValue());
            }
        }
        return allParams;
    }
    
    /**
     * Get all buffers recursively
     */
    public List<Tensor> buffers() {
        List<Tensor> allBuffers = new ArrayList<>(buffers.values());
        for (Module submodule : submodules.values()) {
            allBuffers.addAll(submodule.buffers());
        }
        return allBuffers;
    }
    
    /**
     * Set training mode
     */
    public Module train() {
        return train(true);
    }
    
    public Module train(boolean mode) {
        this.training = mode;
        for (Module submodule : submodules.values()) {
            submodule.train(mode);
        }
        return this;
    }
    
    /**
     * Set evaluation mode
     */
    public Module eval() {
        return train(false);
    }
    
    /**
     * Check if in training mode
     */
    public boolean isTraining() {
        return training;
    }
    
    /**
     * Move module to device
     */
    public Module to(Device device) {
        for (Map.Entry<String, Tensor> entry : parameters.entrySet()) {
            parameters.put(entry.getKey(), entry.getValue().to(device));
        }
        for (Map.Entry<String, Tensor> entry : buffers.entrySet()) {
            buffers.put(entry.getKey(), entry.getValue().to(device));
        }
        for (Module submodule : submodules.values()) {
            submodule.to(device);
        }
        return this;
    }
    
    /**
     * Move module to CUDA
     */
    public Module cuda() {
        return to(Device.CUDA);
    }
    
    /**
     * Move module to CPU
     */
    public Module cpu() {
        return to(Device.CPU);
    }
    
    /**
     * Zero all gradients
     */
    public void zeroGrad() {
        for (Tensor param : parameters()) {
            param.zeroGrad();
        }
    }
    
    /**
     * Apply function to all parameters
     */
    public Module apply(java.util.function.Consumer<Tensor> fn) {
        for (Tensor param : parameters.values()) {
            fn.accept(param);
        }
        for (Tensor buffer : buffers.values()) {
            fn.accept(buffer);
        }
        for (Module submodule : submodules.values()) {
            submodule.apply(fn);
        }
        return this;
    }
    
    /**
     * Get parameter by name
     */
    public Tensor getParameter(String name) {
        return parameters.get(name);
    }
    
    /**
     * Get buffer by name
     */
    public Tensor getBuffer(String name) {
        return buffers.get(name);
    }
    
    /**
     * Get submodule by name
     */
    public Module getSubmodule(String name) {
        return submodules.get(name);
    }
    
    @Override
    public void close() {
        for (Tensor param : parameters.values()) {
            param.close();
        }
        for (Tensor buffer : buffers.values()) {
            buffer.close();
        }
        for (Module submodule : submodules.values()) {
            submodule.close();
        }
        if (arena != null) {
            arena.close();
        }
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName()).append("(\n");
        for (Map.Entry<String, Module> entry : submodules.entrySet()) {
            sb.append("  (").append(entry.getKey()).append("): ")
              .append(entry.getValue().toString().replace("\n", "\n  "))
              .append("\n");
        }
        sb.append(")");
        return sb.toString();
    }
}
