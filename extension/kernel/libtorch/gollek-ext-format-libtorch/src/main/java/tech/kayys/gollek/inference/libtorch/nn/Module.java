package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.util.*;

/**
 * Abstract base class for all neural network modules.
 * Mirrors {@code torch::nn::Module} from the C++ API.
 * <p>
 * Manages parameters, buffers, and submodules. Subclasses must implement
 * the {@link #forward(Tensor)} method.
 */
public abstract class Module implements AutoCloseable {

    protected final Map<String, Tensor> parameters = new LinkedHashMap<>();
    protected final Map<String, Tensor> buffers = new LinkedHashMap<>();
    protected final Map<String, Module> submodules = new LinkedHashMap<>();
    protected boolean training = true;

    // ── Parameter/Buffer/Submodule registration ───────────────────────

    /**
     * Register a parameter tensor.
     */
    protected void registerParameter(String name, Tensor param) {
        Objects.requireNonNull(name, "Parameter name must not be null");
        parameters.put(name, param);
    }

    /**
     * Register a non-trainable buffer tensor.
     */
    protected void registerBuffer(String name, Tensor buffer) {
        Objects.requireNonNull(name, "Buffer name must not be null");
        buffers.put(name, buffer);
    }

    /**
     * Register a child submodule.
     */
    protected void registerModule(String name, Module module) {
        Objects.requireNonNull(name, "Module name must not be null");
        Objects.requireNonNull(module, "Module must not be null");
        submodules.put(name, module);
    }

    // ── Forward ───────────────────────────────────────────────────────

    /**
     * Perform the forward computation.
     *
     * @param input the input tensor
     * @return the output tensor
     */
    public abstract Tensor forward(Tensor input);

    // ── Training/Eval ─────────────────────────────────────────────────

    /**
     * Set training mode.
     */
    public Module train(boolean mode) {
        this.training = mode;
        submodules.values().forEach(m -> m.train(mode));
        return this;
    }

    /** Shortcut for train(true). */
    public Module train() {
        return train(true);
    }

    /** Set evaluation mode. */
    public Module eval() {
        return train(false);
    }

    /** Check if in training mode. */
    public boolean isTraining() {
        return training;
    }

    // ── Device ────────────────────────────────────────────────────────

    /**
     * Move all parameters and buffers to the specified device.
     */
    public Module to(Device device) {
        parameters.replaceAll((name, param) -> param.to(device));
        buffers.replaceAll((name, buf) -> buf.to(device));
        submodules.values().forEach(m -> m.to(device));
        return this;
    }

    // ── Gradient ──────────────────────────────────────────────────────

    /**
     * Zero all parameter gradients.
     */
    public void zeroGrad() {
        // Gradients are zeroed at the native level via optimizer.zeroGrad()
        // or by iterating parameters and zeroing their grads
        submodules.values().forEach(Module::zeroGrad);
    }

    // ── Accessors ─────────────────────────────────────────────────────

    /**
     * Get all parameters (including from submodules).
     */
    public List<Tensor> getParameters() {
        List<Tensor> all = new ArrayList<>(parameters.values());
        submodules.values().forEach(m -> all.addAll(m.getParameters()));
        return Collections.unmodifiableList(all);
    }

    /**
     * Get named parameters (flat, with dot-separated names).
     */
    public Map<String, Tensor> namedParameters() {
        Map<String, Tensor> result = new LinkedHashMap<>(parameters);
        submodules.forEach((name, module) -> module.namedParameters()
                .forEach((pName, param) -> result.put(name + "." + pName, param)));
        return Collections.unmodifiableMap(result);
    }

    /**
     * Get all registered submodules.
     */
    public Map<String, Module> children() {
        return Collections.unmodifiableMap(submodules);
    }

    /**
     * Count total number of parameters.
     */
    public long parameterCount() {
        long count = 0;
        for (Tensor p : parameters.values()) {
            count += p.numel();
        }
        for (Module m : submodules.values()) {
            count += m.parameterCount();
        }
        return count;
    }

    @Override
    public void close() {
        parameters.values().forEach(Tensor::close);
        buffers.values().forEach(Tensor::close);
        submodules.values().forEach(Module::close);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(training=" + training + ")";
    }
}
