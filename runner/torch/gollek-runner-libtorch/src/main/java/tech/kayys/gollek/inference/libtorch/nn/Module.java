package tech.kayys.gollek.inference.libtorch.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Abstract base class for all neural network modules.
 * <p>
 * Mirrors {@code libtorch::nn::Module} from the C++ API. Manages parameters,
 * buffers, and submodules. Subclasses must implement the
 * {@link #forward(TorchTensor)}
 * method to define the module's computation.
 * </p>
 * <p>
 * <h2>Usage Example</h2>
 * 
 * <pre>{@code
 * public class MyNetwork extends Module {
 *     private final Linear fc1;
 *     private final ReLU relu;
 *     
 *     public MyNetwork() {
 *         this.fc1 = new Linear(784, 128);
 *         this.relu = new ReLU();
 *         registerModule("fc1", fc1);
 *         registerModule("relu", relu);
 *     }
 *     
 *     {@literal @Override}
 *     public TorchTensor forward(TorchTensor input) {
 *         return relu.forward(fc1.forward(input));
 *     }
 * }
 * }</pre>
 * <p>
 * <h2>Memory Management</h2>
 * <p>
 * Modules own their parameter tensors. When {@link #close()} is called, all
 * parameters, buffers, and submodules are recursively closed to release native
 * memory.
 * </p>
 *
 * @see TorchTensor
 * @see Linear
 * @see Sequential
 * @since 1.0
 */
public abstract class Module implements AutoCloseable {

    /**
     * Named parameters of this module (trainable weights).
     */
    protected final Map<String, TorchTensor> parameters = new LinkedHashMap<>();

    /**
     * Named buffers (non-trainable persistent tensors).
     */
    protected final Map<String, TorchTensor> buffers = new LinkedHashMap<>();

    /**
     * Named submodules (child modules).
     */
    protected final Map<String, Module> submodules = new LinkedHashMap<>();

    /**
     * Training mode flag. True for training, false for evaluation.
     */
    protected boolean training = true;

    // ── Parameter/Buffer/Submodule registration ───────────────────────

    /**
     * Registers a trainable parameter tensor.
     * <p>
     * Parameters are tensors that are updated during training via backpropagation.
     * </p>
     *
     * @param name  the parameter name (must be unique within this module)
     * @param param the parameter tensor
     * @throws NullPointerException if name or param is null
     */
    protected void registerParameter(String name, TorchTensor param) {
        Objects.requireNonNull(name, "Parameter name must not be null");
        parameters.put(name, param);
    }

    /**
     * Registers a non-trainable buffer tensor.
     * <p>
     * Buffers are persistent tensors that are not updated during training
     * (e.g., running statistics in BatchNorm).
     * </p>
     *
     * @param name   the buffer name (must be unique within this module)
     * @param buffer the buffer tensor
     * @throws NullPointerException if name or buffer is null
     */
    protected void registerBuffer(String name, TorchTensor buffer) {
        Objects.requireNonNull(name, "Buffer name must not be null");
        buffers.put(name, buffer);
    }

    /**
     * Registers a child submodule.
     * <p>
     * Submodules are automatically included in parameter enumeration and
     * device transfers.
     * </p>
     *
     * @param name   the submodule name (must be unique within this module)
     * @param module the submodule instance
     * @throws NullPointerException if name or module is null
     */
    protected void registerModule(String name, Module module) {
        Objects.requireNonNull(name, "Module name must not be null");
        Objects.requireNonNull(module, "Module must not be null");
        submodules.put(name, module);
    }

    // ── Forward ───────────────────────────────────────────────────────

    /**
     * Performs the forward computation for this module.
     * <p>
     * Subclasses must implement this method to define the module's behavior.
     * </p>
     *
     * @param input the input tensor
     * @return the output tensor
     */
    public abstract TorchTensor forward(TorchTensor input);

    // ── Training/Eval ─────────────────────────────────────────────────

    /**
     * Sets the module to training or evaluation mode.
     * <p>
     * Training mode affects the behavior of certain modules like Dropout and
     * BatchNorm. This method recursively sets all submodules to the same mode.
     * </p>
     *
     * @param mode true for training mode, false for evaluation mode
     * @return this module for method chaining
     */
    public Module train(boolean mode) {
        this.training = mode;
        submodules.values().forEach(m -> m.train(mode));
        return this;
    }

    /**
     * Sets the module to training mode.
     *
     * @return this module for method chaining
     * @see #train(boolean)
     */
    public Module train() {
        return train(true);
    }

    /**
     * Sets the module to evaluation mode.
     * <p>
     * This is a convenience method equivalent to {@code train(false)}.
     * </p>
     *
     * @return this module for method chaining
     */
    public Module eval() {
        return train(false);
    }

    /**
     * Returns whether the module is in training mode.
     *
     * @return true if training mode, false if evaluation mode
     */
    public boolean isTraining() {
        return training;
    }

    // ── Device ────────────────────────────────────────────────────────

    /**
     * Moves all parameters, buffers, and submodules to the specified device.
     * <p>
     * This method creates new tensors on the target device and updates all
     * references. The original tensors on the previous device are not closed
     * automatically - callers should manage memory appropriately.
     * </p>
     *
     * @param device the target device
     * @return this module for method chaining
     */
    public Module to(Device device) {
        parameters.replaceAll((name, param) -> param.to(device));
        buffers.replaceAll((name, buf) -> buf.to(device));
        submodules.values().forEach(m -> m.to(device));
        return this;
    }

    // ── Gradient ──────────────────────────────────────────────────────

    /**
     * Zeroes all parameter gradients.
     * <p>
     * This method should be called before each backward pass to accumulate
     * gradients correctly. Gradients are zeroed at the native level via the
     * optimizer or by iterating parameters.
     * </p>
     */
    public void zeroGrad() {
        submodules.values().forEach(Module::zeroGrad);
    }

    // ── Accessors ─────────────────────────────────────────────────────

    /**
     * Returns all parameters including those from submodules.
     *
     * @return unmodifiable list of all parameter tensors
     */
    public List<TorchTensor> getParameters() {
        List<TorchTensor> all = new ArrayList<>(parameters.values());
        submodules.values().forEach(m -> all.addAll(m.getParameters()));
        return Collections.unmodifiableList(all);
    }

    /**
     * Returns all named parameters with dot-separated names for submodules.
     *
     * @return unmodifiable map of parameter name to tensor
     */
    public Map<String, TorchTensor> namedParameters() {
        Map<String, TorchTensor> result = new LinkedHashMap<>(parameters);
        submodules.forEach((name, module) -> module.namedParameters()
                .forEach((pName, param) -> result.put(name + "." + pName, param)));
        return Collections.unmodifiableMap(result);
    }

    /**
     * Returns all registered submodules.
     *
     * @return unmodifiable map of submodule name to module
     */
    public Map<String, Module> children() {
        return Collections.unmodifiableMap(submodules);
    }

    /**
     * Counts the total number of parameters in this module and all submodules.
     *
     * @return total parameter count
     */
    public long parameterCount() {
        long count = 0;
        for (TorchTensor p : parameters.values()) {
            count += p.numel();
        }
        for (Module m : submodules.values()) {
            count += m.parameterCount();
        }
        return count;
    }

    /**
     * Saves the module's state dictionary to a file.
     * <p>
     * The state dictionary contains all parameters and buffers. Use
     * {@link #loadStateDict(java.nio.file.Path)} to restore the state.
     * </p>
     *
     * @param path the file path to save to
     * @throws RuntimeException if saving fails
     */
    public void saveStateDict(java.nio.file.Path path) {
        Map<String, TorchTensor> stateDict = namedParameters();
        if (stateDict.isEmpty())
            return;

        java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined();
        try {
            tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding binding = tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding
                    .getInstance();
            java.lang.invoke.MethodHandle fn = binding.bind(
                    tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.SAVE_STATE_DICT,
                    tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.SAVE_STATE_DICT_DESC);

            int count = stateDict.size();
            java.lang.foreign.MemorySegment keysPtr = arena.allocate(java.lang.foreign.ValueLayout.ADDRESS, count);
            java.lang.foreign.MemorySegment tensorsPtr = arena.allocate(java.lang.foreign.ValueLayout.ADDRESS, count);

            int i = 0;
            for (Map.Entry<String, TorchTensor> entry : stateDict.entrySet()) {
                keysPtr.setAtIndex(java.lang.foreign.ValueLayout.ADDRESS, i, arena.allocateFrom(entry.getKey()));
                tensorsPtr.setAtIndex(java.lang.foreign.ValueLayout.ADDRESS, i, entry.getValue().nativeHandle());
                i++;
            }

            fn.invoke(keysPtr, tensorsPtr, (long) count, arena.allocateFrom(path.toAbsolutePath().toString()));
        } catch (Throwable t) {
            throw new RuntimeException("Failed to save state dict", t);
        } finally {
            arena.close();
        }
    }

    /**
     * Loads the module's state dictionary from a file.
     * <p>
     * The loaded state must match the module's architecture. Parameters are
     * updated in-place with the loaded values.
     * </p>
     *
     * @param path the file path to load from
     * @throws RuntimeException if loading fails
     */
    public void loadStateDict(java.nio.file.Path path) {
        Map<String, TorchTensor> stateDict = namedParameters();
        if (stateDict.isEmpty())
            return;

        java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined();
        try {
            tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding binding = tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding
                    .getInstance();
            java.lang.invoke.MethodHandle loadFn = binding.bind(
                    tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.LOAD_STATE_DICT,
                    tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.LOAD_STATE_DICT_DESC);

            int count = stateDict.size();
            java.lang.foreign.MemorySegment keysPtr = arena.allocate(java.lang.foreign.ValueLayout.ADDRESS, count);
            java.lang.foreign.MemorySegment tensorsPtr = arena.allocate(java.lang.foreign.ValueLayout.ADDRESS, count);

            int i = 0;
            for (Map.Entry<String, TorchTensor> entry : stateDict.entrySet()) {
                keysPtr.setAtIndex(java.lang.foreign.ValueLayout.ADDRESS, i, arena.allocateFrom(entry.getKey()));
                tensorsPtr.setAtIndex(java.lang.foreign.ValueLayout.ADDRESS, i, entry.getValue().nativeHandle());
                i++;
            }

            loadFn.invoke(arena.allocateFrom(path.toAbsolutePath().toString()), keysPtr, tensorsPtr, (long) count);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to load state dict", t);
        } finally {
            arena.close();
        }
    }

    @Override
    public void close() {
        parameters.values().forEach(TorchTensor::close);
        buffers.values().forEach(TorchTensor::close);
        submodules.values().forEach(Module::close);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(training=" + training + ")";
    }
}
