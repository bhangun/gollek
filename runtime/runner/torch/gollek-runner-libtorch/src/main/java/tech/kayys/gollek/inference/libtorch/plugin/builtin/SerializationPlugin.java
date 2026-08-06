package tech.kayys.gollek.inference.libtorch.plugin.builtin;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.plugin.LibTorchPlugin;

import java.util.Set;

/**
 * Built-in plugin providing model serialization and JIT operations.
 * <p>
 * Covers: jit_load, jit_forward, tensor_save, tensor_load.
 */
public class SerializationPlugin implements LibTorchPlugin {

    @Override
    public String id() {
        return "serialization";
    }

    @Override
    public String name() {
        return "Serialization & JIT";
    }

    @Override
    public int priority() {
        return 30;
    }

    @Override
    public void initialize(LibTorchBinding binding) {
        // Serialization ops are lazily resolved
    }

    @Override
    public Set<String> providedOperations() {
        return Set.of(
                "jit_load", "jit_forward", "jit_free",
                "tensor_save", "tensor_load");
    }

    @Override
    public boolean isAvailable(LibTorchBinding binding) {
        return binding.hasSymbol(LibTorchBinding.JIT_LOAD);
    }
}
