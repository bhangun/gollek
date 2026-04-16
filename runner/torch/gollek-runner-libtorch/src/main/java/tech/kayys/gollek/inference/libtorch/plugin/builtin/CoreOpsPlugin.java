package tech.kayys.gollek.inference.libtorch.plugin.builtin;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.plugin.LibTorchPlugin;

import java.util.Set;

/**
 * Built-in plugin providing core tensor operations.
 * <p>
 * Covers: add, sub, mul, div, matmul, reshape, transpose,
 * sum, mean, max, min, argmax, abs, sqrt, log, exp.
 */
public class CoreOpsPlugin implements LibTorchPlugin {

    @Override
    public String id() {
        return "core-ops";
    }

    @Override
    public String name() {
        return "Core TorchTensor Operations";
    }

    @Override
    public int priority() {
        return 10; // Highest priority built-in
    }

    @Override
    public void initialize(LibTorchBinding binding) {
        // Core ops are always available via LibTorchBinding constants
        // No additional initialization needed
    }

    @Override
    public Set<String> providedOperations() {
        return Set.of(
                "add", "sub", "mul", "div", "matmul",
                "reshape", "transpose",
                "sum", "mean", "max", "min", "argmax",
                "neg", "abs", "sqrt", "log", "exp",
                "eq", "gt", "lt",
                "clone", "to_device");
    }

    @Override
    public boolean isAvailable(LibTorchBinding binding) {
        // Core ops require at minimum the add symbol
        return binding.hasSymbol(LibTorchBinding.TENSOR_ADD);
    }
}
