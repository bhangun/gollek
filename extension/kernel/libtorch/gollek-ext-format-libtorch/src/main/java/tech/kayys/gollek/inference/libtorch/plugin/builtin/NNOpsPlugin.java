package tech.kayys.gollek.inference.libtorch.plugin.builtin;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.plugin.LibTorchPlugin;

import java.util.Set;

/**
 * Built-in plugin providing neural network operations.
 * <p>
 * Covers: relu, gelu, sigmoid, tanh, softmax, dropout, conv2d, batch_norm.
 */
public class NNOpsPlugin implements LibTorchPlugin {

    @Override
    public String id() {
        return "nn-ops";
    }

    @Override
    public String name() {
        return "Neural Network Operations";
    }

    @Override
    public int priority() {
        return 20;
    }

    @Override
    public void initialize(LibTorchBinding binding) {
        // NN ops use the binding's lazy resolution — no pre-init needed
    }

    @Override
    public Set<String> providedOperations() {
        return Set.of(
                "relu", "gelu", "sigmoid", "tanh",
                "softmax", "log_softmax",
                "dropout",
                "conv2d", "batch_norm",
                "linear");
    }

    @Override
    public boolean isAvailable(LibTorchBinding binding) {
        return binding.hasSymbol(LibTorchBinding.NN_RELU);
    }
}
