package tech.kayys.gollek.inference.libtorch.plugin.builtin;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.plugin.LibTorchPlugin;

import java.util.Set;

/**
 * Built-in plugin providing CUDA/GPU operations.
 * <p>
 * Only available when CUDA-enabled LibTorch is installed.
 */
public class CudaPlugin implements LibTorchPlugin {

    @Override
    public String id() {
        return "cuda";
    }

    @Override
    public String name() {
        return "CUDA GPU Operations";
    }

    @Override
    public int priority() {
        return 40;
    }

    @Override
    public void initialize(LibTorchBinding binding) {
        // CUDA-specific init if needed
    }

    @Override
    public Set<String> providedOperations() {
        return Set.of(
                "cuda_is_available", "cuda_device_count",
                "to_cuda", "to_cpu");
    }

    @Override
    public boolean isAvailable(LibTorchBinding binding) {
        return binding.hasSymbol(LibTorchBinding.CUDA_IS_AVAILABLE);
    }
}
