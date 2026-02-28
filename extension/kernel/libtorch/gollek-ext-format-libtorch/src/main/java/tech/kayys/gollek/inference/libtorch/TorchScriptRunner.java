package tech.kayys.gollek.inference.libtorch;

import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Manages a loaded TorchScript model and executes forward passes.
 * <p>
 * Each runner owns a JIT-loaded model and an associated Arena for native
 * memory.
 * Not thread-safe — use within a session context.
 */
public class TorchScriptRunner implements AutoCloseable {

    private static final Logger log = Logger.getLogger(TorchScriptRunner.class);

    private final Path modelPath;
    private final Device device;
    private final MemorySegment moduleHandle;
    private final Arena arena;
    private boolean closed = false;

    private TorchScriptRunner(Path modelPath, Device device,
            MemorySegment moduleHandle, Arena arena) {
        this.modelPath = modelPath;
        this.device = device;
        this.moduleHandle = moduleHandle;
        this.arena = arena;
    }

    /**
     * Load a TorchScript model from the given path.
     *
     * @param modelPath path to the .pt/.pts model file
     * @param device    device to load the model onto
     * @return a new TorchScriptRunner
     * @throws RuntimeException if loading fails
     */
    public static TorchScriptRunner load(Path modelPath, Device device) {
        if (!Files.exists(modelPath)) {
            throw new IllegalArgumentException("Model file not found: " + modelPath);
        }

        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle jitLoad = binding.bind(LibTorchBinding.JIT_LOAD, LibTorchBinding.JIT_LOAD_DESC);

            MemorySegment pathStr = arena.allocateFrom(modelPath.toString());
            MemorySegment moduleHandle = (MemorySegment) jitLoad.invoke(pathStr);

            if (moduleHandle == null || moduleHandle.equals(MemorySegment.NULL)) {
                throw new RuntimeException("JIT load returned null for: " + modelPath);
            }

            log.infof("Loaded TorchScript model: %s (device=%s)", modelPath.getFileName(), device);
            return new TorchScriptRunner(modelPath, device, moduleHandle, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to load TorchScript model: " + modelPath, t);
        }
    }

    /**
     * Run a forward pass on the model.
     *
     * @param input input tensor
     * @return output tensor from the model
     */
    public Tensor forward(Tensor input) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle forward = binding.bind(
                    LibTorchBinding.JIT_MODULE_FORWARD,
                    LibTorchBinding.JIT_MODULE_FORWARD_DESC);

            MemorySegment result = (MemorySegment) forward.invoke(moduleHandle, input.nativeHandle());
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Forward pass failed on model: " + modelPath.getFileName(), t);
        }
    }

    /**
     * Get the model file path.
     */
    public Path modelPath() {
        return modelPath;
    }

    /**
     * Get the device the model is loaded on.
     */
    public Device device() {
        return device;
    }

    public boolean isClosed() {
        return closed;
    }

    MemorySegment moduleHandle() {
        checkClosed();
        return moduleHandle;
    }

    private void checkClosed() {
        if (closed) {
            throw new IllegalStateException("TorchScriptRunner has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            try {
                LibTorchBinding binding = LibTorchBinding.getInstance();
                binding.bindOptional(LibTorchBinding.JIT_MODULE_FREE, LibTorchBinding.JIT_MODULE_FREE_DESC)
                        .ifPresent(fn -> {
                            try {
                                fn.invoke(moduleHandle);
                            } catch (Throwable t) {
                                log.warnf(t, "Error freeing JIT module: %s", modelPath.getFileName());
                            }
                        });
            } finally {
                arena.close();
            }
            log.debugf("Closed TorchScript runner: %s", modelPath.getFileName());
        }
    }
}
