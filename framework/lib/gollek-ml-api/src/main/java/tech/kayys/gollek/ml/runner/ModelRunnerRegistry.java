package tech.kayys.gollek.ml.runner;

import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.ml.tensor.RunnerDevice;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Registry that manages ModelRunner instances and provides framework-specific
 * runner creation based on model format.
 *
 * <p>The registry auto-selects the appropriate backend:
 * <pre>
 *   GGUF → GgufRunner
 *   SAFETENSORS → SafetensorRunner
 *   LITERT → LiteRTRunner
 *   ONNX → OnnxRunner
 *   TORCHSCRIPT → TorchScriptRunner
 *   TENSORRT → TensorRTRunner
 * </pre>
 *
 * <p>Custom runners can be registered:
 * <pre>
 *   ModelRunnerRegistry.get().register(ModelFormat.CUSTOM, CustomRunner::new);
 * </pre>
 *
 * @since 0.3.0
 */
public class ModelRunnerRegistry {

    private static volatile ModelRunnerRegistry instance;

    /** Runner factories by format */
    private final Map<ModelFormat, Function<RunnerConfig, ModelRunner>> runnerFactories = new ConcurrentHashMap<>();

    /** Active runner instances by ID */
    private final Map<String, ModelRunner> activeRunners = new ConcurrentHashMap<>();

    private ModelRunnerRegistry() {
        registerDefaults();
    }

    /**
     * Gets the singleton registry instance.
     */
    public static ModelRunnerRegistry get() {
        if (instance == null) {
            synchronized (ModelRunnerRegistry.class) {
                if (instance == null) {
                    instance = new ModelRunnerRegistry();
                }
            }
        }
        return instance;
    }

    /**
     * Resets the singleton instance (useful for testing).
     */
    public static void reset() {
        instance = null;
    }

    /**
     * Registers a runner factory for a model format.
     *
     * @param format the model format
     * @param factory function to create runners for this format
     */
    public void register(ModelFormat format, Function<RunnerConfig, ModelRunner> factory) {
        runnerFactories.put(format, factory);
    }

    /**
     * Creates a new runner for the given format and path.
     *
     * @param id runner identifier
     * @param format model format
     * @param modelPath path to the model file/directory
     * @param device target device
     * @param options framework-specific options
     * @return configured model runner
     */
    public ModelRunner createRunner(String id, ModelFormat format, Path modelPath,
                                    RunnerDevice device, Map<String, Object> options) {
        Function<RunnerConfig, ModelRunner> factory = runnerFactories.get(format);
        if (factory == null) {
            throw new IllegalArgumentException("No runner registered for format: " + format);
        }

        RunnerConfig config = new RunnerConfig(id, format, modelPath, device, options);
        ModelRunner runner = factory.apply(config);

        activeRunners.put(id, runner);
        return runner;
    }

    /**
     * Gets an active runner by ID.
     *
     * @param id runner identifier
     * @return the runner, or null if not found
     */
    public ModelRunner getRunner(String id) {
        return activeRunners.get(id);
    }

    /**
     * Gets all active runners.
     */
    public Map<String, ModelRunner> getActiveRunners() {
        return Map.copyOf(activeRunners);
    }

    /**
     * Removes and closes a runner.
     *
     * @param id runner identifier
     */
    public void removeRunner(String id) {
        ModelRunner runner = activeRunners.remove(id);
        if (runner != null) {
            runner.close();
        }
    }

    /**
     * Closes all active runners.
     */
    public void closeAll() {
        for (ModelRunner runner : activeRunners.values()) {
            try {
                runner.close();
            } catch (Exception e) {
                // Log and continue
            }
        }
        activeRunners.clear();
    }

    /**
     * Registers default runners for all supported formats.
     */
    private void registerDefaults() {
        // GGUF runner
        register(ModelFormat.GGUF, config -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.inference.gguf.GGUFModelRunner");
                return (ModelRunner) clazz.getDeclaredConstructor(RunnerConfig.class).newInstance(config);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create GGUF runner", e);
            }
        });

        // SafeTensors runner
        register(ModelFormat.SAFETENSORS, config -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.inference.safetensor.SafetensorModelRunner");
                return (ModelRunner) clazz.getDeclaredConstructor(RunnerConfig.class).newInstance(config);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create SafeTensors runner", e);
            }
        });

        // LiteRT runner
        register(ModelFormat.LITERT, config -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.inference.litert.LiteRTModelRunner");
                return (ModelRunner) clazz.getDeclaredConstructor(RunnerConfig.class).newInstance(config);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create LiteRT runner", e);
            }
        });

        // ONNX runner
        register(ModelFormat.ONNX, config -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.inference.onnx.OnnxModelRunner");
                return (ModelRunner) clazz.getDeclaredConstructor(RunnerConfig.class).newInstance(config);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create ONNX runner", e);
            }
        });

        // TorchScript runner
        register(ModelFormat.TORCHSCRIPT, config -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.inference.libtorch.TorchScriptModelRunner");
                return (ModelRunner) clazz.getDeclaredConstructor(RunnerConfig.class).newInstance(config);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create TorchScript runner", e);
            }
        });

        // TensorRT runner
        register(ModelFormat.TENSORRT, config -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.inference.tensorrt.TensorRTModelRunner");
                return (ModelRunner) clazz.getDeclaredConstructor(RunnerConfig.class).newInstance(config);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create TensorRT runner", e);
            }
        });
    }

    /**
     * Configuration for creating a model runner.
     */
    public record RunnerConfig(
        String id,
        ModelFormat format,
        Path modelPath,
        RunnerDevice device,
        Map<String, Object> options
    ) {}
}
