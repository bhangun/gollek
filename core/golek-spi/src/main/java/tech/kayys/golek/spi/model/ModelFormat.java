package tech.kayys.golek.spi.model;

import java.util.Arrays;
import java.util.Optional;

/**
 * Enumeration of supported model formats
 */
public enum ModelFormat {
    GGUF("gguf", "llama.cpp"),
    ONNX("onnx", "ONNX Runtime"),
    TENSORRT("trt", "TensorRT"),
    TORCHSCRIPT("pt", "TorchScript"),
    TENSORFLOW_SAVED_MODEL("pb", "TensorFlow");

    private final String extension;
    private final String runtime;

    ModelFormat(String extension, String runtime) {
        this.extension = extension;
        this.runtime = runtime;
    }

    /**
     * Gets the file extension associated with this model format
     *
     * @return the file extension
     */
    public String getExtension() {
        return extension;
    }

    /**
     * Gets the runtime engine associated with this model format
     *
     * @return the runtime engine
     */
    public String getRuntime() {
        return runtime;
    }

    /**
     * Finds a ModelFormat by its extension
     *
     * @param extension the file extension to search for
     * @return Optional containing the ModelFormat if found, empty otherwise
     */
    public static Optional<ModelFormat> findByExtension(String extension) {
        if (extension == null) {
            return Optional.empty();
        }

        return Arrays.stream(values())
                .filter(format -> format.extension.equalsIgnoreCase(extension))
                .findFirst();
    }

    /**
     * Finds a ModelFormat by its runtime
     *
     * @param runtime the runtime to search for
     * @return Optional containing the ModelFormat if found, empty otherwise
     */
    public static Optional<ModelFormat> findByRuntime(String runtime) {
        if (runtime == null) {
            return Optional.empty();
        }

        return Arrays.stream(values())
                .filter(format -> format.runtime.equalsIgnoreCase(runtime))
                .findFirst();
    }

    @Override
    public String toString() {
        return String.format("%s (extension: %s, runtime: %s)",
                           name(), extension, runtime);
    }
}
