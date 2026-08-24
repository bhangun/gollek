package tech.kayys.gollek.runtime.weight;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Factory for creating WeightSource instances.
 */
public final class WeightSourceFactory {
    private WeightSourceFactory() {}

    public static WeightSource create(Path path, ComputeBackend backend) throws IOException {
        if (Files.isDirectory(path)) {
            return new DirectoryWeightSource(path, backend);
        }
        
        String name = path.getFileName().toString().toLowerCase();
        if (name.endsWith(".safetensors")) {
            return new SafetensorWeightSource(path, backend);
        }
        
        if (name.endsWith(".gguf")) {
            // return new GGUFWeightSource(path, backend);
            throw new InferenceException(ErrorCode.CONFIG_UNSUPPORTED, "GGUF support not yet implemented");
        }
        
        throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "Unknown weight source format: " + path);
    }
}
