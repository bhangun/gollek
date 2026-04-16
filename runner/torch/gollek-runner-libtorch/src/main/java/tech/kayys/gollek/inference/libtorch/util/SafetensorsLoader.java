package tech.kayys.gollek.inference.libtorch.util;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

/**
 * Zero-copy loader for <code>.safetensors</code> weight files.
 * <p>
 * Uses {@link java.nio.channels.FileChannel#map} to memory-map tensor data
 * directly from disk and wraps it with LibTorch's {@code at_from_blob}
 * via the FFM API — no data copying occurs.
 *
 * <h3>Usage:</h3>
 * 
 * <pre>{@code
 * Map<String, TorchTensor> weights = loader.loadAll(path);
 * TorchTensor embedding = loader.loadTensor(path, "model.embed_tokens.weight");
 * }</pre>
 */
@ApplicationScoped
public class SafetensorsLoader {

    private static final Logger log = Logger.getLogger(SafetensorsLoader.class);

    @Inject
    SafetensorsHeaderParser headerParser;

    /**
     * Load all tensors from a safetensors file.
     * <p>
     * Each tensor is memory-mapped and wrapped as a native LibTorch tensor.
     * The caller is responsible for closing the returned tensors.
     *
     * @param path path to the .safetensors file
     * @return map of tensor name → TorchTensor
     * @throws IOException if file I/O fails
     */
    public Map<String, TorchTensor> loadAll(Path path) throws IOException {
        Map<String, SafetensorsHeaderParser.TensorMetadata> metadata = headerParser.parse(path);
        Map<String, TorchTensor> tensors = new HashMap<>();

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            for (var entry : metadata.entrySet()) {
                try {
                    TorchTensor tensor = mapTensorFromChannel(channel, entry.getValue());
                    tensors.put(entry.getKey(), tensor);
                } catch (Exception e) {
                    // Cleanup already-loaded tensors on failure
                    tensors.values().forEach(TorchTensor::close);
                    throw new IOException("Failed to load tensor: " + entry.getKey(), e);
                }
            }
        }

        log.infof("Loaded %d tensors from safetensors: %s", tensors.size(), path.getFileName());
        return tensors;
    }

    /**
     * Load a single named tensor from a safetensors file.
     *
     * @param path       path to the .safetensors file
     * @param tensorName name of the tensor to load
     * @return the loaded tensor
     * @throws IOException              if file I/O fails
     * @throws IllegalArgumentException if the tensor name is not found
     */
    public TorchTensor loadTensor(Path path, String tensorName) throws IOException {
        Map<String, SafetensorsHeaderParser.TensorMetadata> metadata = headerParser.parse(path);

        SafetensorsHeaderParser.TensorMetadata info = metadata.get(tensorName);
        if (info == null) {
            throw new IllegalArgumentException(
                    "TorchTensor '" + tensorName + "' not found in " + path.getFileName()
                            + ". Available: " + metadata.keySet());
        }

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return mapTensorFromChannel(channel, info);
        } catch (Exception e) {
            throw new IOException("Failed to load tensor: " + tensorName, e);
        }
    }

    /**
     * List all tensor names and their metadata from a safetensors file.
     *
     * @param path path to the .safetensors file
     * @return map of tensor name → metadata
     * @throws IOException if parsing fails
     */
    public Map<String, SafetensorsHeaderParser.TensorMetadata> inspect(Path path) throws IOException {
        return headerParser.parse(path);
    }

    // ── Internal ──────────────────────────────────────────────────────

    private TorchTensor mapTensorFromChannel(FileChannel channel,
            SafetensorsHeaderParser.TensorMetadata info) throws Exception {
        Arena arena = Arena.ofConfined();
        try {
            // 1. Memory-map the exact tensor slice from disk (zero-copy)
            MemorySegment mapped = channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    info.absoluteStart(),
                    info.length(),
                    arena);

            // 2. Resolve the scalar type
            ScalarType scalarType = ScalarType.fromSafetensors(info.dtype());

            // 3. Prepare shape array as a native segment
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, info.shape());

            // 4. Call at_from_blob to create a LibTorch tensor view over mapped memory
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fromBlob = binding.bind(
                    LibTorchBinding.TENSOR_FROM_BLOB,
                    LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            MemorySegment handle = (MemorySegment) fromBlob.invoke(
                    mapped,
                    shapeSegment,
                    (long) info.shape().length,
                    scalarType.code());

            log.debugf("Mapped tensor: dtype=%s, shape=%s, bytes=%d",
                    info.dtype(), java.util.Arrays.toString(info.shape()), info.length());

            return new TorchTensor(handle, arena);
        } catch (Throwable t) {
            arena.close();
            throw new Exception("Failed to map tensor from disk", t);
        }
    }
}
