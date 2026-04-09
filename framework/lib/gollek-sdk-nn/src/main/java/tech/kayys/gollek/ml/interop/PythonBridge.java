package tech.kayys.gollek.ml.interop;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.StateDict;

import java.io.*;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.nio.file.*;
import java.util.*;

/**
 * Python interoperability bridge — loads PyTorch model weights and NumPy arrays
 * directly into Gollek tensors without requiring Python at runtime.
 *
 * <p>Supports:
 * <ul>
 *   <li>SafeTensors format (via {@link StateDict}) — recommended</li>
 *   <li>NumPy {@code .npy} files — single array</li>
 *   <li>NumPy {@code .npz} files — multiple arrays</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Load PyTorch weights saved as SafeTensors
 * model.loadStateDict(PythonBridge.loadSafeTensors("model.safetensors"));
 *
 * // Load a NumPy array
 * GradTensor tensor = PythonBridge.loadNpy("embeddings.npy");
 *
 * // Load multiple arrays from .npz
 * Map<String, GradTensor> arrays = PythonBridge.loadNpz("data.npz");
 * }</pre>
 */
public final class PythonBridge {

    private PythonBridge() {}

    // ── SafeTensors ───────────────────────────────────────────────────────

    /**
     * Loads a SafeTensors file saved from PyTorch.
     *
     * <p>Compatible with {@code torch.save(model.state_dict(), f, format='safetensors')}
     * and the Python {@code safetensors} library.
     *
     * @param path path to {@code .safetensors} file
     * @return state dict map of parameter name → tensor
     * @throws IOException if the file cannot be read
     */
    public static Map<String, GradTensor> loadSafeTensors(Path path) throws IOException {
        return StateDict.load(path);
    }

    /** @see #loadSafeTensors(Path) */
    public static Map<String, GradTensor> loadSafeTensors(String path) throws IOException {
        return loadSafeTensors(Path.of(path));
    }

    // ── NumPy .npy ────────────────────────────────────────────────────────

    /**
     * Loads a NumPy {@code .npy} file into a {@link GradTensor}.
     *
     * <p>Supports float32 ({@code dtype=np.float32}) arrays.
     * Uses JDK 25 FFM {@link MemorySegment} for zero-copy reading.
     *
     * @param path path to {@code .npy} file
     * @return tensor with the array data and shape
     * @throws IOException if the file cannot be read or is not float32
     */
    public static GradTensor loadNpy(Path path) throws IOException {
        byte[] raw = Files.readAllBytes(path);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(raw.length, 1);
            MemorySegment.copy(MemorySegment.ofArray(raw), 0, seg, 0, raw.length);
            return parseNpy(seg, raw.length);
        }
    }

    /** @see #loadNpy(Path) */
    public static GradTensor loadNpy(String path) throws IOException {
        return loadNpy(Path.of(path));
    }

    /**
     * Loads a NumPy {@code .npz} file (zip of {@code .npy} files).
     *
     * @param path path to {@code .npz} file
     * @return map of array name → tensor
     * @throws IOException if the file cannot be read
     */
    public static Map<String, GradTensor> loadNpz(Path path) throws IOException {
        Map<String, GradTensor> result = new LinkedHashMap<>();
        try (var zip = new java.util.zip.ZipInputStream(Files.newInputStream(path))) {
            java.util.zip.ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                String name = entry.getName();
                if (!name.endsWith(".npy")) continue;
                String key = name.substring(0, name.length() - 4); // strip .npy
                byte[] data = zip.readAllBytes();
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment seg = arena.allocate(data.length, 1);
                    MemorySegment.copy(MemorySegment.ofArray(data), 0, seg, 0, data.length);
                    result.put(key, parseNpy(seg, data.length));
                }
            }
        }
        return result;
    }

    // ── NumPy format parser ───────────────────────────────────────────────

    /**
     * Parses a NumPy .npy binary format from a MemorySegment.
     *
     * <p>Format: {@code \x93NUMPY} magic + version + header_len + header + data
     */
    private static GradTensor parseNpy(MemorySegment seg, int totalLen) throws IOException {
        // Magic: \x93NUMPY (6 bytes)
        if (seg.get(ValueLayout.JAVA_BYTE, 0) != (byte) 0x93)
            throw new IOException("Not a valid .npy file");

        // Header length at offset 8 (little-endian uint16)
        int headerLen = (seg.get(ValueLayout.JAVA_BYTE, 8) & 0xFF)
                      | ((seg.get(ValueLayout.JAVA_BYTE, 9) & 0xFF) << 8);
        int dataOffset = 10 + headerLen;

        // Parse header string for shape and dtype
        byte[] headerBytes = new byte[headerLen];
        MemorySegment.copy(seg, 10, MemorySegment.ofArray(headerBytes), 0, headerLen);
        String header = new String(headerBytes, java.nio.charset.StandardCharsets.US_ASCII);

        long[] shape = parseShape(header);
        boolean isFloat32 = header.contains("'<f4'") || header.contains("'f4'")
                         || header.contains("float32");
        if (!isFloat32) throw new IOException("Only float32 .npy files are supported");

        int numel = 1;
        for (long d : shape) numel *= (int) d;
        float[] data = new float[numel];
        MemorySegment.copy(seg, dataOffset, MemorySegment.ofArray(data), 0, (long) numel * Float.BYTES);
        return GradTensor.of(data, shape);
    }

    /** Extracts shape tuple from NumPy header string. */
    private static long[] parseShape(String header) {
        int start = header.indexOf("'shape': (") + 10;
        int end   = header.indexOf(')', start);
        String shapeStr = header.substring(start, end).trim();
        if (shapeStr.isEmpty()) return new long[]{1}; // scalar
        String[] parts = shapeStr.split(",");
        long[] shape = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String p = parts[i].trim();
            if (!p.isEmpty()) shape[i] = Long.parseLong(p);
        }
        return shape;
    }

    // ── Convenience: load weights into model ─────────────────────────────

    /**
     * Loads PyTorch SafeTensors weights directly into a model.
     *
     * @param model   target model
     * @param path    path to {@code .safetensors} file
     * @param strict  if false, silently skips mismatched keys (for transfer learning)
     * @throws IOException if loading fails
     */
    public static void loadIntoModel(NNModule model, Path path, boolean strict) throws IOException {
        model.loadStateDict(loadSafeTensors(path), strict);
    }
}
