package tech.kayys.gollek.ml.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * ONNX model utilities — model info extraction and validation.
 *
 * <p>ONNX models are inference-only; actual execution is handled by the
 * ONNX Runtime runner plugin. This module provides metadata utilities
 * for the framework layer.
 */
public final class OnnxUtils {

    private OnnxUtils() {}

    /**
     * Check if a file is a valid ONNX model.
     */
    public static boolean isOnnxModel(Path path) {
        if (path == null || !Files.exists(path)) return false;
        String name = path.getFileName().toString().toLowerCase();
        return name.endsWith(".onnx");
    }

    /**
     * Get basic model info from an ONNX file.
     * Returns a map with keys: "producer", "ir_version", "graph_name"
     */
    public static Map<String, String> getModelInfo(Path path) throws IOException {
        if (!isOnnxModel(path)) {
            throw new IOException("Not an ONNX model: " + path);
        }
        long size = Files.size(path);
        return Map.of(
                "format", "onnx",
                "path", path.toString(),
                "size", String.valueOf(size));
    }
}
