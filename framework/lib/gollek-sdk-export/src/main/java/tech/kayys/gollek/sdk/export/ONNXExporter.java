package tech.kayys.gollek.sdk.export;

import tech.kayys.gollek.ml.export.OnnxExporter;
import tech.kayys.gollek.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * ONNX exporter — delegates to the FFM-based {@link OnnxExporter}.
 *
 * <p>Exports a Gollek model's state dict as a valid ONNX protobuf file
 * loadable by ONNX Runtime and the Python {@code onnx} library.
 */
public class ONNXExporter {

    private final NNModule model;
    private final long[] inputShape;

    /**
     * @param model      trained model to export
     * @param inputShape input shape (without batch dim), e.g. {@code {3, 224, 224}}
     * @param metadata   ignored (kept for API compatibility)
     */
    public ONNXExporter(NNModule model, long[] inputShape, Map<String, Object> metadata) {
        this.model      = model;
        this.inputShape = inputShape;
    }

    /**
     * Exports the model to an ONNX file.
     *
     * @param outputPath destination path (e.g. {@code model.onnx})
     * @throws IOException if writing fails
     */
    public void export(Path outputPath) throws IOException {
        OnnxExporter.export(model, inputShape, outputPath);
    }
}
