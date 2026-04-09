package tech.kayys.gollek.sdk.export;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Unified model exporter for deployment.
 *
 * <p>Supports exporting trained models to various formats for production deployment:</p>
 * <ul>
 *   <li><b>ONNX</b> - Open Neural Network Exchange (cross-platform)</li>
 *   <li><b>GGUF</b> - GPT-Generated Unified Format (llama.cpp)</li>
 *   <li><b>LiteRT</b> - Google LiteRT (edge devices)</li>
 * </ul>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * ModelExporter exporter = ModelExporter.builder()
 *     .model(model)
 *     .inputShape(1, 3, 224, 224)
 *     .build();
 *
 * // Export to different formats
 * exporter.toONNX("model.onnx");
 * exporter.toGGUF("model.gguf", Quantization.INT4);
 * exporter.toLiteRT("model.litert");
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class ModelExporter {

    private final NNModule model;
    private final long[] inputShape;
    private final Map<String, Object> metadata;

    private ModelExporter(Builder builder) {
        this.model = builder.model;
        this.inputShape = builder.inputShape;
        this.metadata = builder.metadata;
    }

    /**
     * Create a new builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Export model to ONNX format.
     *
     * @param outputPath path to save ONNX model
     * @throws IOException if export fails
     */
    public void toONNX(Path outputPath) throws IOException {
        ONNXExporter exporter = new ONNXExporter(model, inputShape, metadata);
        exporter.export(outputPath);
    }

    /**
     * Export model to ONNX format (convenience method).
     *
     * @param outputPath path string
     * @throws IOException if export fails
     */
    public void toONNX(String outputPath) throws IOException {
        toONNX(Path.of(outputPath));
    }

    /**
     * Export model to GGUF format.
     *
     * @param outputPath   path to save GGUF model
     * @param quantization quantization method
     * @throws IOException if export fails
     */
    public void toGGUF(Path outputPath, Quantization quantization) throws IOException {
        GGUFExporter exporter = new GGUFExporter(model, inputShape, metadata, quantization);
        exporter.export(outputPath);
    }

    /**
     * Export model to GGUF format (convenience method).
     *
     * @param outputPath   path string
     * @param quantization quantization method
     * @throws IOException if export fails
     */
    public void toGGUF(String outputPath, Quantization quantization) throws IOException {
        toGGUF(Path.of(outputPath), quantization);
    }

    /**
     * Export model to LiteRT format.
     *
     * @param outputPath path to save LiteRT model
     * @throws IOException if export fails
     */
    public void toLiteRT(Path outputPath) throws IOException {
        LiteRTExporter exporter = new LiteRTExporter(model, inputShape, metadata);
        exporter.export(outputPath);
    }

    /**
     * Export model to LiteRT format (convenience method).
     *
     * @param outputPath path string
     * @throws IOException if export fails
     */
    public void toLiteRT(String outputPath) throws IOException {
        toLiteRT(Path.of(outputPath));
    }

    /**
     * Get model size estimate in bytes.
     *
     * @return estimated size
     */
    public long estimateSize() {
        if (model instanceof tech.kayys.gollek.ml.nn.NNModule module) {
            // float32: 4 bytes per parameter
            return module.parameterCount() * 4L;
        }
        return 0L;
    }

    /**
     * Quantization methods for model compression.
     */
    public enum Quantization {
        /** No quantization (FP32) */
        NONE,
        /** 16-bit floating point */
        FP16,
        /** 8-bit integer */
        INT8,
        /** 4-bit integer (GPTQ-style) */
        INT4,
        /** 4-bit NormalFloat (QLoRA-style) */
        NF4
    }

    /**
     * Builder for ModelExporter.
     */
    public static class Builder {
        private NNModule model;
        private long[] inputShape;
        private Map<String, Object> metadata;

        private Builder() {}

        /**
         * Set the model to export.
         *
         * @param model model instance
         * @return this builder
         */
        public Builder model(NNModule model) {
            this.model = model;
            return this;
        }

        /**
         * Set input shape for tracing.
         *
         * @param shape input shape array
         * @return this builder
         */
        public Builder inputShape(long... shape) {
            this.inputShape = shape.clone();
            return this;
        }

        /**
         * Set metadata to include in exported model.
         *
         * @param metadata metadata map
         * @return this builder
         */
        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        /**
         * Build the ModelExporter instance.
         *
         * @return configured exporter
         */
        public ModelExporter build() {
            return new ModelExporter(this);
        }
    }
}
