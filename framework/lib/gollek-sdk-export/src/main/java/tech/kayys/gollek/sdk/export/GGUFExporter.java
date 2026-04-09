package tech.kayys.gollek.sdk.export;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.util.gguf.GgufMetaValue;
import tech.kayys.gollek.ml.nn.util.gguf.GgufWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * GGUF exporter — writes a Gollek model to GGUF format for llama.cpp deployment.
 *
 * <p>Delegates to the existing {@link GgufWriter} which writes the binary GGUF format.
 */
public class GGUFExporter {

    private final NNModule model;
    private final Map<String, Object> metadata;
    private final ModelExporter.Quantization quantization;

    /**
     * @param model        trained model to export
     * @param inputShape   ignored (kept for API compatibility)
     * @param metadata     model metadata (architecture, etc.)
     * @param quantization quantization method (currently F32 only)
     */
    public GGUFExporter(Object model, long[] inputShape, Map<String, Object> metadata,
                        ModelExporter.Quantization quantization) {
        this.model        = model instanceof NNModule m ? m : null;
        this.metadata     = metadata != null ? metadata : Map.of();
        this.quantization = quantization;
    }

    /**
     * Exports the model to GGUF format.
     *
     * @param outputPath destination path (e.g. {@code model.gguf})
     * @throws IOException if writing fails
     * @throws IllegalStateException if model is not an {@link NNModule}
     */
    public void export(Path outputPath) throws IOException {
        if (model == null) throw new IllegalStateException("Model must be a Module instance");

        Map<String, GradTensor> stateDict = model.stateDict();
        Map<String, GgufMetaValue> ggufMeta = new LinkedHashMap<>();
        ggufMeta.put("general.architecture",
            GgufMetaValue.ofString(metadata.getOrDefault("architecture", "gollek").toString()));
        ggufMeta.put("general.quantization",
            GgufMetaValue.ofString(quantization.name()));

        GgufWriter.save(outputPath, stateDict, ggufMeta);
    }
}
