package tech.kayys.gollek.gguf.runner;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;
import tech.kayys.alkhawarizm.core.tensor.WeightAdapter;
import tech.kayys.gollek.gguf.loader.gguf.GGUFFile;
import tech.kayys.gollek.gguf.core.GGUFTensorInfo;
import tech.kayys.gollek.gguf.loader.quant.Dequantizer;

import java.util.HashMap;
import java.util.Map;

public class AljabrWeightAdapter implements WeightAdapter {
    private final GGUFFile gguf;
    private final int numLayers;
    private final int hiddenSize;
    private final int numHeads;
    private final Map<String, Tensor> cache = new HashMap<>();

    public AljabrWeightAdapter(GGUFFile gguf) {
        this.gguf = gguf;

        // Extract architecture metadata
        this.numLayers = getIntDefault(gguf.metadata(), "llama.block_count", 0);
        this.hiddenSize = getIntDefault(gguf.metadata(), "llama.embedding_length", 0);
        this.numHeads = getIntDefault(gguf.metadata(), "llama.attention.head_count", 0);
    }

    private int getIntDefault(Map<String, Object> map, String key, int def) {
        Object val = map.get(key);
        if (val instanceof Number) {
            return ((Number) val).intValue();
        }
        return def;
    }

    @Override
    public Tensor getWeight(String name) {
        if (cache.containsKey(name)) {
            return cache.get(name);
        }

        GGUFTensorInfo info = gguf.findTensor(name).orElse(null);
        if (info == null) {
            // Some models might not have certain tensors (like bias or ffn_gate)
            return null;
        }

        // GGUF tensor dimensions are stored in reverse order (e.g. [cols, rows])
        long[] shape = new long[info.shape().length];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = info.shape()[shape.length - 1 - i];
        }

        tech.kayys.alkhawarizm.core.tensor.DType dtype = mapDType(info.type());

        Tensor tensor;
        if (dtype == tech.kayys.alkhawarizm.core.tensor.DType.F32) {
            // F32 weights: dequantize (copy) into an owned JVM buffer.
            float[] floatData = Dequantizer.dequantize(info);
            tensor = TensorFactory.of(floatData, shape);
        } else if (dtype == tech.kayys.alkhawarizm.core.tensor.DType.F16) {
            // F16 weights (e.g. attention norms in some models): dequantize to F32.
            float[] floatData = Dequantizer.dequantize(info);
            tensor = TensorFactory.of(floatData, shape);
        } else if (info.data() != null) {
            // Block-quantized types (Q4_K, Q8_0, Q4_0 …): zero-copy — keep raw
            // GGUF block data as a MemorySegment slice. Metal kernels decode on-the-fly.
            tensor = TensorFactory.ofQuantized(info.data(), dtype, shape);
        } else {
            throw new IllegalStateException("GGUFTensorInfo has null data segment for tensor: " + info.name());
        }

        cache.put(name, tensor);
        return tensor;
    }

    @Override
    public int numLayers() {
        return numLayers;
    }

    @Override
    public int hiddenSize() {
        return hiddenSize;
    }

    @Override
    public int numHeads() {
        return numHeads;
    }

    private tech.kayys.alkhawarizm.core.tensor.DType mapDType(tech.kayys.gollek.gguf.core.GgmlType ggmlType) {
        switch (ggmlType) {
            case F32:
                return tech.kayys.alkhawarizm.core.tensor.DType.F32;
            case F16:
                return tech.kayys.alkhawarizm.core.tensor.DType.F16;
            case Q4_0:
                return tech.kayys.alkhawarizm.core.tensor.DType.F32; // Dequantize Q4_0 to F32 on load
            case Q4_K:
                return tech.kayys.alkhawarizm.core.tensor.DType.Q4_K;
            case Q8_0:
                return tech.kayys.alkhawarizm.core.tensor.DType.Q8_0;
            case Q6_K:
                return tech.kayys.alkhawarizm.core.tensor.DType.F32; // Dequantize Q6_K to F32 on load
            case Q5_K:
                return tech.kayys.alkhawarizm.core.tensor.DType.F32; // Dequantize Q5_K to F32 on load
            case Q2_K:
                return tech.kayys.alkhawarizm.core.tensor.DType.F32;
            case Q3_K:
                return tech.kayys.alkhawarizm.core.tensor.DType.F32;
            default:
                throw new UnsupportedOperationException("Unsupported GgmlType: " + ggmlType);
        }
    }
}
