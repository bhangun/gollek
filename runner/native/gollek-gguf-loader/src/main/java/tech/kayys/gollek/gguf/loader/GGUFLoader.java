package tech.kayys.gollek.gguf.loader;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;
import tech.kayys.gollek.gguf.loader.GGUFReader;
import tech.kayys.gollek.gguf.loader.GGUFParser;
import tech.kayys.gollek.spi.tensor.weights.TensorData;
import tech.kayys.gollek.spi.tensor.weights.TransformerLayerWeights;
import tech.kayys.gollek.spi.tensor.weights.Dequantizer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import java.nio.file.Path;
import java.io.IOException;

/**
 * High-level loader that maps a parsed GGUF model to the engine's internal 
 * TransformerLayerWeights structures, performing necessary prepacking.
 */
public final class GGUFLoader {

    public static GGUFModel loadModel(Path path) throws IOException {
        Arena arena = Arena.ofAuto();
        try (GGUFReader reader = new GGUFReader(path, arena)) {
            return new GGUFParser().parse(reader.segment(), arena);
        } catch (Throwable t) {
            arena.close();
            throw t;
        }
    }

    public static List<TransformerLayerWeights> loadLayers(GGUFModel model, Arena arena) {
        Map<String, Object> meta = model.metadata();
        String arch = (String) meta.getOrDefault("general.architecture", "llama");
        
        int nLayer = getMetadataInt(meta, arch + ".block_count", "llama.block_count", "general.block_count");
        int hidden = getMetadataInt(meta, arch + ".embedding_length", "llama.embedding_length", "general.embedding_length");
        int nHeads = getMetadataInt(meta, arch + ".attention.head_count", "llama.attention.head_count", "general.attention.head_count");
        
        // Very robust lookup for KV heads
        int nHeadsKv = getMetadataIntOptional(meta, 
            arch + ".attention.head_count_kv", 
            "llama.attention.head_count_kv", 
            "general.attention.head_count_kv",
            "attention.head_count_kv",
            "head_count_kv");
        
        if (nHeadsKv == 0) {
            // Try suffix match for head_count_kv
            for (String key : meta.keySet()) {
                if (key.endsWith(".attention.head_count_kv") || key.endsWith(".head_count_kv")) {
                    Object v = meta.get(key);
                    if (v instanceof Number n) {
                        nHeadsKv = n.intValue();
                        break;
                    }
                }
            }
        }
        if (nHeadsKv == 0) nHeadsKv = nHeads; // Final fallback

        int headDim = getMetadataIntOptional(meta, 
            arch + ".attention.head_dim", 
            "gemma2.attention.head_dim",
            "llama.attention.head_dim",
            "key_length");
            
        if (headDim == 0) {
            // Try to derive from tensor shape: [hidden, nHeads * headDim]
            try {
                GGUFTensorInfo qInfo = model.tensors().stream()
                    .filter(t -> t.name().equals("blk.0.attn_q.weight"))
                    .findFirst().orElse(null);
                if (qInfo != null && qInfo.shape().length >= 2) {
                    headDim = (int) (qInfo.shape()[1] / nHeads);
                }
            } catch (Exception e) {
                // Fallback to traditional calculation
                headDim = hidden / nHeads;
            }
        }
        
        if (headDim == 0) headDim = hidden / nHeads;
 
        int numExperts = getMetadataIntOptional(meta, arch + ".expert_count", "general.expert_count");
        int numExpertsPerTok = getMetadataIntOptional(meta, arch + ".expert_used_count", "general.expert_used_count");
        
        if (numExperts > 0) {
            System.out.println("MoE detected: " + numExperts + " experts, " + numExpertsPerTok + " per token.");
        }

        
        System.out.println("Loading " + nLayer + " layers: arch=" + arch + ", hidden=" + hidden + ", nHeads=" + nHeads + ", nHeadsKv=" + nHeadsKv);
        System.out.println("headDim=" + headDim + " -> Q size=" + (nHeads * headDim) + ", K/V size=" + (nHeadsKv * headDim));

        List<TransformerLayerWeights> layers = new ArrayList<>(nLayer);

        for (int i = 0; i < nLayer; i++) {
            if (i % 10 == 0) {
                System.out.print("Loading layers " + i + "-" + Math.min(i + 9, nLayer - 1) + "... ");
                System.out.flush();
            }
            String prefix = "blk." + i + ".";

            
            MemorySegment rms = findAndPrepareTensor(model, prefix + "attn_norm.weight", arena);
            
            // Collect Q, K, V for prepacking
            MemorySegment q = findAndPrepareTensorOptional(model, prefix + "attn_q.weight", arena);
            MemorySegment k = findAndPrepareTensorOptional(model, prefix + "attn_k.weight", arena);
            MemorySegment v = findAndPrepareTensorOptional(model, prefix + "attn_v.weight", arena);
            
            // Bias handling: Qwen2 and others use biases.
            MemorySegment bq = findAndPrepareTensorOptional(model, prefix + "attn_q.bias", arena);
            MemorySegment bk = findAndPrepareTensorOptional(model, prefix + "attn_k.bias", arena);
            MemorySegment bv = findAndPrepareTensorOptional(model, prefix + "attn_v.bias", arena);

            MemorySegment bqkv = null;
            if (bq != null || bk != null || bv != null) {
                // If any bias exists, we combine them. missing ones are treated as zero.
                // We pass hidden=1 to combineQKV for bias vectors.
                bqkv = combineQKVBias(bq, bk, bv, nHeads, nHeadsKv, headDim, arena);
            } else {
                bqkv = findAndPrepareTensorOptional(model, prefix + "attn_qkv.bias", arena);
            }

            MemorySegment wqkv;
            if (q == null) {
                // Try fused QKV weight
                wqkv = findAndPrepareTensor(model, prefix + "attn_qkv.weight", arena);
            } else {
                wqkv = combineQKV(q, k, v, hidden, nHeads, nHeadsKv, headDim, arena);
            }
            
            MemorySegment packed = QKVPrepacker.prepack(wqkv, hidden, nHeads, nHeadsKv, headDim, arena);

            TensorData wo = findTensorLazy(model, prefix + "attn_output.weight");
            if (i == 0) {
                GGUFTensorInfo woInfo = model.tensors().stream()
                    .filter(t -> t.name().equals(prefix + "attn_output.weight"))
                    .findFirst().orElse(null);
                if (woInfo != null) {
                    System.out.println("wo[0] shape: " + java.util.Arrays.toString(woInfo.shape()) + 
                        " type=" + woInfo.typeId() + " elements=" + wo.numElements());
                }
            }
            MemorySegment bo = findAndPrepareTensorOptional(model, prefix + "attn_output.bias", arena);

            MemorySegment ffnNorm = findAndPrepareTensor(model, prefix + "ffn_norm.weight", arena);
            MemorySegment postAttnNorm = findAndPrepareTensorOptional(model, prefix + "post_attention_layernorm.weight", arena);
            if (postAttnNorm == null) postAttnNorm = findAndPrepareTensorOptional(model, prefix + "ffn_pre_norm.weight", arena);
            
            MemorySegment postFfnNorm = findAndPrepareTensorOptional(model, prefix + "post_feedforward_layernorm.weight", arena);
            if (postFfnNorm == null) postFfnNorm = findAndPrepareTensorOptional(model, prefix + "ffn_post_norm.weight", arena);

            MemorySegment attnQNorm = findAndPrepareTensorOptional(model, prefix + "attn_q_norm.weight", arena);
            MemorySegment attnKNorm = findAndPrepareTensorOptional(model, prefix + "attn_k_norm.weight", arena);
            
            TensorData wG = findTensorLazy(model, prefix + "ffn_gate.weight");
            MemorySegment bg = findAndPrepareTensorOptional(model, prefix + "ffn_gate.bias", arena);
            
            TensorData wU = findTensorLazy(model, prefix + "ffn_up.weight");
            MemorySegment bu = findAndPrepareTensorOptional(model, prefix + "ffn_up.bias", arena);
            
            TensorData wD = findTensorLazy(model, prefix + "ffn_down.weight");
            MemorySegment bd = findAndPrepareTensorOptional(model, prefix + "ffn_down.bias", arena);
 
            MemorySegment ffnGateInp = null;
            TensorData[] wGExps = null;
            TensorData[] wUExps = null;
            TensorData[] wDExps = null;
            
            if (numExperts > 0) {
                ffnGateInp = findAndPrepareTensorOptional(model, prefix + "ffn_gate_inp.weight", arena);
                wGExps = new TensorData[numExperts];
                wUExps = new TensorData[numExperts];
                wDExps = new TensorData[numExperts];
                for (int e = 0; e < numExperts; e++) {
                    wGExps[e] = findTensorLazyOptional(model, prefix + "ffn_gate." + e + ".weight");
                    wUExps[e] = findTensorLazyOptional(model, prefix + "ffn_up." + e + ".weight");
                    wDExps[e] = findTensorLazyOptional(model, prefix + "ffn_down." + e + ".weight");
                }
            }

            layers.add(new TransformerLayerWeights(
                rms,
                packed,
                bqkv,
                wo,
                bo,
                ffnNorm,
                postAttnNorm,
                postFfnNorm,
                attnQNorm,
                attnKNorm,
                wG,
                bg,
                wU,
                bu,
                wD,
                bd,
                ffnGateInp,
                wGExps,
                wUExps,
                wDExps
            ));
        }

        System.out.println("Done.");
        return layers;
    }

    public static TensorData findTensorLazy(GGUFModel model, String name) {
        GGUFTensorInfo info = model.tensors().stream()
            .filter(t -> t.name().equals(name))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("Tensor not found: " + name));

        MemorySegment raw = model.segment().asSlice(model.dataStart() + info.offset(), info.sizeInBytes());
        long numElements = 1;
        for (long d : info.shape()) numElements *= d;
        return new TensorData(raw, info.typeId(), numElements);
    }

    public static TensorData findTensorLazyOptional(GGUFModel model, String name) {
        GGUFTensorInfo info = model.tensors().stream()
            .filter(t -> t.name().equals(name))
            .findFirst()
            .orElse(null);
        if (info == null) return null;

        MemorySegment raw = model.segment().asSlice(model.dataStart() + info.offset(), info.sizeInBytes());
        long numElements = 1;
        for (long d : info.shape()) numElements *= d;
        return new TensorData(raw, info.typeId(), numElements);
    }


    private static MemorySegment combineQKVBias(MemorySegment bq, MemorySegment bk, MemorySegment bv, int nHeads, int nHeadsKv, int headDim, Arena arena) {
        long qLen = (long) nHeads * headDim;
        long kLen = (long) nHeadsKv * headDim;
        long vLen = (long) nHeadsKv * headDim;
        
        MemorySegment combined = arena.allocate((qLen + kLen + vLen) * Float.BYTES, 64);
        
        if (bq != null) MemorySegment.copy(bq, 0, combined, 0, qLen * Float.BYTES);
        if (bk != null) MemorySegment.copy(bk, 0, combined, qLen * Float.BYTES, kLen * Float.BYTES);
        if (bv != null) MemorySegment.copy(bv, 0, combined, (qLen + kLen) * Float.BYTES, vLen * Float.BYTES);
        
        return combined;
    }


    private static MemorySegment findAndPrepareTensor(GGUFModel model, String name, Arena arena) {
        GGUFTensorInfo info = model.tensors().stream()
            .filter(t -> t.name().equals(name))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("Tensor not found: " + name));

        MemorySegment raw = model.segment().asSlice(model.dataStart() + info.offset(), info.sizeInBytes());
        
        if (info.typeId() == 0) { // F32
            return raw;
        }

        long numElements = 1;
        for (long d : info.shape()) numElements *= d;
        MemorySegment f32 = arena.allocate(numElements * Float.BYTES, 64);
        
        if (info.typeId() == 1) { // F16
            Dequantizer.dequantizeF16(raw, f32, numElements);
        } else if (info.typeId() == 8) { // Q8_0
            Dequantizer.dequantizeQ8_0(raw, f32, numElements);
        } else if (info.typeId() == 2) { // Q4_0
            Dequantizer.dequantizeQ4_0(raw, 0, f32, numElements);
        } else if (info.typeId() == 12) { // Q4_K
            Dequantizer.dequantizeQ4_K(raw, 0, f32, numElements);
        } else {
            throw new UnsupportedOperationException("Unsupported tensor type for loader: " + info.typeId() + " (" + info.name() + ")");
        }
        
        return f32;
    }

    private static MemorySegment findAndPrepareTensorOptional(GGUFModel model, String name, Arena arena) {
        GGUFTensorInfo info = model.tensors().stream()
            .filter(t -> t.name().equals(name))
            .findFirst()
            .orElse(null);

        if (info == null) return null;
        
        MemorySegment raw = model.segment().asSlice(model.dataStart() + info.offset(), info.sizeInBytes());
        if (info.typeId() == 0) return raw;

        long numElements = 1;
        for (long d : info.shape()) numElements *= d;
        MemorySegment f32 = arena.allocate(numElements * Float.BYTES, 64);

        if (info.typeId() == 1) {
            Dequantizer.dequantizeF16(raw, f32, numElements);
        } else if (info.typeId() == 8) {
            Dequantizer.dequantizeQ8_0(raw, f32, numElements);
        } else if (info.typeId() == 2) {
            Dequantizer.dequantizeQ4_0(raw, 0, f32, numElements);
        } else if (info.typeId() == 12) {
            Dequantizer.dequantizeQ4_K(raw, 0, f32, numElements);
        } else {
            throw new UnsupportedOperationException("Unsupported optional tensor type: " + info.typeId());
        }
        return f32;
    }

    private static int getMetadataInt(Map<String, Object> meta, String... keys) {
        for (String key : keys) {
            Object val = meta.get(key);
            if (val instanceof Number n) return n.intValue();
        }
        throw new IllegalArgumentException("Metadata keys missing or invalid: " + java.util.Arrays.toString(keys));
    }

    private static int getMetadataIntOptional(Map<String, Object> meta, Object... keysAndDefault) {
        for (int i = 0; i < keysAndDefault.length - 1; i++) {
            Object key = keysAndDefault[i];
            if (key instanceof String s) {
                Object val = meta.get(s);
                if (val instanceof Number n) return n.intValue();
            }
        }
        Object last = keysAndDefault[keysAndDefault.length - 1];
        if (last instanceof Number n) return n.intValue();
        return 0;
    }

    private static MemorySegment combineQKV(MemorySegment q, MemorySegment k, MemorySegment v, int hidden, int nHeads, int nHeadsKv, int headDim, Arena arena) {
        long qBytes = q.byteSize();
        long kBytes = k.byteSize();
        long vBytes = v.byteSize();
        
        MemorySegment combined = arena.allocate(qBytes + kBytes + vBytes, 64);

        
        MemorySegment.copy(q, 0, combined, 0, qBytes);
        MemorySegment.copy(k, 0, combined, qBytes, kBytes);
        MemorySegment.copy(v, 0, combined, qBytes + kBytes, vBytes);
        
        return combined;
    }
}
