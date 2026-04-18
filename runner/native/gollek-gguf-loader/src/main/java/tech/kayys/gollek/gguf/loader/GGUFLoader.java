package tech.kayys.gollek.gguf.loader;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;
import tech.kayys.gollek.gguf.loader.GGUFReader;
import tech.kayys.gollek.gguf.loader.GGUFParser;

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
        Arena arena = Arena.ofShared();
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

        int headDim = hidden / nHeads;
        
        System.out.println("Loading " + nLayer + " layers: arch=" + arch + ", hidden=" + hidden + ", nHeads=" + nHeads + ", nHeadsKv=" + nHeadsKv);

        List<TransformerLayerWeights> layers = new ArrayList<>(nLayer);

        for (int i = 0; i < nLayer; i++) {
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

            MemorySegment wo = findAndPrepareTensor(model, prefix + "attn_output.weight", arena);
            MemorySegment bo = findAndPrepareTensorOptional(model, prefix + "attn_output.bias", arena);

            MemorySegment ffnNorm = findAndPrepareTensor(model, prefix + "ffn_norm.weight", arena);
            
            MemorySegment wG = findAndPrepareTensor(model, prefix + "ffn_gate.weight", arena);
            MemorySegment bg = findAndPrepareTensorOptional(model, prefix + "ffn_gate.bias", arena);
            
            MemorySegment wU = findAndPrepareTensor(model, prefix + "ffn_up.weight", arena);
            MemorySegment bu = findAndPrepareTensorOptional(model, prefix + "ffn_up.bias", arena);
            
            MemorySegment wD = findAndPrepareTensor(model, prefix + "ffn_down.weight", arena);
            MemorySegment bd = findAndPrepareTensorOptional(model, prefix + "ffn_down.bias", arena);

            layers.add(new TransformerLayerWeights(
                rms,
                packed,
                bqkv,
                wo,
                bo,
                ffnNorm,
                wG,
                bg,
                wU,
                bu,
                wD,
                bd
            ));
        }

        return layers;
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
            GGUFDequantizer.dequantizeF16(raw, f32, numElements);
        } else if (info.typeId() == 8) { // Q8_0
            GGUFDequantizer.dequantizeQ8_0(raw, f32, numElements);
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
            GGUFDequantizer.dequantizeF16(raw, f32, numElements);
        } else if (info.typeId() == 8) {
            GGUFDequantizer.dequantizeQ8_0(raw, f32, numElements);
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
        long qBytes = (long) hidden * nHeads * headDim * Float.BYTES;
        long kBytes = (long) hidden * nHeadsKv * headDim * Float.BYTES;
        long vBytes = (long) hidden * nHeadsKv * headDim * Float.BYTES;
        
        MemorySegment combined = arena.allocate(qBytes + kBytes + vBytes, 64);
        
        MemorySegment.copy(q, 0, combined, 0, qBytes);
        MemorySegment.copy(k, 0, combined, qBytes, kBytes);
        MemorySegment.copy(v, 0, combined, qBytes + kBytes, vBytes);
        
        return combined;
    }
}
