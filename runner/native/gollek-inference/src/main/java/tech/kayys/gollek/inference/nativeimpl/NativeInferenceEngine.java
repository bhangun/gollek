package tech.kayys.gollek.inference.nativeimpl;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.TransformerLayerWeights;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;
import tech.kayys.gollek.gguf.loader.GGUFDequantizer;
import tech.kayys.gollek.gguf.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.ExecutorService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The high-level orchestrator for the native inference runtime.
 * Maintains the model weights and oversees the execution of the transformer layers.
 */
public final class NativeInferenceEngine implements AutoCloseable {
    private static final Logger LOG = LoggerFactory.getLogger(NativeInferenceEngine.class);

    private final GGUFModel model;
    private final Arena weightArena;
    private final List<TransformerLayerWeights> layers;
    private final int vocabSize;
    
    private final MemorySegment tokenEmbeddings;
    private final MemorySegment outputNorm;
    private final MemorySegment outputWeight;
    private final RoPECache ropeCache;

    private final boolean isNeox;

    private final int hidden;
    private final int nHeads;
    private final int nHeadsKv;
    private final int headDim;
    private final int nLayers;

    public NativeInferenceEngine(GGUFModel model) {
        this.model = model;
        this.weightArena = Arena.ofShared();
        this.layers = GGUFLoader.loadLayers(model, weightArena);
        
        String arch = (String) model.metadata().getOrDefault("general.architecture", "llama");
        
        this.hidden = getMetaInt(arch + ".embedding_length", "llama.embedding_length");
        this.nHeads = getMetaInt(arch + ".attention.head_count", "llama.attention.head_count");
        this.nHeadsKv = getMetaInt(arch + ".attention.head_count_kv", "llama.attention.head_count_kv", "general.attention.head_count_kv");
        this.headDim = hidden / nHeads;
        this.nLayers = layers.size();
        
        // Auto-detect RoPE style from GGUF metadata
        // In GGML: rope mode bit 1 (value 2) = Neox/split-half, mode 0 = LLaMA/interleaved
        this.isNeox = detectRopeStyle(arch, model.metadata());
        
        this.tokenEmbeddings = findAndPrepareTensor("token_embd.weight", arch + ".token_embd.weight");
        this.outputNorm = findAndPrepareTensor("output_norm.weight", arch + ".output_norm.weight");
        this.outputWeight = findAndPrepareTensor("output.weight", "lm_head.weight", arch + ".output.weight", "token_embd.weight");

        // Determine vocab size robustly
        int vs = 0;
        Object vsObj = model.metadata().get("general.vocab_size");
        if (vsObj == null) vsObj = model.metadata().get(arch + ".vocab_size");
        if (vsObj == null) vsObj = model.metadata().get("llama.vocab_size");
        if (vsObj instanceof Number n) {
            vs = n.intValue();
        } else {
            // Fallback: use outputWeight shape if possible (via GGUFModel tensors)
            vs = model.tensors().stream()
                    .filter(t -> t.name().equals("output.weight") || t.name().equals("lm_head.weight"))
                    .findFirst()
                    .map(t -> (int) t.shape()[0])
                    .orElse(32000); // Absolute fallback
        }
        this.vocabSize = vs;
        
        float defaultFreq = (arch.contains("qwen")) ? 1000000.0f : 10000.0f;
        float ropeFreqBase = getMetaFloat(defaultFreq, 
            arch + ".rope.freq_base", 
            "llama.rope.freq_base", 
            "general.rope.freq_base");
        
        this.eps = getMetaFloat(1e-5f, arch + ".attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon", "general.attention.layer_norm_rms_epsilon");
        
        LOG.debug("Engine Config: arch={}, eps={}, rope_freq_base={}, isNeox={}", arch, eps, ropeFreqBase, isNeox);
        this.ropeCache = RoPECache.build(weightArena, 4096, headDim, ropeFreqBase);
    }

    private final float eps;
    public float getEps() { return eps; }

    private MemorySegment findAndPrepareTensor(String... names) {
        GGUFTensorInfo info = null;
        for (String name : names) {
            info = model.tensors().stream().filter(t -> t.name().equals(name)).findFirst().orElse(null);
            if (info != null) break;
        }
        
        if (info == null) {
            throw new IllegalArgumentException("None of the tensors found in GGUF: " + java.util.Arrays.toString(names));
        }

        MemorySegment raw = model.segment().asSlice(model.dataStart() + info.offset(), info.sizeInBytes());
        
        if (info.typeId() == 0) { // F32
            return raw;
        }

        // Dequantize to F32
        long numElements = 1;
        for (long d : info.shape()) numElements *= d;
        MemorySegment f32 = weightArena.allocate(numElements * Float.BYTES, 64);
        
        if (info.typeId() == 1) { // F16
            GGUFDequantizer.dequantizeF16(raw, f32, numElements);
        } else if (info.typeId() == 8) { // Q8_0
            GGUFDequantizer.dequantizeQ8_0(raw, f32, numElements);
        } else {
            throw new UnsupportedOperationException("Unsupported tensor type for engine: " + info.typeId() + " (" + info.name() + ")");
        }
        
        return f32;
    }

    public void execute(
        MemorySegment inputIds, // [batch, seq]
        int seqLen,
        NativeInferenceSession session,
        ExecutorService executor
    ) {
        // 1. Embedding lookup
        // 2. Loop through FusedTransformerLayer.execute()
        // 3. Final norm and projection
        
        // This will be fleshed out as we implement NativeInferenceSession
    }

    public int getHidden() { return hidden; }
    public int getVocabSize() { return vocabSize; }
    public int getNHeads() { return nHeads; }
    public int getNHeadsKv() { return nHeadsKv; }
    public int getHeadDim() { return headDim; }
    public int getNLayers() { return nLayers; }
    public List<TransformerLayerWeights> getLayers() { return layers; }
    public MemorySegment getTokenEmbeddings() { return tokenEmbeddings; }
    public MemorySegment getOutputNorm() { return outputNorm; }
    public MemorySegment getOutputWeight() { return outputWeight; }
    public RoPECache getRopeCache() { return ropeCache; }
    public GGUFModel getModel() { return model; }
    public boolean isNeox() { return isNeox; }

    private int getMetaInt(String... keys) {
        for (String key : keys) {
            Object val = model.metadata().get(key);
            if (val instanceof Number n) return n.intValue();
        }
        throw new IllegalArgumentException("Metadata keys missing or invalid: " + java.util.Arrays.toString(keys));
    }

    private float getMetaFloat(float defaultVal, String... keys) {
        for (String key : keys) {
            Object val = model.metadata().get(key);
            if (val instanceof Number n) return n.floatValue();
        }
        return defaultVal;
    }

    /**
     * Detects RoPE style from GGUF metadata.
     * In GGML, rope_type/mode:
     *   0 = "normal" (Neox/split-half) - used by most models including Qwen2, LLaMA, etc.
     *   2 = GPT-NeoX explicit
     *   -1 = none
     * 
     * Note: llama.cpp's "normal" rope (type 0) is actually the Neox split-half style.
     * The interleaved (GPT-J) style is rarely used in GGUF files.
     */
    private boolean detectRopeStyle(String arch, java.util.Map<String, Object> metadata) {
        // Try reading explicit rope type from metadata
        for (String key : new String[]{arch + ".rope.type", "rope_type", arch + ".rope_type"}) {
            Object val = metadata.get(key);
            if (val instanceof Number n) {
                int ropeType = n.intValue();
                // In llama.cpp: type 0 = "normal" which is actually Neox/split-half
                // type 2 = explicit NeoX (also split-half)
                // type -1 = none
                boolean neox = (ropeType == 0 || ropeType == 2);
                LOG.debug("RoPE: type={} from key={} → isNeox={}", ropeType, key, neox);
                return neox;
            }
        }

        // Fallback: architecture-based heuristic
        // Most modern models (Qwen2, LLaMA 2/3, Mistral) use Neox-style in GGUF
        boolean neox = switch (arch) {
            case "qwen2", "qwen", "llama", "mistral", "gemma", "phi3" -> true;
            case "gptj" -> false; // GPT-J uses interleaved
            default -> true; // Default to Neox for modern architectures
        };
        LOG.debug("RoPE: no metadata, arch={} → isNeox={}", arch, neox);
        return neox;
    }

    private MemorySegment findTensor(String name) {
        return model.tensors().stream()
            .filter(t -> t.name().equals(name))
            .findFirst()
            .map(t -> model.segment().asSlice(model.dataStart() + t.offset(), t.sizeInBytes()))
            .orElseThrow(() -> new IllegalArgumentException("Tensor not found: " + name));
    }

    @Override
    public void close() {
        if (model != null) {
            model.close();
        }
        weightArena.close();
    }
}
