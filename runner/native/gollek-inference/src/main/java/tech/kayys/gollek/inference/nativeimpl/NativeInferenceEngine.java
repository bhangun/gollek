package tech.kayys.gollek.inference.nativeimpl;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.spi.tensor.weights.TransformerLayerWeights;
import tech.kayys.gollek.spi.tensor.weights.TensorData;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;
import tech.kayys.gollek.spi.tensor.weights.Dequantizer;
import tech.kayys.gollek.gguf.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.ExecutorService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The high-level orchestrator for the native inference runtime.
 */
public final class NativeInferenceEngine implements AutoCloseable {
    private static final Logger LOG = LoggerFactory.getLogger(NativeInferenceEngine.class);

    private final GGUFModel model;
    private final Arena weightArena;
    private final List<TransformerLayerWeights> layers;
    private final int vocabSize;
    
    private final TensorData tokenEmbeddings;
    private final MemorySegment outputNorm;
    private final TensorData outputWeight;
    private final RoPECache ropeCache;

    private final boolean isNeox;

    private final int hidden;
    private final int nHeads;
    private final int nHeadsKv;
    private final int headDim;
    private final int nLayers;

    private final float attnSoftCap;
    private final float finalSoftCap;
    private final float eps;
    private final int numExperts;
    private final int numExpertsPerTok;
    private final tech.kayys.gollek.spi.model.ModelArchitecture architecture;
    private final InferenceMetrics metrics;

    public NativeInferenceEngine(GGUFModel model, tech.kayys.gollek.spi.model.ModelArchitecture architecture) {
        this.model = model;
        this.architecture = architecture;
        this.weightArena = Arena.ofAuto();
        this.layers = GGUFLoader.loadLayers(model, weightArena);
        
        String arch = (String) model.metadata().getOrDefault("general.architecture", "llama");
        System.out.println("Engine Config: arch=" + arch + ", resolved=" + architecture.id());

        this.hidden = getMetaInt(arch + ".embedding_length", "llama.embedding_length");
        this.nHeads = getMetaInt(arch + ".attention.head_count", "llama.attention.head_count");
        this.nHeadsKv = getMetaInt(arch + ".attention.head_count_kv", "llama.attention.head_count_kv", "general.attention.head_count_kv");
        
        System.out.println("Detecting headDim...");
        int hd = getMetaIntOptional(arch + ".attention.head_dim", 
                                  "gemma2.attention.head_dim", 
                                  arch + ".attention.key_length",
                                  "llama.attention.head_dim",
                                  "head_dim",
                                  "key_length");
        if (hd == 0) {
            try {
                GGUFTensorInfo qInfo = model.tensors().stream()
                    .filter(t -> t.name().equals("blk.0.attn_q.weight"))
                    .findFirst().orElse(null);
                if (qInfo != null && qInfo.shape().length >= 2) {
                    hd = (int) (qInfo.shape()[1] / nHeads);
                }
            } catch (Exception e) {
                hd = hidden / nHeads;
            }
        }
        if (hd == 0) hd = hidden / nHeads;
        this.headDim = hd;

        this.nLayers = layers.size();
        
        System.out.println("Detecting RoPE style...");
        this.isNeox = architecture.usesNeoxRope();
        
        this.tokenEmbeddings = findTensorLazy("token_embd.weight", arch + ".token_embd.weight");
        System.out.println("tokenEmbeddings type=" + this.tokenEmbeddings.typeId() + " elements=" + this.tokenEmbeddings.numElements());
        this.outputNorm = findAndPrepareTensor("output_norm.weight", arch + ".output_norm.weight");
        this.outputWeight = findTensorLazy("output.weight", "lm_head.weight", arch + ".output.weight", "token_embd.weight");
        System.out.println("outputWeight type=" + this.outputWeight.typeId() + " elements=" + this.outputWeight.numElements());

        System.out.println("Determining vocab size...");
        int vs = 0;
        Object vsObj = model.metadata().get("general.vocab_size");
        if (vsObj == null) vsObj = model.metadata().get(arch + ".vocab_size");
        if (vsObj == null) vsObj = model.metadata().get("llama.vocab_size");
        if (vsObj instanceof Number n) {
            vs = n.intValue();
        } else {
            vs = model.tensors().stream()
                    .filter(t -> t.name().equals("output.weight") || t.name().equals("lm_head.weight"))
                    .findFirst()
                    .map(t -> (int) t.shape()[0])
                    .orElse(32000);
        }
        this.vocabSize = vs;
        
        float defaultFreq = architecture.defaultRopeFreqBase();
        float ropeFreqBase = getMetaFloat(defaultFreq, 
            arch + ".rope.freq_base", 
            "llama.rope.freq_base", 
            "general.rope.freq_base");
        
        this.eps = getMetaFloat((float) architecture.rmsNormEps(), arch + ".attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon", "general.attention.layer_norm_rms_epsilon");
        
        System.out.println("Building RoPE cache...");
        this.ropeCache = RoPECache.build(weightArena, 4096, headDim, ropeFreqBase);

        // Soft-capping detection
        float aCap = getMetaFloat(architecture.defaultAttnSoftCap(), arch + ".attention.logit_softcapping", "gemma2.attention.logit_softcapping");
        float fCap = getMetaFloat(architecture.defaultFinalSoftCap(), arch + ".logit_softcapping", "gemma2.logit_softcapping");
        
        this.attnSoftCap = aCap;
        this.finalSoftCap = fCap;
        
        if (attnSoftCap > 0 || finalSoftCap > 0) {
            LOG.info("Soft-capping enabled: attn={}, final={}", attnSoftCap, finalSoftCap);
        }
        
        this.numExperts = getMetaInt(arch + ".expert_count", "general.expert_count");
        this.numExpertsPerTok = getMetaInt(arch + ".expert_used_count", "general.expert_used_count");
        
        if (numExperts > 0) {
            LOG.info("MoE initialized: {} experts, {} per token", numExperts, numExpertsPerTok);
        }
        this.metrics = new InferenceMetrics(numExperts);
    }

    public GGUFModel getModel() { return model; }
    public List<TransformerLayerWeights> getLayers() { return layers; }
    public int getVocabSize() { return vocabSize; }
    public TensorData getTokenEmbeddings() { return tokenEmbeddings; }
    public MemorySegment getOutputNorm() { return outputNorm; }
    public TensorData getOutputWeight() { return outputWeight; }
    public RoPECache getRopeCache() { return ropeCache; }
    public int getHidden() { return hidden; }
    public int getNHeads() { return nHeads; }
    public int getNHeadsKv() { return nHeadsKv; }
    public int getHeadDim() { return headDim; }
    public int getNLayers() { return nLayers; }
    public boolean isNeox() { return isNeox; }
    public tech.kayys.gollek.spi.model.ModelArchitecture getArchitecture() { return architecture; }
    public float getAttnSoftCap() { return attnSoftCap; }
    public float getFinalSoftCap() { return finalSoftCap; }
    public float getEps() { return eps; }
    public int getNumExperts() { return numExperts; }
    public int getNumExpertsPerTok() { return numExpertsPerTok; }
    public InferenceMetrics getMetrics() { return metrics; }

    public int getFfnDim() {
        if (layers.isEmpty()) return 0;
        return (int) (layers.get(0).wG.numElements() / hidden);
    }

    private TensorData findTensorLazy(String... names) {
        GGUFTensorInfo info = null;
        for (String name : names) {
            info = model.tensors().stream().filter(t -> t.name().equals(name)).findFirst().orElse(null);
            if (info != null) break;
        }
        
        if (info == null) {
            throw new IllegalArgumentException("None of the tensors found in GGUF: " + java.util.Arrays.toString(names));
        }

        MemorySegment raw = model.segment().asSlice(model.dataStart() + info.offset(), info.sizeInBytes());
        long numElements = 1;
        for (long d : info.shape()) numElements *= d;
        return new TensorData(raw, info.typeId(), numElements);
    }

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

        long numElements = 1;
        for (long d : info.shape()) numElements *= d;
        
        MemorySegment f32 = weightArena.allocate((long) numElements * Float.BYTES, 64);
        if (info.typeId() == 1) { // F16
            Dequantizer.dequantizeF16(raw, f32, numElements);
        } else if (info.typeId() == 8) { // Q8_0
            Dequantizer.dequantizeQ8_0(raw, f32, numElements);
        } else if (info.typeId() == 2) { // Q4_0
            Dequantizer.dequantizeQ4_0(raw, 0, f32, numElements);
        } else if (info.typeId() == 12) { // Q4_K
            Dequantizer.dequantizeQ4_K(raw, 0, f32, numElements);
        } else {
            throw new UnsupportedOperationException("Unsupported tensor type: " + info.typeId());
        }
        return f32;
    }

    private int getMetaInt(String... keys) {
        for (String k : keys) {
            Object v = model.metadata().get(k);
            if (v instanceof Number n) return n.intValue();
        }
        return 0;
    }

    private int getMetaIntOptional(String... keys) {
        return getMetaInt(keys);
    }

    private float getMetaFloat(float defaultValue, String... keys) {
        for (String k : keys) {
            Object v = model.metadata().get(k);
            if (v instanceof Number n) return n.floatValue();
        }
        return defaultValue;
    }


    @Override
    public void close() {
        weightArena.close();
    }
}
