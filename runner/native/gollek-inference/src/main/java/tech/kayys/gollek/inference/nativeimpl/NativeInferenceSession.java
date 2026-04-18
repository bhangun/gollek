package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import tech.kayys.gollek.gguf.loader.TransformerLayerWeights;

/**
 * Manages request-scoped state for the native inference runtime.
 * This includes the KV cache, temporal buffers, and current sequence position.
 */
public final class NativeInferenceSession implements AutoCloseable {

    private final NativeInferenceEngine engine;
    private final Arena arena;
    private final KVCache kvCache;
    private final List<LayerBuffers> layerBuffers;
    
    // Residual and normalization buffers resident in the session to avoid re-allocation
    private final MemorySegment x;     // Main residual stream [hidden]
    private final MemorySegment xNext; // Next state residual stream [hidden]

    private int pos = 0;

    public NativeInferenceSession(NativeInferenceEngine engine, int maxContext) {
        this.engine = engine;
        this.arena = Arena.ofShared();
        
        long kvBytes = (long) engine.getNLayers() * engine.getNHeadsKv() * engine.getHeadDim() * maxContext * Float.BYTES;
        MemorySegment kSeg = arena.allocate(kvBytes, 64);
        MemorySegment vSeg = arena.allocate(kvBytes, 64);
        
        this.kvCache = new KVCache(
            engine.getNLayers(),
            engine.getNHeads(),
            engine.getNHeadsKv(),
            engine.getHeadDim(),
            maxContext,
            kSeg,
            vSeg
        );
        
        this.layerBuffers = new ArrayList<>();
        String arch = (String) engine.getModel().metadata().getOrDefault("general.architecture", "llama");
        int ffnDim = ((Number) engine.getModel().metadata().getOrDefault(arch + ".feed_forward_length", 
                    engine.getModel().metadata().get("llama.feed_forward_length"))).intValue();
        for (int i = 0; i < engine.getNLayers(); i++) {
            layerBuffers.add(new LayerBuffers(arena, engine.getHidden(), engine.getNHeadsKv(), engine.getHeadDim(), ffnDim));
        }
        
        this.x = arena.allocate((long) engine.getHidden() * Float.BYTES, 64);
        this.xNext = arena.allocate((long) engine.getHidden() * Float.BYTES, 64);
    }

    /**
     * Executes a single forward pass for a single token.
     */
    public MemorySegment tick(int tokenId, ExecutorService executor) {
        // 1. Token Embedding Lookup
        lookupEmbedding(tokenId, x);
        
        // 2. Transformer Layers
        for (int i = 0; i < engine.getNLayers(); i++) {
            TransformerLayerWeights weights = engine.getLayers().get(i);
            LayerBuffers buffers = layerBuffers.get(i);
            
            FusedTransformerLayer.execute(
                i, 
                pos, 
                x, 
                xNext, 
                weights, 
                buffers, 
                kvCache, 
                engine.getRopeCache(),
                engine.getHidden(),
                engine.getNHeads(),
                engine.getNHeadsKv(),
                engine.getHeadDim(),
                engine.isNeox(),
                engine.getEps(),
                executor
            );
            
            // Swap buffers for residual stream (or copy back if execute doesn't swap)
            MemorySegment.copy(xNext, 0, x, 0, x.byteSize());
        }
        
        // 3. Final Norm and Logit Projection
        RMSNormKernel.execute(x, xNext, engine.getOutputNorm(), engine.getHidden(), engine.getEps());
        
        // For simplified prototype: we return the normalized output vector.
        // The provider will handle the sampling and logit projection.
        pos++;
        return xNext;
    }

    private void lookupEmbedding(int tokenId, MemorySegment dst) {
        long offset = (long) tokenId * engine.getHidden() * Float.BYTES;
        MemorySegment.copy(engine.getTokenEmbeddings(), offset, dst, 0, dst.byteSize());
    }

    public int getPos() { return pos; }

    @Override
    public void close() {
        arena.close();
    }
}
