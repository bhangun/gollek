package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import tech.kayys.gollek.spi.tensor.weights.TransformerLayerWeights;
import tech.kayys.gollek.spi.tensor.weights.TensorData;
import tech.kayys.gollek.spi.tensor.weights.Dequantizer;

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
        this.arena = Arena.ofAuto();
        
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
        int ffnDim = engine.getFfnDim();
        int numExperts = engine.getNumExperts();
        for (int i = 0; i < engine.getNLayers(); i++) {
            layerBuffers.add(new LayerBuffers(arena, engine.getHidden(), engine.getNHeadsKv(), engine.getHeadDim(), ffnDim, numExperts));
        }
        
        this.x = arena.allocate((long) engine.getHidden() * Float.BYTES, 64);
        this.xNext = arena.allocate((long) engine.getHidden() * Float.BYTES, 64);
    }

    public MemorySegment tick(int tokenId, ExecutorService executor) {
        lookupEmbedding(tokenId, x);
        
        for (int i = 0; i < engine.getNLayers(); i++) {
            TransformerLayerWeights weights = engine.getLayers().get(i);
            LayerBuffers buffers = layerBuffers.get(i);
            
            FusedTransformerLayer.execute(
                i, pos, x, xNext, weights, buffers, kvCache, 
                engine.getRopeCache(), engine.getHidden(), engine.getNHeads(),
                engine.getNHeadsKv(), engine.getHeadDim(), engine.isNeox(),
                engine.getEps(), engine.getAttnSoftCap(),
                engine.getArchitecture().addOneToRmsNormWeight(),
                engine.getArchitecture().activationType(),
                engine.getNumExperts(), engine.getNumExpertsPerTok(),
                engine.getMetrics(),
                executor
            );
            
            MemorySegment.copy(xNext, 0, x, 0, x.byteSize());
        }
        
        RMSNormKernel.execute(x, xNext, engine.getOutputNorm(), engine.getHidden(), engine.getEps(), engine.getArchitecture().addOneToRmsNormWeight());
        pos++;
        return xNext;
    }

    public void setPos(int pos) { this.pos = pos; }
    public Arena getArena() { return arena; }
    public KVCache getKvCache() { return kvCache; }
    public List<LayerBuffers> getLayerBuffers() { return layerBuffers; }
    public MemorySegment getX() { return x; }

    public void lookupEmbedding(int tokenId, MemorySegment dst) {
        TensorData embd = engine.getTokenEmbeddings();
        if (embd.isQ8_0()) {
            long blocksPerRow = engine.getHidden() / 32;
            long bytesPerRow = blocksPerRow * 34;
            long offset = (long) tokenId * bytesPerRow;
            Dequantizer.dequantizeQ8_0(embd.segment(), offset, dst, engine.getHidden());
        } else if (embd.isF16()) {
            long offset = (long) tokenId * engine.getHidden() * 2L;
            Dequantizer.dequantizeF16(embd.segment(), offset, dst, engine.getHidden());
        } else {
            MemorySegment.copy(embd.segment(), (long) tokenId * engine.getHidden() * 4L, dst, 0, (long) engine.getHidden() * 4L);
        }

        float scale = engine.getArchitecture().embeddingScaleFactor(engine.getHidden());
        if (scale != 1.0f) {
            for (int i = 0; i < engine.getHidden(); i++) {
                float v = dst.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
                dst.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, v * scale);
            }
        }
    }


    @Override
    public void close() {
        arena.close();
    }
}
