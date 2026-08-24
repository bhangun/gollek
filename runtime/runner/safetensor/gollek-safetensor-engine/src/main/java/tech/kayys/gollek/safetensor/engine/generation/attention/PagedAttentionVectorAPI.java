package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Paged Attention-2 implementation using Java scalar loops (fallback).
 * Traverses a block-mapped KV cache for memory-efficient inference.
 * Matches the layout: [numHeads, tokensPerBlock, headDim]
 */
public class PagedAttentionVectorAPI {
    /**
     * Compute attention using the Paged Attention algorithm.
     */
    public static AccelTensor compute(
            AccelTensor q,
            ModelConfig config,
            KVCacheManager.KVCacheSession kvSession,
            int layerIdx,
            int kvLayerIdx,
            int startPos,
            int numQHeads,
            int numKVHeads,
            int headDim,
            float scale,
            boolean causal,
            float softCap) {
        return compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numQHeads, numKVHeads, headDim, scale,
                causal, softCap, PagedAttentionVectorOptions.fromSystemProperties());
    }

    /**
     * Compute attention using the Paged Attention algorithm.
     */
    static AccelTensor compute(
            AccelTensor q,
            ModelConfig config,
            KVCacheManager.KVCacheSession kvSession,
            int layerIdx,
            int kvLayerIdx,
            int startPos,
            int numQHeads,
            int numKVHeads,
            int headDim,
            float scale,
            boolean causal,
            float softCap,
            PagedAttentionVectorOptions options) {
        return compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numQHeads, numKVHeads, headDim,
                scale, causal, softCap, options, null);
    }

    /**
     * Compute attention using the Paged Attention algorithm.
     */
    static AccelTensor compute(
            AccelTensor q,
            ModelConfig config,
            KVCacheManager.KVCacheSession kvSession,
            int layerIdx,
            int kvLayerIdx,
            int startPos,
            int numQHeads,
            int numKVHeads,
            int headDim,
            float scale,
            boolean causal,
            float softCap,
            PagedAttentionVectorOptions options,
            MemorySegment attentionContextBuffer) {

        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = startPos + (int) seqLenQ;
        int tokensPerBlock = kvSession.tokensPerBlock();
        List<Integer> blockTable = kvSession.getBlockIndices(kvLayerIdx);
        BlockManager blockManager = kvSession.blockManager();
        PagedKvCacheLayout layout = PagedKvCacheLayout.source(blockManager, numKVHeads, headDim, tokensPerBlock);
        BlockManager.KvStorageType storageType = blockManager.getStorageType();
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        int slidingWindow = slidingLayer ? config.slidingWindowSize() : Integer.MAX_VALUE;
        boolean debugProbe = options != null && options.debugAttentionProbe();

        AccelTensor out = FlashAttentionContextOutputBuffer.viewOrAllocate(attentionContextBuffer, q);
        boolean success = false;
        try {
            AttentionOnlineSoftmax softmax = new AttentionOnlineSoftmax(new float[headDim], headDim);
            PagedAttentionVectorContext context = PagedAttentionVectorContext.create(
                    q,
                    out,
                    blockTable,
                    blockManager,
                    layout,
                    storageType,
                    tokensPerBlock,
                    totalTokens,
                    numKVHeads,
                    headDim,
                    scale,
                    causal,
                    softCap,
                    slidingWindow,
                    debugProbe,
                    layerIdx);

            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < numQHeads; h++) {
                    for (int i = 0; i < seqLenQ; i++) {
                        computePagedHeadQuery(context, context.queryAt(b, h, i), softmax);
                    }
                }
            }
            success = true;
            return out;
        } finally {
            if (!success && !out.isClosed()) {
                out.close();
            }
        }
    }

    private static void computePagedHeadQuery(PagedAttentionVectorContext context, PagedAttentionVectorQuery query,
            AttentionOnlineSoftmax softmax) {
        softmax.reset();

        // Softmax Online Loop over blocks
        int currentTokCount = 0;
        for (int blockIdx : context.blockTable()) {
            MemorySegment kBlock = context.blockManager().getKBlock(blockIdx);
            MemorySegment vBlock = context.blockManager().getVBlock(blockIdx);
            MemorySegment kScaleBlock = context.blockManager().getKScaleBlock(blockIdx);
            MemorySegment vScaleBlock = context.blockManager().getVScaleBlock(blockIdx);
            int tokensInThisBlock = Math.min(context.tokensPerBlock(), context.totalTokens() - currentTokCount);

            for (int tok = 0; tok < tokensInThisBlock; tok++) {
                int absPos = currentTokCount + tok;
                if (absPos < query.minPosition()) {
                    continue;
                }
                if (context.causal() && absPos > query.absolutePosition())
                    break;

                // 1. Compute Dot Product Q_i @ K_j
                long kvElementOff = context.layout().sourceElement(query.kvHeadIndex(), tok);
                long kvOff = context.layout().sourceByteOffset(query.kvHeadIndex(), tok);
                long scaleIndex = context.layout().scaleIndex(query.kvHeadIndex(), tok);
                float score = PagedAttentionVectorMath.score(context.storageType(), context.querySegment(),
                        query.queryByteOffset(), kBlock, kvElementOff, kvOff, kScaleBlock, scaleIndex,
                        context.headDim(), context.scale());
                
                if (context.softCap() > 0.0f) {
                    score = (float) (Math.tanh(score / context.softCap()) * context.softCap());
                }
                if (query.debugProbe() && absPos < 12) {
                    System.err.printf("[DEBUG-ATTN] layer=0 head=0 query=%d key=%d score=%f%n",
                            query.absolutePosition(), absPos, score);
                }

                // 2. Online Softmax Update
                softmax.observe(score);

                // 3. Update Output Accumulator
                PagedAttentionVectorMath.updateAccumulator(context.storageType(), softmax.accumulator(), vBlock,
                        kvElementOff, kvOff, vScaleBlock, scaleIndex, softmax.previousWeight(),
                        softmax.currentWeight(), context.headDim());
            }

            currentTokCount += tokensInThisBlock;
            if (currentTokCount >= context.totalTokens())
                break;
        }

        // 4. Final Normalization
        PagedAttentionVectorMath.writeNormalizedAccumulator(context.outputSegment(), query.outputElementIndex(),
                softmax.accumulator(), softmax.inverseNormalizer(), context.headDim());
    }
}
