/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.kv;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;
import java.util.Map;
import java.util.List;
import jakarta.enterprise.context.ApplicationScoped;

/**
 * FFM-backed KV (key-value) cache manager using {@link AccelTensor}.
 *
 * <p>
 * No LibTorch dependency — pure MemorySegment storage.
 */
@ApplicationScoped
public class KVCacheManager {

    @jakarta.inject.Inject
    BlockManager globalBlockManager;

    public KVCacheSession createSession(int maxSeqLen) {
        return new KVCacheSession(maxSeqLen, globalBlockManager);
    }

    public static class KVCacheSession implements AutoCloseable {
        private final int maxSeqLen;
        private final BlockManager blockManager;

        // Layer -> List of physical block indices
        private final Map<Integer, List<Integer>> blockTables = new java.util.concurrent.ConcurrentHashMap<>();

        private int currentPos = 0;
        private int tokensPerBlock = 16;
        private int numLayers;
        private int numKVHeads;
        private int headDim;

        public KVCacheSession(int maxSeqLen, BlockManager blockManager) {
            this.maxSeqLen = maxSeqLen;
            this.blockManager = blockManager;
        }

        public void allocate(ModelConfig config, tech.kayys.gollek.safetensor.generation.GenerationConfig genConfig) {
            this.numLayers = config.numHiddenLayers();
            this.numKVHeads = config.resolvedNumKvHeads();
            this.headDim = config.resolvedMaxHeadDim();
            this.tokensPerBlock = 16; // Default block size

            // Select layout based on quantization preference
            java.lang.foreign.ValueLayout layout = java.lang.foreign.ValueLayout.JAVA_FLOAT;
            if (genConfig
                    .kvCacheQuant() == tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.INT8) {
                layout = java.lang.foreign.ValueLayout.JAVA_BYTE;
            } else if (genConfig
                    .kvCacheQuant() == tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.TURBO) {
                // TurboQuant uses complex packing, for now we treat as byte and handle logic in
                // kernels
                layout = java.lang.foreign.ValueLayout.JAVA_BYTE;
            }

            // Ensure BlockManager is initialized for this model's dimensions
            blockManager.initialize(tokensPerBlock, numKVHeads, headDim, (maxSeqLen / tokensPerBlock) * numLayers + 100);

            for (int i = 0; i < numLayers; i++) {
                blockTables.put(i, new java.util.ArrayList<>());
                // Allocate first block
                appendBlock(i);
            }
        }

        private void appendBlock(int layerIdx) {
            int block = blockManager.allocateBlock();
            if (block != -1) {
                blockTables.get(layerIdx).add(block);
            } else {
                throw new RuntimeException("KVCache: Out of physical blocks!");
            }
        }

        public List<Integer> getBlockTable(int layerIdx) {
            return blockTables.get(layerIdx);
        }

        /**
         * Proactively ensure that enough blocks are allocated for the given token
         * count.
         * Call this before prefill or decode loops.
         */
        public void ensureCapacity(int totalTokensNeeded) {
            int requiredBlocks = (totalTokensNeeded + tokensPerBlock - 1) / tokensPerBlock;
            if (totalTokensNeeded <= 0)
                return;

            for (int i = 0; i < numLayers; i++) {
                List<Integer> table = blockTables.get(i);
                while (table.size() < requiredBlocks) {
                    appendBlock(i);
                }
            }
        }

        public void advance(int seqLen) {
            int prevBlockCount = (currentPos + tokensPerBlock - 1) / tokensPerBlock;
            if (currentPos == 0)
                prevBlockCount = 0;

            this.currentPos += seqLen;

            int newBlockCount = (currentPos + tokensPerBlock - 1) / tokensPerBlock;

            // Allocate new blocks if we crossed a boundary
            if (newBlockCount > prevBlockCount) {
                for (int i = 0; i < numLayers; i++) {
                    for (int b = prevBlockCount; b < newBlockCount; b++) {
                        if (b >= blockTables.get(i).size()) {
                            appendBlock(i);
                        }
                    }
                }
            }
        }

        public int currentPos() {
            return currentPos;
        }

        public int tokensPerBlock() {
            return tokensPerBlock;
        }

        public int headDim() {
            return headDim;
        }

        public int numKVHeads() {
            return numKVHeads;
        }

        public java.lang.foreign.MemorySegment getRawKPool() {
            return blockManager.getRawKPool();
        }

        public java.lang.foreign.MemorySegment getRawVPool() {
            return blockManager.getRawVPool();
        }

        public int getBlockForToken(int layerIdx, int tokenIdx) {
            List<Integer> table = blockTables.get(layerIdx);
            int blockIdxInTable = tokenIdx / tokensPerBlock;
            if (blockIdxInTable < table.size()) {
                return table.get(blockIdxInTable);
            }
            return -1;
        }

        public List<Integer> getBlockIndices(int layerIdx) {
            return blockTables.get(layerIdx);
        }

        public BlockManager blockManager() {
            return blockManager;
        }

        @Override
        public void close() {
            blockTables.values().forEach(list -> {
                list.forEach(blockManager::freeBlock);
            });
            blockTables.clear();
        }

        // Legacy compatibility - will be replaced by PagedAttention
        public AccelTensor keyCache(int layerIdx) {
            throw new UnsupportedOperationException("KVCache is now paged. Use getBlockTable().");
        }

        public AccelTensor valueCache(int layerIdx) {
            throw new UnsupportedOperationException("KVCache is now paged. Use getBlockTable().");
        }
    }
}
