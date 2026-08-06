/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.kv;

import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
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
        return new KVCacheSession(maxSeqLen, new BlockManager());
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
        private ForwardWorkspace workspace;
        private final Map<Integer, SharedKvState> sharedKvStates =
                new java.util.concurrent.ConcurrentHashMap<>();
        private tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization kvQuantization =
                tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.NONE;

        public KVCacheSession(int maxSeqLen, BlockManager blockManager) {
            this.maxSeqLen = maxSeqLen;
            this.blockManager = blockManager;
        }

        public void allocate(ModelConfig config, tech.kayys.gollek.safetensor.generation.GenerationConfig genConfig) {
            this.numLayers = config.getNumHiddenLayers();
            this.numKVHeads = config.getResolvedMaxKvHeads();
            this.headDim = config.getResolvedMaxHeadDim();
            this.tokensPerBlock = 16; // Default block size
            this.kvQuantization = genConfig.kvCacheQuant();

            BlockManager.KvStorageType storageType =
                    switch (kvQuantization) {
                        case INT8 -> BlockManager.KvStorageType.INT8;
                        case INT4, TURBO -> BlockManager.KvStorageType.INT4;
                        case NONE -> BlockManager.KvStorageType.FP32;
                    };

            // Gemma-4 128K context requires massive memory. We allocate a safe initial pool 
            // and cap it to prevent Metal driver aborts on huge contiguous requests.
            int safeMaxTokens = Math.min(maxSeqLen, 4096); 
            int initialBlocks = (safeMaxTokens / tokensPerBlock) * numLayers + 64;
            
            blockManager.initialize(tokensPerBlock, numKVHeads, headDim, initialBlocks, storageType);

            for (int i = 0; i < numLayers; i++) {
                blockTables.put(i, new java.util.ArrayList<>());
                // Allocate first block
                appendBlock(i);
            }
            
            this.workspace = new ForwardWorkspace();
            this.workspace.ensureCapacity(config.getHiddenSize(), config.getHiddenSize(), config.getIntermediateSize());
        }

        public ForwardWorkspace getWorkspace() {
            return workspace;
        }

        private void appendBlock(int layerIdx) {
            Integer block = blockManager.allocateBlock();
            if (block != null) {
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
            if (table == null || tokenIdx < 0) {
                return -1;
            }
            int blockIdxInTable = tokenIdx / tokensPerBlock;
            while (blockIdxInTable >= table.size()) {
                appendBlock(layerIdx);
            }
            return table.get(blockIdxInTable);
        }

        public List<Integer> getBlockIndices(int layerIdx) {
            return blockTables.get(layerIdx);
        }

        public BlockManager blockManager() {
            return blockManager;
        }

        public Map<Integer, SharedKvState> sharedKvStates() {
            return sharedKvStates;
        }

        public void clearSharedKvStates() {
            for (SharedKvState state : sharedKvStates.values()) {
                if (state != null) {
                    state.close();
                }
            }
            sharedKvStates.clear();
        }

        public tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization kvQuantization() {
            return kvQuantization;
        }

        public boolean isQuantizedInt8() {
            return kvQuantization == tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.INT8;
        }

        public boolean isQuantizedInt4() {
            return kvQuantization == tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.INT4
                    || kvQuantization == tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.TURBO;
        }

        public boolean isQuantized() {
            return kvQuantization != tech.kayys.gollek.safetensor.generation.GenerationConfig.KvCacheQuantization.NONE;
        }

        @Override
        public void close() {
            if (workspace != null) workspace.close();
            clearSharedKvStates();
            blockTables.values().forEach(list -> {
                list.forEach(blockManager::freeBlock);
            });
            blockTables.clear();
            blockManager.close();
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
