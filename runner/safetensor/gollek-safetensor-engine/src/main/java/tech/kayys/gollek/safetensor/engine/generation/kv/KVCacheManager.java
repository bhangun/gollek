/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.kv;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModalityType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * FFM-backed KV (key-value) cache manager for attention layers.
 *
 * <p>Allocates off-heap {@link MemorySegment} buffers for each transformer layer's
 * key and value tensors, enabling zero-copy updates during the attention forward pass.
 * Each inference request gets its own {@link KVCacheSession} which must be closed
 * after the request completes to release native memory.
 *
 * @see KVCacheSession
 */
@ApplicationScoped
public class KVCacheManager {

    /**
     * Creates a new KV cache session for a sequence of up to {@code maxSeqLen} tokens.
     *
     * @param maxSeqLen maximum number of tokens this session will hold
     * @return a new, unallocated {@link KVCacheSession}
     */
    public KVCacheSession createSession(int maxSeqLen) {
        return new KVCacheSession(maxSeqLen);
    }

    /**
     * Per-request KV cache holding pre-allocated off-heap memory for all transformer layers.
     *
     * <p>Call {@link #allocate(ModelConfig)} once after creation to reserve memory,
     * then use {@link #keyCache(int)} / {@link #valueCache(int)} during the forward pass.
     * Always close the session when the request is complete to release native memory.
     */
    public static class KVCacheSession implements AutoCloseable {
        private final int maxSeqLen;
        private final Arena arena = Arena.ofShared();
        private final Map<Integer, MemorySegment> keyCaches = new ConcurrentHashMap<>();
        private final Map<Integer, MemorySegment> valueCaches = new ConcurrentHashMap<>();
        private int currentPos = 0;
        private final Map<ModalityType, Integer> modalityOffsets = new EnumMap<>(ModalityType.class);

        /**
         * Creates a session with the given maximum sequence length.
         *
         * @param maxSeqLen maximum token positions this session can hold
         */
        public KVCacheSession(int maxSeqLen) {
            this.maxSeqLen = maxSeqLen;
        }

        /**
         * Pre-allocates off-heap memory for all layers based on the model configuration.
         *
         * <p>Must be called before any {@link #keyCache} / {@link #valueCache} access.
         * Memory is sized as {@code maxSeqLen × numKVHeads × headDim × 4} bytes per layer.
         *
         * @param config model configuration providing layer count, KV head count, and head dim
         */
        public void allocate(ModelConfig config) {
            allocate(config, null, null);
        }

        /**
         * Pre-allocates off-heap memory for all layers using per-layer head dims
         * inferred from K/V projection weights when available.
         *
         * @param config  model configuration
         * @param weights model weights (optional)
         * @param arch    model architecture (optional)
         */
        public void allocate(ModelConfig config, Map<String, TorchTensor> weights, ModelArchitecture arch) {
            int numLayers = config.numHiddenLayers();
            int numKVHeads = config.resolvedNumKvHeads();
            int fallbackHeadDim = config.resolvedHeadDim();

            for (int i = 0; i < numLayers; i++) {
                int headDim = fallbackHeadDim;
                if (weights != null && arch != null) {
                    TorchTensor kWeight = weights.get(arch.layerKeyWeight(i));
                    if (kWeight == null) {
                        kWeight = weights.get(arch.layerValueWeight(i));
                    }
                    headDim = inferHeadDim(kWeight, numKVHeads, fallbackHeadDim);
                }
                long layerBytes = (long) maxSeqLen * numKVHeads * headDim * 4; // 4 bytes for float32
                keyCaches.put(i, arena.allocate(layerBytes));
                valueCaches.put(i, arena.allocate(layerBytes));
            }
        }

        private int inferHeadDim(TorchTensor weight, int numHeads, int fallback) {
            if (weight == null || numHeads <= 0) {
                return fallback;
            }
            long[] shape = weight.shape();
            if (shape.length != 2) {
                return fallback;
            }
            long out = shape[0];
            if (out <= 0 || out % numHeads != 0) {
                return fallback;
            }
            return (int) (out / numHeads);
        }

        /**
         * Returns the key cache segment for the given transformer layer.
         *
         * @param layerIdx zero-based layer index
         * @return the off-heap {@link MemorySegment} for key tensors
         */
        public MemorySegment keyCache(int layerIdx) {
            return keyCaches.get(layerIdx);
        }

        /**
         * Returns the value cache segment for the given transformer layer.
         *
         * @param layerIdx zero-based layer index
         * @return the off-heap {@link MemorySegment} for value tensors
         */
        public MemorySegment valueCache(int layerIdx) {
            return valueCaches.get(layerIdx);
        }

        /**
         * Advances the current sequence position by {@code seqLen} tokens.
         *
         * @param seqLen number of tokens just processed
         */
        public void advance(int seqLen) {
            this.currentPos += seqLen;
        }

        /**
         * Advances the current sequence position and records the token count
         * for the given modality (used in multimodal sessions).
         *
         * @param seqLen   number of tokens just processed
         * @param modality the modality that produced these tokens
         */
        public void advance(int seqLen, ModalityType modality) {
            this.currentPos += seqLen;
            modalityOffsets.merge(modality, seqLen, Integer::sum);
        }

        /**
         * Returns the current sequence position (total tokens processed so far).
         *
         * @return current position
         */
        public int currentPos() {
            return currentPos;
        }

        /**
         * Returns the cumulative token offset for the given modality.
         *
         * @param modality the modality to query
         * @return token count contributed by this modality, or {@code 0} if none
         */
        public int modalityOffset(ModalityType modality) {
            return modalityOffsets.getOrDefault(modality, 0);
        }

        /**
         * Closes the Arena, releasing all off-heap memory allocated for this session.
         */
        @Override
        public void close() {
            arena.close();
            keyCaches.clear();
            valueCaches.clear();
        }
    }
}
