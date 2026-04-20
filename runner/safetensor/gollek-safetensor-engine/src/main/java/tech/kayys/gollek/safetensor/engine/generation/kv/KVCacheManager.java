/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.kv;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModalityType;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * FFM-backed KV (key-value) cache manager using {@link AccelTensor}.
 *
 * <p>No LibTorch dependency — pure MemorySegment storage.
 */
@ApplicationScoped
public class KVCacheManager {

    public KVCacheSession createSession(int maxSeqLen) {
        return new KVCacheSession(maxSeqLen);
    }

    public static class KVCacheSession implements AutoCloseable {
        private final int maxSeqLen;
        private final Map<Integer, AccelTensor> keyCaches = new ConcurrentHashMap<>();
        private final Map<Integer, AccelTensor> valueCaches = new ConcurrentHashMap<>();
        private int currentPos = 0;
        private final Map<ModalityType, Integer> modalityOffsets = new EnumMap<>(ModalityType.class);

        public KVCacheSession(int maxSeqLen) {
            this.maxSeqLen = maxSeqLen;
        }

        /**
         * Allocates KV cache buffers for all layers.
         */
        public void allocate(ModelConfig config) {
            allocate(config, null, null);
        }

        /**
         * Allocates KV cache buffers, optionally inferring head dim from weights.
         */
        public void allocate(ModelConfig config, Map<String, AccelTensor> weights, ModelArchitecture arch) {
            int numLayers = config.numHiddenLayers();
            int numKVHeads = config.resolvedNumKvHeads();
            int fallbackHeadDim = config.resolvedHeadDim();

            for (int i = 0; i < numLayers; i++) {
                int headDim = fallbackHeadDim;
                if (weights != null && arch != null) {
                    AccelTensor kWeight = weights.get(arch.layerKeyWeight(i));
                    if (kWeight == null) {
                        kWeight = weights.get(arch.layerValueWeight(i));
                    }
                    headDim = inferHeadDim(kWeight, numKVHeads, fallbackHeadDim);
                }
                // Shape: [maxSeqLen, numKVHeads, headDim]
                keyCaches.put(i, AccelTensor.zeros(maxSeqLen, numKVHeads, headDim));
                valueCaches.put(i, AccelTensor.zeros(maxSeqLen, numKVHeads, headDim));
            }
        }

        private int inferHeadDim(AccelTensor weight, int numHeads, int fallback) {
            if (weight == null || numHeads <= 0) return fallback;
            long[] shape = weight.shape();
            if (shape.length != 2) return fallback;
            
            // Check both dimensions: weights might be [out, in] or pre-transposed [in, out]
            long dim0 = shape[0];
            long dim1 = shape[1];
            
            if (dim0 % numHeads == 0 && dim0 / numHeads < dim1) {
                return (int) (dim0 / numHeads);
            }
            if (dim1 % numHeads == 0) {
                return (int) (dim1 / numHeads);
            }
            
            return fallback;
        }

        public AccelTensor keyCache(int layerIdx) {
            return keyCaches.get(layerIdx);
        }

        public AccelTensor valueCache(int layerIdx) {
            return valueCaches.get(layerIdx);
        }

        public void advance(int seqLen) {
            this.currentPos += seqLen;
        }

        public void advance(int seqLen, ModalityType modality) {
            this.currentPos += seqLen;
            modalityOffsets.merge(modality, seqLen, Integer::sum);
        }

        public int currentPos() {
            return currentPos;
        }

        public int modalityOffset(ModalityType modality) {
            return modalityOffsets.getOrDefault(modality, 0);
        }

        @Override
        public void close() {
            keyCaches.values().forEach(t -> { if (t != null && !t.isClosed()) t.close(); });
            valueCaches.values().forEach(t -> { if (t != null && !t.isClosed()) t.close(); });
            keyCaches.clear();
            valueCaches.clear();
        }
    }
}
