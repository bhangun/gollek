package tech.kayys.gollek.safetensor.engine.generation.moe;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor.QuantType;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardFfnService;
import tech.kayys.gollek.safetensor.engine.weights.WeightTensorResolver;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

@ApplicationScoped
public class MoeForwardPass {
    private static final Logger log = Logger.getLogger(MoeForwardPass.class);
    private static final List<String> LAYER_PREFIXES = List.of("model.language_model.layers.", "model.layers.");

    @Inject
    DirectForwardFfnService ffnService;

    /**
     * Bounded LRU cache for materialized MoE expert slices.
     * <p>
     * Key  = System.identityHashCode(stackedTensor) * SHIFT + expertIdx
     *         (multiplied by 4 for gate/up/down/gateUp variants via an offset in [0,3]).
     * Value = contiguous AccelTensor with offset=0, owned by this cache.
     * <p>
     * Bounded to {@code MAX_EXPERT_CACHE_BYTES} total. LRU eviction closes the
     * evicted tensor's arena to release direct memory immediately.
     * This lets hot experts persist across decode steps while keeping
     * total memory usage bounded.
     */
    private long maxExpertCacheBytes = -1L;
    private static final long ENTRY_SHIFT = 1 << 22; // up to 4M entries per tensor
    private long expertCacheBytes = 0L;
    private final java.util.LinkedHashMap<Long, AccelTensor> expertLruCache =
            new java.util.LinkedHashMap<>(256, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Long, AccelTensor> eldest) {
                    if (maxExpertCacheBytes > 0 && expertCacheBytes > maxExpertCacheBytes) {
                        AccelTensor evicted = eldest.getValue();
                        expertCacheBytes -= evictedBytes(evicted);
                        if (!evicted.isClosed()) evicted.close();
                        return true;
                    }
                    return false;
                }
            };



    private static long evictedBytes(AccelTensor t) {
        if (t == null || t.isClosed()) return 0L;
        long bytes = t.numel() * t.elementByteSize();
        if (t.isQuantized()) {
            bytes += t.numel() * 2L;
        }
        return bytes;
    }

    /** Retrieve or materialize a contiguous expert slice, caching it in the LRU. */
    private synchronized AccelTensor getOrMaterializeExpert(AccelTensor stacked, int expertIdx, int variant) {
        long key = (long) System.identityHashCode(stacked) * ENTRY_SHIFT * 4L + (long) expertIdx * 4L + variant;
        AccelTensor cached = expertLruCache.get(key);
        if (cached != null && !cached.isClosed()) {
            return cached;
        }
        // Materialize: slice(0, expertIdx, expertIdx+1).squeeze() then contiguous()
        AccelTensor sliced = stacked.slice(0, expertIdx, expertIdx + 1).squeeze();
        AccelTensor cont   = sliced.contiguous();
        if (cont != sliced) sliced.close();
        
        long bytes = cont.numel() * (long) cont.elementByteSize();
        if (cont.isQuantized()) {
            // Do NOT preemptively dequantize on CPU to avoid SIGILL and memory bloat.
            // Metal kernel will handle the quantized tensors directly.
        }
        
        expertCacheBytes += bytes;
        expertLruCache.put(key, cont);
        return cont;
    }

    /** Retrieve or materialize a contiguous sub-slice of an already-materialized expert. */
    private synchronized AccelTensor getOrMaterializeHalf(AccelTensor expertContiguous,
            long start, long end, int expertIdx, int variant, long stackedKey) {
        long key = stackedKey * ENTRY_SHIFT * 4L + (long) expertIdx * 4L + variant;
        AccelTensor cached = expertLruCache.get(key);
        if (cached != null && !cached.isClosed()) {
            return cached;
        }
        AccelTensor raw  = expertContiguous.slice(0, start, end);
        AccelTensor cont = raw.contiguous();
        if (cont != raw) raw.close();
        
        long bytes = cont.numel() * (long) cont.elementByteSize();
        if (cont.isQuantized()) {
            // Do NOT preemptively dequantize on CPU to avoid SIGILL and memory bloat.
        }
        
        expertCacheBytes += bytes;
        expertLruCache.put(key, cont);
        return cont;
    }

    private synchronized void initMaxExpertCacheBytes(ModelConfig config) {
        if (maxExpertCacheBytes == -1L) {
            long explicit = Long.getLong("gollek.moe.expert_cache_bytes", -1L);
            if (explicit > 0) {
                maxExpertCacheBytes = explicit;
            } else {
                long hiddenSize = config.hiddenSize();
                long intermSize = config.moeIntermediateSize() > 0 
                        ? config.moeIntermediateSize() 
                        : (config.intermediateSize() > 0 ? config.intermediateSize() : hiddenSize * 4L);
                long numExpertsPerTok = config.numExpertsPerTok() > 0
                        ? config.numExpertsPerTok()
                        : 1;
                long numHiddenLayers = config.numHiddenLayers() > 0 ? config.numHiddenLayers() : 32;
                
                // Size of 1 active expert across all layers: 3 matrices (Gate, Up, Down) * 2 bytes (F16) = 6
                long totalActiveBytes = 6L * hiddenSize * intermSize * numExpertsPerTok * numHiddenLayers;
                
                // Minimum 2GB, or totalActiveBytes + 1GB buffer
                maxExpertCacheBytes = Math.max(2_000_000_000L, totalActiveBytes + 1_000_000_000L);
                log.infof("MoE expert cache size bounded to %d bytes (dynamic based on config)", maxExpertCacheBytes);
            }
        }
    }

    public AccelTensor computeAccel(AccelTensor hidden, Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch, int layerIdx) {
        initMaxExpertCacheBytes(config);
        int numExperts = config.numLocalExperts();
        int topK = config.numExpertsPerTok();
        log.tracef("MoE layer %d: %d experts, top-%d routing", layerIdx, numExperts, topK);

        AccelTensor hiddenFlat = flattenBatch(hidden);
        long[] flatShape = hiddenFlat.shape();
        int hiddenSize = (int)flatShape[flatShape.length - 1];
        int numTokens = (int)flatShape[0];
        String prefix = resolveLayerPrefix(weights, layerIdx);

        AccelTensor gateWeight = resolveWeight(weights, prefix, layerIdx, arch.layerMoeGateWeightCandidates(layerIdx), "mlp.gate.weight");
        if (gateWeight == null) {
            log.warnf("MoE router weight not found for layer %d (prefix=%s) — using expert 0 only", layerIdx, prefix);
            // Per-layer-call expert caches (bounded, freed when this call returns)
            Map<Integer, AccelTensor> guCache   = new HashMap<>();
            Map<Integer, AccelTensor> downCache = new HashMap<>();
            Map<Integer, AccelTensor> gateCache = new HashMap<>();
            Map<Integer, AccelTensor> upCache   = new HashMap<>();
            try {
                AccelTensor result = runExpert(hiddenFlat, hiddenSize, weights, prefix, config, arch, layerIdx, 0,
                        guCache, downCache, gateCache, upCache);
                if (hiddenFlat != hidden) hiddenFlat.close();
                return reshapeToOriginal(result, hidden.shape(), hiddenSize);
            } finally {
                // Do not close per-call caches here, as they contain LRU-cached instances
            }
        }

        AccelTensor routerWeight = gateWeight;
        if (gateWeight.isQuantized()) {
            routerWeight = gateWeight.toQuantizedF16CachedUpTo(1024 * 1024 * 1024);
        }
        AccelTensor routerLogits = AccelOps.linear(hiddenFlat, routerWeight);
        String modelType = config.modelType() != null ? config.modelType() : "";
        boolean useSigmoid = modelType.contains("qwen") || modelType.contains("moe");
        float[] probData = routerLogits.toFloatArray();
        routerLogits.close();

        if (useSigmoid) {
            for (int i = 0; i < probData.length; ++i) {
                probData[i] = 1.0f / (1.0f + (float)Math.exp(-probData[i]));
            }
        } else {
            int numExpertsShape = probData.length / numTokens;
            for (int t = 0; t < numTokens; ++t) {
                float max = Float.NEGATIVE_INFINITY;
                for (int e = 0; e < numExpertsShape; ++e) {
                    max = Math.max(max, probData[t * numExpertsShape + e]);
                }
                float sum = 0.0f;
                for (int e = 0; e < numExpertsShape; ++e) {
                    probData[t * numExpertsShape + e] = (float)Math.exp(probData[t * numExpertsShape + e] - max);
                    sum += probData[t * numExpertsShape + e];
                }
                for (int e = 0; e < numExpertsShape; ++e) {
                    probData[t * numExpertsShape + e] /= sum;
                }
            }
        }

        int numExpertsInWeights = probData.length / numTokens;
        int[] expertIndices = new int[numTokens * topK];
        float[] expertWeights = new float[numTokens * topK];
        selectTopK(probData, numTokens, numExpertsInWeights, topK, expertIndices, expertWeights);

        AccelTensor output = AccelTensor.zeros(hiddenFlat.shape());

        // Per-layer-call expert caches: bounded to (active experts × topK) entries.
        // Each value is a contiguous materialized slice (offset=0) that is freed after this call.
        // This prevents the OOM caused by permanently caching all 256×layers experts,
        // while still avoiding redundant materializations within a single layer forward pass.
        Map<Integer, AccelTensor> guCache   = new HashMap<>();
        Map<Integer, AccelTensor> downCache = new HashMap<>();
        Map<Integer, AccelTensor> gateCache = new HashMap<>();
        Map<Integer, AccelTensor> upCache   = new HashMap<>();

        try {
            for (int t = 0; t < numTokens; ++t) {
                AccelTensor tokenHidden = sliceToken(hiddenFlat, t, hiddenSize);
                for (int ki = 0; ki < topK; ++ki) {
                    int expIdx = expertIndices[t * topK + ki];
                    float expWeight = expertWeights[t * topK + ki];
                    if (expIdx < 0 || expIdx >= numExperts || expWeight == 0.0f) continue;

                    AccelTensor expertOut = null;
                    try {
                        expertOut = runExpert(tokenHidden, hiddenSize, weights, prefix, config, arch, layerIdx, expIdx,
                                guCache, downCache, gateCache, upCache);
                        AccelTensor scaled = AccelOps.mulScalar(expertOut, expWeight);
                        addToTokenRow(output, scaled, t, hiddenSize);
                        scaled.close();
                    } catch (Exception e) {
                        log.warnf(e, "Expert %d computation failed at layer %d", expIdx, layerIdx);
                    } finally {
                        if (expertOut != null && expertOut != tokenHidden) {
                            expertOut.close();
                        }
                    }
                }

                AccelTensor sharedOut = runSharedExpert(tokenHidden, weights, prefix, config, arch, layerIdx);
                if (sharedOut != null) {
                    float sharedGate = resolveSharedExpertGate(tokenHidden, weights, prefix, layerIdx);
                    AccelTensor sharedScaled = AccelOps.mulScalar(sharedOut, sharedGate);
                    addToTokenRow(output, sharedScaled, t, hiddenSize);
                    sharedScaled.close();
                    sharedOut.close();
                }
                tokenHidden.close();
            }
        } finally {
            // Do not close per-call caches here, as they contain LRU-cached instances
        }

        if (hiddenFlat != hidden) hiddenFlat.close();
        return reshapeToOriginal(output, hidden.shape(), hiddenSize);
    }

    // ── Utilities ─────────────────────────────────────────────────────

    private static void closeAll(Map<Integer, AccelTensor> cache) {
        for (AccelTensor t : cache.values()) {
            if (t != null && !t.isClosed()) t.close();
        }
    }

    private static String resolveLayerPrefix(Map<String, AccelTensor> weights, int layerIdx) {
        for (String pfx : LAYER_PREFIXES) {
            if (weights.containsKey(pfx + layerIdx + ".mlp.experts.gate_up_proj")) return pfx;
            if (weights.containsKey(pfx + layerIdx + ".mlp.gate.weight")) return pfx;
        }
        return LAYER_PREFIXES.get(0);
    }

    private static AccelTensor resolveWeight(Map<String, AccelTensor> weights, String prefix, int layerIdx, Iterable<String> archCandidates, String suffix) {
        AccelTensor t = WeightTensorResolver.first(weights, archCandidates);
        if (t != null) return t;
        AccelTensor t2 = weights.get(prefix + layerIdx + "." + suffix);
        if (t2 != null) return t2;
        if (suffix.endsWith(".weight")) {
            AccelTensor t3 = weights.get(prefix + layerIdx + "." + suffix.replace(".weight", ".qweight"));
            if (t3 != null) return t3;
        }
        return weights.get(prefix + layerIdx + "." + suffix + ".qweight");
    }

    private static void selectTopK(float[] probs, int numTokens, int numExperts, int k, int[] outIndices, float[] outWeights) {
        Integer[] idx = new Integer[numExperts];
        for (int t = 0; t < numTokens; ++t) {
            int base = t * numExperts;
            for (int e = 0; e < numExperts; ++e) idx[e] = e;
            final int fb = base;
            Arrays.sort(idx, (a, b) -> Float.compare(probs[fb + b], probs[fb + a]));

            float weightSum = 0.0f;
            for (int ki = 0; ki < k && ki < numExperts; ++ki) {
                outIndices[t * k + ki] = idx[ki];
                outWeights[t * k + ki] = probs[base + idx[ki]];
                weightSum += probs[base + idx[ki]];
            }
            if (weightSum > 0.0f) {
                for (int ki = 0; ki < k; ++ki) {
                    outWeights[t * k + ki] /= weightSum;
                }
            }
        }
    }

    // ── Expert dispatch ───────────────────────────────────────────────

    private AccelTensor runExpert(AccelTensor hidden, int hiddenSize,
            Map<String, AccelTensor> weights, String prefix,
            ModelConfig config, ModelArchitecture arch, int layerIdx, int expertIdx,
            Map<Integer, AccelTensor> guCache, Map<Integer, AccelTensor> downCache,
            Map<Integer, AccelTensor> gateCache, Map<Integer, AccelTensor> upCache) {

        AccelTensor stackedGateUp = weights.get(prefix + layerIdx + ".mlp.experts.gate_up_proj");
        if (stackedGateUp == null) stackedGateUp = weights.get(prefix + layerIdx + ".mlp.experts.gate_up_proj.qweight");
        
        AccelTensor stackedDown   = weights.get(prefix + layerIdx + ".mlp.experts.down_proj");
        if (stackedDown == null) stackedDown = weights.get(prefix + layerIdx + ".mlp.experts.down_proj.qweight");

        AccelTensor stackedGate = weights.get(prefix + layerIdx + ".mlp.experts.gate_proj.weight");
        if (stackedGate == null) stackedGate = weights.get(prefix + layerIdx + ".mlp.experts.gate_proj.qweight");
        
        AccelTensor stackedUp   = weights.get(prefix + layerIdx + ".mlp.experts.up_proj.weight");
        if (stackedUp == null) stackedUp = weights.get(prefix + layerIdx + ".mlp.experts.up_proj.qweight");
        if (stackedGate != null && stackedUp != null && stackedDown != null) {
            return runSeparateStackedExpert(hidden, stackedGate, stackedUp, stackedDown, expertIdx, config, arch,
                    gateCache, upCache, downCache);
        }

        if (stackedGateUp != null && stackedDown != null) {
            return runStackedExpert(hidden, stackedGateUp, stackedDown, expertIdx, config, arch,
                    guCache, downCache, gateCache, upCache);
        }

        AccelTensor gateW = WeightTensorResolver.first(weights, arch.expertGateWeightCandidates(layerIdx, expertIdx),
                prefix + layerIdx + ".mlp.experts." + expertIdx + ".gate_proj.weight");
        AccelTensor upW   = WeightTensorResolver.first(weights, arch.expertUpWeightCandidates(layerIdx, expertIdx),
                prefix + layerIdx + ".mlp.experts." + expertIdx + ".up_proj.weight");
        AccelTensor downW = WeightTensorResolver.first(weights, arch.expertDownWeightCandidates(layerIdx, expertIdx),
                prefix + layerIdx + ".mlp.experts." + expertIdx + ".down_proj.weight");
        if (gateW == null || upW == null || downW == null) {
            log.warnf("Expert %d weights not found for layer %d (prefix=%s)", expertIdx, layerIdx, prefix);
            return hidden;
        }
        return ffnService.swigluFfn(hidden, arch, config, gateW, null, upW, null, downW, null);
    }

    private AccelTensor runSeparateStackedExpert(AccelTensor hidden,
            AccelTensor stackedGate, AccelTensor stackedUp, AccelTensor stackedDown,
            int expertIdx, ModelConfig config, ModelArchitecture arch,
            Map<Integer, AccelTensor> gateCache, Map<Integer, AccelTensor> upCache, Map<Integer, AccelTensor> downCache) {
        try {
            long[] gShape = stackedGate.shape();
            if (gShape.length < 3) return hidden;
            long numExperts = gShape[0];
            if (expertIdx < 0 || expertIdx >= numExperts) return hidden;
            
            AccelTensor gateW = gateCache.computeIfAbsent(expertIdx, idx -> getOrMaterializeExpert(stackedGate, idx, 0));
            AccelTensor upW = upCache.computeIfAbsent(expertIdx, idx -> getOrMaterializeExpert(stackedUp, idx, 0));
            AccelTensor downW = downCache.computeIfAbsent(expertIdx, idx -> getOrMaterializeExpert(stackedDown, idx, 0));
            
            return ffnService.swigluFfn(hidden, arch, config, gateW, null, upW, null, downW, null);
        } catch (Exception e) {
            log.warnf(e, "Separate stacked expert %d computation failed", expertIdx);
            return hidden;
        }
    }

    private AccelTensor runStackedExpert(AccelTensor hidden,
            AccelTensor stackedGateUp, AccelTensor stackedDown,
            int expertIdx, ModelConfig config, ModelArchitecture arch,
            Map<Integer, AccelTensor> guCache, Map<Integer, AccelTensor> downCache,
            Map<Integer, AccelTensor> gateCache, Map<Integer, AccelTensor> upCache) {
        try {
            long[] guShape   = stackedGateUp.shape();
            long[] downShape = stackedDown.shape();
            if (guShape.length < 3 || downShape.length < 3) {
                log.warnf("Unexpected stacked expert tensor rank: gate_up=%d, down=%d", guShape.length, downShape.length);
                return hidden;
            }
            long numExperts = guShape[0];
            long twoI = guShape[1];
            long I    = twoI / 2L;
            if (expertIdx < 0 || expertIdx >= numExperts) {
                log.warnf("Expert index %d out of range [0, %d)", expertIdx, numExperts);
                return hidden;
            }

            long guLruKey = (long) System.identityHashCode(stackedGateUp);

            // Use LRU cache (survives across decode steps). Per-call local caches
            // (guCache/etc.) deduplicate within a batched prefill (multiple tokens
            // per layer call). Both layers are freed correctly: LRU evicts via close(),
            // per-call caches are freed by closeAll() in the caller's finally block.
            AccelTensor expertGateUp = guCache.computeIfAbsent(expertIdx,
                    idx -> getOrMaterializeExpert(stackedGateUp, idx, 0));
            AccelTensor expertDown = downCache.computeIfAbsent(expertIdx,
                    idx -> getOrMaterializeExpert(stackedDown, idx, 0));

            // gate (offset=0) is already contiguous — no copy needed.
            // up   (offset=I) needs contiguous() to produce offset=0 buffer.
            AccelTensor gateW = gateCache.computeIfAbsent(expertIdx,
                    idx -> getOrMaterializeHalf(expertGateUp, 0L, I, idx, 1, guLruKey));
            AccelTensor upW = upCache.computeIfAbsent(expertIdx,
                    idx -> getOrMaterializeHalf(expertGateUp, I, twoI, idx, 2, guLruKey));

            return ffnService.swigluFfn(hidden, arch, config, gateW, null, upW, null, expertDown, null);
            // Do NOT close gateW/upW/expertGateUp/expertDown here — owned by LRU or per-call cache.
        } catch (Exception e) {
            log.warnf(e, "Stacked expert %d computation failed", expertIdx);
            return hidden;
        }
    }


    private AccelTensor runSharedExpert(AccelTensor tokenHidden, Map<String, AccelTensor> weights,
            String prefix, ModelConfig config, ModelArchitecture arch, int layerIdx) {
        String base = prefix + layerIdx + ".mlp.shared_expert.";
        AccelTensor gateW = weights.get(base + "gate_proj.weight");
        AccelTensor upW   = weights.get(base + "up_proj.weight");
        AccelTensor downW = weights.get(base + "down_proj.weight");
        if (gateW == null || upW == null || downW == null) return null;
        return ffnService.swigluFfn(tokenHidden, arch, config, gateW, null, upW, null, downW, null);
    }

    private static float resolveSharedExpertGate(AccelTensor tokenHidden, Map<String, AccelTensor> weights,
            String prefix, int layerIdx) {
        AccelTensor gateW = weights.get(prefix + layerIdx + ".mlp.shared_expert_gate.weight");
        if (gateW == null) return 1.0f;
        try {
            AccelTensor logit = AccelOps.linear(tokenHidden, gateW);
            float[] v = logit.toFloatArray();
            logit.close();
            float scalar = v.length > 0 ? v[0] : 0.0f;
            return 1.0f / (1.0f + (float)Math.exp(-scalar));
        } catch (Exception e) {
            log.debugf("shared_expert_gate computation failed at layer %d: %s", layerIdx, e.getMessage());
            return 1.0f;
        }
    }

    // ── Tensor shape helpers ──────────────────────────────────────────

    private static AccelTensor flattenBatch(AccelTensor t) {
        long[] s = t.shape();
        if (s.length == 2) return t;
        long tokens = 1L;
        for (int i = 0; i < s.length - 1; ++i) tokens *= s[i];
        return t.reshape(tokens, s[s.length - 1]);
    }

    private static AccelTensor sliceToken(AccelTensor flat, int tokenIdx, int hiddenSize) {
        AccelTensor row  = flat.slice(0, tokenIdx, tokenIdx + 1);
        AccelTensor cont = row.contiguous();
        if (cont != row) row.close();
        return cont;
    }

    private static void addToTokenRow(AccelTensor output, AccelTensor contribution, int tokenIdx, int hiddenSize) {
        MemorySegment outSeg  = output.dataPtr();
        MemorySegment contSeg = contribution.dataPtr();
        long numContrib = Math.min(contribution.numel(), (long)hiddenSize);
        long dstOffset = (long)tokenIdx * (long)hiddenSize * 4L;
        for (long i = 0L; i < numContrib; ++i) {
            float cur  = outSeg.get(ValueLayout.JAVA_FLOAT, dstOffset + i * 4L);
            float addV = contSeg.get(ValueLayout.JAVA_FLOAT, i * 4L);
            outSeg.set(ValueLayout.JAVA_FLOAT, dstOffset + i * 4L, cur + addV);
        }
    }

    private static AccelTensor reshapeToOriginal(AccelTensor t, long[] origShape, int hiddenSize) {
        if (origShape.length == 2) return t;
        long[] newShape = Arrays.copyOf(origShape, origShape.length);
        newShape[newShape.length - 1] = hiddenSize;
        return t.reshape(newShape);
    }
}
