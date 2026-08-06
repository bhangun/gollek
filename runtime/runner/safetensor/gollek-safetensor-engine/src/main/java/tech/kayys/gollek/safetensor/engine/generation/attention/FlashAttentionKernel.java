package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import org.jboss.logging.Logger;

@ApplicationScoped
public class FlashAttentionKernel {
    private static final Logger LOG = Logger.getLogger(FlashAttentionKernel.class);

    private FlashAttentionStages stages;

    @Inject
    RopeFrequencyCache ropeCache;

    @jakarta.annotation.PostConstruct
    void init() {
        this.stages = FlashAttentionStages.initialize(LOG, ropeCache);
    }

    public AccelTensor compute(AttentionInput in) {
        FlashAttentionStages stages = stages();
        FlashAttentionKvCacheStage kvCache = stages.kvCache();
        FlashAttentionPlan plan = FlashAttentionPlan.resolve(
                in, kvCache, stages.options().normalizationOptions());
        FlashAttentionProjectionStage.PreparedTensors prepared = stages.projection().prepare(
                in,
                plan.config(),
                plan.modelPolicy(),
                plan.headLayout(),
                plan.normalizationPolicy(),
                plan.sharedKvState(),
                plan.sharedKv(),
                plan.useDenseSharedKvState(),
                plan.alternativeAttention());
        AccelTensor q = prepared.query();
        AccelTensor k = prepared.key();
        AccelTensor v = prepared.value();
        AccelTensor sharedOwner = prepared.sharedOwner();
        FlashAttentionKvCacheStage.CachedTensors cached = null;
        boolean queryReleased = false;
        boolean keyValueReleased = false;

        try {
            stages.rope().apply(q, plan.sharedKv() ? null : k, plan.startPos(), plan.config(), plan.modelPolicy(),
                    plan.layerIdx(), plan.headDim());

            // Update cache after RoPE so dispatch sees the same key/value layout as decode.
            KVCacheManager.KVCacheSession kvSession = in.kvCache;
            cached = kvCache.updateCache(
                    in, plan.kvState(), kvSession, k, v, plan.seqLen(), plan.numKeyValueHeads(), plan.headDim());
            k = cached.key();
            v = cached.value();

            // Dispatch after cache update because paged paths read key/value tensors from KV cache.
            AccelTensor attnOut = stages.dispatch().compute(
                    plan.dispatchRequest(q, k, v, kvSession, in.isCausal, in.attentionContextBuffer));
            q.close();
            queryReleased = true;
            kvCache.releaseKeyValueViews(plan.kvState(), cached);
            keyValueReleased = true;

            return plan.applyOutputStage(stages.output(), in, attnOut);
        } finally {
            if (!queryReleased && q != null && !q.isClosed()) {
                q.close();
            }
            if (!keyValueReleased) {
                if (cached != null) {
                    kvCache.releaseKeyValueViews(plan.kvState(), cached);
                } else {
                    kvCache.releasePreparedKeyValueTensors(plan.kvState(), k, v);
                }
            }
            if (sharedOwner != null && !sharedOwner.isClosed()) {
                sharedOwner.close();
            }
        }
    }

    private FlashAttentionStages stages() {
        if (stages == null) {
            stages = FlashAttentionStages.initialize(LOG, ropeCache);
        }
        return stages;
    }

}
