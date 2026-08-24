package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.engine.runtime.ModelRuntimeTraitsResolver;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

final class FlashAttentionModelPolicy {
    private final ModelArchitecture architecture;
    private final ModelConfig config;
    private final ModelRuntimeTraits traits;
    private final ModelRuntimeTraits.AttentionRuntimeTraits attention;

    private FlashAttentionModelPolicy(ModelArchitecture architecture, ModelConfig config, ModelRuntimeTraits traits) {
        this.architecture = architecture;
        this.config = config;
        this.traits = ModelRuntimeTraitsResolver.resolve(config, traits);
        this.attention = this.traits.attention() == null
                ? ModelRuntimeTraits.AttentionRuntimeTraits.EMPTY
                : this.traits.attention();
    }

    static FlashAttentionModelPolicy resolve(ModelArchitecture architecture, ModelConfig config) {
        return new FlashAttentionModelPolicy(architecture, config,
                ModelRuntimeTraitsResolver.resolve(architecture, config));
    }

    static FlashAttentionModelPolicy resolve(ModelArchitecture architecture, ModelConfig config,
            ModelRuntimeTraits traits) {
        return new FlashAttentionModelPolicy(architecture, config, traits);
    }

    boolean nativeBf16Matvec() {
        return traits.nativeBf16Matvec();
    }


    boolean perLayerInputPath() {
        return traits.perLayerInputPath();
    }

    boolean useInterleavedRope(boolean legacyGemma4Interleaved, boolean experimentalGemma4SplitHalf) {
        boolean splitHalf = attention.splitHalfRope();
        if (traits.nativeBf16Matvec()) {
            if (legacyGemma4Interleaved) {
                splitHalf = false;
            }
            if (experimentalGemma4SplitHalf) {
                splitHalf = true;
            }
        }
        boolean architectureUsesSplitHalf = architecture == null || architecture.usesNeoxRope();
        return !splitHalf && !architectureUsesSplitHalf;
    }

    float resolveAttentionSoftCap() {
        if (attention.attentionSoftCapAppliesToFinalLogitsOnly()) {
            return 0.0f;
        }
        Double configured = config != null ? config.attnLogitSoftcapping() : null;
        if (configured != null && configured > 0) {
            return configured.floatValue();
        }
        return architecture == null ? 0.0f : architecture.defaultAttnSoftCap();
    }

    boolean preferMetalPerHeadRmsNorm() {
        return attention.preferMetalPerHeadRmsNorm();
    }

    boolean allowLegacyMetalAttentionBridge(boolean legacyBridgeEnabled) {
        return !restrictsLegacyMetalAttentionBridge() || legacyBridgeEnabled;
    }

    boolean restrictsLegacyMetalAttentionBridge() {
        return attention.restrictLegacyMetalAttentionBridge();
    }

    boolean restrictedMetalDecodeCandidate(int seqLen) {
        return restrictsLegacyMetalAttentionBridge() && seqLen == 1;
    }

    boolean supportsForcedDenseAttention() {
        return attention.supportsForcedDenseAttention();
    }

    boolean preferNativeMetalBf16Linear() {
        return attention.preferNativeMetalBf16Linear();
    }

    boolean disallowBf16ToF16LinearConversion() {
        return attention.disallowBf16ToF16LinearConversion();
    }

    int defaultPagedMetalPrefillMaxTokens(int fallback) {
        int preferred = attention.defaultPagedMetalPrefillMaxTokens();
        return preferred > 0 ? preferred : fallback;
    }

    boolean compactAttentionMatvecCandidate() {
        return attention.compactAttentionMatvecCandidate();
    }

    boolean metalHalfMatvecAutoCandidate() {
        return attention.compactAttentionMatvecCandidate()
                || attention.largeAttentionMatvecCandidate();
    }

    boolean packedQkvProjection() {
        return attention.packedQkvProjection()
                || (architecture != null && architecture.hasFusedQKV());
    }
}
