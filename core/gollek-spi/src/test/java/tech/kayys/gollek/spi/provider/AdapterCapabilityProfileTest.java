package tech.kayys.gollek.spi.provider;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class AdapterCapabilityProfileTest {

    @Test
    void emitsUnsupportedFeatureWhenAdaptersAreDisabled() {
        Set<String> features = AdapterCapabilityProfile.unsupportedWithMetrics().toFeatureFlags();

        assertThat(features).contains(
                AdapterCapabilityProfile.FEATURE_ADAPTER_UNSUPPORTED,
                AdapterCapabilityProfile.FEATURE_ADAPTER_METRICS_SCHEMA_V1);
        assertThat(features).doesNotContain(AdapterCapabilityProfile.FEATURE_ADAPTER_SUPPORTED);
    }

    @Test
    void emitsNormalizedFeaturesForSupportedAdapterProviders() {
        Set<String> features = AdapterCapabilityProfile.builder()
                .adapterSupported(true)
                .adapterTypes(Set.of("lora"))
                .runtimeApply(true)
                .precompiledModelPath(true)
                .rolloutGuard(true)
                .metricsSchema(true)
                .build()
                .toFeatureFlags();

        assertThat(features).contains(
                AdapterCapabilityProfile.FEATURE_ADAPTER_SUPPORTED,
                AdapterCapabilityProfile.FEATURE_ADAPTER_SPEC_V1,
                "adapter_type_lora",
                AdapterCapabilityProfile.FEATURE_ADAPTER_RUNTIME_APPLY,
                AdapterCapabilityProfile.FEATURE_ADAPTER_PRECOMPILED_MODEL_PATH,
                AdapterCapabilityProfile.FEATURE_ADAPTER_ROLLOUT_GUARD,
                AdapterCapabilityProfile.FEATURE_ADAPTER_METRICS_SCHEMA_V1);
        assertThat(features).doesNotContain(AdapterCapabilityProfile.FEATURE_ADAPTER_UNSUPPORTED);
    }
}
