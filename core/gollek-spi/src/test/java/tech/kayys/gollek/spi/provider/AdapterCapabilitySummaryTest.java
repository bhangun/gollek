package tech.kayys.gollek.spi.provider;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class AdapterCapabilitySummaryTest {

    @Test
    void derivesSupportedSummaryWithAdapterTypes() {
        Set<String> features = AdapterCapabilityProfile.builder()
                .adapterSupported(true)
                .adapterTypes(Set.of("lora", "ia3"))
                .runtimeApply(true)
                .precompiledModelPath(true)
                .rolloutGuard(true)
                .metricsSchema(true)
                .build()
                .toFeatureFlags();

        AdapterCapabilitySummary summary = AdapterCapabilitySummary.fromFeatures(features);

        assertThat(summary.adapterSupported()).isTrue();
        assertThat(summary.adapterUnsupported()).isFalse();
        assertThat(summary.metricsSchema()).isTrue();
        assertThat(summary.runtimeApply()).isTrue();
        assertThat(summary.precompiledModelPath()).isTrue();
        assertThat(summary.rolloutGuard()).isTrue();
        assertThat(summary.adapterTypes()).containsExactlyInAnyOrder("lora", "ia3");
    }

    @Test
    void derivesUnsupportedSummary() {
        AdapterCapabilitySummary summary = AdapterCapabilitySummary.fromFeatures(
                AdapterCapabilityProfile.unsupportedWithMetrics().toFeatureFlags());

        assertThat(summary.adapterSupported()).isFalse();
        assertThat(summary.adapterUnsupported()).isTrue();
        assertThat(summary.metricsSchema()).isTrue();
        assertThat(summary.adapterTypes()).isEmpty();
    }

    @Test
    void supportsDirectExtractionFromProviderCapabilities() {
        ProviderCapabilities capabilities = ProviderCapabilities.builder()
                .features(Set.of(
                        "local_inference",
                        AdapterCapabilityProfile.FEATURE_ADAPTER_SUPPORTED,
                        "adapter_type_lora"))
                .build();

        AdapterCapabilitySummary summary = AdapterCapabilitySummary.from(capabilities);
        assertThat(summary.adapterSupported()).isTrue();
        assertThat(summary.adapterTypes()).containsExactly("lora");
    }
}
