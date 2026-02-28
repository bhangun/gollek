package tech.kayys.gollek.spi.provider;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

/**
 * Typed view of adapter-related provider capabilities derived from feature tags.
 */
public record AdapterCapabilitySummary(
        boolean adapterSupported,
        boolean adapterUnsupported,
        boolean metricsSchema,
        boolean runtimeApply,
        boolean precompiledModelPath,
        boolean rolloutGuard,
        Set<String> adapterTypes) {

    public static AdapterCapabilitySummary from(ProviderCapabilities capabilities) {
        return fromFeatures(capabilities == null ? Collections.emptySet() : capabilities.getFeatures());
    }

    public static AdapterCapabilitySummary fromFeatures(Set<String> features) {
        Set<String> normalized = features == null ? Collections.emptySet() : features;
        boolean supported = normalized.contains(AdapterCapabilityProfile.FEATURE_ADAPTER_SUPPORTED);
        boolean unsupported = normalized.contains(AdapterCapabilityProfile.FEATURE_ADAPTER_UNSUPPORTED);
        boolean metrics = normalized.contains(AdapterCapabilityProfile.FEATURE_ADAPTER_METRICS_SCHEMA_V1);
        boolean runtime = normalized.contains(AdapterCapabilityProfile.FEATURE_ADAPTER_RUNTIME_APPLY);
        boolean precompiled = normalized.contains(AdapterCapabilityProfile.FEATURE_ADAPTER_PRECOMPILED_MODEL_PATH);
        boolean rollout = normalized.contains(AdapterCapabilityProfile.FEATURE_ADAPTER_ROLLOUT_GUARD);

        LinkedHashSet<String> types = new LinkedHashSet<>();
        for (String feature : normalized) {
            if (feature == null) {
                continue;
            }
            String lower = feature.toLowerCase(Locale.ROOT);
            if (lower.startsWith("adapter_type_") && lower.length() > "adapter_type_".length()) {
                types.add(lower.substring("adapter_type_".length()));
            }
        }

        return new AdapterCapabilitySummary(
                supported,
                unsupported,
                metrics,
                runtime,
                precompiled,
                rollout,
                Set.copyOf(types));
    }
}
