package tech.kayys.gollek.spi.provider;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Normalized provider capability profile for adapter behavior.
 */
public final class AdapterCapabilityProfile {

    public static final String FEATURE_ADAPTER_SUPPORTED = "adapter_supported";
    public static final String FEATURE_ADAPTER_UNSUPPORTED = "adapter_unsupported";
    public static final String FEATURE_ADAPTER_SPEC_V1 = "adapter_spec_v1";
    public static final String FEATURE_ADAPTER_METRICS_SCHEMA_V1 = "adapter_metrics_schema_v1";
    public static final String FEATURE_ADAPTER_RUNTIME_APPLY = "adapter_runtime_apply";
    public static final String FEATURE_ADAPTER_PRECOMPILED_MODEL_PATH = "adapter_precompiled_model_path";
    public static final String FEATURE_ADAPTER_ROLLOUT_GUARD = "adapter_rollout_guard";

    private final boolean adapterSupported;
    private final Set<String> adapterTypes;
    private final boolean runtimeApply;
    private final boolean precompiledModelPath;
    private final boolean rolloutGuard;
    private final boolean metricsSchema;

    private AdapterCapabilityProfile(Builder builder) {
        this.adapterSupported = builder.adapterSupported;
        this.adapterTypes = Collections.unmodifiableSet(new LinkedHashSet<>(builder.adapterTypes));
        this.runtimeApply = builder.runtimeApply;
        this.precompiledModelPath = builder.precompiledModelPath;
        this.rolloutGuard = builder.rolloutGuard;
        this.metricsSchema = builder.metricsSchema;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static AdapterCapabilityProfile unsupportedWithMetrics() {
        return builder()
                .adapterSupported(false)
                .metricsSchema(true)
                .build();
    }

    public Set<String> toFeatureFlags() {
        LinkedHashSet<String> features = new LinkedHashSet<>();
        if (adapterSupported) {
            features.add(FEATURE_ADAPTER_SUPPORTED);
            features.add(FEATURE_ADAPTER_SPEC_V1);
            for (String type : adapterTypes) {
                if (type != null && !type.isBlank()) {
                    features.add("adapter_type_" + type.toLowerCase(Locale.ROOT));
                }
            }
            if (runtimeApply) {
                features.add(FEATURE_ADAPTER_RUNTIME_APPLY);
            }
            if (precompiledModelPath) {
                features.add(FEATURE_ADAPTER_PRECOMPILED_MODEL_PATH);
            }
            if (rolloutGuard) {
                features.add(FEATURE_ADAPTER_ROLLOUT_GUARD);
            }
        } else {
            features.add(FEATURE_ADAPTER_UNSUPPORTED);
        }
        if (metricsSchema) {
            features.add(FEATURE_ADAPTER_METRICS_SCHEMA_V1);
        }
        return Set.copyOf(features);
    }

    public static final class Builder {
        private boolean adapterSupported;
        private Set<String> adapterTypes = Collections.emptySet();
        private boolean runtimeApply;
        private boolean precompiledModelPath;
        private boolean rolloutGuard;
        private boolean metricsSchema;

        public Builder adapterSupported(boolean adapterSupported) {
            this.adapterSupported = adapterSupported;
            return this;
        }

        public Builder adapterTypes(Set<String> adapterTypes) {
            this.adapterTypes = Objects.requireNonNullElse(adapterTypes, Collections.emptySet());
            return this;
        }

        public Builder runtimeApply(boolean runtimeApply) {
            this.runtimeApply = runtimeApply;
            return this;
        }

        public Builder precompiledModelPath(boolean precompiledModelPath) {
            this.precompiledModelPath = precompiledModelPath;
            return this;
        }

        public Builder rolloutGuard(boolean rolloutGuard) {
            this.rolloutGuard = rolloutGuard;
            return this;
        }

        public Builder metricsSchema(boolean metricsSchema) {
            this.metricsSchema = metricsSchema;
            return this;
        }

        public AdapterCapabilityProfile build() {
            return new AdapterCapabilityProfile(this);
        }
    }
}
