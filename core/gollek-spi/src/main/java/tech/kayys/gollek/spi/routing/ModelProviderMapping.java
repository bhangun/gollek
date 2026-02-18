package tech.kayys.gollek.spi.routing;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Defines which models are available on which providers.
 * Maps model IDs to provider IDs for routing decisions.
 */
public record ModelProviderMapping(
        String modelId,
        String displayName,
        List<String> providerIds,
        String preferredProvider,
        ModelType type,
        Map<String, Object> metadata) {

    /**
     * Model type classification
     */
    public enum ModelType {
        /** ChatLLM models (GPT, Claude, Llama, etc.) */
        CHAT,
        /** Completion models */
        COMPLETION,
        /** Embedding models */
        EMBEDDING,
        /** Image generation models */
        IMAGE,
        /** Multi-modal models */
        MULTIMODAL,
        /** Code generation models */
        CODE,
        /** Custom/Other models */
        CUSTOM
    }

    public ModelProviderMapping {
        Objects.requireNonNull(modelId, "modelId is required");
        providerIds = providerIds != null ? List.copyOf(providerIds) : Collections.emptyList();
        metadata = metadata != null ? Map.copyOf(metadata) : Collections.emptyMap();
        if (type == null) {
            type = ModelType.CHAT;
        }
    }

    /**
     * Check if model is available on a provider
     */
    public boolean isAvailableOn(String providerId) {
        return providerIds.contains(providerId);
    }

    /**
     * Check if model has a preferred provider
     */
    public boolean hasPreferredProvider() {
        return preferredProvider != null && !preferredProvider.isBlank();
    }

    /**
     * Get first available provider
     */
    public String getFirstProvider() {
        return providerIds.isEmpty() ? null : providerIds.get(0);
    }

    /**
     * Create a simple mapping
     */
    public static ModelProviderMapping of(String modelId, String... providers) {
        return new ModelProviderMapping(
                modelId,
                modelId,
                List.of(providers),
                providers.length > 0 ? providers[0] : null,
                ModelType.CHAT,
                Collections.emptyMap());
    }

    /**
     * Builder for complex mappings
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelId;
        private String displayName;
        private List<String> providerIds = Collections.emptyList();
        private String preferredProvider;
        private ModelType type = ModelType.CHAT;
        private Map<String, Object> metadata = Collections.emptyMap();

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            this.displayName = modelId;
            return this;
        }

        public Builder displayName(String displayName) {
            this.displayName = displayName;
            return this;
        }

        public Builder providers(String... providers) {
            this.providerIds = List.of(providers);
            if (providers.length > 0 && preferredProvider == null) {
                this.preferredProvider = providers[0];
            }
            return this;
        }

        public Builder providers(List<String> providers) {
            this.providerIds = providers;
            return this;
        }

        public Builder preferredProvider(String provider) {
            this.preferredProvider = provider;
            return this;
        }

        public Builder type(ModelType type) {
            this.type = type;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public ModelProviderMapping build() {
            return new ModelProviderMapping(
                    modelId, displayName, providerIds, preferredProvider, type, metadata);
        }
    }
}
