package tech.kayys.golek.spi.provider;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import tech.kayys.golek.spi.model.DeviceType;
import tech.kayys.golek.spi.model.ModelFormat;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable provider capabilities descriptor
 */
public final class ProviderCapabilities {

    private final boolean streaming;
    private final boolean functionCalling;
    private final boolean multimodal;
    private final boolean embeddings;
    private final int maxContextTokens;
    private final int maxOutputTokens;
    private final Set<String> supportedModels;
    private final List<String> supportedLanguages;
    private final Set<String> features;
    private final boolean toolCalling;
    private final boolean structuredOutputs;
    private final Set<ModelFormat> supportedFormats;
    private final Set<DeviceType> supportedDevices;

    @JsonCreator
    public ProviderCapabilities(
            @JsonProperty("streaming") boolean streaming,
            @JsonProperty("functionCalling") boolean functionCalling,
            @JsonProperty("multimodal") boolean multimodal,
            @JsonProperty("embeddings") boolean embeddings,
            @JsonProperty("maxContextTokens") int maxContextTokens,
            @JsonProperty("maxOutputTokens") int maxOutputTokens,
            @JsonProperty("supportedModels") Set<String> supportedModels,
            @JsonProperty("supportedLanguages") List<String> supportedLanguages,
            @JsonProperty("features") Set<String> features,
            @JsonProperty("toolCalling") boolean toolCalling,
            @JsonProperty("structuredOutputs") boolean structuredOutputs,
            @JsonProperty("supportedFormats") Set<ModelFormat> supportedFormats,
            @JsonProperty("supportedDevices") Set<DeviceType> supportedDevices) {
        this.streaming = streaming;
        this.functionCalling = functionCalling;
        this.multimodal = multimodal;
        this.embeddings = embeddings;
        this.maxContextTokens = maxContextTokens;
        this.maxOutputTokens = maxOutputTokens;
        this.supportedModels = supportedModels != null
                ? Set.copyOf(supportedModels)
                : Collections.emptySet();
        this.supportedLanguages = supportedLanguages != null
                ? List.copyOf(supportedLanguages)
                : Collections.emptyList();
        this.features = features != null
                ? Set.copyOf(features)
                : Collections.emptySet();
        // Initialize fields used in supports() method
        this.toolCalling = toolCalling;
        this.structuredOutputs = structuredOutputs;
        this.supportedFormats = supportedFormats != null
                ? Set.copyOf(supportedFormats)
                : Collections.emptySet();
        this.supportedDevices = supportedDevices != null
                ? Set.copyOf(supportedDevices)
                : Collections.emptySet();
    }

    public boolean supports(ProviderFeature feature) {
        return switch (feature) {
            case STREAMING -> streaming;
            case TOOL_CALLING -> toolCalling;
            case MULTIMODAL -> multimodal;
            case STRUCTURED_OUTPUTS -> structuredOutputs;
        };
    }

    // Getters
    public boolean isStreaming() {
        return streaming;
    }

    public boolean isFunctionCalling() {
        return functionCalling;
    }

    public boolean isMultimodal() {
        return multimodal;
    }

    public boolean isEmbeddings() {
        return embeddings;
    }

    public int getMaxContextTokens() {
        return maxContextTokens;
    }

    public int getMaxOutputTokens() {
        return maxOutputTokens;
    }

    public Set<String> getSupportedModels() {
        return supportedModels;
    }

    public List<String> getSupportedLanguages() {
        return supportedLanguages;
    }

    public boolean isToolCalling() {
        return toolCalling;
    }

    public boolean isStructuredOutputs() {
        return structuredOutputs;
    }

    public Set<ModelFormat> getSupportedFormats() {
        return supportedFormats;
    }

    public Set<DeviceType> getSupportedDevices() {
        return supportedDevices;
    }

    public boolean supportsModel(String model) {
        return supportedModels.isEmpty() || supportedModels.contains(model);
    }

    public boolean hasFeature(String feature) {
        return features.contains(feature);
    }

    public Set<String> getFeatures() {
        return features;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean streaming = false;
        private boolean functionCalling = false;
        private boolean multimodal = false;
        private boolean embeddings = false;
        private int maxContextTokens = 4096;
        private int maxOutputTokens = 2048;
        private Set<String> supportedModels = Collections.emptySet();
        private List<String> supportedLanguages = List.of("en");
        private Set<String> features = Collections.emptySet();
        private boolean toolCalling = false;
        private boolean structuredOutputs = false;
        private Set<ModelFormat> supportedFormats = Collections.emptySet();
        private Set<DeviceType> supportedDevices = Collections.emptySet();

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder functionCalling(boolean functionCalling) {
            this.functionCalling = functionCalling;
            return this;
        }

        public Builder multimodal(boolean multimodal) {
            this.multimodal = multimodal;
            return this;
        }

        public Builder embeddings(boolean embeddings) {
            this.embeddings = embeddings;
            return this;
        }

        public Builder maxContextTokens(int maxContextTokens) {
            this.maxContextTokens = maxContextTokens;
            return this;
        }

        public Builder maxOutputTokens(int maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder supportedModels(Set<String> supportedModels) {
            this.supportedModels = Set.copyOf(supportedModels);
            return this;
        }

        public Builder supportedLanguages(List<String> supportedLanguages) {
            this.supportedLanguages = List.copyOf(supportedLanguages);
            return this;
        }

        public Builder features(Set<String> features) {
            this.features = Set.copyOf(features);
            return this;
        }

        public Builder toolCalling(boolean toolCalling) {
            this.toolCalling = toolCalling;
            return this;
        }

        public Builder structuredOutputs(boolean structuredOutputs) {
            this.structuredOutputs = structuredOutputs;
            return this;
        }

        public Builder supportedFormats(Set<ModelFormat> supportedFormats) {
            this.supportedFormats = Set.copyOf(supportedFormats);
            return this;
        }

        public Builder supportedDevices(Set<DeviceType> supportedDevices) {
            this.supportedDevices = Set.copyOf(supportedDevices);
            return this;
        }

        public ProviderCapabilities build() {
            return new ProviderCapabilities(
                    streaming, functionCalling, multimodal, embeddings,
                    maxContextTokens, maxOutputTokens, supportedModels,
                    supportedLanguages, features, toolCalling, structuredOutputs,
                    supportedFormats, supportedDevices);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof ProviderCapabilities that))
            return false;
        return streaming == that.streaming &&
                functionCalling == that.functionCalling &&
                multimodal == that.multimodal &&
                maxContextTokens == that.maxContextTokens;
    }

    @Override
    public int hashCode() {
        return Objects.hash(streaming, functionCalling, multimodal, maxContextTokens);
    }

    @Override
    public String toString() {
        return "ProviderCapabilities{" +
                "streaming=" + streaming +
                ", functionCalling=" + functionCalling +
                ", multimodal=" + multimodal +
                ", maxContextTokens=" + maxContextTokens +
                ", features=" + features.size() +
                '}';
    }
}