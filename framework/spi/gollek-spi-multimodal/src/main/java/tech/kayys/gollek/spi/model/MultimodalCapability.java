package tech.kayys.alkhawarizm.spi.model;

import java.util.Collections;
import java.util.Set;

/**
 * Capability descriptor for a multimodal provider.
 */
public class MultimodalCapability {

    private final String modelId;
    private final Set<ModalityType> inputModalities;
    private final Set<ModalityType> outputModalities;
    private final boolean supportsStreaming;
    private final long maxImageSizeBytes;
    private final long maxAudioSizeBytes;
    private final long maxVideoSizeBytes;
    private final int maxAudioDurationSecs;
    private final int maxVideoDurationSecs;
    private final int maxPartsPerRequest;
    private final boolean supportsDocuments;
    private final Set<String> supportedMimeTypes;

    private MultimodalCapability(Builder builder) {
        this.modelId = builder.modelId;
        this.inputModalities = builder.inputModalities;
        this.outputModalities = builder.outputModalities;
        this.supportsStreaming = builder.supportsStreaming;
        this.maxImageSizeBytes = builder.maxImageSizeBytes;
        this.maxAudioSizeBytes = builder.maxAudioSizeBytes;
        this.maxVideoSizeBytes = builder.maxVideoSizeBytes;
        this.maxAudioDurationSecs = builder.maxAudioDurationSecs;
        this.maxVideoDurationSecs = builder.maxVideoDurationSecs;
        this.maxPartsPerRequest = builder.maxPartsPerRequest;
        this.supportsDocuments = builder.supportsDocuments;
        this.supportedMimeTypes = builder.supportedMimeTypes;
    }

    public String getModelId() { return modelId; }
    public Set<ModalityType> getInputModalities() { return inputModalities; }
    public Set<ModalityType> getOutputModalities() { return outputModalities; }
    public boolean isSupportsStreaming() { return supportsStreaming; }
    public long getMaxImageSizeBytes() { return maxImageSizeBytes; }
    public long getMaxAudioSizeBytes() { return maxAudioSizeBytes; }
    public long getMaxVideoSizeBytes() { return maxVideoSizeBytes; }
    public int getMaxAudioDurationSecs() { return maxAudioDurationSecs; }
    public int getMaxVideoDurationSecs() { return maxVideoDurationSecs; }
    public int getMaxPartsPerRequest() { return maxPartsPerRequest; }
    public boolean isSupportsDocuments() { return supportsDocuments; }
    public Set<String> getSupportedMimeTypes() { return supportedMimeTypes; }

    public static Builder builder(String modelId) {
        return new Builder(modelId);
    }

    public static class Builder {
        private final String modelId;
        private Set<ModalityType> inputModalities = Set.of(ModalityType.TEXT);
        private Set<ModalityType> outputModalities = Set.of(ModalityType.TEXT);
        private boolean supportsStreaming = false;
        private long maxImageSizeBytes = 10 * 1024 * 1024;
        private long maxAudioSizeBytes = 20 * 1024 * 1024;
        private long maxVideoSizeBytes = 100 * 1024 * 1024;
        private int maxAudioDurationSecs = 300;
        private int maxVideoDurationSecs = 600;
        private int maxPartsPerRequest = 10;
        private boolean supportsDocuments = false;
        private Set<String> supportedMimeTypes = Collections.emptySet();

        private Builder(String modelId) {
            this.modelId = modelId;
        }

        public Builder inputModalities(ModalityType... modalities) {
            this.inputModalities = Set.of(modalities);
            return this;
        }

        public Builder outputModalities(ModalityType... modalities) {
            this.outputModalities = Set.of(modalities);
            return this;
        }

        public Builder supportsStreaming(boolean supportsStreaming) {
            this.supportsStreaming = supportsStreaming;
            return this;
        }

        public Builder maxImageSizeBytes(long maxImageSizeBytes) {
            this.maxImageSizeBytes = maxImageSizeBytes;
            return this;
        }

        public Builder supportsDocuments(boolean supportsDocuments) {
            this.supportsDocuments = supportsDocuments;
            return this;
        }

        public MultimodalCapability build() {
            return new MultimodalCapability(this);
        }
    }
}
