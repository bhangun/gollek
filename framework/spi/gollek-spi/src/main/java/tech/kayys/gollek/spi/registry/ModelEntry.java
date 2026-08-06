package tech.kayys.gollek.spi.registry;

import tech.kayys.alkhawarizm.core.model.ModelFormat;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;

/**
 * Model entry in the local registry.
 *
 * @param modelId      Unique model identifier
 * @param name         Human-readable model name
 * @param format       Model format enum
 * @param physicalPath Path to model file
 * @param sizeBytes    Model file size in bytes
 * @param provider     Provider ID
 * @param metadata     Additional metadata
 * @param registeredAt Registration timestamp
 *
 * @since 2.1.0
 */
public record ModelEntry(
        String modelId,
        String name,
        ModelFormat format,
        Path physicalPath,
        long sizeBytes,
        String provider,
        Map<String, Object> metadata,
        Instant registeredAt) {

    public ModelEntry {
        if (registeredAt == null) {
            registeredAt = Instant.now();
        }
        if (metadata == null) {
            metadata = Map.of();
        }
    }

    /** Alias for physicalPath() for backwards compatibility. */
    public Path path() {
        return physicalPath();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelId;
        private String name;
        private ModelFormat format;
        private Path physicalPath;
        private long sizeBytes;
        private String provider;
        private Map<String, Object> metadata = Map.of();
        private Instant registeredAt;

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        public Builder physicalPath(Path physicalPath) {
            this.physicalPath = physicalPath;
            return this;
        }

        public Builder sizeBytes(long sizeBytes) {
            this.sizeBytes = sizeBytes;
            return this;
        }

        public Builder provider(String provider) {
            this.provider = provider;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder registeredAt(Instant registeredAt) {
            this.registeredAt = registeredAt;
            return this;
        }

        public ModelEntry build() {
            return new ModelEntry(
                    modelId, name, format, physicalPath, sizeBytes, provider, metadata, registeredAt);
        }
    }
}
