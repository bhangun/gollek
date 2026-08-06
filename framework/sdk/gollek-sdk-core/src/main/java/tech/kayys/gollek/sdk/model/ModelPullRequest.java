package tech.kayys.gollek.sdk.model;

/**
 * Request object for pulling a model with various options.
 */
public record ModelPullRequest(
    String modelSpec,
    String revision,
    String format,
    boolean force,
    boolean convertIfNecessary,
    String quantization,
    String outType
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelSpec;
        private String revision;
        private String format;
        private boolean force = false;
        private boolean convertIfNecessary = true;
        private String quantization = "Q4_K_M";
        private String outType;

        public Builder modelSpec(String modelSpec) {
            this.modelSpec = modelSpec;
            return this;
        }

        public Builder revision(String revision) {
            this.revision = revision;
            return this;
        }

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder force(boolean force) {
            this.force = force;
            return this;
        }

        public Builder convertIfNecessary(boolean convertIfNecessary) {
            this.convertIfNecessary = convertIfNecessary;
            return this;
        }

        public Builder quantization(String quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder outType(String outType) {
            this.outType = outType;
            return this;
        }

        public ModelPullRequest build() {
            if (modelSpec == null || modelSpec.isBlank()) {
                throw new IllegalArgumentException("modelSpec is required");
            }
            return new ModelPullRequest(modelSpec, revision, format, force, convertIfNecessary, quantization, outType);
        }
    }
}
