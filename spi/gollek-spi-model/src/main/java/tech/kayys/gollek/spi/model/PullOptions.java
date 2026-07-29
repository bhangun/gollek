package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.spec.*;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

/**
 * Options for pulling a model from a repository.
 */
public class PullOptions {

    public static final PullOptions DEFAULT = builder().build();

    private final String revision;
    private final boolean force;

    private PullOptions(Builder builder) {
        this.revision = builder.revision;
        this.force = builder.force;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getRevision() {
        return revision;
    }

    public boolean isForce() {
        return force;
    }

    public static class Builder {
        private String revision;
        private boolean force = false;

        public Builder revision(String revision) {
            this.revision = revision;
            return this;
        }

        public Builder force(boolean force) {
            this.force = force;
            return this;
        }

        public PullOptions build() {
            return new PullOptions(this);
        }
    }
}
