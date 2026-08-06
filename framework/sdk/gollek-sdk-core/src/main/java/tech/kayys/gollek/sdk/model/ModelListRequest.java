package tech.kayys.gollek.sdk.model;

/**
 * Request object for listing models with filtering and pagination.
 */
public record ModelListRequest(
        int offset,
        int limit,
        boolean runnableOnly,
        String format,
        String namespace,
        boolean dedupe,
        boolean sort) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int offset = 0;
        private int limit = 50;
        private boolean runnableOnly = false;
        private String format;
        private String namespace;
        private boolean dedupe = true;
        private boolean sort = true;

        public Builder offset(int offset) {
            this.offset = offset;
            return this;
        }

        public Builder limit(int limit) {
            this.limit = limit;
            return this;
        }

        public Builder runnableOnly(boolean runnableOnly) {
            this.runnableOnly = runnableOnly;
            return this;
        }

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder namespace(String namespace) {
            this.namespace = namespace;
            return this;
        }

        public Builder dedupe(boolean dedupe) {
            this.dedupe = dedupe;
            return this;
        }

        public Builder sort(boolean sort) {
            this.sort = sort;
            return this;
        }

        public ModelListRequest build() {
            return new ModelListRequest(offset, limit, runnableOnly, format, namespace, dedupe, sort);
        }
    }
}
