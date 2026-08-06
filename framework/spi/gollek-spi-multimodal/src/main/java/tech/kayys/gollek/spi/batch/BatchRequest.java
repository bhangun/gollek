package tech.kayys.gollek.spi.batch;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Batch request for multimodal inference.
 * Wraps individual requests for batch processing.
 */
public class BatchRequest {

    private final String id;
    private final Object request;
    private final CompletableFuture<Object> future;
    private final long enqueueTime;
    private final int priority;

    private BatchRequest(Builder builder) {
        this.id = builder.id;
        this.request = builder.request;
        this.future = builder.future;
        this.enqueueTime = System.currentTimeMillis();
        this.priority = builder.priority;
    }

    /**
     * Get request ID.
     */
    public String getId() {
        return id;
    }

    /**
     * Get the wrapped request.
     */
    @SuppressWarnings("unchecked")
    public <T> T getRequest() {
        return (T) request;
    }

    /**
     * Get the future for completion.
     */
    public CompletableFuture<Object> getFuture() {
        return future;
    }

    /**
     * Complete the request successfully.
     */
    public void complete(Object response) {
        future.complete(response);
    }

    /**
     * Complete the request with error.
     */
    public void completeExceptionally(Throwable error) {
        future.completeExceptionally(error);
    }

    /**
     * Get time spent in queue (ms).
     */
    public long getQueueTime() {
        return System.currentTimeMillis() - enqueueTime;
    }

    /**
     * Get priority level.
     */
    public int getPriority() {
        return priority;
    }

    /**
     * Check if request is expired.
     */
    public boolean isExpired(long timeoutMs) {
        return getQueueTime() > timeoutMs;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        BatchRequest that = (BatchRequest) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    /**
     * Create a new batch request builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for BatchRequest.
     */
    public static class Builder {
        private String id;
        private Object request;
        private CompletableFuture<Object> future;
        private int priority = 0;

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder request(Object request) {
            this.request = request;
            return this;
        }

        public Builder future(CompletableFuture<Object> future) {
            this.future = future;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public BatchRequest build() {
            Objects.requireNonNull(id, "id is required");
            Objects.requireNonNull(request, "request is required");
            Objects.requireNonNull(future, "future is required");

            if (future == null) {
                future = new CompletableFuture<>();
            }

            return new BatchRequest(this);
        }
    }
}
