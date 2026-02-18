package tech.kayys.gollek.engine.session;

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Represents an active inference session.
 * Sessions manage resources and state for model execution.
 */
public final class Session implements AutoCloseable {

    private final String sessionId;
    private final String modelId;
    private final String requestId;
    private final Instant createdAt;
    private final SessionConfig config;
    private final AtomicBoolean active;
    private final AtomicLong requestCount;
    private final AtomicLong lastAccessTime;

    private volatile Object nativeHandle; // Provider-specific handle

    public Session(String modelId, String requestId, SessionConfig config) {
        this.sessionId = UUID.randomUUID().toString();
        this.modelId = Objects.requireNonNull(modelId, "modelId");
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.config = Objects.requireNonNull(config, "config");
        this.createdAt = Instant.now();
        this.active = new AtomicBoolean(true);
        this.requestCount = new AtomicLong(0);
        this.lastAccessTime = new AtomicLong(System.currentTimeMillis());
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getModelId() {
        return modelId;
    }

    public String getRequestId() {
        return requestId;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public SessionConfig getConfig() {
        return config;
    }

    public boolean isActive() {
        return active.get();
    }

    public long getRequestCount() {
        return requestCount.get();
    }

    public long getLastAccessTime() {
        return lastAccessTime.get();
    }

    public long getIdleTimeMs() {
        return System.currentTimeMillis() - lastAccessTime.get();
    }

    public long getAgeMs() {
        return System.currentTimeMillis() - createdAt.toEpochMilli();
    }

    /**
     * Mark session as accessed
     */
    public void recordAccess() {
        requestCount.incrementAndGet();
        lastAccessTime.set(System.currentTimeMillis());
    }

    /**
     * Set native handle (provider-specific)
     */
    public void setNativeHandle(Object handle) {
        this.nativeHandle = handle;
    }

    /**
     * Get native handle
     */
    @SuppressWarnings("unchecked")
    public <T> T getNativeHandle(Class<T> type) {
        if (nativeHandle != null && type.isInstance(nativeHandle)) {
            return (T) nativeHandle;
        }
        return null;
    }

    /**
     * Check if session has native handle
     */
    public boolean hasNativeHandle() {
        return nativeHandle != null;
    }

    /**
     * Check if session is idle
     */
    public boolean isIdle() {
        return getIdleTimeMs() > config.getMaxIdleTimeMs();
    }

    /**
     * Check if session is expired
     */
    public boolean isExpired() {
        return getAgeMs() > config.getMaxAgeMs();
    }

    /**
     * Check if session should be closed
     */
    public boolean shouldClose() {
        return !active.get() || isIdle() || isExpired();
    }

    @Override
    public void close() {
        if (active.compareAndSet(true, false)) {
            // Native handle cleanup should be done by session manager
            nativeHandle = null;
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof Session session))
            return false;
        return sessionId.equals(session.sessionId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(sessionId);
    }

    @Override
    public String toString() {
        return "Session{" +
                "id='" + sessionId + '\'' +
                ", model='" + modelId + '\'' +
                ", tenant='" + requestId + '\'' +
                ", requests=" + requestCount.get() +
                ", idleMs=" + getIdleTimeMs() +
                ", active=" + active.get() +
                '}';
    }
}