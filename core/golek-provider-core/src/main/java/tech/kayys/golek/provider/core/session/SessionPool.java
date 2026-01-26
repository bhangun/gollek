package tech.kayys.golek.provider.core.session;

import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pool of reusable sessions for a specific model
 */
public class SessionPool {

    private static final Logger LOG = Logger.getLogger(SessionPool.class);

    private final String modelId;
    private final String tenantId;
    private final SessionConfig config;
    private final BlockingQueue<Session> availableSessions;
    private final List<Session> allSessions;
    private final AtomicInteger activeCount;
    private volatile boolean shutdown;

    public SessionPool(String modelId, String tenantId, SessionConfig config) {
        this.modelId = modelId;
        this.tenantId = tenantId;
        this.config = config;
        this.availableSessions = new LinkedBlockingQueue<>();
        this.allSessions = new ArrayList<>();
        this.activeCount = new AtomicInteger(0);
        this.shutdown = false;
    }

    /**
     * Acquire a session from the pool
     */
    public Optional<Session> acquire(long timeoutMs) {
        if (shutdown) {
            return Optional.empty();
        }

        // Try to get from available pool first
        Session session = availableSessions.poll();
        if (session != null) {
            if (session.shouldClose()) {
                closeSession(session);
                return acquire(timeoutMs); // Retry
            }

            activeCount.incrementAndGet();
            session.recordAccess();
            LOG.debugf("Reused session %s for model %s", session.getSessionId(), modelId);
            return Optional.of(session);
        }

        // Create new session if under limit
        if (allSessions.size() < config.getMaxConcurrentSessions()) {
            Session newSession = createSession();
            if (newSession != null) {
                synchronized (allSessions) {
                    allSessions.add(newSession);
                }
                activeCount.incrementAndGet();
                LOG.infof("Created new session %s for model %s (total: %d)",
                        newSession.getSessionId(), modelId, allSessions.size());
                return Optional.of(newSession);
            }
        }

        // Wait for available session
        try {
            session = availableSessions.poll(timeoutMs, TimeUnit.MILLISECONDS);
            if (session != null) {
                if (session.shouldClose()) {
                    closeSession(session);
                    return Optional.empty();
                }
                activeCount.incrementAndGet();
                session.recordAccess();
                return Optional.of(session);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            LOG.warnf("Interrupted while waiting for session");
        }

        return Optional.empty();
    }

    /**
     * Release session back to pool
     */
    public void release(Session session) {
        if (session == null || shutdown) {
            return;
        }

        activeCount.decrementAndGet();

        if (session.shouldClose() || !config.isReuseEnabled()) {
            closeSession(session);
            return;
        }

        boolean offered = availableSessions.offer(session);
        if (!offered) {
            LOG.warnf("Failed to return session %s to pool, closing", session.getSessionId());
            closeSession(session);
        } else {
            LOG.debugf("Released session %s back to pool", session.getSessionId());
        }
    }

    /**
     * Cleanup idle and expired sessions
     */
    public int cleanup() {
        if (shutdown) {
            return 0;
        }

        int cleaned = 0;
        List<Session> toRemove = new ArrayList<>();

        // Check available sessions
        for (Session session : availableSessions) {
            if (session.shouldClose()) {
                toRemove.add(session);
            }
        }

        for (Session session : toRemove) {
            availableSessions.remove(session);
            closeSession(session);
            cleaned++;
        }

        LOG.debugf("Cleaned %d sessions from pool for model %s", cleaned, modelId);
        return cleaned;
    }

    /**
     * Get pool statistics
     */
    public PoolStats getStats() {
        return new PoolStats(
                allSessions.size(),
                activeCount.get(),
                availableSessions.size(),
                modelId,
                tenantId);
    }

    /**
     * Shutdown pool and close all sessions
     */
    public void shutdown() {
        if (shutdown) {
            return;
        }

        shutdown = true;

        LOG.infof("Shutting down session pool for model %s", modelId);

        synchronized (allSessions) {
            for (Session session : allSessions) {
                closeSession(session);
            }
            allSessions.clear();
        }

        availableSessions.clear();
        activeCount.set(0);
    }

    private Session createSession() {
        try {
            return new Session(modelId, tenantId, config);
        } catch (Exception e) {
            LOG.errorf(e, "Failed to create session for model %s", modelId);
            return null;
        }
    }

    private void closeSession(Session session) {
        try {
            session.close();
            synchronized (allSessions) {
                allSessions.remove(session);
            }
            LOG.debugf("Closed session %s", session.getSessionId());
        } catch (Exception e) {
            LOG.warnf(e, "Error closing session %s", session.getSessionId());
        }
    }

    public record PoolStats(
            int totalSessions,
            int activeSessions,
            int availableSessions,
            String modelId,
            String tenantId) {
        @Override
        public String toString() {
            return String.format(
                    "PoolStats{model=%s, total=%d, active=%d, available=%d}",
                    modelId, totalSessions, activeSessions, availableSessions);
        }
    }
}