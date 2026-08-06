package tech.kayys.gollek.safetensor.engine.backend;

import jakarta.enterprise.context.ApplicationScoped;

import java.time.Instant;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Process-local store for direct-backend prepared-generation handles.
 *
 * <p>This intentionally starts with in-process residency only. The purpose is
 * to make {@code resumeSession(...)} real for one backend before Gollek grows
 * true persisted KV snapshots or backend-native prefix caches.
 */
@ApplicationScoped
public class DirectPreparedGenerationRegistry {

    private static final ConcurrentMap<String, DirectPreparedGenerationHandle> HANDLES = new ConcurrentHashMap<>();

    public DirectPreparedGenerationHandle publish(PreparedTextGeneration generation, long[] inputIds) {
        String sessionKey = generation.sessionPlan().sessionKey() != null
                ? generation.sessionPlan().sessionKey()
                : generation.promptFingerprint();
        DirectPreparedGenerationHandle handle = new DirectPreparedGenerationHandle(
                sessionKey,
                generation.promptFingerprint(),
                inputIds,
                generation,
                Instant.now());
        HANDLES.put(sessionKey, handle);
        return handle;
    }

    public Optional<DirectPreparedGenerationHandle> find(String sessionKey) {
        return Optional.ofNullable(HANDLES.get(sessionKey));
    }

    public boolean evict(String sessionKey) {
        return sessionKey != null && HANDLES.remove(sessionKey) != null;
    }
}
