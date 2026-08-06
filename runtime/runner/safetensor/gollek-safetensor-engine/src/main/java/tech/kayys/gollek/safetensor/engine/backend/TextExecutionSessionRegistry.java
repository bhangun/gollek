package tech.kayys.gollek.safetensor.engine.backend;

import jakarta.enterprise.context.ApplicationScoped;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * In-memory registry for reusable/resumable execution sessions.
 *
 * <p>This is intentionally lightweight for now. It lets Gollek attach active
 * session bookkeeping to the new session descriptors without pretending that KV
 * persistence or prefix-cache restoration already exists.
 */
@ApplicationScoped
public class TextExecutionSessionRegistry {

    private final ConcurrentMap<String, ResumableSessionArtifact> sessions = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, Instant> reservations = new ConcurrentHashMap<>();

    public void register(TextExecutionSession session) {
        if (session == null || session.generation() == null || session.generation().artifactPlan() == null
                || !session.generation().artifactPlan().resumeTarget()) {
            return;
        }
        ResumableSessionArtifact artifact = session.artifact();
        if (artifact != null && artifact.reusable()) {
            sessions.put(artifact.sessionKey(), artifact);
        }
    }

    public void unregister(TextExecutionSession session) {
        ResumableSessionArtifact artifact = session.artifact();
        if (artifact != null && !artifact.resumeCapable()) {
            sessions.remove(artifact.sessionKey());
        }
    }

    public Optional<ResumableSessionDescriptor> find(String sessionKey) {
        return findArtifact(sessionKey).map(ResumableSessionArtifact::descriptor);
    }

    public Optional<ResumableSessionArtifact> findArtifact(String sessionKey) {
        return Optional.ofNullable(sessions.get(sessionKey));
    }

    public List<ResumableSessionDescriptor> snapshot() {
        return sessions.values().stream()
                .map(ResumableSessionArtifact::descriptor)
                .toList();
    }

    public List<ResumableSessionArtifact> artifactSnapshot() {
        return new ArrayList<>(sessions.values());
    }

    public List<ReservedSessionKey> reservationSnapshot() {
        List<ReservedSessionKey> snapshot = new ArrayList<>(reservations.size());
        reservations.forEach((key, reservedAt) -> snapshot.add(new ReservedSessionKey(key, reservedAt)));
        return snapshot;
    }

    public boolean evict(String sessionKey) {
        return sessionKey != null && sessions.remove(sessionKey) != null;
    }

    public boolean tryReserve(String sessionKey) {
        return reservations.putIfAbsent(sessionKey, Instant.now()) == null;
    }

    public void releaseReservation(String sessionKey) {
        if (sessionKey != null) {
            reservations.remove(sessionKey);
        }
    }

    public int size() {
        return sessions.size();
    }

    public int reservationCount() {
        return reservations.size();
    }
}
