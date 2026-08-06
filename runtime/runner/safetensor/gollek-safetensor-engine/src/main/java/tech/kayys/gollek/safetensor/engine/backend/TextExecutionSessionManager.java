package tech.kayys.gollek.safetensor.engine.backend;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import io.quarkus.arc.Arc;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * Engine-owned manager for reusable/resumable text execution sessions.
 *
 * <p>This is the boundary future backends can use to resolve, reserve, or
 * publish reusable session state without coupling the rest of the engine to the
 * concrete registry implementation.
 */
@ApplicationScoped
public class TextExecutionSessionManager implements TextExecutionSessionControl {

    private static final TextExecutionSessionRegistry FALLBACK_REGISTRY = new TextExecutionSessionRegistry();

    @Inject
    TextExecutionSessionRegistry registry;

    @Inject
    TextExecutionBackendCatalog backendCatalog;

    public TextExecutionSessionManager() {
    }

    public TextExecutionSessionManager(TextExecutionSessionRegistry registry) {
        this.registry = registry;
    }

    public TextExecutionSessionManager(
            TextExecutionSessionRegistry registry,
            TextExecutionBackendCatalog backendCatalog) {
        this.registry = registry;
        this.backendCatalog = backendCatalog;
    }

    public TextExecutionSessionReuseDecision evaluate(PreparedTextGeneration generation) {
        if (generation == null || generation.sessionPlan() == null) {
            return TextExecutionSessionReuseDecision.notReusable("No session plan");
        }
        if (!generation.sessionPlan().reusable()) {
            return TextExecutionSessionReuseDecision.notReusable(generation.sessionPlan().rationale());
        }
        if (generation.artifactPlan() == null || !generation.artifactPlan().resumeTarget()) {
            return TextExecutionSessionReuseDecision.notReusable(
                    generation.artifactPlan() != null
                            ? generation.artifactPlan().rationale()
                            : "No resumable artifact plan");
        }
        String sessionKey = generation.sessionPlan().sessionKey();
        Optional<ResumableSessionDescriptor> existing = registry().find(sessionKey);
        if (existing.isPresent()) {
            return TextExecutionSessionReuseDecision.hit(existing.get(), "Reusable session candidate found in registry");
        }
        return TextExecutionSessionReuseDecision.miss(sessionKey, "No reusable session candidate registered");
    }

    public Optional<ResumableSessionDescriptor> find(PreparedTextGeneration generation) {
        return evaluate(generation).candidateOptional();
    }

    @Override
    public Optional<ResumableSessionDescriptor> findByKey(String sessionKey) {
        return registry().find(sessionKey);
    }

    @Override
    public Optional<ResumableSessionArtifact> findArtifactByKey(String sessionKey) {
        return registry().findArtifact(sessionKey);
    }

    public TextExecutionSessionReservation reserve(PreparedTextGeneration generation) {
        TextExecutionSessionReuseDecision decision = evaluate(generation);
        return switch (decision.status()) {
            case NOT_REUSABLE -> TextExecutionSessionReservation.ephemeral(decision.rationale());
            case HIT -> TextExecutionSessionReservation.attachedExisting(
                    decision.candidateOptional().orElseThrow(),
                    decision.rationale());
            case MISS -> {
                String key = decision.sessionKey();
                if (key != null && registry().tryReserve(key)) {
                    yield TextExecutionSessionReservation.reservedNew(key, "Reserved new reusable session key");
                }
                yield TextExecutionSessionReservation.contended(key, "Session key already reserved by another execution");
            }
        };
    }

    public TextExecutionSessionAcquisition acquire(PreparedTextGeneration generation) {
        TextExecutionSessionReservation reservation = reserve(generation);
        if (reservation != null && reservation.hasCandidate()) {
            Optional<ResumableSessionArtifact> artifact = registry().findArtifact(reservation.sessionKey());
            if (artifact.isPresent()) {
                Optional<TextExecutionSession> resumed = generation.backend().resumeSession(generation, artifact.get());
                if (resumed.isPresent()) {
                    return TextExecutionSessionAcquisition.resumed(
                            resumed.get(),
                            reservation,
                            "Resumed reusable execution session from backend artifact");
                }
            }
            if (artifact.isPresent()) {
                TextExecutionSession freshSession = generation.backend().openSession(generation);
                return TextExecutionSessionAcquisition.fallbackFresh(
                        freshSession,
                        reservation,
                        "Reusable session artifact existed but backend could not resume state");
            }
            TextExecutionSession freshSession = generation.backend().openSession(generation);
            return TextExecutionSessionAcquisition.fallbackFresh(
                    freshSession,
                    reservation,
                    "Reusable session descriptor existed but no resumable artifact was registered");
        }
        return TextExecutionSessionAcquisition.fresh(
                generation.backend().openSession(generation),
                reservation,
                reservation != null ? reservation.rationale() : "Opened fresh execution session");
    }

    public void register(TextExecutionSession session) {
        registry().register(session);
    }

    public void publish(TextExecutionSession session, TextExecutionSessionReservation reservation) {
        registry().register(session);
        if (reservation != null && reservation.mode() == TextExecutionSessionReservation.Mode.RESERVED_NEW) {
            registry().releaseReservation(reservation.sessionKey());
        }
    }

    public void unregister(TextExecutionSession session) {
        registry().unregister(session);
    }

    public void release(TextExecutionSession session, TextExecutionSessionReservation reservation) {
        registry().unregister(session);
        if (reservation != null
                && (reservation.mode() == TextExecutionSessionReservation.Mode.RESERVED_NEW
                || reservation.mode() == TextExecutionSessionReservation.Mode.CONTENDED)) {
            registry().releaseReservation(reservation.sessionKey());
        }
    }

    public int activeReusableSessions() {
        return registry().size();
    }

    @Override
    public List<ResumableSessionDescriptor> activeSessions() {
        return registry().snapshot();
    }

    @Override
    public List<ResumableSessionArtifact> activeArtifacts() {
        return registry().artifactSnapshot();
    }

    @Override
    public TextExecutionSessionSnapshot snapshot() {
        return new TextExecutionSessionSnapshot(
                Instant.now(),
                registry().snapshot(),
                registry().reservationSnapshot());
    }

    @Override
    public boolean evictByKey(String sessionKey) {
        Optional<ResumableSessionArtifact> artifact = registry().findArtifact(sessionKey);
        TextExecutionBackendCatalog catalog = catalog();
        if (catalog != null) {
            artifact.ifPresent(a -> catalog.find(a.backendId()).ifPresent(backend -> backend.releaseArtifact(a)));
        }
        return registry().evict(sessionKey);
    }

    @Override
    public int activeReservations() {
        return registry().reservationCount();
    }

    private TextExecutionSessionRegistry registry() {
        if (registry != null) {
            return registry;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(TextExecutionSessionRegistry.class);
                if (instance.isAvailable()) {
                    registry = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        return registry != null ? registry : FALLBACK_REGISTRY;
    }

    private TextExecutionBackendCatalog catalog() {
        if (backendCatalog != null) {
            return backendCatalog;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(TextExecutionBackendCatalog.class);
                if (instance.isAvailable()) {
                    backendCatalog = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        return backendCatalog;
    }
}
