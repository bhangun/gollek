package tech.kayys.gollek.safetensor.engine.backend;

import java.util.List;
import java.util.Optional;

/**
 * Control-plane view over active reusable text execution sessions.
 *
 * <p>This is intentionally separate from the hot-path execution interfaces. It
 * gives the rest of Gollek a stable way to inspect or evict reusable session
 * state without depending on the registry implementation details.
 */
public interface TextExecutionSessionControl {

    Optional<ResumableSessionDescriptor> findByKey(String sessionKey);

    Optional<ResumableSessionArtifact> findArtifactByKey(String sessionKey);

    List<ResumableSessionDescriptor> activeSessions();

    List<ResumableSessionArtifact> activeArtifacts();

    TextExecutionSessionSnapshot snapshot();

    boolean evictByKey(String sessionKey);

    int activeReusableSessions();

    int activeReservations();
}
