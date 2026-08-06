package tech.kayys.gollek.safetensor.engine.backend;

import java.util.Objects;

/**
 * Result of backend selection.
 *
 * <p>This keeps backend policy decisions explicit instead of hiding them in
 * provider or planner code. As more backends appear, this record can carry
 * the fallback reason and the original requested backend without changing
 * call sites again.
 */
public record TextExecutionBackendSelection(
        String requestedBackendId,
        TextExecutionBackend selectedBackend,
        String fallbackReason) {

    public TextExecutionBackendSelection {
        Objects.requireNonNull(requestedBackendId, "requestedBackendId");
        Objects.requireNonNull(selectedBackend, "selectedBackend");
    }

    public String selectedBackendId() {
        return selectedBackend.id();
    }

    public boolean fellBack() {
        return fallbackReason != null && !fallbackReason.isBlank();
    }
}
