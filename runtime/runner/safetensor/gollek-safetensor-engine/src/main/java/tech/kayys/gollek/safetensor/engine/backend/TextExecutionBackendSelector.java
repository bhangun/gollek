package tech.kayys.gollek.safetensor.engine.backend;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Selects the active text execution backend according to runtime policy.
 *
 * <p>The current runtime only has one concrete backend, but the selection logic
 * is centralized now so future backends can participate without pushing
 * resolution policy back into planners, providers, or CLI code.
 */
@ApplicationScoped
public class TextExecutionBackendSelector {
    public static final String BACKEND_PROPERTY = "gollek.safetensor.text_backend";
    private static final String DEFAULT_BACKEND = "direct";

    @Inject
    DirectInferenceEngine engine;

    @Inject
    TextExecutionBackendCatalog backendCatalog;

    public TextExecutionBackendSelector() {
    }

    public TextExecutionBackendSelector(DirectInferenceEngine engine) {
        this.backendCatalog = new TextExecutionBackendCatalog(engine);
    }

    public TextExecutionBackendSelection select() {
        String requested = System.getProperty(BACKEND_PROPERTY, DEFAULT_BACKEND)
                .trim()
                .toLowerCase(Locale.ROOT);

        Map<String, TextExecutionBackend> candidates = catalog().available();

        TextExecutionBackend requestedBackend = candidates.get(requested);
        if (requestedBackend != null && requestedBackend.isAvailable()) {
            return new TextExecutionBackendSelection(requested, requestedBackend, null);
        }

        TextExecutionBackend fallback = candidates.get(DEFAULT_BACKEND);
        if (fallback != null && fallback.isAvailable()) {
            String reason = requestedBackend == null
                    ? "unknown backend '%s'; using '%s'".formatted(requested, DEFAULT_BACKEND)
                    : "backend '%s' is not available; using '%s'".formatted(requested, DEFAULT_BACKEND);
            return new TextExecutionBackendSelection(requested, fallback, requested.equals(DEFAULT_BACKEND) ? null : reason);
        }

        throw new IllegalStateException("No available text execution backend");
    }

    public TextExecutionBackend resolve() {
        return select().selectedBackend();
    }

    private TextExecutionBackendCatalog catalog() {
        return backendCatalog != null ? backendCatalog : new TextExecutionBackendCatalog(engine);
    }
}
