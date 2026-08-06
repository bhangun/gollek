package tech.kayys.gollek.safetensor.engine.backend;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Catalog of available text execution backends.
 *
 * <p>This separates backend lookup from backend-selection policy so other
 * engine subsystems, including artifact eviction and future admin tooling, can
 * route work by backend id without re-implementing construction logic.
 */
@ApplicationScoped
public class TextExecutionBackendCatalog {

    @Inject
    DirectInferenceEngine engine;

    @Inject
    DirectPreparedGenerationRegistry directHandleRegistry;

    public TextExecutionBackendCatalog() {
    }

    public TextExecutionBackendCatalog(DirectInferenceEngine engine) {
        this.engine = engine;
    }

    public TextExecutionBackendCatalog(
            DirectInferenceEngine engine,
            DirectPreparedGenerationRegistry directHandleRegistry) {
        this.engine = engine;
        this.directHandleRegistry = directHandleRegistry;
    }

    public Map<String, TextExecutionBackend> available() {
        Map<String, TextExecutionBackend> candidates = new LinkedHashMap<>();
        candidates.put("direct", new DirectTextExecutionBackend(requireEngine(), directHandles()));
        return candidates;
    }

    public Optional<TextExecutionBackend> find(String backendId) {
        if (backendId == null || backendId.isBlank()) {
            return Optional.empty();
        }
        return Optional.ofNullable(available().get(backendId.trim().toLowerCase(Locale.ROOT)));
    }

    private DirectInferenceEngine requireEngine() {
        if (engine == null) {
            throw new IllegalStateException("DirectInferenceEngine is required for backend catalog");
        }
        return engine;
    }

    private DirectPreparedGenerationRegistry directHandles() {
        return directHandleRegistry != null ? directHandleRegistry : new DirectPreparedGenerationRegistry();
    }
}
