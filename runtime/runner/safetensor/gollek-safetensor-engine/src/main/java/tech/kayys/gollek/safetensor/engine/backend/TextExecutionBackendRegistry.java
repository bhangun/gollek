package tech.kayys.gollek.safetensor.engine.backend;

import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;

import java.util.Locale;

/**
 * Resolves the active text execution backend.
 *
 * <p>The selected backend is controlled by {@code gollek.safetensor.text_backend},
 * defaulting to {@code direct}. This keeps provider/CLI orchestration independent
 * from the actual execution engine and makes future backends pluggable.
 */
public final class TextExecutionBackendRegistry {
    public static final String BACKEND_PROPERTY = "gollek.safetensor.text_backend";
    private static final String DEFAULT_BACKEND = "direct";

    private TextExecutionBackendRegistry() {
    }

    public static TextExecutionBackend resolve(DirectInferenceEngine engine) {
        String requested = System.getProperty(BACKEND_PROPERTY, DEFAULT_BACKEND)
                .trim()
                .toLowerCase(Locale.ROOT);
        if (DEFAULT_BACKEND.equals(requested)) {
            return new DirectTextExecutionBackend(engine);
        }
        if ("direct".equals(requested)) {
            return new DirectTextExecutionBackend(engine);
        }
        return new DirectTextExecutionBackend(engine);
    }

    public static String selectedBackendId(DirectInferenceEngine engine) {
        return resolve(engine).id();
    }
}
