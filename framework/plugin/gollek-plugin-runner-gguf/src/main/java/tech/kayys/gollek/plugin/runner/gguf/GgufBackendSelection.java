package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.ModelLoadRequest;
import tech.kayys.gollek.plugin.runner.RunnerContext;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

record GgufBackendSelection(
        Backend backend,
        String requestedValue,
        String source,
        String normalizedValue,
        boolean explicit) {
    enum Backend {
        JAVA,
        LLAMACPP
    }

    private static final String DEFAULT_SOURCE = "default:auto";

    static GgufBackendSelection resolve(ModelLoadRequest request, RunnerContext context) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(context, "context");

        Candidate candidate = firstCandidate(request.getMetadata(), context);
        if (candidate.value().isEmpty()) {
            return auto(DEFAULT_SOURCE, "");
        }

        String raw = candidate.value().get();
        String normalized = normalize(raw);
        return switch (normalized) {
            case "", "auto", "default" -> auto(candidate.source(), raw);
            case "java", "javanative", "java-native" ->
                    new GgufBackendSelection(Backend.JAVA, raw, candidate.source(), normalized, true);
            case "llamacpp", "llama.cpp", "llama-cpp", "llama_cpp", "native", "binding" ->
                    new GgufBackendSelection(Backend.LLAMACPP, raw, candidate.source(), normalized, true);
            default -> new GgufBackendSelection(
                    Backend.LLAMACPP,
                    raw,
                    candidate.source(),
                    normalized,
                    false);
        };
    }

    boolean requestedJavaNative() {
        return backend == Backend.JAVA;
    }

    boolean requestedLlamaCpp() {
        return backend == Backend.LLAMACPP;
    }

    private static GgufBackendSelection auto(String source, String raw) {
        return new GgufBackendSelection(Backend.LLAMACPP, raw, source, "auto", false);
    }

    private static Candidate firstCandidate(Map<String, Object> requestMetadata, RunnerContext context) {
        Optional<String> requestPlugin = stringValue(requestMetadata.get("plugin"));
        if (requestPlugin.isPresent() && isBackendToken(requestPlugin.get())) {
            return new Candidate(requestPlugin, "request.metadata.plugin");
        }

        Optional<String> requestBackend = stringValue(requestMetadata.get("gguf.backend"))
                .or(() -> stringValue(requestMetadata.get("backend")));
        if (requestBackend.isPresent()) {
            return new Candidate(requestBackend, "request.metadata.gguf.backend");
        }

        Optional<String> contextBackend = context.getMetadataValue("gguf.backend")
                .flatMap(GgufBackendSelection::stringValue)
                .or(() -> context.getMetadataValue("backend").flatMap(GgufBackendSelection::stringValue));
        if (contextBackend.isPresent()) {
            return new Candidate(contextBackend, "context.metadata.gguf.backend");
        }

        Optional<String> parameterBackend = context.getParameter("gguf.backend")
                .flatMap(GgufBackendSelection::stringValue)
                .or(() -> context.getParameter("backend").flatMap(GgufBackendSelection::stringValue));
        if (parameterBackend.isPresent()) {
            return new Candidate(parameterBackend, "context.parameter.gguf.backend");
        }

        return new Candidate(Optional.empty(), DEFAULT_SOURCE);
    }

    private static Optional<String> stringValue(Object value) {
        if (value == null) {
            return Optional.empty();
        }
        return Optional.of(value.toString().trim());
    }

    private static boolean isBackendToken(String value) {
        return switch (normalize(value)) {
            case "", "auto", "default", "java", "javanative", "java-native",
                    "llamacpp", "llama.cpp", "llama-cpp", "llama_cpp", "native", "binding" -> true;
            default -> false;
        };
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private record Candidate(Optional<String> value, String source) {
    }
}
