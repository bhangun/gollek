package tech.kayys.gollek.model.repo.kaggle;

import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelRef;

import java.net.URI;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Resolves a Kaggle model reference to a concrete artifact descriptor.
 */
public final class KaggleArtifactResolver {

    private final KaggleClient client;

    public KaggleArtifactResolver(KaggleClient client) {
        this.client = client;
    }

    /**
     * Resolve a model reference to a Kaggle artifact.
     */
    public KaggleArtifact resolve(ModelRef ref) {
        String slug = buildSlug(ref);
        String format = detectFormat(ref);
        String revision = ref.version() != null ? ref.version() : "main";

        return new KaggleArtifact(slug, format, revision);
    }

    private String buildSlug(ModelRef ref) {
        String ns = ref.namespace() != null ? ref.namespace() : "";
        String name = ref.name() != null ? ref.name() : "";
        if (ns.isBlank()) return name;
        return ns + "/" + name;
    }

    private String detectFormat(ModelRef ref) {
        Map<String, String> params = ref.parameters();
        if (params != null && params.containsKey("format")) {
            return params.get("format").toLowerCase(Locale.ROOT);
        }
        String name = ref.name() != null ? ref.name() : "";
        if (name.toLowerCase(Locale.ROOT).endsWith(".gguf")) return "gguf";
        if (name.toLowerCase(Locale.ROOT).endsWith(".safetensors")) return "safetensors";
        if (name.toLowerCase(Locale.ROOT).endsWith(".litertlm")) return "litert";
        if (name.toLowerCase(Locale.ROOT).endsWith(".bin")) return "pytorch";
        return "safetensors";
    }

    public record KaggleArtifact(String slug, String format, String revision) {}
}
