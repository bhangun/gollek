package tech.kayys.gollek.provider.core.routing;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelFormatDetector;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.registry.LocalModelRegistry;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.nio.file.Path;
import java.util.*;

/**
 * Format-aware routing layer that selects the correct {@link StreamingProvider}
 * for a given {@link ProviderRequest}.
 *
 * <p>Selection algorithm (in order):
 * <ol>
 *   <li>Detect the model format via {@link ModelFormatDetector} / {@link LocalModelRegistry}.</li>
 *   <li>Find all providers whose {@link ProviderCapabilities} declares support for that format.</li>
 *   <li>From those, pick the first provider for which {@link StreamingProvider#supports}
 *       returns {@code true} (providers are ordered by {@link #PROVIDER_PRIORITY}).</li>
 *   <li>If no format-specific provider is found, fall back to the legacy "supports()" walk.</li>
 * </ol>
 *
 * <p>The router itself is stateless between calls and thread-safe.
 */
@ApplicationScoped
public class FormatAwareProviderRouter {

    private static final Logger LOG = Logger.getLogger(FormatAwareProviderRouter.class);

    /** Preferred ordering of provider IDs. Providers not in this list are tried last. */
    private static final List<String> PROVIDER_PRIORITY = List.of("gguf", "safetensor", "libtorch");

    @Inject
    Instance<StreamingProvider> providers;

    @Inject
    LocalModelRegistry localModelRegistry;

    // ── Public routing API ────────────────────────────────────────────────────

    /**
     * Route a non-streaming inference request.
     */
    public Uni<InferenceResponse> route(ProviderRequest request) {
        StreamingProvider provider = selectProvider(request);
        LOG.debugf("Routing [%s] → provider=%s", request.getModel(), provider.id());
        return provider.infer(request);
    }

    /**
     * Route a streaming inference request.
     */
    public Multi<StreamChunk> routeStream(ProviderRequest request) {
        StreamingProvider provider = selectProvider(request);
        LOG.debugf("Routing stream [%s] → provider=%s", request.getModel(), provider.id());
        return provider.inferStream(request);
    }

    /**
     * Resolve the format for a model identifier without performing inference.
     *
     * @param modelId model identifier or path
     * @return detected format, or {@link Optional#empty()} when unknown
     */
    public Optional<ModelFormat> resolveFormat(String modelId) {
        return detectFormat(modelId);
    }

    // ── Selection logic ───────────────────────────────────────────────────────

    private StreamingProvider selectProvider(ProviderRequest request) {
        String modelId = request.getModel();

        Optional<ModelFormat> formatOpt = detectFormat(modelId);
        List<StreamingProvider> ordered = orderedProviders();

        if (formatOpt.isPresent()) {
            ModelFormat format = formatOpt.get();
            LOG.debugf("Detected format=%s for model=%s", format, modelId);

            Optional<StreamingProvider> byFormat = ordered.stream()
                    .filter(p -> supportsFormat(p, format))
                    .filter(p -> p.supports(modelId, request))
                    .findFirst();

            if (byFormat.isPresent()) {
                return byFormat.get();
            }

            LOG.warnf("No format-matched provider for format=%s model=%s; falling back", format, modelId);
        }

        // Generic fallback
        return ordered.stream()
                .filter(p -> p.supports(modelId, request))
                .findFirst()
                .orElseThrow(() -> new ProviderException(
                        "router",
                        "No compatible provider found for model: " + modelId
                                + (formatOpt.map(f -> " (format=" + f + ")").orElse("")),
                        null,
                        false));
    }

    private List<StreamingProvider> orderedProviders() {
        List<StreamingProvider> all = new ArrayList<>();
        providers.forEach(all::add);
        all.sort(Comparator.comparingInt(p -> {
            int idx = PROVIDER_PRIORITY.indexOf(p.id());
            return idx < 0 ? Integer.MAX_VALUE : idx;
        }));
        return all;
    }

    private Optional<ModelFormat> detectFormat(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return Optional.empty();
        }

        // 1. Registry lookup
        Optional<ModelFormat> fromRegistry = localModelRegistry.resolve(modelId)
                .map(LocalModelRegistry.ModelEntry::format)
                .filter(f -> f != ModelFormat.UNKNOWN);
        if (fromRegistry.isPresent()) {
            return fromRegistry;
        }

        // 2. Direct file detection
        try {
            Path p = Path.of(modelId);
            Optional<ModelFormat> byFile = ModelFormatDetector.detect(p);
            if (byFile.isPresent()) {
                return byFile;
            }
        } catch (Exception ignored) {}

        // 3. Extension-only
        return ModelFormatDetector.detectByExtension(modelId);
    }

    private static boolean supportsFormat(StreamingProvider provider, ModelFormat format) {
        try {
            ProviderCapabilities caps = provider.capabilities();
            return caps != null
                    && caps.getSupportedFormats() != null
                    && caps.getSupportedFormats().contains(format);
        } catch (Exception e) {
            LOG.warnf("capabilities() threw for provider=%s: %s", provider.id(), e.getMessage());
            return false;
        }
    }
}
