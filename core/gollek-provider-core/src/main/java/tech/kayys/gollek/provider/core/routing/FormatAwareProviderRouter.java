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
import tech.kayys.gollek.spi.registry.LocalModelRegistry;
import tech.kayys.gollek.spi.registry.ModelEntry;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.StreamingProvider;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Format-aware routing layer that selects the correct {@link StreamingProvider}
 * for a given {@link ProviderRequest}.
 *
 * <p>
 * Selection algorithm (in order):
 * <ol>
 * <li>Detect the model format via {@link ModelFormatDetector} /
 * {@link LocalModelRegistry}.</li>
 * <li>Find all providers whose {@link ProviderCapabilities} declares support
 * for that format.</li>
 * <li>From those, pick the first provider for which
 * {@link StreamingProvider#supports}
 * returns {@code true} (providers are ordered by
 * {@link #PROVIDER_PRIORITY}).</li>
 * <li>If no format-specific provider is found, fall back to the legacy
 * "supports()" walk.</li>
 * </ol>
 *
 * <p>
 * The router itself is stateless between calls and thread-safe.
 */
@ApplicationScoped
public class FormatAwareProviderRouter {

    private static final Logger LOG = Logger.getLogger(FormatAwareProviderRouter.class);

    /**
     * Preferred ordering of provider IDs. Providers not in this list are tried
     * last.
     */
    private static final List<String> PROVIDER_PRIORITY = List.of("gguf", "safetensor", "onnx", "libtorch", "tflite",
            "litertlm");

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
        LOG.debugf("Routing [%s] → provider=%s (gguf_requested=%b)",
                request.getModel(),
                provider.id(),
                request.getParameter("gguf", Boolean.class).orElse(false));
        return provider.infer(request);
    }

    /**
     * Route a streaming inference request.
     */
    public Multi<StreamingInferenceChunk> routeStream(ProviderRequest request) {
        StreamingProvider provider = selectProvider(request);
        LOG.debugf("Routing stream [%s] → provider=%s (gguf_requested=%b)",
                request.getModel(),
                provider.id(),
                request.getParameter("gguf", Boolean.class).orElse(false));
        return provider.inferStream(request);
    }

    /**
     * Resolve the format for a model identifier without performing inference.
     *
     * @param modelId model identifier or path
     * @return detected format, or {@link Optional#empty()} when unknown
     */
    public Optional<ModelFormat> resolveFormat(String modelId) {
        return detectFormat(modelId, false);
    }

    // ── Selection logic ───────────────────────────────────────────────────────

    private StreamingProvider selectProvider(ProviderRequest request) {
        String modelId = request.getModel();
        boolean preferGguf = request.getParameter("gguf", Boolean.class).orElse(false);
        Optional<String> preferredProviderId = request.getPreferredProvider();

        Optional<ModelFormat> formatOpt = detectFormat(modelId, preferGguf);
        List<StreamingProvider> ordered = orderedProviders(request);

        // 1. If preferred provider is specified and it supports the model, use it strictly.
        if (preferredProviderId.isPresent()) {
            String preferred = preferredProviderId.get();
            Optional<StreamingProvider> pinned = ordered.stream()
                    .filter(p -> p.id().equalsIgnoreCase(preferred))
                    .filter(p -> p.supports(modelId, request))
                    .findFirst();
            if (pinned.isPresent()) {
                LOG.debugf("Routing to pinned preferred provider: %s", preferred);
                return pinned.get();
            } else {
                LOG.warnf("Preferred provider '%s' not found or does not support model %s", preferred, modelId);
            }
        }

        // 2. Format-aware selection
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
            LOG.debugf("No provider supports detected format=%s for model=%s; falling back to generic supports()",
                    format, modelId);
        }

        // 3. Generic fallback
        return ordered.stream()
                .filter(p -> p.supports(modelId, request))
                .findFirst()
                .orElseThrow(() -> new ProviderException(
                        "router",
                        "No compatible provider found for model: " + modelId
                                + (formatOpt.map(f -> " (format=" + f + ")").orElse("")),
                        null,
                        null, // ErrorCode
                        false));
    }

    private List<StreamingProvider> orderedProviders(ProviderRequest request) {
        String modelId = request.getModel();
        boolean preferGguf = request.getParameter("gguf", Boolean.class).orElse(false);
        Optional<String> preferredProviderId = request.getPreferredProvider();
        
        // Detect Stable Diffusion pipeline to steer away from text-only safetensor provider.
        // We resolve the model path via the registry first to handle relative IDs/aliases.
        boolean isSd = false;
        try {
            Path actualPath = localModelRegistry.resolve(modelId)
                    .map(ModelEntry::physicalPath)
                    .orElseGet(() -> {
                        try { return Path.of(modelId); } catch (Exception e) { return null; }
                    });
            
            if (actualPath != null) {
                isSd = ModelFormatDetector.isStableDiffusion(actualPath);
            }
        } catch (Exception ignored) {}

        List<StreamingProvider> all = new ArrayList<>();
        providers.forEach(all::add);

        List<String> priority = new ArrayList<>(PROVIDER_PRIORITY);
        
        // If it's a Stable Diffusion model, move 'onnx' to the front unless it's already there
        if (isSd && priority.contains("onnx")) {
            priority.remove("onnx");
            priority.add(0, "onnx");
            LOG.debugf("Stable Diffusion detected for model %s; prioritizing 'onnx' provider", modelId);
        }

        // Adjust priority based on flags
        if (preferGguf && priority.contains("gguf")) {
            priority.remove("gguf");
            priority.add(0, "gguf");
        }
        
        // Adjust priority based on explicit preference
        if (preferredProviderId.isPresent()) {
            String preferred = preferredProviderId.get();
            if (priority.contains(preferred)) {
                priority.remove(preferred);
            }
            priority.add(0, preferred);
        }

        all.sort(Comparator.comparingInt(p -> {
            int idx = priority.indexOf(p.id());
            return idx < 0 ? Integer.MAX_VALUE : idx;
        }));
        return all;
    }

    private Optional<ModelFormat> detectFormat(String modelId, boolean preferGguf) {
        if (modelId == null || modelId.isBlank()) {
            return Optional.empty();
        }

        // 1. Registry lookup (Multi-format aware)
        List<ModelEntry> entries = localModelRegistry.resolveAll(modelId);
        if (!entries.isEmpty()) {
            // Priority-aware selection:
            // If we have both SAFETENSORS and GGUF, pick based on preferGguf flag.
            if (preferGguf) {
                // Try for GGUF first
                Optional<ModelFormat> gguf = entries.stream()
                        .map(ModelEntry::format)
                        .filter(f -> f == ModelFormat.GGUF)
                        .findFirst();
                if (gguf.isPresent())
                    return gguf;
            } else {
                // Try for SAFETENSORS first
                Optional<ModelFormat> st = entries.stream()
                        .map(ModelEntry::format)
                        .filter(f -> f == ModelFormat.SAFETENSORS)
                        .findFirst();
                if (st.isPresent())
                    return st;
            }

            // Fallback to first known format in the list
            return entries.stream()
                    .map(ModelEntry::format)
                    .filter(f -> f != ModelFormat.UNKNOWN)
                    .findFirst();
        }

        // 2. Direct file detection
        try {
            Path p = Path.of(modelId);
            Optional<ModelFormat> byFile = ModelFormatDetector.detect(p);
            if (byFile.isPresent()) {
                return byFile;
            }
        } catch (Exception ignored) {
        }

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
