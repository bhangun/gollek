package tech.kayys.gollek.inference.gguf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.AdapterSpec;
import tech.kayys.gollek.spi.inference.AdapterSpecResolver;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Resolves and validates LoRA adapter selections.
 */
@ApplicationScoped
public class LoraAdapterManager {

    private static final Logger log = Logger.getLogger(LoraAdapterManager.class);

    private final GGUFProviderConfig config;
    private final ConcurrentHashMap<String, AdapterSpec> resolved = new ConcurrentHashMap<>();
    private final AtomicInteger activeAdapters = new AtomicInteger(0);

    @Inject
    public LoraAdapterManager(GGUFProviderConfig config) {
        this.config = config;
    }

    public Optional<AdapterSpec> resolve(ProviderRequest request) {
        if (!config.loraEnabled()) {
            return Optional.empty();
        }

        Optional<AdapterSpec> parsed = AdapterSpecResolver.fromProviderRequest(request, config.loraDefaultScale());
        if (parsed.isEmpty()) {
            return Optional.empty();
        }

        AdapterSpec raw = parsed.get();
        if (!raw.isType("lora")) {
            throw new IllegalArgumentException("GGUF provider only supports adapter_type=lora");
        }
        String resolvedPath = resolveAdapterPath(raw.adapterPath());
        if (resolvedPath == null) {
            throw new IllegalArgumentException("LoRA adapter path is required when LoRA is enabled");
        }

        if (!Files.exists(Path.of(resolvedPath))) {
            throw new IllegalArgumentException("LoRA adapter not found: " + resolvedPath);
        }

        AdapterSpec normalized = new AdapterSpec(
                raw.type(),
                raw.adapterId() == null ? resolvedPath : raw.adapterId(),
                resolvedPath,
                raw.scale());

        String cacheKey = normalized.cacheKey();
        AdapterSpec existing = resolved.putIfAbsent(cacheKey, normalized);
        if (existing == null) {
            int count = activeAdapters.incrementAndGet();
            log.debugf("Registered LoRA adapter %s (active=%d)", normalized.adapterId(), count);
        }

        enforceCapacity();
        return Optional.of(existing != null ? existing : normalized);
    }

    public int activeAdapterCount() {
        return activeAdapters.get();
    }

    private String resolveAdapterPath(String adapterPath) {
        if (adapterPath == null || adapterPath.isBlank()) {
            return null;
        }
        Path raw = Path.of(adapterPath);
        if (raw.isAbsolute()) {
            return raw.normalize().toString();
        }
        Path base = Path.of(config.loraAdapterBasePath());
        return base.resolve(raw).normalize().toString();
    }

    private void enforceCapacity() {
        int max = Math.max(1, config.loraMaxActiveAdapters());
        if (resolved.size() <= max) {
            return;
        }
        // Simple bounded map behavior: evict one arbitrary entry.
        String key = resolved.keys().nextElement();
        if (resolved.remove(key) != null) {
            activeAdapters.decrementAndGet();
        }
    }
}
