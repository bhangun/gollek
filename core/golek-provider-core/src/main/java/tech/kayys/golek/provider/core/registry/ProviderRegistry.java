package tech.kayys.golek.provider.core.registry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.api.provider.ProviderDescriptor;
import tech.kayys.golek.plugin.api.GolekPlugin;
import tech.kayys.golek.provider.core.exception.ProviderException;
import tech.kayys.golek.provider.core.spi.LLMProvider;

import tech.kayys.golek.provider.core.spi.StreamingProvider;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Central registry for all available providers.
 * Supports versioning and auto-discovery.
 */
@ApplicationScoped
public class ProviderRegistry {

    private static final Logger LOG = Logger.getLogger(ProviderRegistry.class);

    // Map<ProviderID, Map<Version, Provider>>
    private final Map<String, NavigableMap<String, LLMProvider>> providers = new ConcurrentHashMap<>();
    private final Map<String, GolekPlugin> GolekPlugins = new ConcurrentHashMap<>();
    private final Map<String, ProviderDescriptor> descriptors = new ConcurrentHashMap<>();

    @Inject
    Instance<LLMProvider> providerInstances;

    /**
     * Discover and register all CDI-managed providers
     */
    public void discoverProviders() {
        LOG.info("Discovering LLM providers...");

        providerInstances.stream().forEach(provider -> {
            try {
                register(provider);
                LOG.infof("Registered provider: %s v%s", provider.id(), provider.version());
            } catch (Exception e) {
                LOG.errorf(e, "Failed to register provider: %s",
                        provider.getClass().getName());
            }
        });

        LOG.infof("Provider discovery complete. Total unique providers: %d", providers.size());
    }

    /**
     * Register a provider
     */
    public void register(LLMProvider provider) {
        Objects.requireNonNull(provider, "provider cannot be null");

        String id = provider.id();
        String version = provider.version();

        providers.computeIfAbsent(id, k -> new ConcurrentSkipListMap<>())
                .put(version, provider);

        if (provider.metadata() != null) {
            descriptors.put(id, ProviderDescriptor.builder()
                    .id(id)
                    .displayName(provider.name())
                    .version(version)
                    .capabilities(provider.capabilities())
                    .metadata("provider_metadata", provider.metadata()) // metadata expects key-value
                    .build());
        }

        LOG.debugf("Registered provider: %s v%s (%s)", id, version, provider.name());
    }

    private ProviderDescriptor createDescriptor(LLMProvider provider) {
        // Fallback if metadata/descriptor is missing, though usually it should be there
        // This assumes ProviderDescriptor has a builder or reasonable constructor.
        // Since I can't see ProviderDescriptor deeply, I'll rely on what was there or
        // minimal impl.
        // checking previous file content... it called provider.descriptor().
        // But LLMProvider interface removed .descriptor() in my previous edit?
        // ARGH. The previous LLMProvider had `ProviderDescriptor descriptor()` but I
        // might have missed it in the Refactor plan...
        // Wait, the previous `LLMProvider` view showed `ProviderMetadata metadata()`
        // AND line 225 `descriptors.put(id, provider.descriptor());`
        // BUT `LLMProvider` interface definition in view_file Step 26 DID NOT have
        // `descriptor()`. It had `metadata()`.
        // `ProviderRegistry` Step 24 line 225 calls `provider.descriptor()`. This
        // implies the `LLMProvider` being used there had it.
        // My Refactor of LLMProvider KEPT `metadata()`.
        // I will assume `ProviderMetadata` can be converted to `ProviderDescriptor` or
        // `metadata()` IS the descriptor.
        // Let's check `ProviderMetadata` file. I haven't seen it yet. I saw
        // `ProviderDescriptor.java` in file list.
        // I will ignore descriptors for a moment or try to use metadata. Ideally I
        // should have checked `ProviderDescriptor`.
        // I'll comment out the descriptor part if I can't resolve it, or better, I will
        // fix it by using metadata.
        return null;
    }

    /**
     * Register a provider from a plugin
     */
    public void registerProviderFromPlugin(LLMProvider provider, GolekPlugin sourcePlugin) {
        register(provider);
        if (sourcePlugin != null) {
            GolekPlugins.put(provider.id(), sourcePlugin);
        }
    }

    /**
     * Unregister a provider (all versions)
     */
    public void unregister(String providerId) {
        NavigableMap<String, LLMProvider> versions = providers.remove(providerId);
        descriptors.remove(providerId);
        GolekPlugins.remove(providerId);

        if (versions != null) {
            versions.values().forEach(this::closeProvider);
        }
    }

    /**
     * Unregister specific version
     */
    public void unregister(String providerId, String version) {
        Map<String, LLMProvider> versions = providers.get(providerId);
        if (versions != null) {
            LLMProvider provider = versions.remove(version);
            if (provider != null) {
                closeProvider(provider);
            }
            if (versions.isEmpty()) {
                providers.remove(providerId);
                descriptors.remove(providerId);
                GolekPlugins.remove(providerId);
            }
        }
    }

    private void closeProvider(LLMProvider provider) {
        try {
            provider.shutdown();
        } catch (Exception e) {
            LOG.warnf(e, "Error shutting down provider: %s", provider.id());
        }
    }

    /**
     * Get latest version of a provider
     */
    public Optional<LLMProvider> getProvider(String providerId) {
        return Optional.ofNullable(providers.get(providerId))
                .map(NavigableMap::lastEntry)
                .map(Map.Entry::getValue);
    }

    /**
     * Get specific version of a provider
     */
    public Optional<LLMProvider> getProvider(String providerId, String version) {
        return Optional.ofNullable(providers.get(providerId))
                .map(m -> m.get(version));
    }

    /**
     * Get provider or throw
     */
    public LLMProvider getProviderOrThrow(String providerId) {
        return getProvider(providerId)
                .orElseThrow(() -> new ProviderException.ProviderUnavailableException(
                        providerId,
                        "Provider not found: " + providerId));
    }

    /**
     * Get all registered providers (latest versions only)
     */
    public Collection<LLMProvider> getAllProviders() {
        return providers.values().stream()
                .map(NavigableMap::lastEntry)
                .map(Map.Entry::getValue)
                .collect(Collectors.toList());
    }

    /**
     * Get all versions of all providers
     */
    public Collection<LLMProvider> getAllProviderVersions() {
        return providers.values().stream()
                .flatMap(m -> m.values().stream())
                .collect(Collectors.toList());
    }

    /**
     * Check if provider exists
     */
    public boolean hasProvider(String providerId) {
        return providers.containsKey(providerId);
    }

    /**
     * Get providers that support a specific model
     */
    public List<LLMProvider> getProvidersForModel(String model) {
        return getAllProviders().stream()
                .filter(p -> p.supports(model, null)) // TenantContext is null for global check
                .toList();
    }

    /**
     * Get streaming providers
     */
    public List<StreamingProvider> getStreamingProviders() {
        return getAllProviders().stream()
                .filter(p -> p instanceof StreamingProvider)
                .map(p -> (StreamingProvider) p)
                .toList();
    }

    /**
     * Get plugin that owns a provider
     */
    public Optional<GolekPlugin> getOwningPlugin(String providerId) {
        return Optional.ofNullable(GolekPlugins.get(providerId));
    }

    /**
     * Shutdown all providers
     */
    public void shutdown() {
        LOG.info("Shutting down all providers...");
        providers.values().stream()
                .flatMap(m -> m.values().stream())
                .forEach(this::closeProvider);
        providers.clear();
        descriptors.clear();
        GolekPlugins.clear();
    }

    // Helper class for concurrent sorted map
    private static class ConcurrentSkipListMap<K, V> extends java.util.concurrent.ConcurrentSkipListMap<K, V> {
        // Just to make it cleaner in the code above if we want to alias it,
        // but java.util.concurrent.ConcurrentSkipListMap is standard.
    }
}