package tech.kayys.golek.engine.registry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.plugin.GolekPlugin;
import tech.kayys.golek.spi.provider.ProviderDescriptor;
import tech.kayys.golek.spi.provider.ProviderRegistry;
import tech.kayys.golek.spi.exception.ProviderException;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.spi.provider.StreamingProvider;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Concrete implementation of the ProviderRegistry.
 */
@ApplicationScoped
public class GolekProviderRegistry implements ProviderRegistry {

    private static final Logger LOG = Logger.getLogger(GolekProviderRegistry.class);

    private final Map<String, NavigableMap<String, LLMProvider>> providers = new ConcurrentHashMap<>();
    private final Map<String, GolekPlugin> GolekPlugins = new ConcurrentHashMap<>();
    private final Map<String, ProviderDescriptor> descriptors = new ConcurrentHashMap<>();

    @Inject
    Instance<LLMProvider> providerInstances;

    @jakarta.annotation.PostConstruct
    @Override
    public void discoverProviders() {
        LOG.info("Discovering LLM providers...");

        providerInstances.stream().forEach(provider -> {
            try {

                register(provider);
                LOG.infof("Registered provider: %s v%s", provider.id(), provider.version());

            } catch (Exception e) {

                LOG.errorf(e, "Failed to register provider: %s",
                        provider.getClass().getName());
            } catch (Throwable t) {

                t.printStackTrace();
            }
        });

        LOG.infof("Provider discovery complete. Total unique providers: %d", providers.size());

    }

    @Override
    public void register(LLMProvider provider) {
        Objects.requireNonNull(provider, "provider cannot be null");

        String id = provider.id();
        String version = provider.version();

        providers.computeIfAbsent(id, k -> new java.util.concurrent.ConcurrentSkipListMap<>())
                .put(version, provider);

        if (provider.metadata() != null) {
            descriptors.put(id, ProviderDescriptor.builder()
                    .id(id)
                    .displayName(provider.name())
                    .version(version)
                    .capabilities(provider.capabilities())
                    .metadata("provider_metadata", provider.metadata())
                    .build());
        }

        LOG.debugf("Registered provider: %s v%s (%s)", id, version, provider.name());
    }

    @Override
    public void registerProviderFromPlugin(LLMProvider provider, GolekPlugin sourcePlugin) {
        register(provider);
        if (sourcePlugin != null) {
            GolekPlugins.put(provider.id(), sourcePlugin);
        }
    }

    @Override
    public void unregister(String providerId) {
        NavigableMap<String, LLMProvider> versions = providers.remove(providerId);
        descriptors.remove(providerId);
        GolekPlugins.remove(providerId);

        if (versions != null) {
            versions.values().forEach(this::closeProvider);
        }
    }

    @Override
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

    @Override
    public Optional<LLMProvider> getProvider(String providerId) {
        return Optional.ofNullable(providers.get(providerId))
                .map(NavigableMap::lastEntry)
                .map(Map.Entry::getValue);
    }

    @Override
    public Optional<LLMProvider> getProvider(String providerId, String version) {
        return Optional.ofNullable(providers.get(providerId))
                .map(m -> m.get(version));
    }

    @Override
    public LLMProvider getProviderOrThrow(String providerId) {
        return getProvider(providerId)
                .orElseThrow(() -> new ProviderException.ProviderUnavailableException(
                        providerId,
                        "Provider not found: " + providerId));
    }

    @Override
    public Collection<LLMProvider> getAllProviders() {
        return providers.values().stream()
                .map(NavigableMap::lastEntry)
                .map(Map.Entry::getValue)
                .collect(Collectors.toList());
    }

    @Override
    public Collection<LLMProvider> getAllProviderVersions() {
        return providers.values().stream()
                .flatMap(m -> m.values().stream())
                .collect(Collectors.toList());
    }

    @Override
    public boolean hasProvider(String providerId) {
        return providers.containsKey(providerId);
    }

    @Override
    public List<LLMProvider> getProvidersForModel(String model) {
        return getAllProviders().stream()
                .filter(p -> p.supports(model, null))
                .toList();
    }

    @Override
    public List<StreamingProvider> getStreamingProviders() {
        return getAllProviders().stream()
                .filter(p -> p instanceof StreamingProvider)
                .map(p -> (StreamingProvider) p)
                .toList();
    }

    @Override
    public Optional<GolekPlugin> getOwningPlugin(String providerId) {
        return Optional.ofNullable(GolekPlugins.get(providerId));
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down all providers...");
        providers.values().stream()
                .flatMap(m -> m.values().stream())
                .forEach(this::closeProvider);
        providers.clear();
        descriptors.clear();
        GolekPlugins.clear();
    }
}
