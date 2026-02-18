package tech.kayys.gollek.spi.provider;

import tech.kayys.gollek.spi.plugin.GollekPlugin;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

/**
 * Registry for all available inference providers.
 */
public interface ProviderRegistry {
    void discoverProviders();

    void register(LLMProvider provider);

    void registerProviderFromPlugin(LLMProvider provider, GollekPlugin sourcePlugin);

    void unregister(String providerId);

    void unregister(String providerId, String version);

    Optional<LLMProvider> getProvider(String providerId);

    Optional<LLMProvider> getProvider(String providerId, String version);

    LLMProvider getProviderOrThrow(String providerId);

    Collection<LLMProvider> getAllProviders();

    Collection<LLMProvider> getAllProviderVersions();

    boolean hasProvider(String providerId);

    List<LLMProvider> getProvidersForModel(String model);

    List<StreamingProvider> getStreamingProviders();

    Optional<GollekPlugin> getOwningPlugin(String providerId);

    void shutdown();
}
