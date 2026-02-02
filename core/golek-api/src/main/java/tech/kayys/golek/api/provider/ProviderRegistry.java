package tech.kayys.golek.api.provider;

import tech.kayys.golek.api.plugin.GolekPlugin;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

/**
 * Registry for all available inference providers.
 */
public interface ProviderRegistry {
    void discoverProviders();

    void register(LLMProvider provider);

    void registerProviderFromPlugin(LLMProvider provider, GolekPlugin sourcePlugin);

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

    Optional<GolekPlugin> getOwningPlugin(String providerId);

    void shutdown();
}
