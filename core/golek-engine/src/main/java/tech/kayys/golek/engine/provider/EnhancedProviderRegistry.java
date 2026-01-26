package tech.kayys.golek.engine.provider;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.golek.core.plugin.Plugin;
import tech.kayys.golek.core.plugin.PluginManager;
import tech.kayys.golek.provider.core.PluginProviderWrapper;
import tech.kayys.golek.provider.core.ProviderRegistry;
import tech.kayys.golek.provider.core.plugin.GolekPlugin;
import tech.kayys.golek.provider.core.spi.LLMProvider;

/**
 * Enhanced provider registry that integrates with the plugin system
 */
public class EnhancedProviderRegistry implements ProviderRegistry {
    private final Map<String, LLMProvider> providers = new ConcurrentHashMap<>();
    private final Map<String, Plugin> GolekPlugins = new ConcurrentHashMap<>();
    private final PluginManager pluginManager;

    public EnhancedProviderRegistry(PluginManager pluginManager) {
        this.pluginManager = pluginManager;
    }

    /**
     * Register a provider
     */
    public void registerProvider(LLMProvider provider) {
        String id = provider.providerId();
        providers.put(id, provider);

        // If this provider comes from a plugin, track the relationship
        if (provider instanceof PluginProviderWrapper) {
            GolekPlugin plugin = ((PluginProviderWrapper) provider).getSourcePlugin();
            if (plugin != null) {
                // Look up the actual Plugin instance using the ID from the GolekPlugin
                pluginManager.getPlugin(plugin.id()).ifPresent(p -> {
                    GolekPlugins.put(id, p);
                });
            }
        }
    }

    /**
     * Register a provider from a plugin
     */
    public void registerProviderFromPlugin(LLMProvider provider, Plugin sourcePlugin) {
        String id = provider.providerId();
        // invalid: new PluginProviderWrapper(provider, sourcePlugin) - expects GolekPlugin
        
        if (sourcePlugin instanceof GolekPlugin) {
             providers.put(id, new PluginProviderWrapper(provider, (GolekPlugin) sourcePlugin));
        } else {
            providers.put(id, provider);
        }
        GolekPlugins.put(id, sourcePlugin);
    }

    /**
     * Unregister a provider
     */
    public void unregisterProvider(String providerId) {
        providers.remove(providerId);
        GolekPlugins.remove(providerId);
    }

    /**
     * Get provider by ID
     */
    @Override
    public Optional<LLMProvider> getProvider(String providerId) {
        return Optional.ofNullable(providers.get(providerId));
    }

    /**
     * Get all registered providers
     */
    @Override
    public Collection<LLMProvider> getAllProviders() {
        return Collections.unmodifiableCollection(providers.values());
    }

    /**
     * Check if provider exists
     */
    @Override
    public boolean hasProvider(String providerId) {
        return providers.containsKey(providerId);
    }

    /**
     * Get providers that support a specific model
     */
    @Override
    public List<LLMProvider> getProvidersForModel(String model) {
        return providers.values().stream()
                .filter(p -> p.capabilities().supportsModel(model))
                .toList();
    }

    /**
     * Get providers from a specific plugin
     */
    public List<LLMProvider> getProvidersFromPlugin(String pluginId) {
        return providers.values().stream()
                .filter(p -> {
                    Plugin plugin = GolekPlugins.get(p.providerId());
                    return plugin != null && plugin.getId().equals(pluginId);
                })
                .toList();
    }

    /**
     * Get plugin that owns a provider
     */
    public Optional<GolekPlugin> getOwningPlugin(String providerId) {
        return Optional.ofNullable(GolekPlugins.get(providerId));
    }

    /**
     * Get provider count
     */
    @Override
    public int size() {
        return providers.size();
    }

    /**
     * Check if registry is empty
     */
    @Override
    public boolean isEmpty() {
        return providers.isEmpty();
    }

    /**
     * Clear all providers
     */
    public void clear() {
        providers.clear();
        GolekPlugins.clear();
    }
}
