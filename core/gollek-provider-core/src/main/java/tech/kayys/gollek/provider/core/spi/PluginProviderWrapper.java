package tech.kayys.gollek.provider.core.spi;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.exception.ProviderException;

/**
 * Wrapper for providers that come from plugins
 */
public class PluginProviderWrapper implements LLMProvider {
    private final LLMProvider delegate;
    private final GollekPlugin sourcePlugin;

    public PluginProviderWrapper(LLMProvider delegate, GollekPlugin sourcePlugin) {
        this.delegate = delegate;
        this.sourcePlugin = sourcePlugin;
    }

    public GollekPlugin getSourcePlugin() {
        return sourcePlugin;
    }

    @Override
    public String id() {
        return delegate.id();
    }

    @Override
    public String name() {
        return delegate.name();
    }

    @Override
    public String version() {
        return delegate.version();
    }

    @Override
    public ProviderMetadata metadata() {
        return delegate.metadata();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return delegate.capabilities();
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
        delegate.initialize(config);
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        return delegate.supports(modelId, request);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return delegate.infer(request);
    }

    @Override
    public Uni<ProviderHealth> health() {
        return delegate.health();
    }

    @Override
    public void shutdown() {
        delegate.shutdown();
    }
}