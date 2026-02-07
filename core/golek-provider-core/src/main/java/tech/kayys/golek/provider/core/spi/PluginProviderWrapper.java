package tech.kayys.golek.provider.core.spi;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderConfig;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderMetadata;
import tech.kayys.golek.spi.provider.ProviderRequest;
// import tech.kayys.wayang.tenant.TenantContext; // Temporarily commented out due to missing dependency
import tech.kayys.golek.spi.plugin.GolekPlugin;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.spi.exception.ProviderException;

/**
 * Wrapper for providers that come from plugins
 */
public class PluginProviderWrapper implements LLMProvider {
    private final LLMProvider delegate;
    private final GolekPlugin sourcePlugin;

    public PluginProviderWrapper(LLMProvider delegate, GolekPlugin sourcePlugin) {
        this.delegate = delegate;
        this.sourcePlugin = sourcePlugin;
    }

    public GolekPlugin getSourcePlugin() {
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
    public boolean supports(String modelId, Object tenantContext) { // Using Object temporarily due to missing dependency
        return delegate.supports(modelId, tenantContext);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, Object context) { // Using Object temporarily due to missing dependency
        return delegate.infer(request, context);
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