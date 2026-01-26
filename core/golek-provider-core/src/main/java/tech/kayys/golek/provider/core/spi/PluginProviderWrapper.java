package tech.kayys.golek.provider.core.spi;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.provider.ProviderCapabilities;
import tech.kayys.golek.api.provider.ProviderConfig;
import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.ProviderMetadata;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.plugin.api.GolekPlugin;

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
    public void initialize(ProviderConfig config) throws ProviderInitializationException {
        delegate.initialize(config);
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        return delegate.supports(modelId, tenantContext);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
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