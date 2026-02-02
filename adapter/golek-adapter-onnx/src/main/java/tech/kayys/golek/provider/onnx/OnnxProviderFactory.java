package tech.kayys.golek.provider.onnx;

import tech.kayys.golek.provider.core.spi.ModelRunner;
import tech.kayys.golek.provider.spi.ProviderFactory;
import tech.kayys.golek.provider.spi.model.ModelManifest;
import tech.kayys.golek.provider.spi.model.RequestContext;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;

import java.util.Map;

/**
 * Provider factory for ONNX Runtime adapter
 */
@ApplicationScoped
public class OnnxProviderFactory implements ProviderFactory {

    @Produces
    @ApplicationScoped
    public OnnxRuntimeRunner onnxRuntimeRunner() {
        return new OnnxRuntimeRunner();
    }

    @Override
    public String providerId() {
        return "onnx-runtime";
    }

    @Override
    public boolean supports(ModelManifest manifest) {
        return manifest.supportedFormats().contains(tech.kayys.golek.provider.spi.model.ModelFormat.ONNX);
    }

    @Override
    public ModelRunner createRunner(ModelManifest manifest, Map<String, Object> config, RequestContext context) {
        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        runner.initialize(manifest, config, context);
        return runner;
    }
}