package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;

@ApplicationScoped
public class ModelFamilyPluginRegistryProducer {

    @Produces
    @ApplicationScoped
    public ModelFamilyPluginRegistry produceRegistry() {
        return ModelFamilyPluginRegistry.global();
    }
}
