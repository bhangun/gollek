package tech.kayys.golek.runtime.service;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.runtime.ModelManifest;
import tech.kayys.golek.runtime.repository.ModelRepository;
import tech.kayys.golek.runtime.service.ModelRunnerFactory;

import java.util.List;

/**
 * Model management service for lifecycle operations
 */
@ApplicationScoped
public class ModelManagementService {

    private static final Logger LOG = Logger.getLogger(ModelManagementService.class);

    @Inject
    ModelRepository modelRepository;

    @Inject
    ModelRunnerFactory runnerFactory;

    public Uni<List<ModelManifest>> listModels(
            TenantContext tenantContext,
            int page,
            int size) {
        return Uni.createFrom().item(() -> modelRepository.findByTenant(
                tenantContext.getTenantId(),
                Pageable.of(page, size)));
    }

    public Uni<ModelManifest> getModel(
            String modelId,
            TenantContext tenantContext) {
        return Uni.createFrom().optional(
                modelRepository.findById(modelId, tenantContext.getTenantId()));
    }

    public Uni<ModelManifest> registerModel(
            ModelManifest manifest,
            TenantContext tenantContext) {
        LOG.infof("Registering model: %s for tenant: %s",
                manifest.modelId(), tenantContext.getTenantId());

        return Uni.createFrom().item(() -> modelRepository.save(manifest));
    }

    public Uni<ModelManifest> updateModel(
            String modelId,
            ModelManifest manifest,
            TenantContext tenantContext) {
        return Uni.createFrom().item(() -> modelRepository.save(manifest));
    }

    public Uni<Void> deleteModel(
            String modelId,
            TenantContext tenantContext) {
        return Uni.createFrom().voidItem()
                .invoke(() -> modelRepository.delete(modelId, tenantContext.getTenantId()));
    }

    public Uni<Void> warmup(
            String modelId,
            TenantContext tenantContext) {
        return Uni.createFrom().item(() -> modelRepository.findById(modelId, tenantContext.getTenantId()))
                .onItem().ifNotNull().invoke(manifest -> runnerFactory.prewarm(
                        manifest,
                        List.of("default"),
                        tenantContext))
                .replaceWithVoid();
    }
}