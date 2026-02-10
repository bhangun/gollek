package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.Pageable;
import tech.kayys.wayang.tenant.TenantContext;

import java.util.List;

/**
 * Model management service for lifecycle operations
 */
@ApplicationScoped
public class ModelManagementService {

        private static final Logger LOG = Logger.getLogger(ModelManagementService.class);

        @Inject
        CachedModelRepository modelRepository;

        @Inject
        ModelRunnerFactory runnerFactory;

        public Uni<List<ModelManifest>> listModels(
                        TenantContext tenantContext,
                        int page,
                        int size) {
                return modelRepository.list(tenantContext.getTenantId().value(), Pageable.of(page, size));
        }

        public Uni<ModelManifest> getModel(
                        String modelId,
                        TenantContext tenantContext) {
                return modelRepository.findById(modelId, tenantContext.getTenantId().value());
        }

        public Uni<ModelManifest> registerModel(
                        ModelManifest manifest,
                        TenantContext tenantContext) {
                LOG.infof("Registering model: %s for tenant: %s",
                                manifest.modelId(), tenantContext.getTenantId().value());

                return modelRepository.save(manifest);
        }

        public Uni<ModelManifest> updateModel(
                        String modelId,
                        ModelManifest manifest,
                        TenantContext tenantContext) {
                return modelRepository.save(manifest);
        }

        public Uni<Void> deleteModel(
                        String modelId,
                        TenantContext tenantContext) {
                return modelRepository.delete(modelId, tenantContext.getTenantId().value());
        }

        public Uni<Void> warmup(
                        String modelId,
                        TenantContext tenantContext) {
                return modelRepository.findById(modelId, tenantContext.getTenantId().value())
                                .onItem().ifNotNull().invoke(manifest -> runnerFactory.prewarm(
                                                List.of(manifest.modelId()),
                                                List.of("default")))
                                .replaceWithVoid();
        }
}