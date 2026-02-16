package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.context.RequestContext;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.Pageable;

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
                        RequestContext requestContext,
                        int page,
                        int size) {
                return modelRepository.list(requestContext.getRequestId(), Pageable.of(page, size));
        }

        public Uni<ModelManifest> getModel(
                        String modelId,
                        RequestContext requestContext) {
                return modelRepository.findById(modelId, requestContext.getRequestId());
        }

        public Uni<ModelManifest> registerModel(
                        ModelManifest manifest,
                        RequestContext requestContext) {
                LOG.infof("Registering model: %s for tenant: %s",
                                manifest.modelId(), requestContext.getRequestId());

                return modelRepository.save(manifest);
        }

        public Uni<ModelManifest> updateModel(
                        String modelId,
                        ModelManifest manifest,
                        RequestContext requestContext) {
                return modelRepository.save(manifest);
        }

        public Uni<Void> deleteModel(
                        String modelId,
                        RequestContext requestContext) {
                return modelRepository.delete(modelId, requestContext.getRequestId());
        }

        public Uni<Void> warmup(
                        String modelId,
                        RequestContext requestContext) {
                return modelRepository.findById(modelId, requestContext.getRequestId())
                                .onItem().ifNotNull().invoke(manifest -> runnerFactory.prewarm(
                                                List.of(manifest.modelId()),
                                                List.of("default")))
                                .replaceWithVoid();
        }
}