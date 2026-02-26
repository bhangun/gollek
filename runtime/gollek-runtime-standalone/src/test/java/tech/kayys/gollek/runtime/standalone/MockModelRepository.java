package tech.kayys.gollek.runtime.standalone;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.model.core.ModelRepository;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.Pageable;
import tech.kayys.gollek.spi.model.ModelDescriptor;
import tech.kayys.gollek.spi.model.ModelRef;
import tech.kayys.gollek.spi.model.ModelArtifact;
import java.util.List;
import java.nio.file.Path;

@ApplicationScoped
public class MockModelRepository implements ModelRepository {

    @Override
    public Uni<ModelManifest> findById(String modelId, String requestId) {
        if ("tinyllama".equals(modelId)) {
            return Uni.createFrom().item(ModelManifest.builder()
                    .modelId("tinyllama")
                    .name("tinyllama")
                    .version("1.0")
                    .path("local")
                    .apiKey("test-key")
                    .requestId(requestId != null ? requestId : "community")
                    .build());
        }
        return Uni.createFrom().nullItem();
    }

    @Override
    public Uni<List<ModelManifest>> list(String requestId, Pageable pageable) {
        return Uni.createFrom().item(List.of());
    }

    @Override
    public Uni<ModelManifest> save(ModelManifest manifest) {
        return Uni.createFrom().nullItem();
    }

    @Override
    public Uni<Void> delete(String modelId, String requestId) {
        return Uni.createFrom().nullItem();
    }

    @Override
    public ModelDescriptor resolve(ModelRef ref) {
        return null;
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        return null;
    }

    @Override
    public boolean supports(ModelRef ref) {
        return false;
    }

    @Override
    public Path downloadArtifact(ModelManifest manifest, ModelFormat format) {
        return null;
    }

    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        return false;
    }

    @Override
    public void evictCache(String modelId, ModelFormat format) {
    }
}
