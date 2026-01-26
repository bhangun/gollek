package tech.kayys.golek.model.repository.local;

import java.net.URI;
import java.nio.file.Path;
import java.util.Map;

import tech.kayys.golek.model.core.ModelArtifact;
import tech.kayys.golek.model.core.ModelDescriptor;
import tech.kayys.golek.model.core.ModelRef;
import tech.kayys.golek.model.core.ModelRepository;

public final class LocalModelRepository implements ModelRepository {

    @Override
    public boolean supports(ModelRef ref) {
        return "local".equalsIgnoreCase(ref.scheme());
    }

    @Override
    public ModelDescriptor resolve(ModelRef ref) {
        return new ModelDescriptor(
                ref.name(),
                ref.parameters().getOrDefault("format", "auto"),
                URI.create("file://" + ref.parameters().get("path")),
                Map.of("provider", "local"));
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        Path path = Path.of(descriptor.source());
        return new ModelArtifact(path, "local", descriptor.metadata());
    }
}