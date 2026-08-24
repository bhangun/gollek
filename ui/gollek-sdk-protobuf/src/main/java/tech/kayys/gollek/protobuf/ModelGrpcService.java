package tech.kayys.gollek.protobuf;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.registry.service.ModelRegistryService;

import java.util.stream.Collectors;

@GrpcService
public class ModelGrpcService implements ModelService {

    private static final Logger LOG = LoggerFactory.getLogger(ModelGrpcService.class);

    @Inject
    ModelRegistryService registryService;

    @Inject
    MeterRegistry meterRegistry;

    @Override
    public Uni<ListModelsResponse> listModels(ListModelsRequest request) {
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received listModels request with query: {}", request.getQuery());
        
        return registryService.findByTenant("default", null)
                .map(models -> {
                    sample.stop(meterRegistry.timer("gollek.grpc.model.list.duration"));
                    meterRegistry.counter("gollek.grpc.model.list.count").increment();
                    LOG.info("Returning {} models for listModels request", models.size());
                    
                    return ListModelsResponse.newBuilder()
                            .addAllModels(models.stream()
                                    .map(m -> ModelMetadata.newBuilder()
                                            .setId(m.modelId())
                                            .setName(m.name())
                                            .setDescription(m.architecture() != null ? m.architecture() : "")
                                            .setFormat(m.artifacts() != null && !m.artifacts().isEmpty() ? m.artifacts().keySet().iterator().next().name() : "UNKNOWN")
                                            .build())
                                    .collect(Collectors.toList()))
                            .build();
                })
                .onFailure().invoke(th -> {
                    LOG.error("Failed listModels request", th);
                    meterRegistry.counter("gollek.grpc.model.list.error").increment();
                });
    }

    @Override
    public Uni<GetModelManifestResponse> getModelManifest(GetModelManifestRequest request) {
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received getModelManifest request for model: {}", request.getModelId());
        
        return registryService.getManifest("default", request.getModelId(), "latest")
                .map(manifest -> {
                    sample.stop(meterRegistry.timer("gollek.grpc.model.manifest.duration"));
                    meterRegistry.counter("gollek.grpc.model.manifest.count").increment();
                    LOG.info("Returning manifest for model: {}", request.getModelId());
                    
                    return GetModelManifestResponse.newBuilder()
                            .setManifestJson(manifest.toString())
                            .build();
                })
                .onFailure().invoke(th -> {
                    LOG.error("Failed getModelManifest request for model: {}", request.getModelId(), th);
                    meterRegistry.counter("gollek.grpc.model.manifest.error").increment();
                });
    }
}
