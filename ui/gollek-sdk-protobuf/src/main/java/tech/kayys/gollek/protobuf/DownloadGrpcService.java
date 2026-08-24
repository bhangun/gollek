package tech.kayys.gollek.protobuf;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.alkhawarizm.spi.model.ModelManifest;
import tech.kayys.gollek.model.domain.download.ModelDownloadManager;
import tech.kayys.gollek.model.domain.download.ModelDownloadState;
import tech.kayys.gollek.registry.service.ModelRegistryService;

import java.util.Optional;

@GrpcService
public class DownloadGrpcService implements DownloadService {

    private static final Logger LOG = LoggerFactory.getLogger(DownloadGrpcService.class);

    @Inject
    ModelDownloadManager downloadManager;

    @Inject
    ModelRegistryService registryService;

    @Inject
    MeterRegistry meterRegistry;

    @Override
    public Uni<DownloadState> startDownload(StartDownloadRequest request) {
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received startDownload request for model: {}", request.getModelId());
        
        return registryService.getManifest("default", request.getModelId(), "latest")
                .onItem().transformToUni(manifest -> {
                    String downloadUri = resolveUri(manifest);
                    if (downloadUri == null) {
                        LOG.error("No valid artifacts found to download for model: {}", request.getModelId());
                        throw new RuntimeException("Model has no downloadable artifacts");
                    }
                    
                    downloadManager.startDownload(request.getModelId(), downloadUri);
                    ModelDownloadState state = downloadManager.getDownloadState(request.getModelId());
                    
                    sample.stop(meterRegistry.timer("gollek.grpc.download.start.duration"));
                    meterRegistry.counter("gollek.grpc.download.start.count").increment();
                    return Uni.createFrom().item(mapState(state));
                })
                .onFailure().invoke(th -> {
                    LOG.error("Failed to start download for model: {}", request.getModelId(), th);
                    meterRegistry.counter("gollek.grpc.download.start.error").increment();
                });
    }

    private String resolveUri(ModelManifest manifest) {
        if (manifest.artifacts() != null && !manifest.artifacts().isEmpty()) {
            return manifest.artifacts().values().iterator().next().uri();
        }
        return null;
    }

    @Override
    public Uni<DownloadState> pauseDownload(PauseDownloadRequest request) {
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received pauseDownload request for model: {}", request.getModelId());
        
        return Uni.createFrom().item(() -> {
            ModelDownloadState state = downloadManager.getDownloadState(request.getModelId());
            if (state != null) {
                state.setStatus("PAUSED");
            }
            
            sample.stop(meterRegistry.timer("gollek.grpc.download.pause.duration"));
            meterRegistry.counter("gollek.grpc.download.pause.count").increment();
            return mapState(state);
        });
    }

    @Override
    public Uni<DownloadState> resumeDownload(ResumeDownloadRequest request) {
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received resumeDownload request for model: {}", request.getModelId());
        
        return Uni.createFrom().item(() -> {
            ModelDownloadState state = downloadManager.getDownloadState(request.getModelId());
            if (state != null) {
                state.setStatus("DOWNLOADING");
            }
            
            sample.stop(meterRegistry.timer("gollek.grpc.download.resume.duration"));
            meterRegistry.counter("gollek.grpc.download.resume.count").increment();
            return mapState(state);
        });
    }

    @Override
    public Uni<DownloadState> getDownloadStatus(GetDownloadStatusRequest request) {
        return Uni.createFrom().item(() -> {
            ModelDownloadState state = downloadManager.getDownloadState(request.getModelId());
            if (state == null) {
                throw new RuntimeException("Download not found");
            }
            return mapState(state);
        });
    }

    private DownloadState mapState(ModelDownloadState state) {
        if (state == null) {
            return DownloadState.newBuilder().build();
        }
        return DownloadState.newBuilder()
                .setModelId(state.getModelId())
                .setTotalBytes(state.getTotalBytes())
                .setBytesDownloaded(state.getBytesDownloaded())
                .setPercentage(state.getPercentage())
                .setStatus(state.getStatus())
                .setErrorMessage(state.getErrorMessage() != null ? state.getErrorMessage() : "")
                .build();
    }
}
