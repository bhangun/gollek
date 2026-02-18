package tech.kayys.gollek.cluster;

import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.cluster.proto.CacheHandoffRequest;
import tech.kayys.gollek.cluster.proto.CacheHandoffResponse;
import tech.kayys.gollek.cluster.proto.GollekInternal;
import tech.kayys.gollek.cluster.proto.PingRequest;
import tech.kayys.gollek.cluster.proto.PingResponse;
import org.jboss.logging.Logger;

@GrpcService
public class CacheHandoffService implements GollekInternal {

    private static final Logger LOG = Logger.getLogger(CacheHandoffService.class);

    private final GollekClusterManager clusterManager;

    // In a real implementation, we'd inject PagedKVCacheManager here to reconstitute the cache
    // private final PagedKVCacheManager kvCacheManager;

    public CacheHandoffService(GollekClusterManager clusterManager) {
        this.clusterManager = clusterManager;
    }

    @Override
    public Uni<CacheHandoffResponse> handoffCache(CacheHandoffRequest request) {
        LOG.infof("Received cache handoff request for %s (model: %s, blocks: %d)",
                request.getRequestId(), request.getModelId(), request.getBlockIdsCount());

        if (!clusterManager.isDecodeNode()) {
            return Uni.createFrom().item(CacheHandoffResponse.newBuilder()
                    .setSuccess(false)
                    .setErrorMessage("Node is not a DECODE node")
                    .build());
        }

        // TODO: Reconstitute the KV-cache from the request data
        // For now, just acknowledge receipt
        
        return Uni.createFrom().item(CacheHandoffResponse.newBuilder()
                .setSuccess(true)
                .setDecodeNodeId(clusterManager.getNodeId())
                .build());
    }

    @Override
    public Uni<PingResponse> ping(PingRequest request) {
        return Uni.createFrom().item(PingResponse.newBuilder()
                .setNodeId(clusterManager.getNodeId())
                .setRole(clusterManager.getCurrentRole().name())
                .setReady(true)
                .build());
    }
}
