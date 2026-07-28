package tech.kayys.gollek.prefilldecode;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.LocalInferenceEngine;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

/**
 * A local inference engine that implements disaggregated prefill/decode logic.
 * It uses the PrefillDecodeDisaggService to orchestrate the handoff.
 */
@ApplicationScoped
public class DisaggregatedLLMProvider implements LocalInferenceEngine {

    private static final Logger LOG = Logger.getLogger(DisaggregatedLLMProvider.class);
    public static final String ID = "pd-disagg";

    @Inject
    PrefillDecodeDisaggService disaggService;

    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        return Multi.createBy().concatenating().streams(
                stream(request)).collect().asList().map(chunks -> {
                    StringBuilder sb = new StringBuilder();
                    for (StreamingInferenceChunk chunk : chunks) {
                        if (chunk.delta() != null) {
                            sb.append(chunk.delta());
                        }
                    }
                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .content(sb.toString())
                            .model(request.getModel())
                            .build();
                });
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        // 1. Trigger Prefill
        return disaggService.executePrefillAsync(request)
                .onItem().transformToMulti(kvTransferId ->
                    // 2. Trigger Decode Stream using the handoff token
                    disaggService.executeDecodeStream(kvTransferId, request)
                );
    }

    @Override
    public void shutdown() {
        try {
            disaggService.stop();
            LOG.info("[DisaggProvider] Shutdown complete");
        } catch (Exception e) {
            LOG.errorf("[DisaggProvider] Shutdown error: %s", e.getMessage());
        }
    }

    @Override
    public boolean isHealthy() {
        return disaggService.isActive();
    }
}
