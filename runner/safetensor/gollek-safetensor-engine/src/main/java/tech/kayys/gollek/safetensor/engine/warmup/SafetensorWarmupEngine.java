package tech.kayys.gollek.safetensor.engine.warmup;

import tech.kayys.gollek.spi.inference.LocalInferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.model.HealthStatus;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;

public class SafetensorWarmupEngine implements LocalInferenceEngine {
    public void initialize() {}
    public void shutdown() {}
    public String name() { return "SafetensorWarmupEngine"; }
    public HealthStatus health() { return HealthStatus.healthy("OK"); }
    public Uni<InferenceResponse> infer(InferenceRequest request) { return null; }
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) { return null; }
}
