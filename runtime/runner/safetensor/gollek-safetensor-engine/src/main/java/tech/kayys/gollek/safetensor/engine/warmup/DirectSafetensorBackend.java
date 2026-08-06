package tech.kayys.gollek.safetensor.engine.warmup;

import tech.kayys.gollek.spi.inference.LocalInferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.alkhawarizm.spi.model.HealthStatus;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class DirectSafetensorBackend implements LocalInferenceEngine {
    public void initialize() {}
    public void shutdown() {}
    public String name() { return "DirectSafetensorBackend"; }
    public HealthStatus health() { return HealthStatus.healthy("OK"); }
    public Uni<InferenceResponse> infer(InferenceRequest request) { return null; }
    public io.smallrye.mutiny.Multi<tech.kayys.gollek.spi.inference.StreamingInferenceChunk> inferStream(InferenceRequest request) { return null; }
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) { return null; }
}
