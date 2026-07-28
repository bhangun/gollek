package tech.kayys.gollek.safetensor.engine.warmup;

import tech.kayys.gollek.spi.inference.InferenceRequest;
import java.util.Optional;
import java.nio.file.Path;

public class LoraAdapterRouter {
    public Optional<Object> resolve(InferenceRequest request) { return Optional.empty(); }
    public Optional<Path> resolveAdapterPath(InferenceRequest request) { return Optional.empty(); }
}
