package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;
import tech.kayys.gollek.spi.inference.InferenceRequest;

/**
 * GGUF backend powered by the existing llama.cpp provider.
 */
final class LlamaCppGgufBackend implements GgufBackend {
    private final Object provider;

    LlamaCppGgufBackend(Object provider) {
        this.provider = provider;
    }

    @Override
    public String name() {
        return "llamacpp";
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        if (request.getInferenceRequest().isEmpty()) {
            return RunnerResult.failed("Unsupported request type for llama.cpp GGUF backend");
        }

        InferenceRequest inferenceRequest = request.getInferenceRequest().get();

        try {
            Object uni = provider.getClass()
                    .getMethod("infer", InferenceRequest.class)
                    .invoke(provider, inferenceRequest);
            Object awaiter = uni.getClass().getMethod("await").invoke(uni);
            Object response = awaiter.getClass().getMethod("indefinitely").invoke(awaiter);
            return (RunnerResult<T>) RunnerResult.success(response);
        } catch (Exception e) {
            return RunnerResult.failed("Llama.cpp GGUF inference failed: " + e.getMessage());
        }
    }

    @Override
    public void close() {
        // LlamaCppProvider lifecycle is managed by CDI or the embedding application.
    }
}
