package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;

import java.util.Map;

public interface GgufBackend extends AutoCloseable {
    String name();

    default Map<String, Object> metadata() {
        return Map.of("backend", name());
    }

    <T> RunnerResult<T> execute(RunnerRequest request);

    @Override
    void close();
}
