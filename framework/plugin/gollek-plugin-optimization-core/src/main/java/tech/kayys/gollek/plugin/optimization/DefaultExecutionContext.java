package tech.kayys.gollek.plugin.optimization;

import tech.kayys.gollek.plugin.runner.RunnerSession;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Simple ExecutionContext implementation used by the engine to provide
 * necessary information to OptimizationPlugins. Currently supports only the
 * attributes required by KVCacheOptimizationPlugin.
 */
public class DefaultExecutionContext implements ExecutionContext {
    private final RunnerSession runnerSession;
    private final Map<String, Object> attributes = new HashMap<>();

    public DefaultExecutionContext(RunnerSession runnerSession) {
        this.runnerSession = runnerSession;
        // expose the runner session for plugins that need it
        setAttribute("runnerSession", runnerSession);
    }

    @Override
    public <T> Optional<T> getParameter(String key, Class<T> type) {
        return Optional.empty();
    }

    @Override
    public Optional<MemoryBuffer> getBuffer(String name) {
        return Optional.empty();
    }

    @Override
    public Map<String, MemoryBuffer> getBuffers() {
        return Collections.emptyMap();
    }

    @Override
    public long getCudaStream() {
        return 0L;
    }

    @Override
    public long getHipStream() {
        return 0L;
    }

    @Override
    public int getBatchSize() {
        return 1; // default batch size
    }

    @Override
    public int getSequenceLength() {
        return 0; // not used by current plugins
    }

    @Override
    public Optional<String> getModelArchitecture() {
        return Optional.empty();
    }

    @Override
    public int getDeviceId() {
        return 0;
    }

    @Override
    public boolean isGpu() {
        return true; // assume GPU when available
    }

    @Override
    public ExecutionPhase getPhase() {
        return ExecutionPhase.DECODE;
    }

    @Override
    public void setAttribute(String key, Object value) {
        attributes.put(key, value);
    }

    @Override
    public <T> Optional<T> getAttribute(String key, Class<T> type) {
        Object val = attributes.get(key);
        if (type.isInstance(val)) {
            return Optional.of(type.cast(val));
        }
        return Optional.empty();
    }
}
