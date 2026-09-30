package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.*;

import java.util.Set;

/**
 * Backward-compatible facade delegating to {@link tech.kayys.alkhawarizm.gguf.api.GgufRunnerPlugin}.
 */
public class GgufRunnerPlugin implements RunnerPlugin {
    public static final String ID = tech.kayys.alkhawarizm.gguf.api.GgufRunnerPlugin.ID;

    private final tech.kayys.alkhawarizm.gguf.api.GgufRunnerPlugin delegate =
            new tech.kayys.alkhawarizm.gguf.api.GgufRunnerPlugin();

    @Override
    public String id() {
        return delegate.id();
    }

    @Override
    public String name() {
        return delegate.name();
    }

    @Override
    public String version() {
        return delegate.version();
    }

    @Override
    public String description() {
        return delegate.description();
    }

    @Override
    public String format() {
        return delegate.format();
    }

    @Override
    public Set<String> supportedFormats() {
        return delegate.supportedFormats();
    }

    @Override
    public Set<String> supportedArchitectures() {
        return delegate.supportedArchitectures();
    }

    @Override
    public void initialize(RunnerContext context) throws RunnerException {
        delegate.initialize(context);
    }

    @Override
    public boolean isAvailable() {
        return delegate.isAvailable();
    }

    @Override
    public ModelHandle loadModel(ModelLoadRequest request, RunnerContext context) throws RunnerException {
        return delegate.loadModel(request, context);
    }

    @Override
    public void unloadModel(ModelHandle handle, RunnerContext context) {
        delegate.unloadModel(handle, context);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request, RunnerContext context) throws RunnerException {
        return delegate.execute(request, context);
    }

    @Override
    public void shutdown() {
        delegate.shutdown();
    }
}
