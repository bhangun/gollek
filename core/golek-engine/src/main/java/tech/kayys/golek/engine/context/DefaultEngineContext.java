package tech.kayys.golek.engine.context;

import java.time.Instant;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import tech.kayys.golek.core.exception.IllegalStateTransitionException;
import tech.kayys.golek.core.plugin.GolekPluginRegistry;
import tech.kayys.golek.core.provider.ProviderRegistry;
import tech.kayys.golek.core.engine.EngineContext;

/**
 * Default implementation of engine context.
 */
public class DefaultEngineContext implements EngineContext {

    private final GolekPluginRegistry pluginRegistry;
    private final ProviderRegistry providerRegistry;
    private final ExecutorService executorService;
    private final Map<String, Object> config;
    private final Instant startTime;
    private final String version;
    private volatile boolean running;

    private DefaultEngineContext(Builder builder) {
        this.pluginRegistry = Objects.requireNonNull(builder.pluginRegistry);
        this.providerRegistry = Objects.requireNonNull(builder.providerRegistry);
        this.executorService = Objects.requireNonNull(builder.executorService);
        this.config = Collections.unmodifiableMap(builder.config);
        this.startTime = builder.startTime;
        this.version = builder.version;
        this.running = false;
    }

    @Override
    public GolekPluginRegistry pluginRegistry() {
        return pluginRegistry;
    }

    @Override
    public ProviderRegistry providerRegistry() {
        return providerRegistry;
    }

    @Override
    public ExecutorService executorService() {
        return executorService;
    }

    @Override
    public Map<String, Object> config() {
        return config;
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> T getConfig(String key, Class<T> type) {
        Object value = config.get(key);
        if (value == null) {
            return null;
        }
        if (type.isInstance(value)) {
            return (T) value;
        }
        throw new IllegalArgumentException(
                "Config value for key '" + key + "' is not of type " + type.getName());
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> T getConfig(String key, T defaultValue) {
        Object value = config.get(key);
        if (value == null) {
            return defaultValue;
        }
        return (T) value;
    }

    @Override
    public boolean isRunning() {
        return running;
    }

    public void setRunning(boolean running) {
        this.running = running;
    }

    @Override
    public Instant startTime() {
        return startTime;
    }

    @Override
    public String version() {
        return version;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private GolekPluginRegistry pluginRegistry;
        private ProviderRegistry providerRegistry;
        private ExecutorService executorService;
        private Map<String, Object> config = Map.of();
        private Instant startTime = Instant.now();
        private String version = "1.0.0";

        public Builder pluginRegistry(GolekPluginRegistry pluginRegistry) {
            this.pluginRegistry = pluginRegistry;
            return this;
        }

        public Builder providerRegistry(ProviderRegistry providerRegistry) {
            this.providerRegistry = providerRegistry;
            return this;
        }

        public Builder executorService(ExecutorService executorService) {
            this.executorService = executorService;
            return this;
        }

        public Builder config(Map<String, Object> config) {
            this.config = config;
            return this;
        }

        public Builder startTime(Instant startTime) {
            this.startTime = startTime;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public DefaultEngineContext build() {
            return new DefaultEngineContext(this);
        }
    }
}