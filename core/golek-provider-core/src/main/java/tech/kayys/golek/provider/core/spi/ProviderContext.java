package tech.kayys.golek.provider.core.spi;

import io.micrometer.core.instrument.MeterRegistry;
import io.opentelemetry.api.trace.Tracer;
import tech.kayys.golek.provider.core.registry.ProviderRegistry;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Context provided to providers during initialization.
 * Contains shared resources and configuration.
 */
public final class ProviderContext {

    private final ProviderRegistry registry;
    private final MeterRegistry meterRegistry;
    private final Tracer tracer;
    private final Map<String, Object> sharedResources;
    private final Map<String, Object> attributes;

    private ProviderContext(Builder builder) {
        this.registry = builder.registry;
        this.meterRegistry = builder.meterRegistry;
        this.tracer = builder.tracer;
        this.sharedResources = new ConcurrentHashMap<>(builder.sharedResources);
        this.attributes = new ConcurrentHashMap<>(builder.attributes);
    }

    public ProviderRegistry getRegistry() {
        return registry;
    }

    public MeterRegistry getMeterRegistry() {
        return meterRegistry;
    }

    public Tracer getTracer() {
        return tracer;
    }

    public Map<String, Object> getSharedResources() {
        return sharedResources;
    }

    @SuppressWarnings("unchecked")
    public <T> Optional<T> getSharedResource(String key, Class<T> type) {
        Object value = sharedResources.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    public void putSharedResource(String key, Object value) {
        sharedResources.put(key, value);
    }

    public Map<String, Object> getAttributes() {
        return attributes;
    }

    @SuppressWarnings("unchecked")
    public <T> Optional<T> getAttribute(String key, Class<T> type) {
        Object value = attributes.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    public void setAttribute(String key, Object value) {
        attributes.put(key, value);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private ProviderRegistry registry;
        private MeterRegistry meterRegistry;
        private Tracer tracer;
        private final Map<String, Object> sharedResources = new ConcurrentHashMap<>();
        private final Map<String, Object> attributes = new ConcurrentHashMap<>();

        public Builder registry(ProviderRegistry registry) {
            this.registry = registry;
            return this;
        }

        public Builder meterRegistry(MeterRegistry meterRegistry) {
            this.meterRegistry = meterRegistry;
            return this;
        }

        public Builder tracer(Tracer tracer) {
            this.tracer = tracer;
            return this;
        }

        public Builder sharedResource(String key, Object value) {
            this.sharedResources.put(key, value);
            return this;
        }

        public Builder attribute(String key, Object value) {
            this.attributes.put(key, value);
            return this;
        }

        public ProviderContext build() {
            return new ProviderContext(this);
        }
    }
}