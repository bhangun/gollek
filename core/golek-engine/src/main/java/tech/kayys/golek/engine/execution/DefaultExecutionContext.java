package tech.kayys.golek.engine.execution;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.execution.ExecutionToken;
import tech.kayys.golek.spi.execution.ExecutionStatus;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.inference.InferencePhase;

/**
 * Default implementation of execution context.
 */
public class DefaultExecutionContext implements ExecutionContext {

    private final EngineContext engineContext;
    private final TenantContext tenantContext;
    private final AtomicReference<ExecutionToken> tokenRef;
    private final AtomicReference<Throwable> errorRef;

    public DefaultExecutionContext(
            EngineContext engineContext,
            TenantContext tenantContext,
            ExecutionToken initialToken) {
        this.engineContext = engineContext;
        this.tenantContext = tenantContext != null ? tenantContext : TenantContext.of("default");
        this.tokenRef = new AtomicReference<>(initialToken);
        this.errorRef = new AtomicReference<>();
    }

    @Override
    public EngineContext engine() {
        return engineContext;
    }

    @Override
    public ExecutionToken token() {
        return tokenRef.get();
    }

    @Override
    public TenantContext tenantContext() {
        return tenantContext;
    }

    @Override
    public void updateStatus(ExecutionStatus status) {
        tokenRef.updateAndGet(token -> token.withStatus(status));
    }

    @Override
    public void updatePhase(InferencePhase phase) {
        tokenRef.updateAndGet(token -> token.withPhase(phase));
    }

    @Override
    public void incrementAttempt() {
        tokenRef.updateAndGet(ExecutionToken::withNextAttempt);
    }

    @Override
    public Map<String, Object> variables() {
        return token().getVariables();
    }

    @Override
    public void putVariable(String key, Object value) {
        tokenRef.updateAndGet(token -> token.withVariable(key, value));
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getVariable(String key, Class<T> type) {
        Object value = variables().get(key);
        if (value == null) {
            return Optional.empty();
        }
        if (type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    @Override
    public void removeVariable(String key) {
        Map<String, Object> vars = token().getVariables();
        vars.remove(key);
    }

    @Override
    public Map<String, Object> metadata() {
        return token().getMetadata();
    }

    @Override
    public void putMetadata(String key, Object value) {
        tokenRef.updateAndGet(token -> token.withMetadata(key, value));
    }

    @Override
    public void replaceToken(ExecutionToken newToken) {
        tokenRef.set(newToken);
    }

    @Override
    public boolean hasError() {
        return errorRef.get() != null;
    }

    @Override
    public Optional<Throwable> getError() {
        return Optional.ofNullable(errorRef.get());
    }

    @Override
    public void setError(Throwable error) {
        errorRef.set(error);
    }

    @Override
    public void clearError() {
        errorRef.set(null);
    }

    @Override
    public String toString() {
        return "DefaultExecutionContext{" +
                "token=" + token() +
                ", tenant=" + tenantContext.getTenantId() +
                ", hasError=" + hasError() +
                '}';
    }
}
