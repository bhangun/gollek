package tech.kayys.golek.engine.resource;

import io.smallrye.mutiny.Uni;

public interface CircuitBreaker {
    <T> Uni<T> call(java.util.function.Supplier<Uni<T>> supplier);
}