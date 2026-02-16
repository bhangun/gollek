package tech.kayys.golek.runtime.unified.impl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;
import jakarta.enterprise.inject.Produces;
import tech.kayys.golek.spi.context.RequestContext;

@ApplicationScoped
public class RequestContextProducer {

    @Produces
    @Default
    public RequestContext produceRequestContext() {
        return DefaultRequestContext.INSTANCE;
    }
}