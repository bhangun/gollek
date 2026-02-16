package tech.kayys.golek.runtime.unified.impl;

import tech.kayys.golek.spi.context.RequestContext;

public class DefaultRequestContext {

    public static final RequestContext INSTANCE = RequestContext.of("unified-runtime");
}