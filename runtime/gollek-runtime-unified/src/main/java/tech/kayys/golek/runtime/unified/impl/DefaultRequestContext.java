package tech.kayys.gollek.runtime.unified.impl;

import tech.kayys.gollek.spi.context.RequestContext;

public class DefaultRequestContext {

    public static final RequestContext INSTANCE = RequestContext.of("unified-runtime");
}