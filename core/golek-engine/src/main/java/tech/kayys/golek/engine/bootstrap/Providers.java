package tech.kayys.golek.engine.bootstrap;

import java.util.ServiceLoader;

import tech.kayys.golek.engine.model.ModelRepositoryProvider;

public final class Providers {

    public static Iterable<ModelRepositoryProvider> load() {
        return ServiceLoader.load(ModelRepositoryProvider.class);
    }
}