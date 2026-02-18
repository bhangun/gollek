package tech.kayys.gollek.engine.bootstrap;

import java.util.ServiceLoader;

import tech.kayys.gollek.engine.model.ModelRepositoryProvider;

public final class Providers {

    public static Iterable<ModelRepositoryProvider> load() {
        return ServiceLoader.load(ModelRepositoryProvider.class);
    }
}