package tech.kayys.golek.model.bootstrap;

import java.util.ServiceLoader;

import tech.kayys.golek.model.repository.ModelRepositoryProvider;

public final class Providers {

    public static Iterable<ModelRepositoryProvider> load() {
        return ServiceLoader.load(ModelRepositoryProvider.class);
    }
}