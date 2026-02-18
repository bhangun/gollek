package tech.kayys.gollek.engine.model;

import tech.kayys.gollek.model.core.RepositoryContext;

public interface ModelRepositoryProvider {

    String scheme(); // hf, local, s3, etc

    tech.kayys.gollek.model.core.ModelRepository create(RepositoryContext context);
}