package tech.kayys.golek.engine.model;

import tech.kayys.golek.model.core.RepositoryContext;

public interface ModelRepositoryProvider {

    String scheme(); // hf, local, s3, etc

    tech.kayys.golek.model.core.ModelRepository create(RepositoryContext context);
}