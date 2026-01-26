package tech.kayys.golek.model.repository;

import tech.kayys.golek.model.core.RepositoryContext;

public interface ModelRepositoryProvider {

    String scheme(); // hf, local, s3, etc

    ModelRepository create(RepositoryContext context);
}