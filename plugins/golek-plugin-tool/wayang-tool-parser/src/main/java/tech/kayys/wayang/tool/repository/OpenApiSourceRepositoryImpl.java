package tech.kayys.wayang.tool.repository;

import io.quarkus.hibernate.reactive.panache.PanacheRepository;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.OpenApiSource;

import java.util.List;
import java.util.UUID;

@ApplicationScoped
public class OpenApiSourceRepositoryImpl implements PanacheRepository<OpenApiSource>, OpenApiSourceRepository {

    @Override
    public Uni<List<OpenApiSource>> listAllSources() {
        return listAll();
    }

    @Override
    public Uni<List<OpenApiSource>> findByRequestId(String requestId) {
        return list("requestId", requestId);
    }

    @Override
    public Uni<List<OpenApiSource>> findByNamespace(String namespace) {
        return list("namespace", namespace);
    }

    @Override
    public Uni<List<OpenApiSource>> findByRequestIdAndNamespace(String requestId, String namespace) {
        return list("requestId = ?1 AND namespace = ?2", requestId, namespace);
    }

    @Override
    public Uni<OpenApiSource> findById(UUID sourceId) {
        return find("sourceId", sourceId).firstResult();
    }

    @Override
    public Uni<OpenApiSource> findByRequestIdAndSourceId(String requestId, UUID sourceId) {
        return find("requestId = ?1 AND sourceId = ?2", requestId, sourceId).firstResult();
    }

    @Override
    public Uni<OpenApiSource> save(OpenApiSource source) {
        return persist(source);
    }

    @Override
    public Uni<OpenApiSource> update(OpenApiSource source) {
        return persist(source);
    }

    @Override
    public Uni<Boolean> deleteById(UUID sourceId) {
        return delete("sourceId", sourceId).map(deleted -> deleted > 0);
    }

    @Override
    public Uni<List<OpenApiSource>> searchSources(String query, Object... params) {
        return list(query, params);
    }

    @Override
    public Uni<Long> count() {
        return PanacheRepository.super.count();
    }

    @Override
    public Uni<Long> countByRequestId(String requestId) {
        return count("requestId", requestId);
    }
}