package tech.kayys.wayang.mcp.repository;

import io.quarkus.hibernate.reactive.panache.PanacheRepository;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.mcp.domain.OpenApiSource;

import java.util.List;
import java.util.UUID;

@ApplicationScoped
public class OpenApiSourceRepositoryImpl implements PanacheRepository<OpenApiSource>, OpenApiSourceRepository {

    @Override
    public Uni<List<OpenApiSource>> listAllSources() {
        return listAll();
    }

    @Override
    public Uni<List<OpenApiSource>> findByTenantId(String tenantId) {
        return list("tenantId", tenantId);
    }

    @Override
    public Uni<List<OpenApiSource>> findByNamespace(String namespace) {
        return list("namespace", namespace);
    }

    @Override
    public Uni<List<OpenApiSource>> findByTenantIdAndNamespace(String tenantId, String namespace) {
        return list("tenantId = ?1 AND namespace = ?2", tenantId, namespace);
    }

    @Override
    public Uni<OpenApiSource> findById(UUID sourceId) {
        return find("sourceId", sourceId).firstResult();
    }

    @Override
    public Uni<OpenApiSource> findByTenantIdAndSourceId(String tenantId, UUID sourceId) {
        return find("tenantId = ?1 AND sourceId = ?2", tenantId, sourceId).firstResult();
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
    public Uni<Long> countByTenantId(String tenantId) {
        return count("tenantId", tenantId);
    }
}