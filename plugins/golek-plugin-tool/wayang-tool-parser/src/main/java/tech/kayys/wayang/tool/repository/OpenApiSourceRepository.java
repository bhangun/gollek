package tech.kayys.wayang.mcp.repository;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.mcp.domain.OpenApiSource;

import java.util.List;
import java.util.UUID;

public interface OpenApiSourceRepository {

    Uni<List<OpenApiSource>> listAllSources();

    Uni<List<OpenApiSource>> findByTenantId(String tenantId);

    Uni<List<OpenApiSource>> findByNamespace(String namespace);

    Uni<List<OpenApiSource>> findByTenantIdAndNamespace(String tenantId, String namespace);

    Uni<OpenApiSource> findById(UUID sourceId);

    Uni<OpenApiSource> findByTenantIdAndSourceId(String tenantId, UUID sourceId);

    Uni<OpenApiSource> save(OpenApiSource source);

    Uni<OpenApiSource> update(OpenApiSource source);

    Uni<Boolean> deleteById(UUID sourceId);

    Uni<List<OpenApiSource>> searchSources(String query, Object... params);

    Uni<Long> count();

    Uni<Long> countByTenantId(String tenantId);
}