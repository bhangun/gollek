package tech.kayys.wayang.tool.repository;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tool.entity.OpenApiSource;

import java.util.List;
import java.util.UUID;

public interface OpenApiSourceRepository {

    Uni<List<OpenApiSource>> listAllSources();

    Uni<List<OpenApiSource>> findByRequestId(String requestId);

    Uni<List<OpenApiSource>> findByNamespace(String namespace);

    Uni<List<OpenApiSource>> findByRequestIdAndNamespace(String requestId, String namespace);

    Uni<OpenApiSource> findById(UUID sourceId);

    Uni<OpenApiSource> findByRequestIdAndSourceId(String requestId, UUID sourceId);

    Uni<OpenApiSource> save(OpenApiSource source);

    Uni<OpenApiSource> update(OpenApiSource source);

    Uni<Boolean> deleteById(UUID sourceId);

    Uni<List<OpenApiSource>> searchSources(String query, Object... params);

    Uni<Long> count();

    Uni<Long> countByRequestId(String requestId);
}