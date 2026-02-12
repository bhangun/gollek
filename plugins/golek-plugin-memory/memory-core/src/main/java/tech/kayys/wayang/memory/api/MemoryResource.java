package tech.kayys.wayang.memory.api;

import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import tech.kayys.wayang.memory.dto.*;
import tech.kayys.wayang.memory.model.*;
import tech.kayys.wayang.memory.service.*;
import tech.kayys.wayang.error.ErrorCode;
import tech.kayys.wayang.error.WayangException;

import java.util.List;
import java.util.Map;

/**
 * Memory executor REST API
 */
@Path("/api/memory")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class MemoryResource {

    @Inject
    VectorMemoryStore memoryStore;

    @Inject
    EmbeddingServiceFactory embeddingFactory;

    @Inject
    ContextEngineeringService contextService;

    /**
     * Store a new memory
     */
    @POST
    @Path("/store")
    public Uni<StoreResponse> storeMemory(StoreRequest request) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        return embeddingService.embed(request.content())
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace(request.namespace() != null ? request.namespace() : "default")
                            .content(request.content())
                            .embedding(embedding)
                            .type(request.type() != null ? request.type() : MemoryType.EPISODIC)
                            .importance(request.importance() != null ? request.importance() : 0.5)
                            .metadata(request.metadata() != null ? request.metadata() : Map.of())
                            .build();

                    return memoryStore.store(memory);
                })
                .map(memoryId -> new StoreResponse(true, memoryId, "Memory stored successfully"))
                .onFailure().transform(error ->
                        new WayangException(ErrorCode.MEMORY_STORE_FAILED, error.getMessage(), error));
    }

    /**
     * Search for similar memories
     */
    @POST
    @Path("/search")
    public Uni<SearchResponse> search(SearchRequest request) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        return embeddingService.embed(request.query())
                .flatMap(queryEmbedding -> memoryStore.search(
                        queryEmbedding,
                        request.limit() != null ? request.limit() : 10,
                        request.minSimilarity() != null ? request.minSimilarity() : 0.5,
                        Map.of("namespace", request.namespace() != null ? request.namespace() : "default")))
                .map(results -> {
                    List<MemoryResult> memoryResults = results.stream()
                            .map(scored -> new MemoryResult(
                                    scored.getMemory().getId(),
                                    scored.getMemory().getContent(),
                                    scored.getScore(),
                                    scored.getMemory().getType().name(),
                                    scored.getMemory().getImportance()))
                            .toList();

                    return new SearchResponse(true, memoryResults, memoryResults.size());
                })
                .onFailure().transform(error ->
                        new WayangException(ErrorCode.VECTOR_QUERY_FAILED, error.getMessage(), error));
    }

    /**
     * Build context for a query
     */
    @POST
    @Path("/context")
    public Uni<ContextResponse> buildContext(ContextRequest request) {
        ContextConfig config = ContextConfig.builder()
                .maxMemories(request.maxMemories() != null ? request.maxMemories() : 10)
                .systemPrompt(request.systemPrompt())
                .taskInstructions(request.taskInstructions())
                .includeMetadata(request.includeMetadata() != null ? request.includeMetadata() : true)
                .build();

        return contextService.buildContext(
                request.query(),
                request.namespace() != null ? request.namespace() : "default",
                config)
                .map(context -> new ContextResponse(
                        true,
                        context.toPrompt(),
                        context.getTotalTokens(),
                        context.getUtilization(),
                        context.getSections().size()))
                .onFailure().transform(error ->
                        new WayangException(ErrorCode.MEMORY_STORE_FAILED, error.getMessage(), error));
    }

    /**
     * Get memory statistics
     */
    @GET
    @Path("/stats/{namespace}")
    public Uni<StatsResponse> getStatistics(@PathParam("namespace") String namespace) {
        return memoryStore.getStatistics(namespace)
                .map(stats -> new StatsResponse(
                        true,
                        stats.getTotalMemories(),
                        stats.getEpisodicCount(),
                        stats.getSemanticCount(),
                        stats.getProceduralCount(),
                        stats.getWorkingCount(),
                        stats.getAvgImportance()))
                .onFailure().transform(error ->
                        new WayangException(ErrorCode.STORAGE_READ_FAILED, error.getMessage(), error));
    }

    /**
     * Run examples
     */
    @GET
    @Path("/examples/run")
    public Uni<ExampleResponse> runExamples() {
        // This would inject and run the examples
        return Uni.createFrom().item(
                new ExampleResponse(true, "Examples would run here. Check logs.", 7));
    }
}
