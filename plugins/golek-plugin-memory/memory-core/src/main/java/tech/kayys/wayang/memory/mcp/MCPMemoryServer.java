package tech.kayys.wayang.memory.mcp;

import tech.kayys.wayang.memory.model.MemoryContext;
import tech.kayys.wayang.memory.service.MemoryService;
import tech.kayys.wayang.memory.service.MemoryServiceImpl;
import tech.kayys.wayang.memory.model.ConversationMemory;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.lsp4j.jsonrpc.Launcher;
import org.eclipse.lsp4j.jsonrpc.MessageConsumer;
import org.eclipse.lsp4j.jsonrpc.json.JsonRpcMethod;
import org.eclipse.lsp4j.jsonrpc.json.JsonRpcMethodProvider;
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest;
import org.eclipse.lsp4j.jsonrpc.services.JsonRpcService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.error.ErrorCode;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

@ApplicationScoped
@JsonRpcService
public class MCPMemoryServer implements JsonRpcMethodProvider {
    
    private static final Logger LOG = LoggerFactory.getLogger(MCPMemoryServer.class);
    
    @Inject
    MemoryService memoryService;
    
    @Inject
    ObjectMapper objectMapper;

    @JsonRequest("memory/getContext")
    public CompletableFuture<MCPResponse<MemoryContext>> getContext(MCPRequest<GetContextParams> request) {
        LOG.info("MCP: getContext called with sessionId: {}", request.params.sessionId);
        
        return memoryService.getContext(request.params.sessionId, request.params.userId)
            .onItem().transform(context -> new MCPResponse<>(context, null))
            .onFailure().recoverWithItem(throwable -> 
                new MCPResponse<>(null, new MCPError(ErrorCode.MEMORY_NOT_FOUND.getCode(), throwable.getMessage()))
            )
            .subscribe().asCompletionStage()
            .toCompletableFuture();
    }

    @JsonRequest("memory/storeContext")
    public CompletableFuture<MCPResponse<Void>> storeContext(MCPRequest<StoreContextParams> request) {
        LOG.info("MCP: storeContext called for sessionId: {}", request.params.context.getSessionId());
        
        return memoryService.storeContext(request.params.context)
            .onItem().transform(unused -> new MCPResponse<Void>(null, null))
            .onFailure().recoverWithItem(throwable -> 
                new MCPResponse<Void>(null, new MCPError(ErrorCode.MEMORY_STORE_FAILED.getCode(), throwable.getMessage()))
            )
            .subscribe().asCompletionStage()
            .toCompletableFuture();
    }

    @JsonRequest("memory/findSimilar")
    public CompletableFuture<MCPResponse<List<ConversationMemory>>> findSimilar(MCPRequest<FindSimilarParams> request) {
        LOG.info("MCP: findSimilar called for sessionId: {}, query: {}", 
                request.params.sessionId, request.params.query);
        
        return ((MemoryServiceImpl) memoryService).findSimilarMemories(
                request.params.sessionId, 
                request.params.query, 
                request.params.limit != null ? request.params.limit : 10
            )
            .onItem().transform(memories -> new MCPResponse<>(memories, null))
            .onFailure().recoverWithItem(throwable -> 
                new MCPResponse<>(null, new MCPError(ErrorCode.VECTOR_QUERY_FAILED.getCode(), throwable.getMessage()))
            )
            .subscribe().asCompletionStage()
            .toCompletableFuture();
    }

    @JsonRequest("memory/summarize")
    public CompletableFuture<MCPResponse<MemoryContext>> summarizeAndCompact(MCPRequest<SummarizeParams> request) {
        LOG.info("MCP: summarizeAndCompact called for sessionId: {}", request.params.sessionId);
        
        return ((MemoryServiceImpl) memoryService).summarizeAndCompact(request.params.sessionId)
            .onItem().transform(context -> new MCPResponse<>(context, null))
            .onFailure().recoverWithItem(throwable -> 
                new MCPResponse<>(null, new MCPError(ErrorCode.MEMORY_STORE_FAILED.getCode(), throwable.getMessage()))
            )
            .subscribe().asCompletionStage()
            .toCompletableFuture();
    }

    @Override
    public Map<String, JsonRpcMethod> supportedMethods() {
        return Map.of(
            "memory/getContext", JsonRpcMethod.request("memory/getContext", GetContextParams.class, MemoryContext.class),
            "memory/storeContext", JsonRpcMethod.request("memory/storeContext", StoreContextParams.class, Void.class),
            "memory/findSimilar", JsonRpcMethod.request("memory/findSimilar", FindSimilarParams.class, List.class),
            "memory/summarize", JsonRpcMethod.request("memory/summarize", SummarizeParams.class, MemoryContext.class)
        );
    }

    // MCP Protocol Models
    public static class MCPRequest<T> {
        public String id;
        public String method;
        public T params;
    }

    public static class MCPResponse<T> {
        public final T result;
        public final MCPError error;

        public MCPResponse(T result, MCPError error) {
            this.result = result;
            this.error = error;
        }
    }

    public static class MCPError {
        public final String code;
        public final String message;

        public MCPError(String code, String message) {
            this.code = code;
            this.message = message;
        }
    }

    public static class GetContextParams {
        public String sessionId;
        public String userId;
    }

    public static class StoreContextParams {
        public MemoryContext context;
    }

    public static class FindSimilarParams {
        public String sessionId;
        public String query;
        public Integer limit;
    }

    public static class SummarizeParams {
        public String sessionId;
    }
}
