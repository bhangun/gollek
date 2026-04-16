package tech.kayys.gollek.mcp.resource;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.mcp.dto.MCPResource;
import tech.kayys.gollek.mcp.dto.MCPResourceContent;
import tech.kayys.gollek.mcp.client.MCPClient;
import tech.kayys.gollek.mcp.dto.MCPConnection;
import tech.kayys.gollek.mcp.dto.MCPResponse;
import tech.kayys.gollek.mcp.exception.ConnectionNotFoundException;

import java.time.Duration;
import java.util.*;

/**
 * Provider for reading MCP resources with caching support.
 */
@ApplicationScoped
public class MCPResourceProvider {

    private static final Logger LOG = Logger.getLogger(MCPResourceProvider.class);

    @Inject
    MCPClient mcpClient;

    @Inject
    MCPResourceRegistry resourceRegistry;

    /**
     * Register resources from a connection
     */
    public void registerConnection(MCPConnection connection) {
        resourceRegistry.registerConnection(connection);
    }

    @Inject
    MCPResourceCache resourceCache;

    /**
     * Read resource content
     */
    public Uni<MCPResourceContent> readResource(String uri) {
        return readResource(uri, false);
    }

    /**
     * Read resource with cache control
     */
    public Uni<MCPResourceContent> readResource(String uri, boolean bypassCache) {
        LOG.debugf("Reading resource: %s (bypassCache=%s)", uri, bypassCache);

        // Check cache first (unless bypassing)
        if (!bypassCache) {
            Optional<MCPResourceContent> cached = resourceCache.get(uri);
            if (cached.isPresent()) {
                LOG.debugf("Resource cache hit: %s", uri);
                return Uni.createFrom().item(cached.get());
            }
        }

        // Validate resource exists
        Optional<MCPResource> resourceOpt = resourceRegistry.getResource(uri);
        if (resourceOpt.isEmpty()) {
            return Uni.createFrom().failure(
                    new ResourceNotFoundException("Resource not found: " + uri));
        }

        // Get connection
        Optional<String> connectionNameOpt = resourceRegistry.getConnectionForResource(uri);
        if (connectionNameOpt.isEmpty()) {
            return Uni.createFrom().failure(
                    new ResourceNotFoundException("No connection for resource: " + uri));
        }

        String connectionName = connectionNameOpt.get();
        Optional<MCPConnection> connectionOpt = mcpClient.getConnection(connectionName);
        if (connectionOpt.isEmpty()) {
            return Uni.createFrom().failure(
                    new ConnectionNotFoundException("Connection not found: " + connectionName));
        }

        MCPConnection connection = connectionOpt.get();

        // Read resource
        return connection.readResource(uri)
                .ifNoItem().after(Duration.ofSeconds(30)).fail()
                .onItem().transform(response -> convertResponse(uri, response))
                .onItem().invoke(content -> resourceCache.put(uri, content))
                .onFailure().invoke(error -> LOG.errorf(error, "Failed to read resource: %s", uri));
    }

    /**
     * Convert MCP response to resource content
     */
    private MCPResourceContent convertResponse(String uri, MCPResponse response) {
        if (!response.isSuccess()) {
            throw new ResourceReadException(
                    "Failed to read resource: " + response.getError());
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) response.getResult();
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> contents = (List<Map<String, Object>>) result.get("contents");

        if (contents == null || contents.isEmpty()) {
            return MCPResourceContent.text(uri, "");
        }

        // Take first content item
        Map<String, Object> content = contents.get(0);
        String mimeType = (String) content.get("mimeType");
        String text = (String) content.get("text");
        String blob = (String) content.get("blob");

        return MCPResourceContent.builder()
                .uri(uri)
                .mimeType(mimeType)
                .text(text)
                .blob(blob)
                .build();
    }

    /**
     * Read multiple resources in parallel
     */
    public Uni<Map<String, MCPResourceContent>> readResources(List<String> uris) {
        if (uris == null || uris.isEmpty()) {
            return Uni.createFrom().item(Map.of());
        }

        return Uni.combine().all().unis(
                uris.stream()
                        .map(uri -> readResource(uri)
                                .onItem().transform(content -> Map.entry(uri, content))
                                .onFailure().recoverWithItem(error -> {
                                    LOG.warnf("Failed to read resource %s: %s", uri, error.getMessage());
                                    return Map.entry(uri, null);
                                }))
                        .toList())
                .with(results -> results.stream()
                        .filter(Objects::nonNull)
                        .map(entry -> (Map.Entry<String, MCPResourceContent>) entry)
                        .filter(entry -> entry.getValue() != null)
                        .collect(java.util.stream.Collectors.toMap(
                                Map.Entry::getKey,
                                Map.Entry::getValue,
                                (first, second) -> first,
                                LinkedHashMap::new)));
    }

    /**
     * Invalidate cache for resource
     */
    public void invalidateCache(String uri) {
        resourceCache.invalidate(uri);
    }

    /**
     * Clear all cache
     */
    public void clearCache() {
        resourceCache.clear();
    }
}
