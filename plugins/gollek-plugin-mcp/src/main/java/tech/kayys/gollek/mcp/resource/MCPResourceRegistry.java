package tech.kayys.gollek.mcp.resource;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.mcp.dto.MCPConnection;
import tech.kayys.gollek.mcp.dto.MCPResource;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for all MCP resources across connections.
 * Thread-safe and supports resource discovery and caching.
 */
@ApplicationScoped
public class MCPResourceRegistry {

    private static final Logger LOG = Logger.getLogger(MCPResourceRegistry.class);

    // connectionName -> resources
    private final Map<String, Map<String, MCPResource>> resourcesByConnection = new ConcurrentHashMap<>();

    // uri -> connectionName
    private final Map<String, String> uriToConnection = new ConcurrentHashMap<>();

    /**
     * Register resources from a connection
     */
    public void registerConnection(MCPConnection connection) {
        String connectionName = connection.getConfig().getName();
        Map<String, MCPResource> resources = connection.getResources();

        resourcesByConnection.put(connectionName, new ConcurrentHashMap<>(resources));

        // Update quick lookup map
        resources.values().forEach(resource -> uriToConnection.put(resource.getUri(), connectionName));

        LOG.infof("Registered %d resources from connection: %s", resources.size(), connectionName);
    }

    /**
     * Unregister connection
     */
    public void unregisterConnection(String connectionName) {
        Map<String, MCPResource> resources = resourcesByConnection.remove(connectionName);
        if (resources != null) {
            resources.values().forEach(resource -> uriToConnection.remove(resource.getUri()));
            LOG.infof("Unregistered connection: %s", connectionName);
        }
    }

    /**
     * Get resource by URI
     */
    public Optional<MCPResource> getResource(String uri) {
        String connectionName = uriToConnection.get(uri);
        if (connectionName == null) {
            return Optional.empty();
        }

        Map<String, MCPResource> resources = resourcesByConnection.get(connectionName);
        return Optional.ofNullable(resources != null ? resources.get(uri) : null);
    }

    /**
     * Get connection for resource
     */
    public Optional<String> getConnectionForResource(String uri) {
        return Optional.ofNullable(uriToConnection.get(uri));
    }

    /**
     * Get all resources
     */
    public List<MCPResource> getAllResources() {
        return resourcesByConnection.values().stream()
                .flatMap(resources -> resources.values().stream())
                .toList();
    }

    /**
     * Get resources by connection
     */
    public List<MCPResource> getResourcesByConnection(String connectionName) {
        Map<String, MCPResource> resources = resourcesByConnection.get(connectionName);
        return resources != null ? new ArrayList<>(resources.values()) : Collections.emptyList();
    }

    /**
     * Search resources by keyword
     */
    public List<MCPResource> searchResources(String keyword) {
        String lowerKeyword = keyword.toLowerCase();
        return getAllResources().stream()
                .filter(resource -> resource.getUri().toLowerCase().contains(lowerKeyword) ||
                        (resource.getName() != null && resource.getName().toLowerCase().contains(lowerKeyword)) ||
                        (resource.getDescription() != null
                                && resource.getDescription().toLowerCase().contains(lowerKeyword)))
                .toList();
    }

    /**
     * Filter resources by MIME type
     */
    public List<MCPResource> filterByMimeType(String mimeType) {
        return getAllResources().stream()
                .filter(resource -> mimeType.equals(resource.getMimeType()))
                .toList();
    }

    /**
     * Get resource count
     */
    public int getTotalResourceCount() {
        return uriToConnection.size();
    }

    /**
     * Check if resource exists
     */
    public boolean hasResource(String uri) {
        return uriToConnection.containsKey(uri);
    }

    /**
     * Clear all registrations
     */
    public void clear() {
        resourcesByConnection.clear();
        uriToConnection.clear();
        LOG.info("Cleared all resource registrations");
    }
}