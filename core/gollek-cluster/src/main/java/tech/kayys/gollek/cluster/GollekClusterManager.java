package tech.kayys.gollek.cluster;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Optional;
import java.util.UUID;

/**
 * Manages the identity and role of this node within the Gollek cluster.
 */
@ApplicationScoped
public class GollekClusterManager {

    private static final Logger LOG = Logger.getLogger(GollekClusterManager.class);

    private final NodeRole currentRole;
    private final String nodeId;
    private final String nodeAddress;

    @Inject
    public GollekClusterManager(
            @ConfigProperty(name = "gollek.cluster.role", defaultValue = "BOTH") NodeRole role,
            @ConfigProperty(name = "gollek.cluster.advertised-address") Optional<String> advertisedAddress,
            @ConfigProperty(name = "gollek.cluster.node-id") Optional<String> configNodeId) {
        
        this.currentRole = role;
        this.nodeId = configNodeId.orElseGet(() -> UUID.randomUUID().toString());
        this.nodeAddress = advertisedAddress.orElseGet(this::resolveLocalAddress);
        
        LOG.infof("Gollek Cluster Manager initialized. Node ID: %s, Role: %s, Address: %s",
                nodeId, currentRole, nodeAddress);
    }

    public NodeRole getCurrentRole() {
        return currentRole;
    }

    public String getNodeId() {
        return nodeId;
    }

    public String getNodeAddress() {
        return nodeAddress;
    }

    public boolean isPrefillNode() {
        return currentRole.canPrefill();
    }

    public boolean isDecodeNode() {
        return currentRole.canDecode();
    }

    private String resolveLocalAddress() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            LOG.warn("Could not resolve local host address, defaulting to 127.0.0.1", e);
            return "127.0.0.1";
        }
    }
}
