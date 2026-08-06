package tech.kayys.gollek.sdk.api;

/**
 * Defines the deployment architecture mode for the GollekClient.
 * Each mode has different latency, isolation, and throughput tradeoffs.
 */
public enum DeploymentMode {
    /**
     * In-process CDI injection. Zero network latency, shared heap.
     * Best for single-node deployments and fast development.
     */
    EMBEDDED,

    /**
     * HTTP/2 Protobuf communication. High throughput, streaming, strict isolation.
     * Recommended for production distributed and Kubernetes deployments.
     */
    GRPC,

    /**
     * HTTP/1.1 JSON communication.
     * Useful for exposing public REST endpoints and simple integrations.
     */
    REST,

    /**
     * Subprocess execution via gollek-cli.
     * High latency, strict isolation. Only recommended for tools, converters, or diagnostics.
     */
    CLI
}
