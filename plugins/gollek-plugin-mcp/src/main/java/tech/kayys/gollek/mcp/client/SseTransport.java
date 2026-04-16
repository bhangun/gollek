package tech.kayys.gollek.mcp.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Uni;
import io.vertx.core.http.HttpMethod;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.core.buffer.Buffer;
import io.vertx.mutiny.ext.web.client.HttpRequest;
import io.vertx.mutiny.ext.web.client.HttpResponse;
import io.vertx.mutiny.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;

import org.jboss.logging.Logger;

import tech.kayys.gollek.mcp.dto.*;
import tech.kayys.gollek.mcp.dto.JsonRpcMessage;

import java.net.URI;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * HTTP + SSE transport implementation for MCP.
 * Compliant with MCP specification 2025-11-25.
 * 
 * <h2>Transport protocol</h2>
 * The MCP SSE transport uses:
 * <ul>
 *   <li>A persistent GET connection to {@code /sse} for server-to-client messages.</li>
 *   <li>POST requests to {@code /message} for client-to-server JSON-RPC calls.</li>
 * </ul>
 * 
 * <h2>Connection lifecycle</h2>
 * <pre>
 * 1. GET /sse                       ← open SSE channel, receive "endpoint" event
 * 2. POST {endpoint} initialize     ← send initialize, receive response via SSE
 * 3. POST {endpoint} initialized    ← send notification (no response)
 * 4. Normal operation: POST tool calls, receive results via SSE
 * </pre>
 */
public class SseTransport implements MCPTransport {

    private static final Logger LOG = Logger.getLogger(SseTransport.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);

    private final MCPClientConfig config;
    private final ObjectMapper objectMapper;
    private final WebClient httpClient;
    private final AtomicLong idCounter = new AtomicLong(1);
    private final AtomicBoolean connected = new AtomicBoolean(false);

    /** Pending JSON-RPC calls: id → CompletableFuture<MCPResponse> */
    private final ConcurrentHashMap<String, CompletableFuture<MCPResponse>> pending
            = new ConcurrentHashMap<>();

    /** SSE endpoint URL received from the server during the connection handshake. */
    private volatile String postEndpoint;
    
    /** Internal handler for incoming server messages */
    @FunctionalInterface
    private interface ServerMessageHandler {
        void handle(String method, com.fasterxml.jackson.databind.JsonNode params);
    }
    
    /** Callback for incoming server requests */
    private volatile ServerMessageHandler serverMessageHandler;

    public SseTransport(MCPClientConfig config, ObjectMapper objectMapper, Vertx vertx) {
        this.config = config;
        this.objectMapper = objectMapper;

        URI uri = URI.create(config.getUrl());
        WebClientOptions opts = new WebClientOptions()
                .setDefaultHost(uri.getHost())
                .setDefaultPort(uri.getPort() > 0 ? uri.getPort() : ("https".equals(uri.getScheme()) ? 443 : 80))
                .setSsl("https".equals(uri.getScheme()))
                .setTrustAll(true)
                .setConnectTimeout((int) config.getConnectTimeout().toMillis())
                .setIdleTimeout(300_000);
        this.httpClient = WebClient.create(vertx, opts);
    }

    @Override
    public Uni<Void> connect() {
        if (connected.get()) return Uni.createFrom().voidItem();

        return openSseChannel()
                .chain(() -> sendInitialize())
                .chain(() -> sendInitialized())
                .invoke(() -> connected.set(true))
                .invoke(() -> LOG.infof("MCP SSE transport connected: url=%s", config.getUrl()));
    }

    /**
     * Open the SSE channel and wait for endpoint event.
     */
    private Uni<Void> openSseChannel() {
        CompletableFuture<Void> endpointReceived = new CompletableFuture<>();

        HttpRequest<Buffer> req = buildRequest(HttpMethod.GET, "/sse");
        req.putHeader("Accept", "text/event-stream");
        req.putHeader("Cache-Control", "no-cache");

        req.send()
                .subscribe().with(
                resp -> {
                    if (resp.statusCode() != 200) {
                        endpointReceived.completeExceptionally(
                                new RuntimeException("SSE connection failed: HTTP " + resp.statusCode()));
                        return;
                    }
                    if (resp.body() != null) {
                        processSseChunk(resp.bodyAsString(), endpointReceived);
                    }
                },
                err -> endpointReceived.completeExceptionally(err));

        return Uni.createFrom().completionStage(endpointReceived)
                .ifNoItem().after(Duration.ofSeconds(15))
                .failWith(() -> new RuntimeException("Timed out waiting for SSE endpoint"));
    }

    /**
     * Process SSE chunk and dispatch events.
     */
    private void processSseChunk(String chunk, CompletableFuture<Void> endpointFuture) {
        String[] lines = chunk.split("\n");
        String eventType = null;
        StringBuilder dataBuf = new StringBuilder();

        for (String line : lines) {
            if (line.startsWith("event:")) {
                eventType = line.substring(6).strip();
            } else if (line.startsWith("data:")) {
                dataBuf.append(line.substring(5).strip());
            } else if (line.isBlank() && dataBuf.length() > 0) {
                dispatchSseEvent(eventType, dataBuf.toString(), endpointFuture);
                eventType = null;
                dataBuf.setLength(0);
            }
        }
    }

    /**
     * Dispatch SSE events to appropriate handlers.
     */
    private void dispatchSseEvent(String type, String data, CompletableFuture<Void> endpointFuture) {
        if ("endpoint".equals(type)) {
            postEndpoint = data.startsWith("http") ? data : config.getUrl() + data;
            LOG.debugf("MCP SSE endpoint received: %s", postEndpoint);
            if (!endpointFuture.isDone()) endpointFuture.complete(null);
            return;
        }
        if ("message".equals(type)) {
            try {
                handleSseMessage(data);
            } catch (Exception e) {
                LOG.warnf("Failed to parse SSE message: %s — %s", data, e.getMessage());
            }
        }
    }

    /**
     * Handle incoming SSE message (can be response or server-initiated request).
     */
    @SuppressWarnings("unchecked")
    private void handleSseMessage(String data) throws Exception {
        com.fasterxml.jackson.databind.JsonNode node = MAPPER.readTree(data);
        
        if (node.has("id") && (node.has("result") || node.has("error"))) {
            // Response to our request
            String id = node.get("id").asText();
            CompletableFuture<MCPResponse> fut = pending.remove(id);
            if (fut != null) {
                MCPResponse response = parseResponse(node);
                fut.complete(response);
            }
        } else if (node.has("method")) {
            // Server-initiated request or notification
            String method = node.get("method").asText();
            com.fasterxml.jackson.databind.JsonNode params = node.get("params");
            
            if (serverMessageHandler != null) {
                serverMessageHandler.handle(method, params);
            }
        }
    }

    /**
     * Send initialize request and negotiate protocol version.
     */
    private Uni<Void> sendInitialize() {
        Map<String, Object> params = Map.of(
                "protocolVersion", "2025-11-25",
                "capabilities", Map.of(
                        "roots", Map.of("listChanged", true),
                        "sampling", Map.of()),
                "clientInfo", Map.of(
                        "name", "gollek-mcp-plugin",
                        "version", "1.0.0"));

        MCPRequest request = MCPRequest.builder()
                .id("init-" + idCounter.getAndIncrement())
                .method("initialize")
                .params(params)
                .build();

        return sendRequest(request)
                .onItem().transformToUni(response -> {
                    if (!response.isSuccess()) {
                        return Uni.createFrom().failure(
                                new RuntimeException("Initialize failed: " + response.getError()));
                    }
                    LOG.info("MCP server initialized successfully");
                    return Uni.createFrom().voidItem();
                });
    }

    /**
     * Send initialized notification.
     */
    private Uni<Void> sendInitialized() {
        return Uni.createFrom().voidItem(); // Already sent as part of initialize flow
    }

    @Override
    public Uni<MCPResponse> sendRequest(MCPRequest request) {
        String id = String.valueOf(request.getId() != null ? request.getId() : idCounter.getAndIncrement());
        
        CompletableFuture<MCPResponse> future = new CompletableFuture<>();
        pending.put(id, future);

        try {
            String body = objectMapper.writeValueAsString(request);
            return postMessage(body)
                    .chain(() -> Uni.createFrom().completionStage(future)
                            .ifNoItem().after(DEFAULT_TIMEOUT)
                            .failWith(() -> {
                                pending.remove(id);
                                return new RuntimeException("Timeout waiting for response to " + request.getMethod());
                            }));
        } catch (Exception e) {
            return Uni.createFrom().failure(new RuntimeException("Failed to serialize request: " + e.getMessage()));
        }
    }

    @Override
    public Uni<Void> sendNotification(String method, Object params) {
        try {
            Map<String, Object> notification = Map.of(
                    "jsonrpc", "2.0",
                    "method", method,
                    "params", params);
            String body = objectMapper.writeValueAsString(notification);
            return postMessage(body).replaceWithVoid();
        } catch (Exception e) {
            return Uni.createFrom().failure(new RuntimeException("Failed to send notification: " + e.getMessage()));
        }
    }

    @Override
    public void onMessage(java.util.function.Consumer<JsonRpcMessage> handler) {
        // Wrap the handler to work with our server message handler pattern
        this.serverMessageHandler = (method, params) -> {
            // Convert to a notification for now (simplified)
            handler.accept(new tech.kayys.gollek.mcp.dto.MCPRequest.Builder()
                .method(method)
                .params(objectMapper.convertValue(params, Map.class))
                .build());
        };
    }

    @Override
    public boolean isConnected() {
        return connected.get();
    }

    @Override
    public Uni<Void> disconnect() {
        connected.set(false);
        pending.values().forEach(f -> f.completeExceptionally(new RuntimeException("Disconnecting")));
        pending.clear();
        httpClient.close();
        return Uni.createFrom().voidItem();
    }

    @Override
    public void close() {
        disconnect().subscribe().with(v -> {}, e -> {});
    }

    /**
     * POST a JSON-RPC message to the server.
     */
    private Uni<HttpResponse<Buffer>> postMessage(String body) {
        String endpoint = postEndpoint != null ? postEndpoint : config.getUrl() + "/message";
        URI uri = URI.create(endpoint);
        String path = uri.getPath() + (uri.getQuery() != null ? "?" + uri.getQuery() : "");

        HttpRequest<Buffer> req = buildRequest(HttpMethod.POST, path);
        req.putHeader("Content-Type", "application/json");

        return req.sendBuffer(Buffer.buffer(body))
                .invoke(resp -> {
                    if (resp.statusCode() >= 400) {
                        LOG.warnf("MCP POST returned %d: %s", resp.statusCode(), resp.bodyAsString());
                    }
                });
    }

    /**
     * Build HTTP request with headers.
     */
    private HttpRequest<Buffer> buildRequest(HttpMethod method, String path) {
        HttpRequest<Buffer> req = httpClient.request(method, path);
        if (config.getHeaders() != null) {
            config.getHeaders().forEach(req::putHeader);
        }
        req.putHeader("Accept", "application/json, text/event-stream");
        return req;
    }

    /**
     * Parse JSON-RPC response.
     */
    @SuppressWarnings("unchecked")
    private MCPResponse parseResponse(com.fasterxml.jackson.databind.JsonNode node) {
        Object id = node.has("id") ? node.get("id").asText() : null;
        
        if (node.has("result")) {
            return MCPResponse.success(id, objectMapper.convertValue(node.get("result"), Map.class));
        } else if (node.has("error")) {
            com.fasterxml.jackson.databind.JsonNode errorNode = node.get("error");
            MCPError error = new MCPError(
                    errorNode.get("code").asInt(),
                    errorNode.get("message").asText(),
                    errorNode.has("data") ? objectMapper.convertValue(errorNode.get("data"), Map.class) : null);
            return MCPResponse.error(id, error);
        }
        
        return MCPResponse.error(id, new MCPError(-32600, "Invalid response", null));
    }
}
