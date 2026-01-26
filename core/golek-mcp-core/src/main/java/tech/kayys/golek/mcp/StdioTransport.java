package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;
import org.zeroturnaround.exec.ProcessExecutor;
import org.zeroturnaround.exec.StartedProcess;
import org.zeroturnaround.exec.stream.LogOutputStream;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Stdio transport for local MCP servers.
 * Launches a process and communicates via stdin/stdout.
 */
public class StdioTransport implements MCPTransport {

    private static final Logger LOG = Logger.getLogger(StdioTransport.class);

    private final MCPClientConfig config;
    private final ObjectMapper objectMapper;
    private final AtomicBoolean connected = new AtomicBoolean(false);
    private final Map<Object, CompletableFuture<MCPResponse>> pendingRequests = new ConcurrentHashMap<>();
    private final AtomicLong requestIdCounter = new AtomicLong(0);

    private StartedProcess process;
    private OutputStream processInput;
    private Consumer<JsonRpcMessage> messageHandler;

    public StdioTransport(MCPClientConfig config, ObjectMapper objectMapper) {
        this.config = config;
        this.objectMapper = objectMapper;
    }

    @Override
    public Uni<Void> connect() {
        if (connected.get()) {
            return Uni.createFrom().voidItem();
        }

        return Uni.createFrom().item(() -> {
            try {
                LOG.infof("Starting MCP server process: %s", config.getCommand());

                process = new ProcessExecutor()
                        .command(config.getCommand())
                        .environment(System.getenv())
                        .redirectOutput(new LogOutputStream() {
                            @Override
                            protected void processLine(String line) {
                                handleOutput(line);
                            }
                        })
                        .redirectError(new LogOutputStream() {
                            @Override
                            protected void processLine(String line) {
                                LOG.warnf("MCP server stderr: %s", line);
                            }
                        })
                        .start();

                processInput = process.getProcess().getOutputStream();
                connected.set(true);

                LOG.infof("MCP server process started successfully");
                return null;

            } catch (Exception e) {
                throw new MCPTransportException("Failed to start MCP server process", e);
            }
        });
    }

    @Override
    public Uni<MCPResponse> sendRequest(MCPRequest request) {
        if (!connected.get()) {
            return Uni.createFrom().failure(
                    new MCPTransportException("Not connected to MCP server"));
        }

        CompletableFuture<MCPResponse> future = new CompletableFuture<>();
        pendingRequests.put(request.getId(), future);

        try {
            String json = objectMapper.writeValueAsString(request);
            writeToProcess(json);

            return Uni.createFrom().completionStage(future)
                    .ifNoItem().after(config.getTimeout())
                    .failWith(() -> {
                        pendingRequests.remove(request.getId());
                        return new MCPTransportException("Request timeout: " + request.getMethod());
                    });

        } catch (Exception e) {
            pendingRequests.remove(request.getId());
            return Uni.createFrom().failure(
                    new MCPTransportException("Failed to send request", e));
        }
    }

    @Override
    public Uni<Void> sendNotification(String method, Object params) {
        if (!connected.get()) {
            return Uni.createFrom().failure(
                    new MCPTransportException("Not connected to MCP server"));
        }

        return Uni.createFrom().item(() -> {
            try {
                MCPNotification notification = new MCPNotification(method,
                        params instanceof Map ? (Map<String, Object>) params : Map.of());
                String json = objectMapper.writeValueAsString(notification);
                writeToProcess(json);
                return null;
            } catch (Exception e) {
                throw new MCPTransportException("Failed to send notification", e);
            }
        });
    }

    private void writeToProcess(String json) throws IOException {
        if (processInput != null) {
            processInput.write(json.getBytes(StandardCharsets.UTF_8));
            processInput.write('\n');
            processInput.flush();
            LOG.debugf("Sent to MCP server: %s", json);
        }
    }

    private void handleOutput(String line) {
        if (line == null || line.trim().isEmpty()) {
            return;
        }

        LOG.debugf("Received from MCP server: %s", line);

        try {
            JsonRpcMessage message = objectMapper.readValue(line, JsonRpcMessage.class);

            if (message.isResponse()) {
                handleResponse((MCPResponse) message);
            } else if (messageHandler != null) {
                messageHandler.accept(message);
            }

        } catch (Exception e) {
            LOG.errorf(e, "Failed to parse MCP message: %s", line);
        }
    }

    private void handleResponse(MCPResponse response) {
        CompletableFuture<MCPResponse> future = pendingRequests.remove(response.getId());
        if (future != null) {
            future.complete(response);
        } else {
            LOG.warnf("Received response for unknown request ID: %s", response.getId());
        }
    }

    @Override
    public void onMessage(Consumer<JsonRpcMessage> handler) {
        this.messageHandler = handler;
    }

    @Override
    public boolean isConnected() {
        return connected.get() && process != null && process.getProcess().isAlive();
    }

    @Override
    public Uni<Void> disconnect() {
        return Uni.createFrom().item(() -> {
            if (process != null) {
                process.getProcess().destroy();
            }
            connected.set(false);
            pendingRequests.clear();
            return null;
        });
    }

    @Override
    public void close() {
        disconnect().await().indefinitely();
    }

    public static class MCPTransportException extends RuntimeException {
        public MCPTransportException(String message) {
            super(message);
        }

        public MCPTransportException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}