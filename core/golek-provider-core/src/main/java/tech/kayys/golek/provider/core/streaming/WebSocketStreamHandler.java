package tech.kayys.golek.provider.core.streaming;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.operators.multi.processors.BroadcastProcessor;
import io.vertx.core.http.WebSocketConnectOptions;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.core.http.HttpClient;
import io.vertx.mutiny.core.http.WebSocket;
import org.jboss.logging.Logger;

import java.net.URI;
import java.time.Duration;

/**
 * Handles WebSocket streaming
 */
public class WebSocketStreamHandler implements StreamHandler {

    private static final Logger LOG = Logger.getLogger(WebSocketStreamHandler.class);

    private final Vertx vertx;
    private final HttpClient httpClient;
    private final Duration timeout;
    private WebSocket activeWebSocket;

    public WebSocketStreamHandler(Vertx vertx, Duration timeout) {
        this.vertx = vertx;
        this.timeout = timeout;
        this.httpClient = vertx.createHttpClient();
    }

    @Override
    public Multi<String> handleSSE(String url, String data) {
        throw new UnsupportedOperationException(
                "WebSocketStreamHandler does not support SSE");
    }

    @Override
    public Multi<String> handleWebSocket(String url, String data) {
        try {
            URI uri = URI.create(url);
            BroadcastProcessor<String> processor = BroadcastProcessor.create();

            WebSocketConnectOptions options = new WebSocketConnectOptions()
                    .setHost(uri.getHost())
                    .setPort(uri.getPort() > 0 ? uri.getPort() : 443)
                    .setURI(uri.getPath())
                    .setSsl("wss".equals(uri.getScheme()));

            httpClient.webSocket(options)
                    .subscribe().with(
                            ws -> {
                                this.activeWebSocket = ws;

                                ws.textMessageHandler(message -> {
                                    processor.onNext(message);
                                });

                                ws.closeHandler(() -> {
                                    processor.onComplete();
                                });

                                ws.exceptionHandler(ex -> {
                                    LOG.errorf(ex, "WebSocket error");
                                    processor.onError(ex);
                                });

                                // Send initial data
                                ws.writeTextMessage(data);
                            },
                            error -> {
                                LOG.errorf(error, "Failed to connect WebSocket");
                                processor.onError(error);
                            });

            return Multi.createFrom().publisher(processor);

        } catch (Exception e) {
            LOG.errorf(e, "Failed to initiate WebSocket stream to: %s", url);
            return Multi.createFrom().failure(e);
        }
    }

    @Override
    public void close() {
        if (activeWebSocket != null && !activeWebSocket.isClosed()) {
            activeWebSocket.close();
        }
        if (httpClient != null) {
            httpClient.close();
        }
    }
}