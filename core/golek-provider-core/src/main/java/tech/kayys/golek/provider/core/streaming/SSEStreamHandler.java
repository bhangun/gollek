package tech.kayys.golek.provider.core.streaming;

import io.smallrye.mutiny.Multi;
import io.vertx.core.http.HttpClientOptions;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.core.buffer.Buffer;
import io.vertx.mutiny.core.http.HttpClient;
import io.vertx.mutiny.core.http.HttpClientResponse;
import org.jboss.logging.Logger;

import java.net.URI;
import java.time.Duration;

/**
 * Handles Server-Sent Events (SSE) streaming
 */
public class SSEStreamHandler implements StreamHandler {

    private static final Logger LOG = Logger.getLogger(SSEStreamHandler.class);

    private final Vertx vertx;
    private final HttpClient httpClient;
    private final Duration timeout;

    public SSEStreamHandler(Vertx vertx, Duration timeout) {
        this.vertx = vertx;
        this.timeout = timeout;

        HttpClientOptions options = new HttpClientOptions()
                .setConnectTimeout((int) timeout.toMillis())
                .setIdleTimeout((int) timeout.toSeconds())
                .setKeepAlive(true);

        this.httpClient = vertx.createHttpClient(options);
    }

    @Override
    public Multi<String> handleSSE(String url, String data) {
        try {
            URI uri = URI.create(url);
            int port = uri.getPort() > 0 ? uri.getPort() : ("https".equals(uri.getScheme()) ? 443 : 80);

            return httpClient.request(
                    io.vertx.core.http.HttpMethod.POST,
                    port,
                    uri.getHost(),
                    uri.getPath())
                    .onItem().transformToUni(request -> {
                        request.putHeader("Accept", "text/event-stream");
                        request.putHeader("Content-Type", "application/json");
                        request.putHeader("Cache-Control", "no-cache");

                        return request.send(Buffer.buffer(data));
                    })
                    .onItem().transformToMulti(this::processSSEResponse)
                    .onFailure().invoke(ex -> LOG.errorf(ex, "SSE streaming failed for URL: %s", url));

        } catch (Exception e) {
            LOG.errorf(e, "Failed to initiate SSE stream to: %s", url);
            return Multi.createFrom().failure(e);
        }
    }

    @Override
    public Multi<String> handleWebSocket(String url, String data) {
        throw new UnsupportedOperationException(
                "SSEStreamHandler does not support WebSocket");
    }

    @Override
    public void close() {
        if (httpClient != null) {
            httpClient.close();
        }
    }

    private Multi<String> processSSEResponse(HttpClientResponse response) {
        if (response.statusCode() != 200) {
            return Multi.createFrom().failure(new RuntimeException(
                    "SSE stream failed with status: " + response.statusCode()));
        }

        return response.toMulti()
                .onItem().transform(Buffer::toString)
                .onItem().transformToMultiAndConcatenate(this::parseSSEData)
                .select().where(line -> !line.isBlank());
    }

    private Multi<String> parseSSEData(String chunk) {
        // SSE format: "data: {json}\n\n"
        String[] lines = chunk.split("\n");

        return Multi.createFrom().items(lines)
                .filter(line -> line.startsWith("data: "))
                .map(line -> line.substring(6).trim())
                .filter(data -> !data.isEmpty() && !data.equals("[DONE]"));
    }
}