package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.internal.util.Json;
import tech.kayys.gollek.sdk.model.GenerationRequest;
import tech.kayys.gollek.sdk.model.GenerationStream;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * SSE (Server-Sent Events) based streaming client for Gollek token generation.
 *
 * <p>Sends a POST request to the streaming endpoint and processes each
 * {@code data:} line as a token event. Implements {@link GenerationStream} so
 * callers can register token, completion, and error listeners.
 *
 * <p>Cancellation is cooperative: once {@link #cancel()} is called, incoming
 * SSE lines are silently dropped and the underlying {@link CompletableFuture}
 * is cancelled.
 */
public class HttpStreamClient implements GenerationStream {

    private final HttpClient httpClient;
    private final URI uri;
    private final String apiKey;
    private final Duration timeout;

    private final List<Consumer<String>> tokenListeners = new CopyOnWriteArrayList<>();
    private final List<Runnable> completeListeners = new CopyOnWriteArrayList<>();
    private final List<Consumer<Throwable>> errorListeners = new CopyOnWriteArrayList<>();

    private CompletableFuture<Void> future;
    private volatile boolean cancelled = false;

    /**
     * Creates a new streaming client.
     *
     * @param httpClient the shared HTTP client
     * @param uri        the streaming endpoint URI (e.g. {@code /v1/generate/stream})
     * @param apiKey     bearer token for authentication
     * @param timeout    per-request timeout
     */
    public HttpStreamClient(HttpClient httpClient, URI uri, String apiKey, Duration timeout) {
        this.httpClient = httpClient;
        this.uri = uri;
        this.apiKey = apiKey;
        this.timeout = timeout;
    }

    /**
     * Initiates the streaming request asynchronously.
     * Token, completion, and error listeners should be registered <em>before</em>
     * calling this method to avoid missing early events.
     *
     * @param request the generation parameters
     * @return {@code this} for chaining
     */
    public GenerationStream stream(GenerationRequest request) {
        this.future = CompletableFuture.runAsync(() -> {
            try {
                HttpRequest httpRequest = HttpRequest.newBuilder()
                        .uri(uri)
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + apiKey)
                        .timeout(timeout)
                        .POST(HttpRequest.BodyPublishers.ofString(Json.stringify(request)))
                        .build();

                httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofLines())
                        .body()
                        .forEach(line -> {
                            if (cancelled) return;
                            if (line.startsWith("data:")) {
                                String data = line.substring(5).trim();
                                if (!"[DONE]".equals(data)) {
                                    // Parse token from data if it's JSON, or use raw if simplified
                                    tokenListeners.forEach(l -> l.accept(data));
                                }
                            }
                        });

                if (!cancelled) {
                    completeListeners.forEach(Runnable::run);
                }
            } catch (Exception e) {
                if (!cancelled) {
                    errorListeners.forEach(l -> l.accept(e));
                }
            }
        });
        return this;
    }

    @Override
    public GenerationStream onToken(Consumer<String> tokenListener) {
        tokenListeners.add(tokenListener);
        return this;
    }

    @Override
    public GenerationStream onComplete(Runnable completeListener) {
        completeListeners.add(completeListener);
        return this;
    }

    @Override
    public GenerationStream onError(Consumer<Throwable> errorListener) {
        errorListeners.add(errorListener);
        return this;
    }

    @Override
    public void cancel() {
        cancelled = true;
        if (future != null) {
            future.cancel(true);
        }
    }
}
