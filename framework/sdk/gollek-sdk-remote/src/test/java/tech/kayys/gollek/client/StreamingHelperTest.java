package tech.kayys.gollek.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.client.agent.AgentRequestOptions;
import tech.kayys.gollek.client.agent.AgentStreamEvent;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.Authenticator;
import java.net.CookieHandler;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLParameters;
import javax.net.ssl.SSLSession;

import static org.junit.jupiter.api.Assertions.assertEquals;

class StreamingHelperTest {

    @Test
    void acceptsSseDataLinesWithAndWithoutSpace() {
        assertEquals("{\"ok\":true}", StreamingHelper.dataFromSseLine("data:{\"ok\":true}"));
        assertEquals("{\"ok\":true}", StreamingHelper.dataFromSseLine("data: {\"ok\":true}"));
    }

    @Test
    void streamsTypedOpenAiChatEvents() throws Exception {
        HttpClient httpClient = new FakeHttpClient(new ByteArrayInputStream("""
                data:{"id":"chatcmpl-req-1","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"hello"}}],"metadata":{"gollek_stream":{"surface":"chat.completions","sequence_number":0,"final":false}}}

                data: {"id":"chatcmpl-req-1","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}

                data:[DONE]
                """.getBytes(StandardCharsets.UTF_8)));

        StreamingHelper helper = new StreamingHelper(
                httpClient,
                new ObjectMapper(),
                "http://localhost:8080",
                "community");

        List<AgentStreamEvent> events = helper.createOpenAiChatStreamPublisher("{}")
                .collect().asList()
                .await().atMost(Duration.ofSeconds(2));

        assertEquals(2, events.size());
        assertEquals("hello", events.get(0).delta());
        assertEquals("stop", events.get(1).finishReason());
        assertEquals(3, events.get(1).usage().totalTokens());
    }

    @Test
    void propagatesTraceHeadersOnOpenAiStreams() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(new ByteArrayInputStream("""
                data:{"type":"response.output_text.delta","sequence_number":0,"delta":"hello"}

                data:[DONE]
                """.getBytes(StandardCharsets.UTF_8)));

        StreamingHelper helper = new StreamingHelper(
                httpClient,
                new ObjectMapper(),
                "http://localhost:8080",
                "community");

        helper.createOpenAiResponsesStreamPublisher("{}", AgentRequestOptions.builder()
                        .requestId("req-stream")
                        .traceId("trace-stream")
                        .build())
                .collect().asList()
                .await().atMost(Duration.ofSeconds(2));

        assertEquals("req-stream", httpClient.lastRequest().headers()
                .firstValue(AgentRequestOptions.REQUEST_ID_HEADER).orElseThrow());
        assertEquals("trace-stream", httpClient.lastRequest().headers()
                .firstValue(AgentRequestOptions.TRACE_ID_HEADER).orElseThrow());
    }

    private static final class FakeHttpClient extends HttpClient {
        private final InputStream body;
        private HttpRequest lastRequest;

        private FakeHttpClient(InputStream body) {
            this.body = body;
        }

        private HttpRequest lastRequest() {
            return lastRequest;
        }

        @Override
        public Optional<CookieHandler> cookieHandler() {
            return Optional.empty();
        }

        @Override
        public Optional<Duration> connectTimeout() {
            return Optional.empty();
        }

        @Override
        public Redirect followRedirects() {
            return Redirect.NEVER;
        }

        @Override
        public Optional<ProxySelector> proxy() {
            return Optional.empty();
        }

        @Override
        public SSLContext sslContext() {
            return null;
        }

        @Override
        public SSLParameters sslParameters() {
            return null;
        }

        @Override
        public Optional<Authenticator> authenticator() {
            return Optional.empty();
        }

        @Override
        public Version version() {
            return Version.HTTP_1_1;
        }

        @Override
        public Optional<Executor> executor() {
            return Optional.empty();
        }

        @Override
        @SuppressWarnings("unchecked")
        public <T> HttpResponse<T> send(HttpRequest request, HttpResponse.BodyHandler<T> responseBodyHandler)
                throws IOException, InterruptedException {
            lastRequest = request;
            return (HttpResponse<T>) new FakeResponse(request, body);
        }

        @Override
        public <T> CompletableFuture<HttpResponse<T>> sendAsync(
                HttpRequest request,
                HttpResponse.BodyHandler<T> responseBodyHandler) {
            throw new UnsupportedOperationException();
        }

        @Override
        public <T> CompletableFuture<HttpResponse<T>> sendAsync(
                HttpRequest request,
                HttpResponse.BodyHandler<T> responseBodyHandler,
                HttpResponse.PushPromiseHandler<T> pushPromiseHandler) {
            throw new UnsupportedOperationException();
        }
    }

    private record FakeResponse(HttpRequest request, InputStream body) implements HttpResponse<InputStream> {
        @Override
        public int statusCode() {
            return 200;
        }

        @Override
        public Optional<HttpResponse<InputStream>> previousResponse() {
            return Optional.empty();
        }

        @Override
        public HttpHeaders headers() {
            return HttpHeaders.of(Map.of(), (name, value) -> true);
        }

        @Override
        public Optional<SSLSession> sslSession() {
            return Optional.empty();
        }

        @Override
        public URI uri() {
            return request.uri();
        }

        @Override
        public HttpClient.Version version() {
            return HttpClient.Version.HTTP_1_1;
        }
    }
}
