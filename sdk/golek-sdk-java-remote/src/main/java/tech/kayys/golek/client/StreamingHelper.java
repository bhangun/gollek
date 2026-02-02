package tech.kayys.golek.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.reactivex.Flowable;
import org.reactivestreams.Publisher;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.client.exception.GolekClientException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.Flow;

/**
 * Helper class for handling Server-Sent Events (SSE) for streaming inference.
 */
public class StreamingHelper {
    
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String apiKey;
    private final String defaultTenantId;
    
    public StreamingHelper(HttpClient httpClient, ObjectMapper objectMapper, String baseUrl, String apiKey, String defaultTenantId) {
        this.httpClient = httpClient;
        this.objectMapper = objectMapper;
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
        this.defaultTenantId = defaultTenantId;
    }
    
    /**
     * Creates a publisher for streaming inference chunks.
     * 
     * @param requestBody The JSON request body for the inference request
     * @return A Publisher that emits StreamChunk objects
     */
    public Publisher<StreamChunk> createStreamPublisher(String requestBody) {
        return Flowable.fromPublisher(subscriber -> {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/v1/inference/completions/stream"))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + apiKey)
                        .header("X-Tenant-ID", defaultTenantId)
                        .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                        .timeout(Duration.ofMinutes(10)) // Longer timeout for streaming
                        .build();
                
                HttpResponse<java.io.InputStream> response = httpClient.send(request, 
                        HttpResponse.BodyHandlers.ofInputStream());
                
                if (response.statusCode() != 200) {
                    subscriber.onError(new GolekClientException("Streaming request failed with status: " + response.statusCode()));
                    return;
                }
                
                BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()));
                String line;
                
                while ((line = reader.readLine()) != null && !subscriber.isCancelled()) {
                    if (line.startsWith("data: ")) {
                        String data = line.substring(6); // Remove "data: " prefix
                        
                        if ("[DONE]".equals(data)) {
                            break; // End of stream
                        }
                        
                        try {
                            StreamChunk chunk = objectMapper.readValue(data, StreamChunk.class);
                            subscriber.onNext(chunk);
                        } catch (Exception e) {
                            subscriber.onError(new GolekClientException("Error parsing stream chunk: " + e.getMessage(), e));
                            return;
                        }
                    }
                }
                
                subscriber.onComplete();
            } catch (IOException | InterruptedException e) {
                if (!subscriber.isCancelled()) {
                    subscriber.onError(new GolekClientException("Error during streaming: " + e.getMessage(), e));
                }
            }
        });
    }
}