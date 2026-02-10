package tech.kayys.golek.client.example;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.inference.AsyncJobStatus;
import tech.kayys.golek.spi.inference.BatchInferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.golek.client.GolekClient;
import tech.kayys.golek.client.builder.InferenceRequest;
import tech.kayys.golek.client.exception.AuthenticationException;
import tech.kayys.golek.client.exception.GolekClientException;
import tech.kayys.golek.client.exception.RateLimitException;

import java.time.Duration;
import java.util.Arrays;
import java.util.List;

/**
 * Example usage of the Golek Java SDK.
 */
public class GolekClientExample {

    public static void main(String[] args) {
        // Initialize the client
        GolekClient client = GolekClient.builder()
                .baseUrl("http://localhost:8080") // Replace with your API endpoint
                .apiKey("your-api-key") // Replace with your API key
                .build();

        // Example 1: Simple inference
        simpleInferenceExample(client);

        // Example 2: Streaming inference
        streamingInferenceExample(client);

        // Example 3: Async job
        asyncJobExample(client);

        // Example 4: Batch inference
        batchInferenceExample(client);
    }

    /**
     * Example of simple synchronous inference.
     */
    private static void simpleInferenceExample(GolekClient client) {
        System.out.println("=== Simple Inference Example ===");

        try {
            var request = InferenceRequest.builder()
                    .model("llama3:latest")
                    .userMessage("What is the capital of France?")
                    .temperature(0.7)
                    .maxTokens(100)
                    .build();

            InferenceResponse response = client.createCompletion(request);
            System.out.println("Response: " + response.getContent());
        } catch (AuthenticationException e) {
            System.err.println("Authentication failed: " + e.getMessage());
        } catch (RateLimitException e) {
            System.err.println("Rate limited: " + e.getMessage());
        } catch (GolekClientException e) {
            System.err.println("Client error: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example of streaming inference.
     */
    private static void streamingInferenceExample(GolekClient client) {
        System.out.println("=== Streaming Inference Example ===");

        try {
            var request = InferenceRequest.builder()
                    .model("llama3:latest")
                    .userMessage("Write a short poem about AI.")
                    .temperature(0.7)
                    .maxTokens(200)
                    .streaming(true)
                    .build();

            Multi<StreamChunk> stream = client.streamCompletion(request);

            StringBuilder fullResponse = new StringBuilder();
            stream.subscribe().with(
                    chunk -> {
                        System.out.print(chunk.getContent()); // Print each chunk as it arrives
                        fullResponse.append(chunk.getContent());
                    },
                    failure -> System.err.println("Streaming error: " + failure.getMessage()),
                    () -> System.out.println("\nStreaming completed!"));

            // Wait a bit for the stream to complete
            Thread.sleep(5000);
        } catch (Exception e) {
            System.err.println("Streaming error: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example of async job submission and monitoring.
     */
    private static void asyncJobExample(GolekClient client) {
        System.out.println("=== Async Job Example ===");

        try {
            var request = InferenceRequest.builder()
                    .model("llama3:latest")
                    .userMessage("Analyze this long document: [very long document content here]")
                    .maxTokens(1000)
                    .build();

            // Submit the job
            String jobId = client.submitAsyncJob(request);
            System.out.println("Job submitted with ID: " + jobId);

            // Wait for the job to complete
            AsyncJobStatus status = client.waitForJob(
                    jobId,
                    Duration.ofMinutes(5), // Max wait time
                    Duration.ofSeconds(2) // Poll interval
            );

            if (status.getResult() != null) {
                System.out.println("Job completed successfully:");
                System.out.println("Response: " + status.getResult().getContent());
            } else {
                System.out.println("Job failed: " + status.getError());
            }
        } catch (GolekClientException e) {
            System.err.println("Async job error: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example of batch inference.
     */
    private static void batchInferenceExample(GolekClient client) {
        System.out.println("=== Batch Inference Example ===");

        try {
            // Create multiple requests
            var request1 = InferenceRequest.builder()
                    .model("llama3:latest")
                    .userMessage("What is 2+2?")
                    .maxTokens(10)
                    .build();

            var request2 = InferenceRequest.builder()
                    .model("llama3:latest")
                    .userMessage("What is the largest planet?")
                    .maxTokens(20)
                    .build();

            var request3 = InferenceRequest.builder()
                    .model("llama3:latest")
                    .userMessage("Who wrote Romeo and Juliet?")
                    .maxTokens(20)
                    .build();

            // Create batch request
            var batchRequest = BatchInferenceRequest.builder()
                    .requests(Arrays.asList(request1, request2, request3))
                    .maxConcurrent(3)
                    .build();

            // Execute batch inference
            List<InferenceResponse> responses = client.batchInference(batchRequest);

            System.out.println("Batch inference completed with " + responses.size() + " responses:");
            for (int i = 0; i < responses.size(); i++) {
                System.out.println((i + 1) + ". " + responses.get(i).getContent());
            }
        } catch (GolekClientException e) {
            System.err.println("Batch inference error: " + e.getMessage());
        }

        System.out.println();
    }
}
