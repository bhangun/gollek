package tech.kayys.gollek.client.example;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.sdk.core.model.AsyncJobStatus;
import tech.kayys.gollek.sdk.core.model.BatchInferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.stream.StreamChunk;
import tech.kayys.gollek.client.GollekClient;
import tech.kayys.gollek.sdk.core.exception.SdkException;

import java.time.Duration;
import java.util.Arrays;
import java.util.List;

/**
 * Example usage of the Gollek Java SDK.
 */
public class GollekClientExample {

    public static void main(String[] args) {
        // Initialize the client
        GollekClient client = GollekClient.builder()
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
    private static void simpleInferenceExample(GollekClient client) {
        System.out.println("=== Simple Inference Example ===");

        try {
            var request = InferenceRequest.builder()
                    .model("llama3:latest")
                    .message(Message.user("What is the capital of France?"))
                    .temperature(0.7)
                    .maxTokens(100)
                    .build();

            InferenceResponse response = client.createCompletion(request);
            System.out.println("Response: " + response.getContent());
        } catch (SdkException e) {
            System.err.println("Gollek error [" + e.getErrorCode() + "]: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example of streaming inference.
     */
    private static void streamingInferenceExample(GollekClient client) {
        System.out.println("=== Streaming Inference Example ===");

        try {
            var request = InferenceRequest.builder()
                    .model("llama3:latest")
                    .message(Message.user("Write a short poem about AI."))
                    .temperature(0.7)
                    .maxTokens(200)
                    .streaming(true)
                    .build();

            Multi<StreamChunk> stream = client.streamCompletion(request);

            StringBuilder fullResponse = new StringBuilder();
            stream.subscribe().with(
                    chunk -> {
                        System.out.print(chunk.getDelta()); // Print each chunk as it arrives
                        fullResponse.append(chunk.getDelta());
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
    private static void asyncJobExample(GollekClient client) {
        System.out.println("=== Async Job Example ===");

        try {
            var request = InferenceRequest.builder()
                    .model("llama3:latest")
                    .message(Message.user("Analyze this long document: [very long document content here]"))
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
        } catch (SdkException e) {
            System.err.println("Async job error [" + e.getErrorCode() + "]: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example of batch inference.
     */
    private static void batchInferenceExample(GollekClient client) {
        System.out.println("=== Batch Inference Example ===");

        try {
            // Create multiple requests
            var request1 = InferenceRequest.builder()
                    .model("llama3:latest")
                    .message(Message.user("What is 2+2?"))
                    .maxTokens(10)
                    .build();

            var request2 = InferenceRequest.builder()
                    .model("llama3:latest")
                    .message(Message.user("What is the largest planet?"))
                    .maxTokens(20)
                    .build();

            var request3 = InferenceRequest.builder()
                    .model("llama3:latest")
                    .message(Message.user("Who wrote Romeo and Juliet?"))
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
        } catch (SdkException e) {
            System.err.println("Batch inference error [" + e.getErrorCode() + "]: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
        }

        System.out.println();
    }
}
