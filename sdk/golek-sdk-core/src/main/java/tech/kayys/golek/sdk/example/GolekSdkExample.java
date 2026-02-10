package tech.kayys.golek.sdk.example;

import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.factory.GolekSdkFactory;

/**
 * Example usage of the Golek SDK factory to create different implementations.
 */
public class GolekSdkExample {
    
    public static void main(String[] args) {
        // Example 1: Using the local SDK (when running embedded in the same JVM)
        System.out.println("=== Local SDK Example ===");
        try {
            // This would work when the SDK is embedded within the inference engine
            // GolekSdk localSdk = GolekSdkFactory.createLocalSdk();
            // For demonstration purposes, we'll show the remote example instead
            System.out.println("Local SDK would be used when embedded in the same JVM as the inference engine");
        } catch (Exception e) {
            System.err.println("Error creating local SDK: " + e.getMessage());
        }
        
        // Example 2: Using the remote SDK (for external applications)
        System.out.println("\n=== Remote SDK Example ===");
        try {
            // GolekSdk remoteSdk = GolekSdkFactory.createRemoteSdk(
            //     "http://localhost:8080",  // Replace with your API endpoint
            //     "your-api-key"            // Replace with your API key (use "community" for standalone)
            // );
            
            // // Create an inference request
            // InferenceRequest request = InferenceRequest.builder()
            //         .model("llama3:latest")
            //         .message(Message.user("What is the capital of France?"))
            //         .temperature(0.7)
            //         .maxTokens(100)
            //         .build();
            
            // // Execute the request
            // InferenceResponse response = remoteSdk.createCompletion(request);
            // System.out.println("Response: " + response.getContent());
            
        } catch (Exception e) {
            System.err.println("Error creating or using remote SDK: " + e.getMessage());
            e.printStackTrace();
        }
        
        // Example 3: Using the remote SDK with async operations
        System.out.println("\n=== Remote SDK Async Example ===");
        try {
            // GolekSdk remoteSdk = GolekSdkFactory.createRemoteSdk(
            //     "http://localhost:8080",  // Replace with your API endpoint
            //     "your-api-key"            // Replace with your API key (use "community" for standalone)
            // );
            
            // InferenceRequest request = InferenceRequest.builder()
            //         .model("llama3:latest")
            //         .message(Message.user("Count from 1 to 10"))
            //         .temperature(0.7)
            //         .maxTokens(100)
            //         .build();
            
            // // Execute asynchronously
            // var future = remoteSdk.createCompletionAsync(request);
            // InferenceResponse response = future.join(); // Wait for completion
            // System.out.println("Async response: " + response.getContent());
            
        } catch (Exception e) {
            System.err.println("Error with async remote SDK: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
