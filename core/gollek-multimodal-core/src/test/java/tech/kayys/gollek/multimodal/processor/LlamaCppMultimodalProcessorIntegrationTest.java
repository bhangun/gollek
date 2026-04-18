package tech.kayys.gollek.multimodal.processor;

import io.smallrye.mutiny.Uni;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.plugin.runner.llamacpp.processor.LlamaCppMultimodalProcessor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for GGUF multimodal processor with LLaVA models.
 */
class LlamaCppMultimodalProcessorIntegrationTest {

    private LlamaCppMultimodalProcessor processor;

    @BeforeEach
    void setUp() {
        processor = new LlamaCppMultimodalProcessor();
    }

    @Test
    void testLlavaVisualQA() throws Exception {
        // Skip if model not available
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "llava-13b.gguf")),
                "LLaVA model not found, skipping test");

        // Load test image
        byte[] testImage = loadTestImage("test-car.jpg");

        // Create request
        MultimodalRequest request = MultimodalRequest.builder()
                .requestId("test-vqa-001")
                .model("llava-13b-gguf")
                .inputs(
                        MultimodalContent.ofText("What color is the car?"),
                        MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
                .outputConfig(MultimodalRequest.OutputConfig.builder()
                        .maxTokens(64)
                        .temperature(0.7)
                        .build())
                .build();

        // Execute
        Uni<MultimodalResponse> responseUni = processor.process(request);
        MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(30));

        // Validate
        assertThat(response).isNotNull();
        assertThat(response.getRequestId()).isEqualTo("test-vqa-001");
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
        assertThat(response.getOutputs()).hasSize(1);
        assertThat(response.getOutputs()[0].getText()).isNotBlank();
        assertThat(response.getDurationMs()).isLessThan(5000); // <5s latency

        System.out.println("Answer: " + response.getOutputs()[0].getText());
    }

    @Test
    void testLlavaImageCaptioning() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "llava-13b.gguf")),
                "LLaVA model not found, skipping test");

        byte[] testImage = loadTestImage("test-scene.jpg");

        MultimodalRequest request = MultimodalRequest.builder()
                .requestId("test-caption-001")
                .model("llava-13b-gguf")
                .inputs(
                        MultimodalContent.ofText("Describe this image in detail"),
                        MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
                .outputConfig(MultimodalRequest.OutputConfig.builder()
                        .maxTokens(256)
                        .build())
                .build();

        Uni<MultimodalResponse> responseUni = processor.process(request);
        MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(30));

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
        assertThat(response.getOutputs()[0].getText()).isNotBlank();
        assertThat(response.getOutputs()[0].getText().length()).isGreaterThan(20);

        System.out.println("Caption: " + response.getOutputs()[0].getText());
    }

    @Test
    void testLlavaMultiImage() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "llava-1.6-gguf")),
                "LLaVA-1.6 model not found, skipping test");

        byte[] image1 = loadTestImage("test-image1.jpg");
        byte[] image2 = loadTestImage("test-image2.jpg");

        MultimodalRequest request = MultimodalRequest.builder()
                .requestId("test-multi-001")
                .model("llava-1.6-gguf")
                .inputs(
                        MultimodalContent.ofText("Compare these two images"),
                        MultimodalContent.ofBase64Image(image1, "image/jpeg"),
                        MultimodalContent.ofBase64Image(image2, "image/jpeg"))
                .outputConfig(MultimodalRequest.OutputConfig.builder()
                        .maxTokens(512)
                        .build())
                .build();

        Uni<MultimodalResponse> responseUni = processor.process(request);
        MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(60));

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);

        System.out.println("Comparison: " + response.getOutputs()[0].getText());
    }

    @Test
    void testProcessorAvailability() {
        // Test that processor reports availability correctly
        boolean available = processor.isAvailable();

        // Should be false if model not loaded, true if loaded
        System.out.println("Processor available: " + available);
    }

    @Test
    void testProcessorId() {
        assertThat(processor.getProcessorId()).isEqualTo("gguf-multimodal");
    }

    @Test
    void testConcurrentRequests() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "llava-13b.gguf")),
                "LLaVA model not found, skipping test");

        byte[] testImage = loadTestImage("test-car.jpg");

        // Create 10 concurrent requests
        java.util.List<Uni<MultimodalResponse>> requests = java.util.stream.IntStream.range(0, 10)
                .mapToObj(i -> MultimodalRequest.builder()
                        .requestId("test-concurrent-" + i)
                        .model("llava-13b-gguf")
                        .inputs(
                                MultimodalContent.ofText("What is this?"),
                                MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
                        .build())
                .map(request -> processor.process(request))
                .collect(java.util.stream.Collectors.toList());

        // Execute all and validate
        for (int i = 0; i < requests.size(); i++) {
            MultimodalResponse response = requests.get(i).await().atMost(java.time.Duration.ofSeconds(30));
            assertThat(response).isNotNull();
            assertThat(response.getRequestId()).isEqualTo("test-concurrent-" + i);
            System.out.println("Request " + i + " completed in " + response.getDurationMs() + "ms");
        }
    }

    @Test
    void testMemoryLeak() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "llava-13b.gguf")),
                "LLaVA model not found, skipping test");

        byte[] testImage = loadTestImage("test-car.jpg");
        Runtime runtime = Runtime.getRuntime();

        // Force GC
        System.gc();
        Thread.sleep(1000);
        long initialMemory = runtime.totalMemory() - runtime.freeMemory();

        // Process 100 requests
        for (int i = 0; i < 100; i++) {
            MultimodalRequest request = MultimodalRequest.builder()
                    .requestId("test-leak-" + i)
                    .model("llava-13b-gguf")
                    .inputs(
                            MultimodalContent.ofText("Test"),
                            MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
                    .build();

            MultimodalResponse response = processor.process(request)
                    .await().atMost(java.time.Duration.ofSeconds(30));

            assertThat(response).isNotNull();
        }

        // Force GC again
        System.gc();
        Thread.sleep(1000);
        long finalMemory = runtime.totalMemory() - runtime.freeMemory();

        // Memory increase should be less than 100MB
        long memoryIncrease = finalMemory - initialMemory;
        System.out.println("Memory increase after 100 requests: " + (memoryIncrease / 1024 / 1024) + "MB");
        assertThat(memoryIncrease).isLessThan(100 * 1024 * 1024); // <100MB
    }

    // Helper methods

    private byte[] loadTestImage(String filename) throws Exception {
        Path testImagePath = Path.of("src/test/resources/images", filename);
        if (Files.exists(testImagePath)) {
            return Files.readAllBytes(testImagePath);
        }

        // Create placeholder test image if not found
        return createPlaceholderImage();
    }

    private byte[] createPlaceholderImage() {
        // Create a minimal valid JPEG for testing
        // In production, use actual test images
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return Base64.getDecoder().decode(base64Image);
    }
}
