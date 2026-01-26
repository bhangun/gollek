package tech.kayys.golek;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;

import tech.kayys.golek.engine.LlamaConfig;
import tech.kayys.golek.engine.LlamaEngine;
import tech.kayys.golek.model.ChatMessage;
import tech.kayys.golek.model.GenerationResult;
import tech.kayys.golek.model.ModelInfo;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Execution(ExecutionMode.SAME_THREAD) // Ensure tests run sequentially for shared engine
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class LlamaEngineTest {
    
    private static LlamaEngine engine;
    private static final AtomicReference<Boolean> ENGINE_AVAILABLE = new AtomicReference<>(false);
    
    @BeforeAll
    static void setup() {
        String libraryPath = System.getenv("LLAMA_LIBRARY_PATH");
        String modelPath = System.getenv("TEST_MODEL_PATH");
        
        // Skip tests if environment variables are not set
        assumeTrue(libraryPath != null && !libraryPath.trim().isEmpty(), 
            "LLAMA_LIBRARY_PATH environment variable must be set");
        assumeTrue(modelPath != null && !modelPath.trim().isEmpty(), 
            "TEST_MODEL_PATH environment variable must be set");
        
        try {
            LlamaConfig config = LlamaConfig.builder()
                .libraryPath(libraryPath)
                .modelPath(modelPath)
                .contextSize(2048)
                .batchSize(128)
                .threads(4)
                .gpuLayers(0)
                .useMmap(true)
                .useMlock(false)
                .seed(42) // Fixed seed for reproducible tests
                .sampling(LlamaConfig.SamplingConfig.builder()
                    .temperature(0.8f)
                    .topK(40)
                    .topP(0.9f)
                    .repeatPenalty(1.1f)
                    .build())
                .build();
            
            engine = new LlamaEngine(config);
            ENGINE_AVAILABLE.set(true);
            
            // Verify engine is working with a quick health check
            ModelInfo info = engine.getModelInfo();
            assertNotNull(info, "Engine should return model info");
            logEngineInfo(info);
            
        } catch (Exception e) {
            System.err.println("Failed to initialize LlamaEngine: " + e.getMessage());
            ENGINE_AVAILABLE.set(false);
            throw new RuntimeException("Engine initialization failed", e);
        }
    }
    
    private static void logEngineInfo(ModelInfo info) {
        System.out.println("=== Engine Test Configuration ===");
        System.out.println("Model: " + info.name());
        System.out.println("Description: " + info.description());
        System.out.println("Vocab Size: " + info.vocabSize());
        System.out.println("Context Length: " + info.contextLength());
        System.out.println("Parameters: " + info.parameterCount());
        System.out.println("=================================");
    }
    
    private void ensureEngineAvailable() {
        assumeTrue(ENGINE_AVAILABLE.get(), "Engine is not available for testing");
        assertNotNull(engine, "Engine should be initialized");
    }
    
    @Test
    @Order(1)
    @DisplayName("Should load model and return valid model information")
    void testModelInfo() {
        ensureEngineAvailable();
        
        ModelInfo info = engine.getModelInfo();
        
        assertNotNull(info, "ModelInfo should not be null");
        assertTrue(info.vocabSize() > 0, "Vocabulary size should be positive");
        assertTrue(info.contextLength() > 0, "Context length should be positive");
        assertNotNull(info.name(), "Model name should not be null");
        assertNotNull(info.description(), "Model description should not be null");
        
        // Verify metadata if available
        if (info.metadata() != null) {
            assertFalse(info.metadata().containsKey("error"), 
                "Model metadata should not contain errors");
        }
    }
    
    @Test
    @Order(2)
    @DisplayName("Should generate text from prompt")
    void testGeneration() {
        ensureEngineAvailable();
        
        assertTimeoutPreemptively(Duration.ofSeconds(30), () -> {
            GenerationResult result = engine.generate(
                "The capital of France is",
                engine.getConfig().sampling(),
                10,
                List.of("."), // Stop on period
                null // No streaming
            );
            
            assertNotNull(result, "Generation result should not be null");
            assertTrue(result.tokensGenerated() > 0, "Should generate at least one token");
            assertTrue(result.tokensGenerated() <= 10, "Should not exceed max tokens");
            assertFalse(result.text().isEmpty(), "Generated text should not be empty");
            assertTrue(result.promptTokens() > 0, "Should tokenize prompt");
            assertTrue(result.timeMs() > 0, "Should measure generation time");
            assertNotNull(result.finishReason(), "Finish reason should not be null");
            
            System.out.println("Generated: " + result.text());
        });
    }
    
    @Test
    @Order(3)
    @DisplayName("Should handle chat conversations")
    void testChat() {
        ensureEngineAvailable();
        
        assertTimeoutPreemptively(Duration.ofSeconds(30), () -> {
            List<ChatMessage> messages = List.of(
                new ChatMessage("system", "You are a helpful assistant."),
                new ChatMessage("user", "What is 2+2?")
            );
            
            GenerationResult result = engine.chat(
                messages,
                engine.getConfig().sampling(),
                15,
                null
            );
            
            assertNotNull(result, "Chat result should not be null");
            assertTrue(result.tokensGenerated() > 0, "Should generate response tokens");
            assertFalse(result.text().isEmpty(), "Chat response should not be empty");
            assertTrue(result.timeMs() > 0, "Should measure chat time");
            
            System.out.println("Chat response: " + result.text());
        });
    }
    
    @Test
    @Order(4)
    @DisplayName("Should handle empty and edge case prompts")
    void testEdgeCases() {
        ensureEngineAvailable();
        
        // Test empty prompt
        assertDoesNotThrow(() -> {
            GenerationResult result = engine.generate("", engine.getConfig().sampling(), 5, null, null);
            assertNotNull(result);
        });
        
        // Test very short prompt
        GenerationResult shortResult = engine.generate("A", engine.getConfig().sampling(), 3, null, null);
        assertNotNull(shortResult);
        assertTrue(shortResult.tokensGenerated() <= 3);
    }
    
    @ParameterizedTest
    @Order(5)
    @ValueSource(strings = {
        "The weather today is",
        "Artificial intelligence is",
        "The meaning of life is"
    })
    @DisplayName("Should generate different responses for various prompts")
    void testMultiplePrompts(String prompt) {
        ensureEngineAvailable();
        
        assertTimeoutPreemptively(Duration.ofSeconds(15), () -> {
            GenerationResult result = engine.generate(
                prompt,
                engine.getConfig().sampling(),
                8,
                null,
                null
            );
            
            assertNotNull(result, "Result should not be null for prompt: " + prompt);
            assertTrue(result.tokensGenerated() > 0, "Should generate tokens for prompt: " + prompt);
            assertFalse(result.text().isEmpty(), "Should generate text for prompt: " + prompt);
        });
    }
    
    @Test
    @Order(6)
    @DisplayName("Should handle streaming callback")
    void testStreaming() {
        ensureEngineAvailable();
        
        StringBuilder streamedText = new StringBuilder();
        AtomicReference<Integer> tokenCount = new AtomicReference<>(0);
        
        GenerationResult result = engine.generate(
            "Count to three:",
            engine.getConfig().sampling(),
            10,
            null,
            token -> {
                streamedText.append(token);
                tokenCount.set(tokenCount.get() + 1);
            }
        );
        
        assertNotNull(result);
        assertEquals(streamedText.toString(), result.text(), 
            "Streamed text should match final result");
        assertTrue(tokenCount.get() > 0, "Should receive streaming tokens");
    }
    
    @Test
    @Order(7)
    @DisplayName("Should respect stop sequences")
    void testStopSequences() {
        ensureEngineAvailable();
        
        GenerationResult result = engine.generate(
            "List three colors: red, blue,",
            engine.getConfig().sampling(),
            20,
            List.of("green", "stop"), // Stop on these sequences
            null
        );
        
        assertNotNull(result);
        assertTrue(result.text().toLowerCase().contains("red"));
        assertTrue(result.text().toLowerCase().contains("blue"));
        // Should stop before generating too much more
        assertTrue(result.tokensGenerated() < 15, "Should stop early due to stop sequences");
    }
    
    @Test
    @Order(8)
    @DisplayName("Should return valid configuration")
    void testConfiguration() {
        ensureEngineAvailable();
        
        LlamaConfig config = engine.getConfig();
        assertNotNull(config, "Engine configuration should not be null");
        assertNotNull(config.modelPath(), "Model path should be set");
        assertNotNull(config.libraryPath(), "Library path should be set");
        assertTrue(config.contextSize() > 0, "Context size should be positive");
        assertTrue(config.threads() > 0, "Thread count should be positive");
        
        // Test sampling config
        LlamaConfig.SamplingConfig sampling = config.sampling();
        assertNotNull(sampling, "Sampling config should not be null");
        assertTrue(sampling.temperature() >= 0, "Temperature should be non-negative");
        assertTrue(sampling.topK() == -1 || sampling.topK() > 0, "TopK should be positive or -1");
    }
    
    @Test
    @Order(9)
    @DisplayName("Should handle concurrent requests")
    void testConcurrentAccess() {
        ensureEngineAvailable();
        
        assertTimeoutPreemptively(Duration.ofSeconds(10), () -> {
            // Test that we can make multiple quick requests
            for (int i = 0; i < 3; i++) {
                GenerationResult result = engine.generate(
                    "Test " + i + ":",
                    engine.getConfig().sampling(),
                    2,
                    null,
                    null
                );
                assertNotNull(result);
                assertTrue(result.tokensGenerated() > 0);
            }
        });
    }
    
    @Test
    @Order(10)
    @DisplayName("Should properly close engine resources")
    void testEngineClose() {
        ensureEngineAvailable();
        
        assertDoesNotThrow(() -> {
            // Create a separate engine instance for close test
            LlamaConfig tempConfig = LlamaConfig.builder()
                .libraryPath(System.getenv("LLAMA_LIBRARY_PATH"))
                .modelPath(System.getenv("TEST_MODEL_PATH"))
                .contextSize(1024)
                .threads(2)
                .build();
            
            try (LlamaEngine tempEngine = new LlamaEngine(tempConfig)) {
                // Use the temporary engine
                GenerationResult result = tempEngine.generate("Test", tempConfig.sampling(), 5, null, null);
                assertNotNull(result);
            } // Engine should auto-close here
        });
    }
    
    @AfterAll
    static void cleanup() {
        if (engine != null) {
            try {
                engine.close();
                System.out.println("Engine closed successfully");
            } catch (Exception e) {
                System.err.println("Error closing engine: " + e.getMessage());
            }
        }
    }
    
    // Nested test class for specific configurations
    @Nested
    @DisplayName("With Different Sampling Configurations")
    class SamplingConfigTests {
        
        @ParameterizedTest
        @CsvSource({
            "0.1, 20, 0.5",
            "0.8, 40, 0.9", 
            "1.2, 60, 0.95"
        })
        @DisplayName("Should generate with different sampling parameters")
        void testSamplingParameters(float temperature, int topK, float topP) {
            ensureEngineAvailable();
            
            LlamaConfig.SamplingConfig sampling = LlamaConfig.SamplingConfig.builder()
                .temperature(temperature)
                .topK(topK)
                .topP(topP)
                .build();
            
            GenerationResult result = engine.generate(
                "The sky is",
                sampling,
                5,
                null,
                null
            );
            
            assertNotNull(result);
            assertTrue(result.tokensGenerated() > 0);
        }
    }
}
