package tech.kayys.golek;

import tech.kayys.golek.engine.LlamaConfig;
import tech.kayys.golek.engine.LlamaEngine;
import tech.kayys.golek.model.ChatMessage;
import tech.kayys.golek.model.GenerationResult;
import tech.kayys.golek.model.ModelInfo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class Example {

    public static void main(String[] args) {
        try {
            // Configuration with validation and fallbacks
            LlamaConfig config = createConfig();

            // Engine with proper resource management
            try (LlamaEngine engine = new LlamaEngine(config)) {
                runExample(engine);
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    
     
 private static LlamaConfig createConfig() {
    String libraryPath = "/Users/bhangun/Workspace/workkayys/Products/AI/core/gollek/third_party/llama.cpp/build/bin/libllama.dylib";
    String modelPath = "/Users/bhangun/Workspace/workkayys/Products/AI/core/gollek/third_party/llama.cpp/models/ggml-vocab-qwen2.gguf";
    
    // Use the most basic configuration possible
    return new LlamaConfig() {
        @Override public String libraryPath() { return libraryPath; }
        @Override public String modelPath() { return modelPath; }
        @Override public int contextSize() { return 2048; }
        @Override public int batchSize() { return 512; }
        @Override public int threads() { return 4; }
        @Override public int gpuLayers() { return 0; }
        @Override public float ropeFreqBase() { return 10000.0f; }
        @Override public float ropeFreqScale() { return 1.0f; }
        @Override public int seed() { return 42; }
        @Override public boolean useMmap() { return true; }
        @Override public boolean useMlock() { return false; }
        @Override public boolean embeddings() { return false; }
        @Override public boolean flashAttention() { return false; }
        @Override public SamplingConfig sampling() { 
            return new SamplingConfig() {
                @Override public float temperature() { return 0.7f; }
                @Override public int topK() { return 40; }
                @Override public float topP() { return 0.9f; }
                @Override public float minP() { return 0.05f; }
                @Override public float repeatPenalty() { return 1.1f; }
                @Override public int repeatLastN() { return 64; }
                @Override public float presencePenalty() { return 0.0f; }
                @Override public float frequencyPenalty() { return 0.0f; }
                @Override public float typicalP() { return 1.0f; }
                @Override public float tfsZ() { return 1.0f; }
                @Override public float mirostat() { return 0.0f; }
                @Override public float mirostatTau() { return 5.0f; }
                @Override public float mirostatEta() { return 0.1f; }
                @Override public String grammar() { return ""; }
                @Override public boolean grammarAcceptToken() { return false; }
                @Override public void validate() {}
            };
        }
        
        // Add any other required methods from LlamaConfig interface
        @Override public int uBatchSize() { return 512; }
        @Override public int threadsBatch() { return 4; }
        @Override public int ropeScalingType() { return -1; }
        @Override public int nSeqMax() { return 16; }
        @Override public int poolingType() { return 0; }
        @Override public int attentionType() { return 1; }
        @Override public boolean offloadKqv() { return false; }
        @Override public float defragThreshold() { return -1.0f; }
        @Override public float yarnExtFactor() { return -1.0f; }
        @Override public float yarnAttnFactor() { return 1.0f; }
        @Override public float yarnBetaFast() { return 32.0f; }
        @Override public float yarnBetaSlow() { return 1.0f; }
        @Override public int yarnOrigCtx() { return 0; }
        @Override public String tensorSplit() { return ""; }
        @Override public int mainGpu() { return 0; }
        @Override public int splitMode() { return 0; }
        @Override public boolean vocabOnly() { return false; }
    };
} 
     


    private static String getLibraryPath() {
    // Expand home directory and try multiple possible library locations
    String home = System.getProperty("user.home");
    
    String[] possiblePaths = {
        System.getenv("LLAMA_LIBRARY_PATH"),
      //  home + "/Workspace/workkayys/Products/AI/core/gollek/third_party/llama.cpp/build/bin/libllama.dylib",
     
      "/usr/local/lib/libllama.dylib",
        "/usr/local/lib/libllama.so",
        "/usr/lib/libllama.so",
        "./libllama.so",
        "llama.cpp/libllama.so"
    };

    for (String path : possiblePaths) {
        if (path != null && Files.exists(Path.of(path))) {
            System.out.println("Found library at: " + path);
            return path;
        } else if (path != null) {
            System.out.println("Library not found at: " + path);
        }
    }

    throw new IllegalStateException("""
        Llama library not found. Please set LLAMA_LIBRARY_PATH environment variable.
        Checked locations:
        - ~/Workspace/workkayys/Products/AI/core/gollek/third_party/llama.cpp/build/bin/libllama.dylib
        - /usr/local/lib/libllama.so
        - /usr/lib/libllama.so
        - ./libllama.so
        """);
}

    private static String getModelPath() {
        String modelPath = System.getenv("LLAMA_MODEL_PATH");
        if (modelPath != null && Files.exists(Path.of(modelPath))) {
            return modelPath;
        }

        // Check common model locations
        String[] commonPaths = {
          
                "./models/llama-2-7b-chat.gguf",
                "./models/codellama-7b.gguf",
                "./models/mistral-7b.gguf",
                "./model.gguf"
        };

        for (String path : commonPaths) {
            if (Files.exists(Path.of(path))) {
                return path;
            }
        }

        throw new IllegalStateException("""
                Model file not found. Please set LLAMA_MODEL_PATH environment variable.
                Expected: GGUF format model file
                Common locations:
                - ./models/llama-2-7b-chat.gguf
                - ./models/codellama-7b.gguf
                - ./model.gguf
                """);
    }

    private static int getGpuLayers() {
        // Auto-detect GPU capability
        if (System.getenv("LLAMA_GPU_LAYERS") != null) {
            return Integer.parseInt(System.getenv("LLAMA_GPU_LAYERS"));
        }

        // Default: 0 for CPU-only, or auto-detect based on available GPU memory
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("nix") || os.contains("nux") || os.contains("mac")) {
            // Try to detect NVIDIA GPU on Linux/Mac
            try {
                Process process = Runtime.getRuntime()
                        .exec("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits");
                process.waitFor();
                String output = new String(process.getInputStream().readAllBytes()).trim();
                if (!output.isEmpty()) {
                    int vramGB = Integer.parseInt(output.split("\n")[0]) / 1024;
                    // Allocate layers based on VRAM (rough heuristic)
                    return Math.min(40, vramGB * 5); // ~5 layers per GB
                }
            } catch (Exception e) {
                // GPU not available or nvidia-smi not found
            }
        }

        return 0; // CPU-only fallback
    }

    private static LlamaConfig.SamplingConfig createSamplingConfig() {
        return LlamaConfig.SamplingConfig.builder()
                .temperature(0.7f) // Balanced creativity
                .topK(40)
                .topP(0.9f)
                .minP(0.05f)
                .repeatPenalty(1.1f)
                .repeatLastN(64)
                .presencePenalty(0.0f)
                .frequencyPenalty(0.0f)
                .build();
    }

    private static void runExample(LlamaEngine engine) {
        System.out.println("=== Llama.cpp Java Example ===\n");

        // Display model information
        ModelInfo modelInfo = engine.getModelInfo();
        System.out.println("Model: " + modelInfo.name());
        System.out.println("Description: " + modelInfo.description());
        System.out.println("Context: " + modelInfo.contextLength() + " tokens");
        System.out.println("Vocab: " + modelInfo.vocabSize() + " tokens");
        System.out.println("Parameters: " + formatNumber(modelInfo.parameterCount()));
        System.out.println();

        // Example 1: Simple generation with streaming
        System.out.println("1. Simple Generation (Streaming):");
        System.out.println("Prompt: Write a short poem about Java programming");
        System.out.println("Response:");

        AtomicInteger tokenCount = new AtomicInteger(0);
        GenerationResult result1 = engine.generate(
                "Write a short poem about Java programming",
                engine.getConfig().sampling(),
                100,
                List.of("\n\n"), // Stop on double newline
                token -> {
                    System.out.print(token);
                    tokenCount.incrementAndGet();
                });

        System.out.println("\n\nGenerated " + tokenCount.get() + " tokens");
        System.out.println("Finish reason: " + result1.finishReason());
        System.out.println("Time: " + result1.timeMs() + "ms");
        System.out.println();

        // Example 2: Chat conversation
        System.out.println("2. Chat Conversation:");
        List<ChatMessage> messages = List.of(
                new ChatMessage("system", "You are a helpful programming assistant."),
                new ChatMessage("user", "Explain polymorphism in object-oriented programming in simple terms."));

        GenerationResult result2 = engine.chat(
                messages,
                LlamaConfig.SamplingConfig.builder()
                        .temperature(0.3f) // Lower temperature for factual responses
                        .topP(0.95f)
                        .build(),
                150,
                null);

        System.out.println("Response: " + result2.text());
        System.out.println("Generated " + result2.tokensGenerated() + " tokens");
        System.out.println();

        // Example 3: Code generation
        System.out.println("3. Code Generation:");
        GenerationResult result3 = engine.generate(
                "Write a Java function to calculate factorial:",
                LlamaConfig.SamplingConfig.builder()
                        .temperature(0.2f) // Very low temperature for code
                        .topP(0.9f)
                        .build(),
                80,
                List.of("```"), // Stop at code block end
                null);

        System.out.println("Generated code:");
        System.out.println(result3.text());
        System.out.println();

        // Example 4: Using stop sequences
        System.out.println("4. Stop Sequence Example:");
        GenerationResult result4 = engine.generate(
                "List three programming languages:",
                engine.getConfig().sampling(),
                50,
                List.of("4.", "four", "4"), // Stop before listing too many
                null);

        System.out.println("Response: " + result4.text());
        System.out.println("Stopped after: " + result4.tokensGenerated() + " tokens");
        System.out.println();

        // Performance summary
        System.out.println("=== Performance Summary ===");
        System.out.println("Total tokens processed: " + engine.getTotalTokensProcessed());
        System.out.println("Total generations: " + engine.getTotalGenerations());
    }

    private static String formatNumber(long number) {
        if (number >= 1_000_000_000) {
            return String.format("%.1fB", number / 1_000_000_000.0);
        } else if (number >= 1_000_000) {
            return String.format("%.1fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.1fK", number / 1_000.0);
        }
        return String.valueOf(number);
    }

    // Additional utility class for batch processing
    static class BatchProcessor {
        private final LlamaEngine engine;

        public BatchProcessor(LlamaEngine engine) {
            this.engine = engine;
        }

        public void processPrompts(List<String> prompts) {
            System.out.println("Processing " + prompts.size() + " prompts...");

            for (int i = 0; i < prompts.size(); i++) {
                System.out.println("\n--- Prompt " + (i + 1) + " ---");
                System.out.println("Input: " + prompts.get(i));

                GenerationResult result = engine.generate(
                        prompts.get(i),
                        engine.getConfig().sampling(),
                        100,
                        null,
                        null);

                System.out.println("Output: " + result.text());
                System.out.println("Tokens: " + result.tokensGenerated());
                System.out.println("Time: " + result.timeMs() + "ms");
            }
        }
    }
}
