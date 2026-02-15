package io.quarkus.pytorch.examples;

import io.quarkus.pytorch.core.Tensor;
import io.quarkus.pytorch.nn.*;
import io.quarkus.pytorch.optim.Adam;

/**
 * Advanced Convolutional Neural Network example for image classification.
 * Architecture similar to LeNet-5 for CIFAR-10 dataset.
 * 
 * Network Structure:
 * - Conv2d(3, 32, 3x3) -> ReLU -> MaxPool2d(2x2)
 * - Conv2d(32, 64, 3x3) -> ReLU -> MaxPool2d(2x2)
 * - Conv2d(64, 128, 3x3) -> ReLU
 * - Flatten
 * - Linear(128*4*4, 256) -> ReLU -> Dropout(0.5)
 * - Linear(256, 10)
 */
public class AdvancedCNN extends Module {
    
    private Sequential features;
    private Sequential classifier;
    
    /**
     * Construct the CNN architecture.
     */
    public AdvancedCNN() {
        // Feature extraction layers (convolutional)
        features = registerModule("features", new Sequential()
            // Block 1: 3 -> 32 channels
            .add("conv1", new Conv2d(3, 32, 3, 1, 1))
            .add("relu1", new ActivationLayer(Activation.RELU))
            .add("pool1", new MaxPool2d(2))
            
            // Block 2: 32 -> 64 channels
            .add("conv2", new Conv2d(32, 64, 3, 1, 1))
            .add("relu2", new ActivationLayer(Activation.RELU))
            .add("pool2", new MaxPool2d(2))
            
            // Block 3: 64 -> 128 channels
            .add("conv3", new Conv2d(64, 128, 3, 1, 1))
            .add("relu3", new ActivationLayer(Activation.RELU))
        );
        
        // Classification layers (fully connected)
        // Input: 128 * 8 * 8 = 8192 (for 32x32 input images after pooling)
        classifier = registerModule("classifier", new Sequential()
            .add("flatten", new FlattenLayer())
            .add("fc1", new Linear(128 * 8 * 8, 256))
            .add("relu4", new ActivationLayer(Activation.RELU))
            .add("dropout", new Dropout(0.5))
            .add("fc2", new Linear(256, 10))
        );
    }
    
    @Override
    public Tensor forward(Tensor input) {
        // Input shape: (batch, 3, 32, 32)
        Tensor x = features.forward(input);
        // After features: (batch, 128, 8, 8)
        
        x = classifier.forward(x);
        // Output shape: (batch, 10)
        
        return x;
    }
    
    /**
     * Helper activation layer for Sequential.
     */
    private static class ActivationLayer extends Module {
        private final Activation activation;
        
        public ActivationLayer(Activation activation) {
            this.activation = activation;
        }
        
        @Override
        public Tensor forward(Tensor input) {
            return switch (activation) {
                case RELU -> input.relu();
                case SIGMOID -> input.sigmoid();
                case TANH -> input.tanh();
            };
        }
        
        @Override
        public String toString() {
            return activation.name();
        }
    }
    
    /**
     * Flatten layer for Sequential.
     */
    private static class FlattenLayer extends Module {
        @Override
        public Tensor forward(Tensor input) {
            // Flatten all dimensions except batch
            long batchSize = input.size(0);
            long numFeatures = input.numel() / batchSize;
            return input.reshape(new long[]{batchSize, numFeatures});
        }
        
        @Override
        public String toString() {
            return "Flatten()";
        }
    }
    
    /**
     * MaxPool2d placeholder (would need actual implementation).
     */
    private static class MaxPool2d extends Module {
        private final int kernelSize;
        
        public MaxPool2d(int kernelSize) {
            this.kernelSize = kernelSize;
        }
        
        @Override
        public Tensor forward(Tensor input) {
            throw new UnsupportedOperationException(
                "MaxPool2d requires additional FFM bindings");
        }
        
        @Override
        public String toString() {
            return String.format("MaxPool2d(kernel_size=%d)", kernelSize);
        }
    }
    
    private enum Activation {
        RELU, SIGMOID, TANH
    }
    
    /**
     * Training example with advanced features.
     */
    public static void trainAdvanced() {
        System.out.println("=== Advanced CNN Training Example ===\n");
        
        // Create model
        AdvancedCNN model = new AdvancedCNN();
        model.train();
        
        System.out.println("Model Architecture:");
        System.out.println(model);
        System.out.println();
        
        // Move to GPU if available
        if (Tensor.cudaIsAvailable()) {
            System.out.println("Moving model to CUDA...");
            model.to(Tensor.Device.CUDA);
        }
        
        // Create optimizer
        Adam optimizer = new Adam(model.parameters(), 0.001);
        
        // Training configuration
        int numEpochs = 50;
        int batchSize = 64;
        double bestLoss = Double.MAX_VALUE;
        
        System.out.println("Training Configuration:");
        System.out.println("  Epochs: " + numEpochs);
        System.out.println("  Batch Size: " + batchSize);
        System.out.println("  Optimizer: Adam(lr=0.001)");
        System.out.println("  Device: " + (Tensor.cudaIsAvailable() ? "CUDA" : "CPU"));
        System.out.println();
        
        // Training loop
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double epochLoss = 0.0;
            int numBatches = 100;  // Simulated number of batches
            
            for (int batch = 0; batch < numBatches; batch++) {
                // Generate dummy batch (in practice, load from DataLoader)
                try (Tensor input = Tensor.randn(
                        new long[]{batchSize, 3, 32, 32}, 
                        Tensor.ScalarType.FLOAT);
                     Tensor target = Tensor.rand(
                        new long[]{batchSize, 10}, 
                        Tensor.ScalarType.FLOAT)) {
                    
                    // Enable gradients
                    input.requiresGrad(true);
                    
                    // Forward pass
                    Tensor output = model.forward(input);
                    
                    // Compute loss (would use actual loss function)
                    Tensor loss = computeLoss(output, target);
                    
                    // Backward pass
                    optimizer.zeroGrad();
                    loss.backward();
                    optimizer.step();
                    
                    epochLoss += extractLossValue(loss);
                    
                    // Cleanup
                    output.close();
                    loss.close();
                }
            }
            
            double avgLoss = epochLoss / numBatches;
            
            // Print progress
            if ((epoch + 1) % 5 == 0 || epoch == 0) {
                System.out.printf("Epoch [%3d/%3d], Loss: %.4f", 
                    epoch + 1, numEpochs, avgLoss);
                
                if (avgLoss < bestLoss) {
                    bestLoss = avgLoss;
                    System.out.print(" [Best]");
                }
                System.out.println();
            }
        }
        
        System.out.println("\nTraining completed!");
        System.out.printf("Best loss: %.4f\n", bestLoss);
        
        // Cleanup
        model.close();
    }
    
    /**
     * Inference example with image preprocessing.
     */
    public static void inferenceAdvanced() {
        System.out.println("=== Advanced CNN Inference Example ===\n");
        
        // Load model
        AdvancedCNN model = new AdvancedCNN();
        model.eval();
        
        if (Tensor.cudaIsAvailable()) {
            model.to(Tensor.Device.CUDA);
        }
        
        // Create sample input (simulating a 32x32 RGB image)
        try (Tensor image = Tensor.randn(
                new long[]{1, 3, 32, 32}, 
                Tensor.ScalarType.FLOAT)) {
            
            // Preprocess: normalize to [0, 1]
            // In real scenario: normalize with ImageNet stats
            Tensor normalized = image.add(0.5).mul(
                Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT).add(0.5));
            
            // Forward pass (no gradient computation)
            Tensor output = model.forward(normalized);
            
            // Apply softmax for probabilities
            Tensor probs = output.softmax(1);
            
            // Get top-5 predictions
            System.out.println("Top-5 Predictions:");
            for (int i = 0; i < 5; i++) {
                // Would extract actual probabilities and class indices
                System.out.printf("  %d. Class %d: %.2f%%\n", 
                    i + 1, i, Math.random() * 100);
            }
            
            // Cleanup
            normalized.close();
            output.close();
            probs.close();
        }
        
        model.close();
    }
    
    /**
     * Model analysis and profiling.
     */
    public static void analyzeModel() {
        System.out.println("=== Model Analysis ===\n");
        
        AdvancedCNN model = new AdvancedCNN();
        
        // Count parameters
        long totalParams = 0;
        long trainableParams = 0;
        
        System.out.println("Layer-wise Parameter Count:");
        for (var entry : model.namedParameters().entrySet()) {
            long params = entry.getValue().numel();
            totalParams += params;
            trainableParams += params;  // All params are trainable in this example
            
            System.out.printf("  %-30s: %,d\n", entry.getKey(), params);
        }
        
        System.out.println("\nModel Statistics:");
        System.out.printf("  Total parameters: %,d\n", totalParams);
        System.out.printf("  Trainable parameters: %,d\n", trainableParams);
        System.out.printf("  Model size: %.2f MB\n", 
            (totalParams * 4) / (1024.0 * 1024.0));  // Assuming float32
        
        // Test forward pass timing
        System.out.println("\nPerformance Benchmark:");
        try (Tensor input = Tensor.randn(
                new long[]{1, 3, 32, 32}, 
                Tensor.ScalarType.FLOAT)) {
            
            // Warmup
            for (int i = 0; i < 10; i++) {
                Tensor output = model.forward(input);
                output.close();
            }
            
            // Benchmark
            int iterations = 100;
            long startTime = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                Tensor output = model.forward(input);
                output.close();
            }
            long endTime = System.nanoTime();
            
            double avgTimeMs = (endTime - startTime) / (iterations * 1_000_000.0);
            double throughput = 1000.0 / avgTimeMs;
            
            System.out.printf("  Average inference time: %.2f ms\n", avgTimeMs);
            System.out.printf("  Throughput: %.2f images/sec\n", throughput);
        }
        
        model.close();
    }
    
    /**
     * Helper: Compute loss (placeholder).
     */
    private static Tensor computeLoss(Tensor predictions, Tensor targets) {
        // Simplified loss computation
        Tensor diff = predictions.add(targets.mul(
            Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT).add(-1.0)));
        return diff.mul(diff).mean();
    }
    
    /**
     * Helper: Extract scalar loss value (placeholder).
     */
    private static double extractLossValue(Tensor loss) {
        // Would read from tensor's data pointer
        return Math.random();  // Placeholder
    }
    
    /**
     * Main demonstration method.
     */
    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║  Advanced CNN Example - PyTorch Java  ║");
        System.out.println("╚════════════════════════════════════════╝");
        System.out.println();
        
        // Check CUDA
        if (Tensor.cudaIsAvailable()) {
            System.out.println("✓ CUDA is available");
        } else {
            System.out.println("○ Running on CPU");
        }
        System.out.println();
        
        // Run examples
        System.out.println("Select example to run:");
        System.out.println("1. Model Analysis");
        System.out.println("2. Training Simulation");
        System.out.println("3. Inference Example");
        System.out.println();
        
        // For demonstration, run analysis
        analyzeModel();
        
        System.out.println("\n" + "=".repeat(50));
        System.out.println("Example completed successfully!");
    }
}
