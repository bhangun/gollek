package io.quarkus.pytorch.examples;

import io.quarkus.pytorch.core.Tensor;
import io.quarkus.pytorch.nn.Linear;
import io.quarkus.pytorch.nn.Module;
import io.quarkus.pytorch.nn.Sequential;
import io.quarkus.pytorch.optim.SGD;
import io.quarkus.pytorch.optim.Adam;

/**
 * Complete end-to-end example: Neural network for MNIST digit classification.
 * Mirrors the PyTorch C++ frontend example from the documentation.
 * 
 * Network architecture:
 * - Input: 784 (28x28 flattened image)
 * - Hidden layer 1: 64 neurons with ReLU
 * - Hidden layer 2: 32 neurons with ReLU
 * - Output: 10 classes (digits 0-9)
 */
public class MNISTNet extends Module {
    
    private Linear fc1;
    private Linear fc2;
    private Linear fc3;
    
    /**
     * Construct the network and register submodules.
     */
    public MNISTNet() {
        // Define layers
        fc1 = registerModule("fc1", new Linear(784, 64));
        fc2 = registerModule("fc2", new Linear(64, 32));
        fc3 = registerModule("fc3", new Linear(32, 10));
    }
    
    /**
     * Forward pass through the network.
     * 
     * @param input Input tensor of shape (batch_size, 784)
     * @return Output tensor of shape (batch_size, 10)
     */
    @Override
    public Tensor forward(Tensor input) {
        // x = relu(fc1(input))
        Tensor x = fc1.forward(input).relu();
        
        // x = relu(fc2(x))
        x = fc2.forward(x).relu();
        
        // output = fc3(x)
        return fc3.forward(x);
    }
    
    /**
     * Alternative constructor using Sequential for cleaner syntax.
     */
    public static Module createSequential() {
        return new Sequential()
            .add("fc1", new Linear(784, 64))
            .add("relu1", new ActivationModule("relu"))
            .add("fc2", new Linear(64, 32))
            .add("relu2", new ActivationModule("relu"))
            .add("fc3", new Linear(32, 10));
    }
    
    /**
     * Helper class for activation functions in Sequential.
     */
    private static class ActivationModule extends Module {
        private final String activation;
        
        ActivationModule(String activation) {
            this.activation = activation;
        }
        
        @Override
        public Tensor forward(Tensor input) {
            return switch (activation.toLowerCase()) {
                case "relu" -> input.relu();
                case "sigmoid" -> input.sigmoid();
                case "tanh" -> input.tanh();
                default -> throw new IllegalArgumentException("Unknown activation: " + activation);
            };
        }
    }
    
    /**
     * Training loop example.
     */
    public static void trainExample() {
        // Create the network
        MNISTNet net = new MNISTNet();
        net.train();  // Set to training mode
        
        // Create optimizer
        SGD optimizer = new SGD(net.parameters(), 0.01, 0.9);  // lr=0.01, momentum=0.9
        
        // Training loop
        int numEpochs = 10;
        int batchSize = 64;
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // In real implementation, iterate over data loader
            
            // Generate dummy batch (in practice, load from dataset)
            try (Tensor input = Tensor.randn(new long[]{batchSize, 784}, Tensor.ScalarType.FLOAT);
                 Tensor target = Tensor.rand(new long[]{batchSize, 10}, Tensor.ScalarType.FLOAT)) {
                
                // Enable gradient computation
                input.requiresGrad(true);
                
                // Forward pass
                Tensor output = net.forward(input);
                
                // Compute loss (MSE for demonstration)
                Tensor loss = computeMSELoss(output, target);
                
                // Backward pass
                optimizer.zeroGrad();
                loss.backward();
                
                // Update weights
                optimizer.step();
                
                System.out.printf("Epoch [%d/%d], Loss: %.4f%n", 
                    epoch + 1, numEpochs, getLossValue(loss));
                
                // Clean up
                output.close();
                loss.close();
            }
        }
        
        // Cleanup
        net.close();
    }
    
    /**
     * Inference example.
     */
    public static void inferenceExample() {
        // Load trained model
        MNISTNet net = new MNISTNet();
        net.eval();  // Set to evaluation mode
        
        // Create input
        try (Tensor input = Tensor.randn(new long[]{1, 784}, Tensor.ScalarType.FLOAT)) {
            
            // Forward pass (no gradient computation needed)
            Tensor output = net.forward(input);
            
            // Apply softmax to get probabilities
            Tensor probs = output.softmax(1);
            
            // Get prediction (argmax)
            int prediction = argmax(probs);
            
            System.out.printf("Predicted digit: %d%n", prediction);
            
            // Cleanup
            output.close();
            probs.close();
        }
        
        net.close();
    }
    
    /**
     * Helper: Compute MSE loss.
     */
    private static Tensor computeMSELoss(Tensor predictions, Tensor targets) {
        // Simplified loss computation
        // In real implementation, use proper loss function
        Tensor diff = predictions.add(targets.mul(
            Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT).add(-1.0)));
        return diff.mul(diff);
    }
    
    /**
     * Helper: Extract scalar value from loss tensor.
     */
    private static double getLossValue(Tensor loss) {
        // In real implementation, extract scalar value from tensor
        // This would use dataPtr() and read the value
        return 0.0;  // Placeholder
    }
    
    /**
     * Helper: Get argmax index.
     */
    private static int argmax(Tensor tensor) {
        // In real implementation, find index of maximum value
        return 0;  // Placeholder
    }
    
    /**
     * Main method demonstrating usage.
     */
    public static void main(String[] args) {
        System.out.println("=== PyTorch Java Binding Example ===");
        System.out.println("CUDA available: " + Tensor.cudaIsAvailable());
        
        // Create network
        System.out.println("\nCreating MNIST network...");
        MNISTNet net = new MNISTNet();
        System.out.println(net);
        System.out.println("Total parameters: " + net.parameters().size());
        
        // Test forward pass
        System.out.println("\nTesting forward pass...");
        try (Tensor input = Tensor.randn(new long[]{2, 784}, Tensor.ScalarType.FLOAT)) {
            Tensor output = net.forward(input);
            System.out.println("Input shape: " + java.util.Arrays.toString(input.shape()));
            System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
            output.close();
        }
        
        // Move to GPU if available
        if (Tensor.cudaIsAvailable()) {
            System.out.println("\nMoving network to CUDA...");
            net.to(Tensor.Device.CUDA);
        }
        
        net.close();
        System.out.println("\nExample completed!");
    }
}
