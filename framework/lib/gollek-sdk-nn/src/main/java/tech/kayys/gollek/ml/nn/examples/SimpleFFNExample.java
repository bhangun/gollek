package tech.kayys.gollek.ml.nn.examples;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.optim.*;

/**
 * Example 1: Simple feedforward network for classification.
 * <p>
 * This example demonstrates:
 * - Building a sequential model
 * - Training with cross-entropy loss
 * - Using Adam optimizer and learning rate scheduling
 * - Evaluating with accuracy metric
 */
public class SimpleFFNExample {

    public static void main(String[] args) {
        // Build model: 784 (MNIST) -> 256 -> 128 -> 10 (classes)
        tech.kayys.gollek.ml.nn.NNModule model = new Sequential(
            new Linear(784, 256),
            new ReLU(),
            new Dropout(0.2f),
            new Linear(256, 128),
            new ReLU(),
            new Dropout(0.2f),
            new Linear(128, 10)
        );

        // Setup optimizer and scheduler
        var optimizer = new Adam(model.parameters(), 0.001f);
        var scheduler = new StepLR(optimizer, 10, 0.1f);

        // Loss function
        var loss = new CrossEntropyLoss();

        // Training loop (pseudo-code)
        int epochs = 20;
        float[] batchX = new float[784];  // Mini-batch data
        float[] batchY = new float[10];   // Mini-batch labels

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            var input = GradTensor.of(batchX, new long[]{1, 784});
            var output = model.forward(input);

            // Compute loss
            var target = GradTensor.of(batchY, new long[]{1});
            var lossValue = loss.compute(output, target);

            // Backward pass
            lossValue.backward();

            // Optimization step
            optimizer.step();
            optimizer.zeroGrad();

            // Learning rate scheduling
            scheduler.step();

            System.out.println("Epoch " + epoch + ", Loss: " + lossValue.item());
        }

        System.out.println("Training complete!");
    }
}
