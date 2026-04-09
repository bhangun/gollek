package tech.kayys.gollek.ml.nn.examples;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.optim.*;

/**
 * Example 2: Transformer encoder for sequence processing.
 * <p>
 * This example demonstrates:
 * - Building a transformer encoder stack
 * - Multi-head attention with causal masking (optional)
 * - Positional embeddings
 * - Sequence-to-sequence processing
 */
public class TransformerEncoderExample {

    public static void main(String[] args) {
        int embedDim = 64;
        int numHeads = 4;
        int ffDim = 256;
        int numLayers = 3;
        int vocabSize = 1000;
        int maxSeqLen = 128;

        // Build transformer encoder
        tech.kayys.gollek.ml.nn.NNModule model = new Sequential(
            // Embedding layer
            new Embedding(vocabSize, embedDim),

            // Stack of transformer layers
            new TransformerEncoderLayer(embedDim, numHeads, ffDim, 0.1f),
            new TransformerEncoderLayer(embedDim, numHeads, ffDim, 0.1f),
            new TransformerEncoderLayer(embedDim, numHeads, ffDim, 0.1f),

            // Output projection
            new Linear(embedDim, 10)  // 10 output classes
        );

        // Setup training
        var optimizer = new Adam(model.parameters(), 0.0001f);
        var scheduler = new CosineAnnealingLR(optimizer, 100, 1e-5f);

        // Training loop (pseudo-code)
        int epochs = 5;
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Create dummy sequence: [batch_size, seq_len]
            int batchSize = 4;
            int seqLen = 32;
            long[] shape = {batchSize, seqLen};
            float[] seqData = new float[(int)(batchSize * seqLen)];
            for (int i = 0; i < seqData.length; i++) {
                seqData[i] = i % vocabSize;  // Token indices
            }

            // Forward pass
            var input = GradTensor.of(seqData, shape);
            var output = model.forward(input);

            // Loss and backward (simplified)
            var loss = new CrossEntropyLoss();
            int[] targets = new int[batchSize];
            float[] targetData = new float[batchSize];
            for (int i = 0; i < batchSize; i++) {
                targetData[i] = targets[i];
            }
            var target = GradTensor.of(targetData, new long[]{batchSize});
            var lossValue = loss.compute(output, target);

            lossValue.backward();
            optimizer.step();
            optimizer.zeroGrad();
            scheduler.step();

            System.out.println("Epoch " + epoch + ", Loss: " + lossValue.item());
        }

        System.out.println("Transformer training complete!");
    }
}
