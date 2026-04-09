# Gollek SDK-NN Getting Started Guide

Welcome! This guide will help you quickly get started with the enhanced gollek-sdk-nn neural network framework.

## Quick Links

- 📚 **[API Reference](API_REFERENCE.md)** - Complete API documentation
- 📊 **[Enhancement Summary](ENHANCEMENT_SUMMARY.md)** - What's new and changed
- 💻 **[SimpleFFN Example](src/main/java/tech/kayys/gollek/ml/nn/examples/SimpleFFNExample.java)** - 3-layer network
- 🔄 **[Transformer Example](src/main/java/tech/kayys/gollek/ml/nn/examples/TransformerEncoderExample.java)** - Transformer encoder
- 🧪 **[Integration Tests](src/test/java/tech/kayys/gollek/ml/nn/IntegrationTest.java)** - Test examples
- 🌐 **[Website Docs](../../website/wayang.github.io/GOLLEK_SDK_NN_ENHANCEMENTS.md)** - Comprehensive enhancement guide

## Installation

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>tech.kayys</groupId>
    <artifactId>gollek-sdk-nn</artifactId>
    <version>1.0.0</version>
</dependency>
```

Build:
```bash
mvn clean compile
```

## Your First Network

```java
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.optim.*;
import tech.kayys.gollek.ml.autograd.GradTensor;

// 1. Build model
Module model = new Sequential(
    new Linear(784, 256),      // Input: 784, Hidden: 256
    new ReLU(),                 // Activation
    new Dropout(0.2f),          // Regularization
    new Linear(256, 10)         // Output: 10 classes
);

// 2. Setup optimizer and loss
var optimizer = new Adam(model.parameters(), 0.001f);
var loss = new CrossEntropyLoss();

// 3. Create dummy data
float[] input = new float[784];
for (int i = 0; i < 784; i++) {
    input[i] = (float) Math.random();
}
float[] target = new float[1];
target[0] = 3;  // Class 3

// 4. Training step
var x = GradTensor.of(input, new long[]{1, 784});
var y = model.forward(x);
var lossVal = loss.compute(y, GradTensor.of(target, new long[]{1}));

// 5. Backward and optimize
lossVal.backward();
optimizer.step();
optimizer.zeroGrad();

System.out.println("Loss: " + lossVal.item());
```

## Common Patterns

### Using Learning Rate Scheduling

```java
var optimizer = new Adam(model.parameters(), 0.001f);

// Option 1: Step decay
var scheduler = new StepLR(optimizer, 10, 0.1f);

// Option 2: Cosine annealing
var scheduler = new CosineAnnealingLR(optimizer, 100, 1e-6f);

for (int epoch = 0; epoch < 100; epoch++) {
    // Training...
    scheduler.step();
}
```

### Evaluating with Metrics

```java
var metric = new tech.kayys.gollek.ml.nn.metrics.Accuracy();

for (int i = 0; i < batchSize; i++) {
    var pred = model.forward(batchInput);
    metric.update(predictions, targets);
}

float accuracy = metric.compute();
System.out.println("Accuracy: " + (accuracy * 100) + "%");
```

### Early Stopping

```java
var earlyStopping = new EarlyStopping(
    10,      // patience: stop after 10 epochs without improvement
    true,    // restore best weights
    0,       // minDelta: any improvement counts
    "min"    // mode: minimize validation loss
);

for (int epoch = 0; epoch < 200; epoch++) {
    // Training...
    float valLoss = evaluateOnValidationSet();
    
    if (earlyStopping.check(valLoss)) {
        System.out.println("Stopped at epoch " + epoch);
        break;
    }
}
```

### Batch Normalization

```java
var model = new Sequential(
    new Linear(784, 256),
    new BatchNorm1d(256),  // After linear layer
    new ReLU()
);

// Important: Switch modes during training/evaluation
model.train();   // Use batch statistics
// Training loop...

model.eval();    // Use running statistics
// Evaluation...
```

### Residual Blocks

```java
var resblock = new ResidualBlock(
    new Sequential(
        new Linear(256, 256),
        new ReLU(),
        new Linear(256, 256)
    )
);

var output = resblock.forward(input);
// output = input + transformation(input)
```

## Available Activations

| Activation | Formula | When to Use |
|-----------|---------|------------|
| **ReLU** | max(0, x) | Standard choice |
| **LeakyReLU** | x if x>0 else αx | Prevent dying neurons |
| **ELU** | x if x>0 else α(exp(x)-1) | Smooth gradients |
| **GELU** | x * Φ(x) | Transformers |
| **SiLU** | x * sigmoid(x) | Modern networks |
| **Mish** | x * tanh(softplus(x)) | Vision + NLP |
| **Tanh** | tanh(x) | Legacy/specific use |
| **Sigmoid** | 1 / (1 + exp(-x)) | Binary classification |

## Available Loss Functions

| Loss | Use Case |
|------|----------|
| **CrossEntropyLoss** | Multi-class classification |
| **BCEWithLogitsLoss** | Binary classification |
| **MSELoss** | Regression |
| **L1Loss** | Robust regression (outlier-resistant) |
| **SmoothL1Loss** | Object detection bounding boxes |
| **CosineEmbeddingLoss** | Similarity/embedding learning |

## Testing

Run all tests:
```bash
mvn test
```

Run specific test:
```bash
mvn test -Dtest=IntegrationTest
```

## Build and Compile

Check for compilation errors:
```bash
mvn clean compile
```

Build without running tests:
```bash
mvn clean compile -DskipTests
```

## What's New?

### Phase 1: Critical Fixes ✅
- Fixed CosineEmbeddingLoss backward pass
- Implemented causal masking in MultiHeadAttention
- Added 800+ LOC of documentation
- Added comprehensive input validation

### Phase 2: Core Enhancements ✅
- 3 new activations (LeakyReLU, ELU, Mish)
- 3 new losses (L1Loss, SmoothL1Loss, BCEWithLogitsLoss)
- BatchNorm1d with running statistics
- Learning rate schedulers (StepLR, CosineAnnealingLR)
- ResidualBlock for skip connections

### Phase 3: Extended Features ✅
- Adam optimizer
- Accuracy metric
- EarlyStopping callback
- LRScheduler base class

### Phase 4: Testing & Docs ✅
- Comprehensive integration tests
- Complete API reference (8,700+ lines)
- 2 working example programs
- Full JavaDoc coverage

## Dimension Conventions

All tensors: `[batch, ...features...]`

```
Linear: [batch, in] → [batch, out]
Embedding: [batch, seq] → [batch, seq, embed]
LayerNorm: [batch, seq, embed] → [batch, seq, embed]
Attention: [batch, seq, embed] → [batch, seq, embed]
Dropout: [batch, ...] → [batch, ...]
```

## Performance Tips

1. **Use BatchNorm** - Stabilizes training and allows higher learning rates
2. **Use Dropout** - Prevents overfitting (typically 0.1-0.5)
3. **Learning Rate Scheduling** - Gradual decay improves convergence
4. **Early Stopping** - Monitor validation loss to prevent overfitting
5. **Proper Initialization** - Linear layers use Kaiming initialization

## Troubleshooting

### NaN Loss
- Check input ranges (very large values cause overflow)
- Verify label format matches loss function
- Reduce learning rate

### Exploding Gradients
- Use gradient clipping
- Reduce learning rate
- Add batch normalization

### Poor Convergence
- Increase training time
- Try different learning rate
- Add batch normalization
- Use better scheduler (cosine annealing)

## Documentation

- **API_REFERENCE.md** - Detailed API documentation
- **ENHANCEMENT_SUMMARY.md** - What's new and statistics
- **JavaDoc** - In each source file
- **Examples** - SimpleFFNExample.java, TransformerEncoderExample.java
- **Tests** - IntegrationTest.java for usage patterns

## Next Steps

1. Read [API_REFERENCE.md](API_REFERENCE.md) for detailed component info
2. Check [SimpleFFNExample.java](src/main/java/tech/kayys/gollek/ml/nn/examples/SimpleFFNExample.java)
3. Explore [ENHANCEMENT_SUMMARY.md](ENHANCEMENT_SUMMARY.md) for what's new
4. Run tests: `mvn test`
5. Build your own models!

## Support

- 📖 See API_REFERENCE.md for comprehensive documentation
- 💡 Check examples directory for working code
- 🧪 Look at IntegrationTest.java for test patterns
- 📝 Review JavaDoc comments in source files

Happy neural network building! 🚀
