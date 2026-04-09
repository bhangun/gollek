# Gollek SDK-NN API Reference

## Overview

The Gollek SDK-NN module provides a comprehensive neural network framework for building and training deep learning models in Java. It includes layers, activations, loss functions, optimizers, and utilities with full JavaDoc coverage and proper error handling.

## Module Architecture

### Core Components

#### 1. **Layers** (`tech.kayys.gollek.ml.nn`)

**Dense Layers**
- `Linear`: Fully connected layer with Kaiming initialization
- `Embedding`: Token embedding lookup table for NLP tasks

**Normalization**
- `LayerNorm`: Normalizes activations along feature dimension
- `BatchNorm1d`: Batch normalization with training/eval modes and running statistics

**Attention**
- `MultiHeadAttention`: Multi-head self-attention with causal masking support
- `TransformerEncoderLayer`: Complete encoder block (attention + FFN)
- `TransformerDecoderLayer`: Decoder with causal attention + cross-attention

**Architectural Helpers**
- `ResidualBlock`: Skip connections for deep networks
- `Sequential`: Container for stacking modules

**Regularization**
- `Dropout`: Inverted dropout with configurable drop rate
- `Parameter`: Learnable parameters with gradient tracking

#### 2. **Activation Functions** (`tech.kayys.gollek.ml.nn`)

- `ReLU`: Rectified Linear Unit (max(0, x))
- `Sigmoid`: Logistic sigmoid (1 / (1 + exp(-x)))
- `Tanh`: Hyperbolic tangent (-1 to 1 range)
- `GELU`: Gaussian Error Linear Unit (approximation via tanh)
- `SiLU`: Sigmoid Linear Unit (self-gating)
- `LeakyReLU`: Leaky ReLU with configurable negative slope
- `ELU`: Exponential Linear Unit with saturation
- `Mish`: Modern self-regularizing activation

#### 3. **Loss Functions** (`tech.kayys.gollek.ml.nn.loss`)

**Classification**
- `CrossEntropyLoss`: Multi-class classification (log-softmax + NLL)
- `BCEWithLogitsLoss`: Binary classification with numerically stable sigmoid

**Regression**
- `MSELoss`: Mean Squared Error
- `L1Loss`: Mean Absolute Error (robust to outliers)
- `SmoothL1Loss`: Huber loss (hybrid L1/MSE)

**Embedding**
- `CosineEmbeddingLoss`: Similarity loss for embeddings (with fixed backward pass)

#### 4. **Optimizers** (`tech.kayys.gollek.ml.nn.optim`)

- `SGD`: Stochastic Gradient Descent with momentum
- `Adam`: Adaptive Moment Estimation (default for most tasks)
- (Future: RMSprop, AdamW, LAMB with weight decay)

#### 5. **Learning Rate Schedulers** (`tech.kayys.gollek.ml.nn.optim`)

- `StepLR`: Step decay (multiply by gamma every N steps)
- `CosineAnnealingLR`: Cosine annealing over T_max iterations
- (Future: ReduceLROnPlateau, ExponentialLR, WarmupScheduler)

#### 6. **Metrics** (`tech.kayys.gollek.ml.nn.metrics`)

- `Accuracy`: Classification accuracy (0-1 scale)
- (Future: Precision, Recall, F1Score, ConfusionMatrix)

#### 7. **Training Utilities** (`tech.kayys.gollek.ml.nn`)

- `EarlyStopping`: Monitor validation metric and stop training when plateauing
- (Future: ModelCheckpoint, TensorBoard integration)

## Dimension Conventions

All tensors follow [batch, ...feature dimensions...] convention:

```
Linear:             [batch, in_features] → [batch, out_features]
LayerNorm:          [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]
MultiHeadAttention: [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]
Embedding:          [batch, seq_len] → [batch, seq_len, embed_dim]
Dropout:            [batch, ...] → [batch, ...]
BatchNorm1d:        [batch, features] → [batch, features]
```

## Common Patterns

### Training Loop

```java
// Build model
Module model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
);

// Setup training
var optimizer = new Adam(model.parameters(), 0.001f);
var scheduler = new StepLR(optimizer, 10, 0.1f);
var loss = new CrossEntropyLoss();
var metric = new Accuracy();

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    for (var batch : dataloader) {
        // Forward
        var logits = model.forward(batch.input);
        
        // Compute loss
        var lossVal = loss.compute(logits, batch.target);
        
        // Backward
        lossVal.backward();
        
        // Update
        optimizer.step();
        optimizer.zeroGrad();
        
        // Track metrics
        metric.update(predictions, batch.target);
    }
    
    // Schedule learning rate
    scheduler.step();
    System.out.println("Epoch " + epoch + ", Loss: " + lossVal.item());
}
```

### Batch Normalization in Training

```java
// Training mode (default)
model.train();  // Compute batch statistics, update running stats
// Forward pass...

// Evaluation mode
model.eval();   // Use running statistics
// Forward pass...
```

### Multi-Head Attention with Causal Masking

```java
var attention = new MultiHeadAttention(embedDim, numHeads);

// For autoregressive generation (decoder)
var output = attention.forward(input, createCausalMask(seqLen, seqLen));
```

### Learning Rate Scheduling

```java
var optimizer = new Adam(params, 0.001f);

// Step decay: multiply by 0.1 every 30 epochs
var scheduler = new StepLR(optimizer, 30, 0.1f);

// Cosine annealing: smooth decay to 1e-6 over 100 epochs
var scheduler = new CosineAnnealingLR(optimizer, 100, 1e-6f);

for (int epoch = 0; epoch < 100; epoch++) {
    // Training...
    scheduler.step();  // Update learning rate
}
```

## Key Implementation Details

### Numerical Stability

All loss functions use techniques to prevent overflow/underflow:

- **Log-sum-exp trick**: Subtract max before exp in softmax
- **Sigmoid stability**: Two forms prevent exp(-x) overflow
- **Epsilon additions**: 1e-8 for log(), 1e-5 for division

### Gradient Computation

All backward passes use the closure pattern:

```java
gradTensor.setGradFn(new Function.Context("name") {
    @Override
    public void backward(GradTensor upstream) {
        // Compute gradients using closure variables
    }
});
```

### Training vs Evaluation Modes

Modules that behave differently:

- `Dropout`: Applies masking in train mode, identity in eval mode
- `BatchNorm1d`: Uses batch stats in train, running stats in eval
- `LayerNorm`: Always normalizes (no mode difference)

## Error Handling

All modules validate inputs with helpful error messages:

```java
// Shape validation
if (input.shape()[0] != expectedBatch) {
    throw new IllegalArgumentException(
        "Expected batch size " + expectedBatch + 
        " but got " + input.shape()[0]
    );
}

// Range validation
if (dropoutRate < 0 || dropoutRate > 1) {
    throw new IllegalArgumentException(
        "dropout rate must be in [0, 1], got: " + dropoutRate
    );
}
```

## Testing

### Unit Tests
Each component has comprehensive unit tests covering:
- Forward pass correctness
- Shape validation
- Gradient computation
- Edge cases (empty tensors, boundary values)

### Integration Tests
End-to-end tests combining multiple modules:
- Feedforward networks
- Transformer encoders
- Residual blocks
- Complete training loops

Run tests with:
```bash
mvn test
```

## Performance Considerations

### Memory Usage
- `Adam`: 2x parameter memory (for m and v estimates)
- `MultiHeadAttention`: O(seq_len²) memory for attention weights
- `BatchNorm1d`: ~3x parameter memory (weights, bias, running stats)

### Computational Complexity
- `Linear`: O(in_features × out_features)
- `MultiHeadAttention`: O(seq_len² × embed_dim)
- `LayerNorm`: O(embed_dim)

## Known Limitations

1. No distributed training support
2. No GPU acceleration
3. No KV cache for efficient inference
4. BatchNorm1d backward pass simplified
5. No seed control for reproducibility
6. No mixed precision training

## Future Enhancements

- [ ] Conv1D/Conv2D layers
- [ ] RNN/LSTM layers
- [ ] More optimizers (RMSprop, AdamW, LAMB)
- [ ] More metrics (Precision, Recall, F1)
- [ ] Model checkpointing and save/load
- [ ] DataLoader with batching and shuffling
- [ ] Distributed training
- [ ] GPU support
- [ ] Mixed precision
- [ ] Quantization support

## Troubleshooting

### NaN Loss
- Check input ranges (very large values cause overflow)
- Verify label format matches loss function expectations
- Try reducing learning rate

### Exploding Gradients
- Use gradient clipping
- Reduce learning rate
- Check for numerical instabilities in custom modules

### Poor Model Performance
- Verify data preprocessing and normalization
- Check learning rate schedule
- Ensure proper train/eval mode switching
- Verify layer initialization

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)
- [GELU: Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
- [Mish: A Self Regularized Activation Function](https://arxiv.org/abs/1908.03682)
