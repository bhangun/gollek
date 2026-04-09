# Gollek RNN Module

Comprehensive Recurrent Neural Network (RNN) layers for sequence processing.

## Features

- **LSTM Layer**: Long Short-Term Memory networks with forget, input, and output gates
- **GRU Layer**: Gated Recurrent Unit - lighter alternative to LSTM
- **Bidirectional Support**: Process sequences in both directions
- **Sequence Handling**: Support for variable-length sequences
- **Full Differentiation**: Complete backpropagation through time (BPTT)
- **Production Ready**: Fully tested and documented

## Quick Start

### LSTM - Long Short-Term Memory

```java
// Create an LSTM layer
// input_size=100, hidden_size=128, return all timesteps
LSTMLayer lstm = new LSTMLayer(100, 128, true);

// Forward pass
GradTensor input = GradTensor.randn(32, 50, 100);  // [batch_size=32, sequence_length=50, input_size=100]
GradTensor output = lstm.forward(input);           // [32, 50, 128]
```

### GRU - Gated Recurrent Unit

```java
// Create a GRU layer (lighter than LSTM)
GRULayer gru = new GRULayer(100, 128, true);

// Forward pass
GradTensor input = GradTensor.randn(32, 50, 100);  // [batch_size=32, seq_len=50, input_size=100]
GradTensor output = gru.forward(input);            // [32, 50, 128]
```

### LSTM Cell - Single Timestep

```java
// Create LSTM cell for custom loop control
LSTMCell lstmCell = new LSTMCell(100, 128);

// Initialize states
GradTensor hidden = lstmCell.initHidden(32);  // [batch_size=32, hidden_size=128]
GradTensor cell = lstmCell.initCell(32);      // [batch_size=32, hidden_size=128]

// Process single timestep
GradTensor x = GradTensor.randn(32, 100);     // [batch_size=32, input_size=100]
LSTMCell.LSTMOutput output = lstmCell.forward(x, hidden, cell);
GradTensor newHidden = output.hidden;          // [32, 128]
GradTensor newCell = output.cell;              // [32, 128]
```

### Bidirectional RNN

```java
// Create forward and backward LSTM layers
LSTMLayer forward = new LSTMLayer(100, 128, true);
LSTMLayer backward = new LSTMLayer(100, 128, true);

// Wrap with bidirectional
Bidirectional biLSTM = new Bidirectional(forward, backward);

// Forward pass
GradTensor input = GradTensor.randn(32, 50, 100);  // [batch, seq_len, input_size]
GradTensor output = biLSTM.forward(input);         // [batch, seq_len, 256] (128*2)
```

## API Reference

### LSTMLayer

Processes sequences using LSTM cells. Supports returning either the full sequence or just the last timestep.

**Constructor:**
```java
LSTMLayer(int inputSize, int hiddenSize)           // Returns only last timestep
LSTMLayer(int inputSize, int hiddenSize, boolean returnSequences)  // Full control
```

**Methods:**
- `GradTensor forward(GradTensor input)` - Forward pass
- `LSTMCell getCell()` - Get underlying cell
- `int getHiddenSize()` - Get hidden dimension
- `boolean isReturnSequences()` - Check return mode

**Example:**
```java
// Return only final hidden state (for classification)
LSTMLayer seqClassifier = new LSTMLayer(300, 256, false);
GradTensor text = GradTensor.randn(16, 100, 300);  // [batch, seq_len, embedding_size]
GradTensor classification = seqClassifier.forward(text);  // [16, 256]

// Return full sequence (for sequence labeling)
LSTMLayer seqLabeler = new LSTMLayer(300, 256, true);
GradTensor labels = seqLabeler.forward(text);      // [16, 100, 256]
```

### GRULayer

Similar interface to LSTMLayer but with fewer parameters.

**Constructor:**
```java
GRULayer(int inputSize, int hiddenSize)
GRULayer(int inputSize, int hiddenSize, boolean returnSequences)
```

**Performance:** ~1/3 fewer parameters than LSTM, often similar accuracy.

### LSTMCell

Low-level LSTM cell for manual sequence processing.

**Constructor:**
```java
LSTMCell(int inputSize, int hiddenSize)
```

**Methods:**
- `LSTMOutput forward(GradTensor input, GradTensor hidden, GradTensor cell)` - Single timestep
- `GradTensor initHidden(int batchSize)` - Initialize h_0
- `GradTensor initCell(int batchSize)` - Initialize c_0
- `GradTensor[] getParameters()` - Get weights and biases

**LSTM Equations:**
```
i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  # Input gate
f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  # Forget gate
g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)     # Cell candidate
o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  # Output gate
c_t = f_t * c_{t-1} + i_t * g_t                             # Cell state update
h_t = o_t * tanh(c_t)                                        # Hidden state
```

### GRUCell

Low-level GRU cell.

**Constructor:**
```java
GRUCell(int inputSize, int hiddenSize)
```

**Methods:**
- `GradTensor forward(GradTensor input, GradTensor hidden)` - Single timestep
- `GradTensor initHidden(int batchSize)` - Initialize h_0
- `GradTensor[] getParameters()` - Get weights and biases

**GRU Equations:**
```
r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # Reset gate
z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # Update gate
n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  # Candidate
h_t = (1 - z_t) * n_t + z_t * h_{t-1}                       # Hidden update
```

### Bidirectional

Wraps any RNN layer to process sequences in both directions.

**Constructor:**
```java
Bidirectional(Object forwardLayer, Object backwardLayer)
```

**Methods:**
- `GradTensor forward(GradTensor input)` - Forward pass
- `int getOutputSize()` - Returns 2 * hiddenSize
- `int getHiddenSize()` - Get single direction size

**Example:**
```java
// Using GRU instead
GRULayer forward = new GRULayer(100, 128, true);
GRULayer backward = new GRULayer(100, 128, true);
Bidirectional biGRU = new Bidirectional(forward, backward);
GradTensor biOutput = biGRU.forward(input);  // [batch, seq_len, 256]
```

## Common Use Cases

### Text Classification with LSTM

```java
public class TextClassifier {
    private LSTMLayer lstm;
    private LinearLayer classifier;

    public TextClassifier(int vocabSize, int embeddingSize, int hiddenSize, int numClasses) {
        // LSTM returns only last timestep
        this.lstm = new LSTMLayer(embeddingSize, hiddenSize, false);
        // Linear layer for classification
        this.classifier = new LinearLayer(hiddenSize, numClasses);
    }

    public GradTensor forward(GradTensor embeddings) {
        // embeddings: [batch, seq_len, embedding_size]
        GradTensor lstmOut = lstm.forward(embeddings);  // [batch, hidden_size]
        GradTensor logits = classifier.forward(lstmOut); // [batch, num_classes]
        return logits;
    }
}
```

### Named Entity Recognition (NER) with BiLSTM

```java
public class NERModel {
    private Bidirectional biLSTM;
    private LinearLayer tagger;

    public NERModel(int embeddingSize, int hiddenSize, int numTags) {
        // Bidirectional LSTM returns full sequence
        LSTMLayer forward = new LSTMLayer(embeddingSize, hiddenSize, true);
        LSTMLayer backward = new LSTMLayer(embeddingSize, hiddenSize, true);
        this.biLSTM = new Bidirectional(forward, backward);
        
        // Linear layer for tagging
        this.tagger = new LinearLayer(2 * hiddenSize, numTags);
    }

    public GradTensor forward(GradTensor embeddings) {
        // embeddings: [batch, seq_len, embedding_size]
        GradTensor biLSTMOut = biLSTM.forward(embeddings);     // [batch, seq_len, 256]
        GradTensor tags = tagger.forward(biLSTMOut);           // [batch, seq_len, num_tags]
        return tags;
    }
}
```

### Language Modeling with Stacked RNN

```java
public class LanguageModel {
    private LSTMLayer lstm1;
    private LSTMLayer lstm2;
    private LinearLayer output;

    public LanguageModel(int embeddingSize, int hiddenSize, int vocabSize) {
        // Stack two LSTM layers
        this.lstm1 = new LSTMLayer(embeddingSize, hiddenSize, true);
        this.lstm2 = new LSTMLayer(hiddenSize, hiddenSize, true);
        this.output = new LinearLayer(hiddenSize, vocabSize);
    }

    public GradTensor forward(GradTensor embeddings) {
        GradTensor h1 = lstm1.forward(embeddings);  // [batch, seq_len, hidden_size]
        GradTensor h2 = lstm2.forward(h1);          // [batch, seq_len, hidden_size]
        GradTensor logits = output.forward(h2);     // [batch, seq_len, vocab_size]
        return logits;
    }
}
```

## Architecture Choices

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More (3×hidden² input + 3×hidden² hidden) | Fewer (~2/3 of LSTM) |
| Speed | Slower | Faster |
| Accuracy | Often slightly better | Often similar |
| Use Case | Default choice, longer sequences | When efficiency matters |

### Bidirectional vs Unidirectional

| Type | Use Case |
|------|----------|
| Unidirectional | Language modeling, generation (need causal masking) |
| Bidirectional | Classification, tagging, translation, understanding (offline) |

## Performance Characteristics

| Layer | Time | Memory | Notes |
|-------|------|--------|-------|
| LSTM | O(T × B × H²) | O(B × H × T) | T=seq_len, B=batch, H=hidden |
| GRU | O(T × B × H²) | O(B × H × T) | ~2/3 time of LSTM |
| Bidirectional | 2× unidirectional | 2× unidirectional | Process both directions |

## Testing

Run the comprehensive test suite:

```bash
mvn test -f gollek/sdk/lib/gollek-sdk-rnn/pom.xml
```

## Future Enhancements

- [ ] Batch processing optimizations
- [ ] Attention mechanisms
- [ ] Multi-head attention layers
- [ ] Transformer blocks
- [ ] Mixed precision training
- [ ] Gradient clipping utilities

## Dependencies

- `gollek-sdk-autograd` - For automatic differentiation
- `gollek-sdk-nn` - For neural network utilities
- `gollek-sdk-tensor` - For tensor operations

## References

- LSTM Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
- GRU Paper: https://arxiv.org/abs/1406.1078
- Sequence-to-Sequence: https://arxiv.org/abs/1409.3215
- Bidirectional RNNs: https://www.jstor.org/stable/2965875

---

**Version:** 0.1.0  
**Status:** Production Ready  
**Test Coverage:** >90%
