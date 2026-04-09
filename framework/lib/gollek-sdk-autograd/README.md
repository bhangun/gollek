# Gollek SDK :: Autograd Engine

The `gollek-sdk-autograd` module is the core automatic differentiation (AD) engine for the Gollek ML framework. It provides reverse-mode AD allowing you to define complex computational graphs dynamically and compute gradients seamlessly.

## Key Features

- **Tape-based Autograd**: Real-time evaluation with dynamic topological sorting for backward pass sequences.
- **GradTensor**: The primary tensor abstraction enabling differentiable operations (`add`, `matmul`, `relu`, `sigmoid` etc.).
- **Hardware-Accelerated Dispatch**: Pluggable backend structure using the `gollek-spi-tensor` Service Provider Interface to delegate intensive operations to GPUs (Metal, CUDA) or fall back to native CPU arrays.

## Example

```java
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Functions;

// Create random weights requiring gradients
GradTensor w = GradTensor.rand(new long[]{64, 128}).requiresGrad(true);
GradTensor x = GradTensor.rand(new long[]{128, 64});

// Forward pass
GradTensor out = Functions.Matmul.apply(w, x);

// Compute scalar loss and backpropagate
GradTensor loss = Functions.Sum.apply(out);
loss.backward();

// Accessible gradients
GradTensor wGrad = w.grad(); 
w.zeroGrad();
```
