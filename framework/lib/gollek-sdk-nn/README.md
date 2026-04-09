# Gollek SDK :: Neural Network Modules

The `gollek-sdk-nn` component offers an object-oriented Module API (akin to PyTorch's `nn.Module`) built on top of the underlying Autograd tape.

## Modules provided:
- `Linear`: Computes affine transformations $y = xW^T + b$ using gradients.
- `Module`: The core abstract class that organizes `Parameter` containers sequentially. 

## Utilities
- Provides state dictionary parsing for module parameter freezing and gradient initialization.

```java
import tech.kayys.gollek.ml.nn.*;

class MLP extends Module {
    Linear fc1 = new Linear(128, 64);
    Linear fc2 = new Linear(64, 10);
}
```
