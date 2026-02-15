# PyTorch Java FFM Binding for Quarkus

A comprehensive Java binding for PyTorch/LibTorch using JDK 25's Foreign Function & Memory (FFM) API, designed for Quarkus applications.

## Features

- ✅ **Modern FFM API**: Uses JDK 25 Foreign Function & Memory API instead of JNI
- ✅ **Zero-copy interop**: Direct memory access between Java and native code
- ✅ **Complete PyTorch Frontend**: Comprehensive bindings for tensors, neural networks, autograd, and optimizers
- ✅ **Quarkus Integration**: Designed to work seamlessly with Quarkus framework
- ✅ **Type-safe**: Leverages Java's type system for compile-time safety
- ✅ **AutoCloseable**: Proper resource management with try-with-resources
- ✅ **GPU Support**: CUDA operations for GPU acceleration

## Architecture

```
io.github.pytorch
├── ffm/                    # Low-level FFM bindings
│   └── LibTorchFFM.java   # Native function bindings
├── core/                   # Core tensor operations
│   ├── Tensor.java        # High-level tensor API
│   ├── ScalarType.java    # Data types
│   └── Device.java        # Device management
├── nn/                     # Neural network modules
│   ├── Module.java        # Base module class
│   ├── Linear.java        # Linear layer
│   ├── Sequential.java    # Sequential container
│   └── functional/        # Functional API
│       └── Functional.java
└── optim/                  # Optimizers
    ├── Optimizer.java     # Base optimizer
    ├── SGD.java           # SGD optimizer
    └── Adam.java          # Adam optimizer
```

## Requirements

- **JDK 25** or later (for FFM API)
- **LibTorch 2.5.1** or later
- **Quarkus 3.17.3** or later
- **Maven 3.8+**
- Optional: **CUDA** for GPU support

## Installation

### 1. Install LibTorch

Download LibTorch from https://pytorch.org/get-started/locally/

```bash
# For CPU-only (Linux)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip

# For CUDA 12.1 (Linux)
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu121.zip
```

### 2. Set Library Path

```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

### 3. Build the Project

```bash
mvn clean install
```

## Usage

### Basic Tensor Operations

```java
import io.github.pytorch.core.Tensor;

// Create tensors
try (Tensor a = Tensor.randn(new long[]{2, 3});
     Tensor b = Tensor.ones(new long[]{2, 3})) {
    
    // Arithmetic operations
    Tensor c = a.add(b);
    Tensor d = a.mul(2.0);
    
    // Matrix operations
    Tensor result = a.matmul(b.transpose(0, 1));
    
    // Statistical operations
    Tensor mean = a.mean();
    Tensor sum = a.sum();
}
```

### Neural Network Definition

```java
import io.github.pytorch.nn.*;

class MyModel extends Module {
    private final Linear fc1;
    private final Linear fc2;
    
    public MyModel() {
        super();
        this.fc1 = new Linear(784, 128);
        this.fc2 = new Linear(128, 10);
        
        registerModule("fc1", fc1);
        registerModule("fc2", fc2);
    }
    
    @Override
    public Tensor forward(Tensor x) {
        x = fc1.forward(x);
        x = Functional.relu(x);
        x = fc2.forward(x);
        return Functional.softmax(x, -1);
    }
}
```

### Training Loop

```java
import io.github.pytorch.optim.Adam;

// Create model and optimizer
MyModel model = new MyModel();
Adam optimizer = new Adam(model.parameters(), 0.001);

// Training loop
model.train();
for (int epoch = 0; epoch < 10; epoch++) {
    Tensor input = Tensor.randn(new long[]{32, 784});
    Tensor target = Tensor.randint(0, 10, new long[]{32});
    
    // Forward pass
    Tensor output = model.forward(input);
    Tensor loss = Functional.crossEntropy(output, target);
    
    // Backward pass
    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();
    
    System.out.printf("Epoch %d, Loss: %.4f%n", epoch, loss.item());
}
```

### GPU Operations

```java
// Check CUDA availability
boolean cudaAvailable = CUDA.isAvailable();

if (cudaAvailable) {
    // Move tensors to GPU
    Tensor cpuTensor = Tensor.randn(new long[]{1000, 1000});
    Tensor gpuTensor = cpuTensor.cuda();
    
    // Perform GPU computation
    Tensor result = gpuTensor.matmul(gpuTensor);
    
    // Move back to CPU
    Tensor cpuResult = result.cpu();
}
```

### Autograd Example

```java
// Create tensor with gradient tracking
Tensor x = Tensor.ones(new long[]{2, 2}).requiresGrad(true);

// Perform operations
Tensor y = x.add(2);
Tensor z = y.mul(y).mul(3);
Tensor out = z.mean();

// Compute gradients
out.backward();

// Access gradients
Tensor grad = x.grad();
System.out.println("Gradient: " + grad);
```

## Quarkus Integration

### REST Endpoint Example

```java
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;

@Path("/ml")
public class MLResource {
    
    private final MyModel model = new MyModel();
    
    @POST
    @Path("/predict")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public PredictionResponse predict(InputData input) {
        try (Tensor inputTensor = Tensor.fromArray(input.getData(), 
                                                     new long[]{1, 784})) {
            model.eval();
            Tensor output = model.forward(inputTensor);
            
            // Get prediction
            Tensor maxResult = output.max(-1, false);
            int prediction = maxResult.item().intValue();
            
            return new PredictionResponse(prediction);
        }
    }
}
```

## Running the Examples

### Dev Mode

```bash
mvn quarkus:dev --enable-preview --enable-native-access=ALL-UNNAMED
```

### Run End-to-End Example

```bash
mvn compile exec:java \
    -Dexec.mainClass="io.github.pytorch.example.EndToEndExample" \
    -Dexec.classpathScope=compile \
    --enable-preview \
    --enable-native-access=ALL-UNNAMED
```

### Build Native Image

```bash
mvn package -Pnative \
    -Dquarkus.native.additional-build-args=\
    "--enable-preview,--enable-native-access=ALL-UNNAMED"
```

## JVM Arguments

When running applications using this library, you must enable preview features and native access:

```bash
java --enable-preview \
     --enable-native-access=ALL-UNNAMED \
     -Djava.library.path=/path/to/libtorch/lib \
     -jar your-app.jar
```

## Performance Considerations

1. **Memory Management**: Use try-with-resources or explicit close() calls to prevent memory leaks
2. **Batch Operations**: Process data in batches for better GPU utilization
3. **In-place Operations**: Use in-place variants when possible (e.g., `relu(x, true)`)
4. **Device Placement**: Keep tensors on the same device to avoid unnecessary transfers

## API Coverage

### Tensor Operations
- ✅ Creation: empty, zeros, ones, randn, rand, arange, linspace, fromArray
- ✅ Arithmetic: add, sub, mul, div, matmul, mm, bmm
- ✅ Shape: reshape, view, transpose, permute, squeeze, unsqueeze
- ✅ Reduction: sum, mean, max, min
- ✅ Device: to, cuda, cpu, clone, detach

### Neural Network Modules
- ✅ Linear
- ✅ Sequential
- ✅ Module (base class with parameter management)

### Activation Functions
- ✅ ReLU
- ✅ GELU
- ✅ Softmax
- ✅ LogSoftmax

### Loss Functions
- ✅ CrossEntropyLoss
- ✅ MSELoss
- ✅ BCELoss

### Optimizers
- ✅ SGD
- ✅ Adam

### Autograd
- ✅ backward()
- ✅ grad()
- ✅ requiresGrad()
- ✅ zeroGrad()

## Troubleshooting

### Library Not Found

```bash
# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

# Or copy libraries to system path
sudo cp /path/to/libtorch/lib/*.so* /usr/local/lib/
sudo ldconfig
```

### Preview Features Error

Make sure you're using JDK 25 and enabling preview features:

```bash
java --version  # Should show version 25 or later
java --enable-preview ...
```

### Native Access Error

Enable native access for all modules:

```bash
--enable-native-access=ALL-UNNAMED
```

## Limitations

- Native compilation requires additional configuration
- Some advanced PyTorch features may not be exposed
- Performance overhead compared to pure C++ (minimal with FFM)

## Contributing

Contributions are welcome! Areas for improvement:
- Additional neural network layers (Conv2D, LSTM, etc.)
- More optimizers (AdamW, RMSprop, etc.)
- Data loading utilities
- Distributed training support
- TorchScript integration

## License

MIT License - see LICENSE file for details

## References

- [PyTorch C++ Frontend Documentation](https://pytorch.org/cppdocs/frontend.html)
- [JDK FFM API](https://openjdk.org/jeps/454)
- [Quarkus Documentation](https://quarkus.io/)

## Authors

Created as a demonstration of PyTorch/LibTorch integration with Java using modern FFM API.
