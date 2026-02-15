# PyTorch Java FFM API Documentation

## Table of Contents
1. [Core API](#core-api)
2. [Neural Network Modules](#neural-network-modules)
3. [Optimizers](#optimizers)
4. [Loss Functions](#loss-functions)
5. [Examples](#examples)

---

## Core API

### Tensor

The `Tensor` class is the fundamental building block for all operations.

#### Creation Methods

```java
// Create tensors filled with specific values
Tensor zeros = Tensor.zeros(new long[]{3, 4}, ScalarType.FLOAT);
Tensor ones = Tensor.ones(new long[]{2, 3}, ScalarType.FLOAT);

// Random tensors
Tensor uniform = Tensor.rand(new long[]{5, 5}, ScalarType.FLOAT);  // Uniform [0, 1)
Tensor normal = Tensor.randn(new long[]{10, 20}, ScalarType.FLOAT); // Normal N(0, 1)
```

#### Arithmetic Operations

```java
Tensor a = Tensor.rand(new long[]{3, 3}, ScalarType.FLOAT);
Tensor b = Tensor.ones(new long[]{3, 3}, ScalarType.FLOAT);

// Element-wise operations
Tensor sum = a.add(b);           // a + b
Tensor product = a.mul(b);       // a * b (element-wise)
Tensor scaled = a.add(2.0);      // a + 2

// Matrix operations
Tensor matmul = a.matmul(b);     // Matrix multiplication
```

#### Activation Functions

```java
Tensor x = Tensor.randn(new long[]{10, 20}, ScalarType.FLOAT);

Tensor relu = x.relu();          // ReLU(x) = max(0, x)
Tensor sigmoid = x.sigmoid();    // σ(x) = 1/(1 + e^(-x))
Tensor tanh = x.tanh();          // tanh(x)
Tensor softmax = x.softmax(1);   // Softmax along dimension 1
```

#### Autograd Operations

```java
// Enable gradient tracking
Tensor x = Tensor.randn(new long[]{5, 3}, ScalarType.FLOAT);
x.requiresGrad(true);

// Forward computation
Tensor y = x.mul(x).add(x);

// Backward pass
y.backward();

// Access gradients
Tensor grad = x.grad();
System.out.println("Gradient: " + grad);
```

#### Shape Manipulation

```java
Tensor x = Tensor.rand(new long[]{2, 3, 4}, ScalarType.FLOAT);

// Get shape information
long[] shape = x.shape();        // [2, 3, 4]
long dims = x.dim();             // 3
long numel = x.numel();          // 24
long size = x.size(0);           // 2

// Reshape operations
Tensor reshaped = x.reshape(new long[]{2, 12});
Tensor transposed = x.transpose(0, 1);
```

#### Device Management

```java
// Check CUDA availability
boolean hasCuda = Tensor.cudaIsAvailable();

// Create tensor on GPU
Tensor gpu = Tensor.rand(new long[]{1000, 1000}, ScalarType.FLOAT)
                   .to(Tensor.Device.CUDA);

// Move back to CPU
Tensor cpu = gpu.to(Tensor.Device.CPU);
```

#### Scalar Types

```java
enum ScalarType {
    FLOAT,      // 32-bit floating point
    DOUBLE,     // 64-bit floating point
    INT32,      // 32-bit integer
    INT64,      // 64-bit integer
    UINT8,      // 8-bit unsigned integer
    INT8,       // 8-bit signed integer
    BOOL        // Boolean
}
```

---

## Neural Network Modules

### Module (Base Class)

All neural network layers inherit from `Module`.

```java
public class MyModule extends Module {
    @Override
    public Tensor forward(Tensor input) {
        // Define forward pass
        return input;
    }
}

// Usage
MyModule module = new MyModule();
module.train();                   // Training mode
module.eval();                    // Evaluation mode

List<Tensor> params = module.parameters();
Map<String, Tensor> named = module.namedParameters();

module.zeroGrad();                // Zero all gradients
module.to(Tensor.Device.CUDA);    // Move to GPU
```

### Linear Layer

Fully connected (dense) layer: `y = xW^T + b`

```java
// Create layer: 784 inputs -> 128 outputs
Linear fc = new Linear(784, 128);

// Forward pass
Tensor input = Tensor.randn(new long[]{32, 784}, ScalarType.FLOAT);
Tensor output = fc.forward(input);  // Shape: [32, 128]

// Without bias
Linear fcNoBias = new Linear(784, 128, false);
```

### Conv2d Layer

2D Convolutional layer for image processing.

```java
// Conv2d(in_channels, out_channels, kernel_size)
Conv2d conv = new Conv2d(3, 64, 3);

// With stride and padding
Conv2d conv2 = new Conv2d(64, 128, 3, 2, 1);  // stride=2, padding=1

// Forward pass
Tensor input = Tensor.randn(new long[]{32, 3, 64, 64}, ScalarType.FLOAT);
Tensor output = conv.forward(input);  // Shape: [32, 64, 64, 64]
```

### BatchNorm2d Layer

Batch normalization for 2D inputs.

```java
// BatchNorm2d(num_features)
BatchNorm2d bn = new BatchNorm2d(64);

// Custom parameters
BatchNorm2d bn2 = new BatchNorm2d(
    64,          // num_features
    1e-5,        // eps
    0.1,         // momentum
    true,        // affine
    true         // track_running_stats
);
```

### Dropout Layer

Regularization via random dropout.

```java
// Dropout with p=0.5
Dropout dropout = new Dropout(0.5);

// During training (randomly zeros elements)
Tensor x = Tensor.randn(new long[]{32, 128}, ScalarType.FLOAT);
Tensor out = dropout.forward(x);

// During evaluation (no dropout applied)
dropout.eval();
Tensor outEval = dropout.forward(x);
```

### Sequential Container

Chain multiple modules together.

```java
Sequential model = new Sequential()
    .add("fc1", new Linear(784, 256))
    .add("relu1", new ReLUModule())
    .add("dropout", new Dropout(0.5))
    .add("fc2", new Linear(256, 10));

// Forward pass through entire sequence
Tensor input = Tensor.randn(new long[]{32, 784}, ScalarType.FLOAT);
Tensor output = model.forward(input);

// Access individual layers
Module fc1 = model.get(0);
```

---

## Optimizers

### SGD (Stochastic Gradient Descent)

```java
// Basic SGD
SGD optimizer = new SGD(model.parameters(), 0.01);

// With momentum
SGD sgdMomentum = new SGD(
    model.parameters(),
    0.01,        // learning rate
    0.9          // momentum
);

// Full configuration
SGD sgdFull = new SGD(
    model.parameters(),
    0.01,        // learning rate
    0.9,         // momentum
    0.0,         // dampening
    0.0001,      // weight decay
    false        // nesterov
);

// Training step
optimizer.zeroGrad();
loss.backward();
optimizer.step();
```

### Adam

Adaptive moment estimation optimizer.

```java
// Basic Adam
Adam optimizer = new Adam(model.parameters(), 0.001);

// Full configuration
Adam adamFull = new Adam(
    model.parameters(),
    0.001,       // learning rate
    0.9,         // beta1
    0.999,       // beta2
    1e-8,        // eps
    0.0,         // weight decay
    false        // amsgrad
);

// Training step
optimizer.zeroGrad();
loss.backward();
optimizer.step();
```

---

## Loss Functions

### Mean Squared Error (MSE)

```java
Tensor predictions = model.forward(input);
Tensor targets = Tensor.randn(new long[]{32, 10}, ScalarType.FLOAT);

Tensor loss = Loss.mse(predictions, targets);
```

### Cross Entropy

```java
// For classification
Tensor logits = model.forward(input);  // Shape: [batch, num_classes]
Tensor targets = Tensor.rand(new long[]{32}, ScalarType.INT64);

Tensor loss = Loss.crossEntropy(logits, targets);
```

### Binary Cross Entropy

```java
// For binary classification
Tensor predictions = model.forward(input).sigmoid();
Tensor targets = Tensor.rand(new long[]{32, 1}, ScalarType.FLOAT);

Tensor loss = Loss.binaryCrossEntropy(predictions, targets);
```

---

## Examples

### Complete Training Loop

```java
// Create model
public class Net extends Module {
    private Linear fc1, fc2, fc3;
    
    public Net() {
        fc1 = registerModule("fc1", new Linear(784, 128));
        fc2 = registerModule("fc2", new Linear(128, 64));
        fc3 = registerModule("fc3", new Linear(64, 10));
    }
    
    @Override
    public Tensor forward(Tensor input) {
        return fc3.forward(fc2.forward(fc1.forward(input).relu()).relu());
    }
}

// Training
Net model = new Net();
model.train();

Adam optimizer = new Adam(model.parameters(), 0.001);

for (int epoch = 0; epoch < 10; epoch++) {
    double totalLoss = 0.0;
    
    for (Batch batch : dataLoader) {
        try (Tensor input = batch.getInput();
             Tensor target = batch.getTarget()) {
            
            // Forward
            Tensor output = model.forward(input);
            Tensor loss = Loss.crossEntropy(output, target);
            
            // Backward
            optimizer.zeroGrad();
            loss.backward();
            optimizer.step();
            
            totalLoss += extractScalar(loss);
            
            output.close();
            loss.close();
        }
    }
    
    System.out.printf("Epoch %d, Loss: %.4f%n", epoch, totalLoss);
}
```

### Transfer Learning

```java
// Load pretrained model
PretrainedModel pretrained = PretrainedModel.load("resnet18.pt");

// Freeze early layers
for (int i = 0; i < 7; i++) {
    Module layer = pretrained.getLayer(i);
    for (Tensor param : layer.parameters()) {
        param.requiresGrad(false);
    }
}

// Replace final layer
Linear newClassifier = new Linear(512, numClasses);
pretrained.setClassifier(newClassifier);

// Train only new layer
List<Tensor> trainableParams = newClassifier.parameters();
Adam optimizer = new Adam(trainableParams, 0.001);
```

### Model Evaluation

```java
model.eval();  // Set to evaluation mode

int correct = 0;
int total = 0;

try (Arena arena = Arena.ofConfined()) {
    for (Batch batch : testLoader) {
        Tensor input = batch.getInput();
        Tensor target = batch.getTarget();
        
        // Forward pass (no gradients needed)
        Tensor output = model.forward(input);
        Tensor predictions = output.argmax(1);
        
        // Count correct predictions
        correct += countCorrect(predictions, target);
        total += target.size(0);
        
        input.close();
        target.close();
        output.close();
        predictions.close();
    }
}

double accuracy = 100.0 * correct / total;
System.out.printf("Test Accuracy: %.2f%%\n", accuracy);
```

### Custom Layer

```java
public class CustomLayer extends Module {
    private Linear linear;
    private Tensor alpha;
    
    public CustomLayer(int inFeatures, int outFeatures) {
        linear = registerModule("linear", new Linear(inFeatures, outFeatures));
        alpha = Tensor.ones(new long[]{1}, ScalarType.FLOAT);
        alpha.requiresGrad(true);
        registerParameter("alpha", alpha);
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Tensor x = linear.forward(input);
        return x.mul(alpha).relu();
    }
}
```

### Memory Management Best Practices

```java
// Always use try-with-resources
try (Tensor a = Tensor.rand(new long[]{100, 100}, ScalarType.FLOAT);
     Tensor b = Tensor.rand(new long[]{100, 100}, ScalarType.FLOAT)) {
    
    Tensor c = a.matmul(b);
    
    // Process c...
    
    c.close();  // Explicit close
}  // a and b automatically closed

// Or use Arena for batch management
try (Arena arena = Arena.ofConfined()) {
    List<Tensor> tensors = new ArrayList<>();
    
    for (int i = 0; i < 100; i++) {
        tensors.add(Tensor.rand(new long[]{10, 10}, ScalarType.FLOAT));
    }
    
    // Process tensors...
    
}  // All tensors cleaned up together
```

---

## Performance Tips

1. **Batch Operations**: Process multiple samples together for better GPU utilization
   ```java
   // Good: batch of 32
   Tensor batch = Tensor.randn(new long[]{32, 784}, ScalarType.FLOAT);
   
   // Avoid: processing one at a time
   for (int i = 0; i < 32; i++) {
       Tensor single = Tensor.randn(new long[]{1, 784}, ScalarType.FLOAT);
       // ...
   }
   ```

2. **Keep Data on GPU**: Minimize CPU-GPU transfers
   ```java
   // Move model to GPU once
   model.to(Tensor.Device.CUDA);
   
   // Process batches on GPU
   for (Batch batch : dataLoader) {
       Tensor gpuInput = batch.getInput().to(Tensor.Device.CUDA);
       // ... process on GPU ...
   }
   ```

3. **Use Appropriate Data Types**: `float32` is usually sufficient
   ```java
   // Use FLOAT for most ML tasks
   Tensor x = Tensor.rand(new long[]{100, 100}, ScalarType.FLOAT);
   
   // Use DOUBLE only when needed for numerical precision
   Tensor precise = Tensor.rand(new long[]{10, 10}, ScalarType.DOUBLE);
   ```

4. **Disable Gradients for Inference**:
   ```java
   model.eval();  // Disables dropout, batchnorm updates
   
   // No gradient computation
   Tensor output = model.forward(input);
   ```

---

## API Status

✅ **Fully Implemented**
- Tensor creation (zeros, ones, rand, randn)
- Basic arithmetic operations
- Activation functions (relu, sigmoid, tanh, softmax)
- Autograd (backward, grad, requiresGrad)
- Shape operations (shape, dim, numel, size)
- Device management (CPU/CUDA)
- Module system (parameters, train/eval, device movement)
- Linear layer
- Optimizers (SGD, Adam base structure)

⚠️ **Partially Implemented** (requires additional FFM bindings)
- Conv2d (structure complete, operation pending)
- BatchNorm2d (structure complete, operation pending)
- Dropout (structure complete, operation pending)
- Loss functions (structure complete, operations pending)
- Optimizer step functions (structure complete, in-place ops pending)

🚧 **To Be Implemented**
- Data loading utilities
- Model serialization (save/load)
- More layers (RNN, LSTM, Attention)
- Additional optimizers (RMSprop, AdaGrad)
- Learning rate schedulers
- Gradient clipping
- Mixed precision training

---

For complete examples, see the `/examples` directory in the source code.
