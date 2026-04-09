# Jupyter Setup Guide - Gollek SDK

Complete guide for setting up and using Gollek SDK with Jupyter notebooks.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage Examples](#usage-examples)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- Python 3.6+
- Jupyter 4.0+
- Java 11+ (ideally Java 25)
- Maven 3.6+

### Optional
- Conda (recommended for Python environment management)
- JupyterLab (modern Jupyter interface)

## Installation

### Step 1: Install Jupyter

**Using pip:**
```bash
pip install jupyter
```

**Using conda:**
```bash
conda install jupyter
```

### Step 2: Install IJava Kernel (Jupyter Java Support)

```bash
# Install IJava kernel
git clone https://github.com/SpencerPark/IJava.git
cd IJava/
mvn clean install

# Install kernel
python install.py --sys-prefix
```

Alternatively, use the pre-built kernel:
```bash
pip install ijava
# or
conda install ijava
```

### Step 3: Setup Gollek SDK Kernel

```bash
# From gollek/sdk directory
cd gollek/sdk

# Build Gollek SDK
mvn clean install -DskipTests

# Install Gollek kernel
bash jupyter-kernel/install.sh
```

### Step 4: Verify Installation

```bash
# List available kernels
jupyter kernelspec list

# You should see "gollek-sdk" in the list
```

## Quick Start

### 1. Start Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### 2. Create New Notebook

- Click "New" button
- Select "Gollek SDK (Java)" from kernel list
- A new notebook opens with Java kernel

### 3. First Code Cell

```java
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.*;

// Create a simple neural network layer
Linear layer = new Linear(10, 5);
System.out.println("Layer created: " + layer);
```

Press Shift+Enter to execute.

## Usage Examples

### Example 1: Building a Neural Network

```java
// Create model
Module model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Dropout(0.2f),
    new Linear(256, 10)
);

System.out.println("Model created successfully!");
```

### Example 2: Training Loop

```java
// Setup
var optimizer = new tech.kayys.gollek.ml.nn.optim.Adam(model.parameters(), 0.001f);
var loss = new tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss();

// Create dummy data
float[] input = new float[784];
for (int i = 0; i < 784; i++) {
    input[i] = (float) Math.random();
}

float[] target = new float[1];
target[0] = 3; // Target class

// Forward pass
var x = GradTensor.of(input, new long[]{1, 784});
var y = model.forward(x);
var y_target = GradTensor.of(target, new long[]{1});

// Loss
var lossVal = loss.compute(y, y_target);
System.out.println("Loss: " + lossVal.item());

// Backward
lossVal.backward();

// Optimize
optimizer.step();
optimizer.zeroGrad();

System.out.println("Training step completed!");
```

### Example 3: Activation Functions

```java
// Create different activations
ReLU relu = new ReLU();
Sigmoid sigmoid = new Sigmoid();
var leakyReLU = new tech.kayys.gollek.ml.nn.LeakyReLU(0.01f);
var elu = new tech.kayys.gollek.ml.nn.ELU(1.0f);
var mish = new tech.kayys.gollek.ml.nn.Mish();

System.out.println("Available activations:");
System.out.println("✓ ReLU");
System.out.println("✓ Sigmoid");
System.out.println("✓ LeakyReLU");
System.out.println("✓ ELU");
System.out.println("✓ Mish");
```

### Example 4: Loss Functions

```java
// Classification
var ce = new tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss();
var bce = new tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss();

// Regression
var mse = new tech.kayys.gollek.ml.nn.loss.MSELoss();
var l1 = new tech.kayys.gollek.ml.nn.loss.L1Loss();
var smoothL1 = new tech.kayys.gollek.ml.nn.loss.SmoothL1Loss();

System.out.println("Available loss functions:");
System.out.println("✓ CrossEntropyLoss (classification)");
System.out.println("✓ BCEWithLogitsLoss (binary)");
System.out.println("✓ MSELoss (regression)");
System.out.println("✓ L1Loss (robust regression)");
System.out.println("✓ SmoothL1Loss (object detection)");
```

### Example 5: Learning Rate Scheduling

```java
var optimizer = new tech.kayys.gollek.ml.nn.optim.Adam(params, 0.001f);

// Step decay schedule
var scheduler = new tech.kayys.gollek.ml.nn.optim.StepLR(optimizer, 10, 0.1f);

// Or cosine annealing
var cosineScheduler = new tech.kayys.gollek.ml.nn.optim.CosineAnnealingLR(
    optimizer, 100, 1e-6f
);

for (int epoch = 0; epoch < 100; epoch++) {
    // Training...
    scheduler.step();  // Update LR
}

System.out.println("Learning rate scheduling: " + optimizer.learningRate());
```

### Example 6: Metrics and Early Stopping

```java
var metric = new tech.kayys.gollek.ml.nn.metrics.Accuracy();
var earlyStopping = new tech.kayys.gollek.ml.nn.EarlyStopping(10, true, 0, "min");

for (int epoch = 0; epoch < 100; epoch++) {
    // Validation
    float valLoss = 0.5f;  // Dummy value
    
    // Update metrics
    if (earlyStopping.check(valLoss)) {
        System.out.println("Early stopping at epoch: " + epoch);
        break;
    }
}
```

## Tips & Tricks

### 1. Import All Components at Once

```java
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.optim.*;
import tech.kayys.gollek.ml.nn.loss.*;
import tech.kayys.gollek.ml.nn.metrics.*;
import tech.kayys.gollek.ml.autograd.*;
```

### 2. Print Model Architecture

```java
Module model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
);

System.out.println(model);
```

### 3. Count Parameters

```java
long paramCount = 0;
for (var param : model.parameters()) {
    paramCount += param.data().numel();
}
System.out.println("Total parameters: " + paramCount);
```

### 4. Use Markdown Cells for Documentation

- Click "+ Markdown" to add text cells
- Document your experiments
- Include math formulas with LaTeX: `$equation$`

### 5. Visualize with Print Statements

```java
System.out.println("Epoch: " + epoch);
System.out.println("Loss: " + lossVal);
System.out.println("Accuracy: " + accuracy.compute());
```

## Troubleshooting

### Issue: Kernel not found

**Solution:**
```bash
# Reinstall kernel
jupyter kernelspec remove gollek-sdk
bash jupyter-kernel/install.sh
```

### Issue: Import errors (class not found)

**Solution:**
```bash
# Rebuild Gollek SDK
cd gollek/sdk
mvn clean install -DskipTests

# Reinstall kernel
bash jupyter-kernel/install.sh
```

### Issue: Out of memory (OOM)

**Solution:**
```bash
# Increase Java heap size in kernel.json
# Modify jupyter-kernel/kernel.json:
"argv": [
    "java",
    "-Xmx2G",  # Increase heap to 2GB
    "-cp",
    ...
]
```

### Issue: Very slow first execution

**Solution:**
- First execution compiles bytecode (normal)
- Subsequent cells run faster
- Be patient with first few imports

## Performance Tips

1. **Batch Operations**: Use batch processing instead of single samples
2. **Avoid Re-imports**: Import once at the start
3. **Use Variables**: Store model references to avoid recreating
4. **Chunk Processing**: For large datasets, process in chunks

## Next Steps

1. Review [example notebooks](../examples/notebooks/)
2. Check [API reference](../API_REFERENCE.md)
3. Try your own experiments!

## Additional Resources

- [IJava GitHub](https://github.com/SpencerPark/IJava)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Gollek SDK Documentation](../README.md)

## Support

For issues or questions:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review example notebooks
3. Check Gollek SDK documentation

---

**Happy Data Science with Gollek SDK! 🚀**
