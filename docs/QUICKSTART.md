# Gollek Quickstart

Get started with Gollek ML framework in 5 minutes.

---

## 1. Install

### Option A: JBang (No build tools)

```bash
curl -Ls https://sh.jbang.dev | bash -s -
```

### Option B: Maven

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-ml-api</artifactId>
    <version>0.3.0-SNAPSHOT</version>
</dependency>
```

### Option C: Clone & Build

```bash
git clone https://github.com/bhangun/gollek.git
cd gollek
mvn clean install -DskipTests
```

---

## 2. Verify Installation

```bash
# Via JBang
jbang examples/jbang/common/hello_gollek.java

# Via Maven
cd framework/lib/gollek-ml-examples
mvn exec:java -Dexec.mainClass=tech.kayys.gollek.train.examples.SimpleFFNExample
```

Expected output:
```
✓ Gollek SDK installed successfully
✓ Tensor operations working
```

---

## 3. Create Your First Model

### Using JBang

Create `my_model.java`:

```java
///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.gollek:gollek-ml-ml:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.tensor.Tensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.ReLU;
import tech.kayys.gollek.ml.nn.Sequential;

public class my_model {
    public static void main(String[] args) {
        // Build a simple neural network
        Sequential model = Sequential.builder()
            .add(new Linear(784, 256))
            .add(new ReLU())
            .add(new Linear(256, 10))
            .build();

        // Create random input (simulating a 28x28 image)
        Tensor input = Gollek.randn(1, 784);

        // Forward pass
        Tensor output = model.forward(input);

        System.out.println("Model: 784 → 256 → 10");
        System.out.println("Input shape: [1, 784]");
        System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
    }
}
```

Run:
```bash
jbang my_model.java
```

---

## 4. Run Inference

```java
import tech.kayys.gollek.ml.runner.*;

ModelRunner runner = ModelRunner.builder()
    .modelPath("model.onnx")
    .device(RunnerDevice.CUDA)
    .build();

InferenceResult result = runner.infer(
    InferenceInput.fromFloats(inputData, 1, 3, 224, 224)
);

System.out.println("Output: " + result.primaryOutput().asFloats());
System.out.println("Latency: " + result.latency());

runner.close();
```

---

## 5. Explore Examples

```bash
# Browse all examples
ls examples/jbang/

# Run comprehensive demo
jbang examples/jbang/sdk/unified_framework_demo.java --demo all

# Open Jupyter notebooks
cd examples/jupyter/
jupyter notebook
```

---

## Next Steps

- **[Examples Setup](examples/docs/SETUP.md) - Detailed example setup guide
- **[SDK Installation](docs/setup/sdk-installation.md) - Full installation guide
- **[Framework Guide](website/gollek-ai.github.io/docs/framework/) - ML framework docs
- **[Inference Guide](website/gollek-ai.github.io/docs/framework/inference.md) - Model inference docs
