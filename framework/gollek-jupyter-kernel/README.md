# gollek-kernel

Standalone Jupyter kernel for the Gollek AI platform.

Extends [JJava](https://github.com/dflib/jjava)'s `BaseKernel` (Jupyter wire protocol over ZMQ/JeroMQ) and bootstraps the Gollek inference engine **in-process** via CDI SE (Weld) — no separate server process required.

## Architecture

```
Jupyter ←──ZMQ──→ KernelLauncher
                       │
                  GollekKernel (extends BaseKernel)
                  ├── CodeEvaluator (JShell)
                  ├── CompletionProvider (JShell sourceCodeAnalysis)
                  ├── TensorDisplay (HTML heatmap rendering)
                  └── GollekKernelContext ──→ GollekSdk (CDI SE / Weld)
```

## Build & Install

```bash
# 1. Build fat jar
cd gollek/gollek-kernel
mvn clean package

# 2. Install kernel
python3 install.py --user

# 3. Verify
jupyter kernelspec list
```

## Usage

Open any notebook and select **Gollek (Java 25 + AI/ML)** as the kernel.

A `GollekSdk sdk` variable is pre-bound in every cell:

```java
// List models
sdk.listModels().forEach(m -> System.out.println(m.id()));

// Generate text
var resp = sdk.createCompletion(
    InferenceRequest.builder()
        .model("qwen2.5-0.5b-instruct-q4_0")
        .prompt("Hello!")
        .build()
);
System.out.println(resp.getText());
```

## JVM options

Edit `kernel.json` (in the installed kernelspec directory) to tune memory:

```json
"-Xmx8g",
"-XX:MaxDirectMemorySize=4g"
```

## License

Apache 2.0
