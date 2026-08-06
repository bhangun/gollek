## ONNX Runtime Runner

Local ONNX Runtime runner with optional accelerator Execution Providers (EPs).

### CoreML (Apple Silicon)

When running on Apple Silicon and the CoreML EP is available, the runner can
select `coreml` or `auto` and will attach the CoreML execution provider.

```properties
# Explicit CoreML
onnx.runner.execution_provider=coreml

# Auto (tries CoreML → CUDA → ROCm → CPU)
onnx.runner.execution_provider=auto
```

Notes:
- CoreML EP availability depends on your ONNX Runtime build.
- The runner detects the CoreML append symbol at runtime and falls back to CPU if missing.

### Install ONNX Runtime (prebuilt)

Gollek ships a helper Makefile that downloads a matching ONNX Runtime release
and installs the shared library into `~/.gollek/libs/`.

```bash
make -C inference-gollek/extension/runner/onnx/gollek-runner-onnx/src/main/cpp/onnxruntime install
```

You can override the version or package:

```bash
ORT_VERSION=1.19.2 make -C inference-gollek/extension/runner/onnx/gollek-runner-onnx/src/main/cpp/onnxruntime install
ORT_PACKAGE=onnxruntime-osx-arm64-1.19.2.tgz make -C inference-gollek/extension/runner/onnx/gollek-runner-onnx/src/main/cpp/onnxruntime install
```


