# Golek Adapters

This folder contains runtime adapters for local and specialized inference backends.

## Available Adapters

* `golek-adapter-gguf` — llama.cpp bindings for GGUF models
* `golek-adapter-onnx` — ONNX runtime adapter
* `golek-adapter-pytorch` — PyTorch adapter
* `golek-adapter-tensorrt` — TensorRT adapter
* `golek-adapter-tpu` — TPU adapter
* `golek-adapter-triton` — Triton inference server adapter

## Notes

Adapters are typically used by providers and the runtime engine to execute local models.
