# Golek ONNX Runtime Adapter

The Golek ONNX Runtime Adapter provides ONNX model execution capabilities for the Golek inference platform.

## Overview

This adapter enables the execution of ONNX (Open Neural Network Exchange) models using Microsoft's ONNX Runtime. It supports multiple execution providers including CPU, CUDA, TensorRT, OpenVINO, and DirectML for optimal performance across different hardware configurations.

## Features

- **Multi-Platform Support**: Runs on Windows, Linux, and macOS
- **Hardware Acceleration**: Supports CPU, GPU (CUDA, TensorRT), and specialized accelerators
- **Execution Provider Selection**: Automatic or manual selection of optimal execution provider
- **Resource Management**: Efficient memory and thread management
- **Model Format Support**: Native ONNX model format support
- **Flexible Configuration**: Configurable threading, optimization, and execution parameters

## Architecture

```
┌─────────────────────────────────────┐
│   OnnxRuntimeRunner                 │
│                                    │
│   • Implements ModelRunner         │
│   • Manages ONNX Runtime session   │
│   • Handles input/output conversion│
│   • Provides health metrics        │
└─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│   ExecutionProviderSelector         │
│                                    │
│   • Detects available providers    │
│   • Selects optimal provider       │
│   • Validates provider availability│
└─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│   ONNX Runtime (Microsoft)          │
│                                    │
│   • Cross-platform inference engine│
│   • Optimized for multiple hardware│
│   • Extensive operator support     │
└─────────────────────────────────────┘
```

## Execution Providers

The adapter supports multiple execution providers:

- **CPUExecutionProvider**: Default CPU-based execution
- **CUDAExecutionProvider**: NVIDIA GPU acceleration
- **TensorrtExecutionProvider**: NVIDIA TensorRT optimization
- **OpenVINOExecutionProvider**: Intel OpenVINO toolkit
- **DirectMLExecutionProvider**: Windows DirectML acceleration

## Configuration

The adapter can be configured using the following properties:

```properties
# Enable the ONNX adapter
inference.adapter.onnx.enabled=true

# Preferred execution provider (auto-detected if not specified)
inference.adapter.onnx.execution-provider=CUDAExecutionProvider

# Thread configuration
inference.adapter.onnx.inter-op-threads=1
inference.adapter.onnx.intra-op-threads=4

# Additional optimization settings can be passed via model configuration
```

## Input/Output Conversion

The adapter handles automatic conversion between Golek's inference request/response format and ONNX tensor format:

- **Input Types Supported**: 
  - `long[]` → ONNX Int64 tensor
  - `float[]` → ONNX Float tensor
  - `int[]` → ONNX Int32 tensor
  - `double[]` → ONNX Double tensor
  - `List<Number>` → Converted to appropriate tensor type

- **Output Types Supported**: ONNX tensor values converted to corresponding Java types

## Performance Optimization

The adapter includes several optimization features:

- **Memory Pattern Optimization**: Reduces memory allocation overhead
- **Thread Pool Management**: Configurable inter-op and intra-op thread counts
- **Execution Mode Selection**: Sequential or parallel execution modes
- **Model Optimization Levels**: Different optimization strategies based on use case

## Health Monitoring

The adapter provides health status and resource metrics:

- **Health Status**: Reports initialization and runtime status
- **Resource Metrics**: Memory usage estimates
- **Runtime Metadata**: Version information and capabilities

## Integration

The adapter integrates with the Golek provider SPI and follows these conventions:

- **Provider Type**: ModelRunner implementation
- **Model Format**: ONNX (.onnx files)
- **Configuration**: MicroProfile Config compliant
- **Dependency Injection**: CDI compatible

## Usage

The adapter works automatically when registered in the Golek provider system. Models in ONNX format will be handled by this adapter when available.

For custom configuration, implement custom model manifests that specify ONNX format support and pass provider-specific configuration options.

## Requirements

- Java 21+
- ONNX Runtime native libraries (automatically included via Maven)
- Compatible hardware for specific execution providers (GPU for CUDA/TensorRT, etc.)