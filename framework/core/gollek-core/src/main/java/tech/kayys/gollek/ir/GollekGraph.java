package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

import java.util.List;

/**
 * [ Model Formats ]
 * (safetensors / gguf / onnx / tflite / torch / etc.)
 * ↓
 * [ Loaders ]
 * (format-specific parsing)
 * ↓
 * [ Adapters ]
 * (normalize → unified structure)
 * ↓
 * 🔥 [ GOLLEK IR ] 🔥 ← THIS IS THE CORE YOU ARE MISSING
 * ↓
 * [ Optimizer / Compiler ]
 * (fusion, quantization, layout)
 * ↓
 * [ Runtime ]
 * (your current work: tensor + kernels + kv + adaptive)
 * ↓
 * [ Backend ]
 * (CPU / GPU / Metal / etc.)
 * 
 * class MatMulNode implements GollekNode {}
 * class AttentionNode implements GollekNode {}
 * class Conv2DNode implements GollekNode {}
 * class RMSNormNode implements GollekNode {}
 */
public final class GollekGraph {

    private final List<GollekNode> nodes;

    public GollekGraph(List<GollekNode> nodes) {
        this.nodes = nodes;
    }

    public List<GollekNode> nodes() {
        return nodes;
    }
}