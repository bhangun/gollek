package tech.kayys.gollek.tensorflow.loader;

import io.quarkus.runtime.annotations.RegisterForReflection;
import java.util.List;
import java.util.Map;
import java.lang.foreign.MemorySegment;

/** Minimal TFModel record used by the TF module skeleton. */
@RegisterForReflection
public record TFModel(
    String name,
    Map<String, TFNodeInfo> nodes,
    List<TFTensorInfo> inputs,
    List<TFTensorInfo> outputs,
    Map<String,Object> metadata,
    MemorySegment segment
) {}

@RegisterForReflection
record TFNodeInfo(String name, String op, java.util.List<String> inputs, java.util.Map<String, TFAttrValue> attrs, String device) {}
@RegisterForReflection
record TFAttrValue(String name, Object value) {}
@RegisterForReflection
record TFTensorInfo(String name, int dtype, long[] shape, long byteSize) {}
