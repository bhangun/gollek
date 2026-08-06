package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

record OnnxTensorDataView(MemorySegment data, long byteLength) {

    OnnxTensorDataView {
        Objects.requireNonNull(data, "data");
        if (byteLength < 0L || byteLength > data.byteSize()) {
            throw new IllegalArgumentException("Tensor data byte length " + byteLength
                    + " is outside segment byte size " + data.byteSize());
        }
    }

    static OnnxTensorDataView int64Elements(MemorySegment data, long elementCount) {
        if (elementCount < 0L) {
            throw new IllegalArgumentException("Tensor int64 element count must be non-negative: " + elementCount);
        }
        return new OnnxTensorDataView(data, Math.multiplyExact(elementCount, Long.BYTES));
    }

    MemorySegment asSizedSegment() {
        return data.asSlice(0L, byteLength);
    }
}
