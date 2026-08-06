package tech.kayys.gollek.runtime.weight;


import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import tech.kayys.gollek.core.weight.WeightStore;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.DType;

/**
 * A weight store that handles quantized weights.
 * It can dequantize on-the-fly or return quantized tensors directly.
 */
public class QuantizedWeightStore implements WeightStore {
    private final WeightStore delegate;
    private final boolean autoDequantize;

    public QuantizedWeightStore(WeightStore delegate) {
        this(delegate, false);
    }

    public QuantizedWeightStore(WeightStore delegate, boolean autoDequantize) {
        this.delegate = delegate;
        this.autoDequantize = autoDequantize;
    }

    @Override
    public Tensor get(String key) {
        Tensor tensor = delegate.get(key);
        if (autoDequantize && isQuantized(tensor.dtype())) {
            return dequantize(tensor);
        }
        return tensor;
    }

    @Override
    public boolean contains(String key) {
        return delegate.contains(key);
    }

    private boolean isQuantized(DType dtype) {
        return dtype == DType.Q4_K || dtype == DType.Q8_0 || dtype == DType.I8;
    }

    private Tensor dequantize(Tensor tensor) {
        // This would normally call a specialized kernel or backend method
        // For now, we'll throw an exception or return a placeholder if not implemented
        throw new InferenceException(ErrorCode.CONFIG_UNSUPPORTED, "On-the-fly dequantization for " + tensor.dtype() + " not implemented yet");
    }
}
