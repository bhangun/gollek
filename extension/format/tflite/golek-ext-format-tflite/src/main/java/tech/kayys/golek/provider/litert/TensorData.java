package tech.kayys.golek.provider.litert;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

/**
 * Platform-agnostic tensor representation.
 * 
 * @author Bhangun
 * @since 1.0.0
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TensorData {

    /**
     * Tensor name.
     */
    private String name;

    /**
     * Tensor shape (e.g., [1, 224, 224, 3] for batch=1, 224x224 RGB image).
     */
    private long[] shape;

    /**
     * Data type.
     */
    private TensorDataType dtype;

    /**
     * Raw byte data (most flexible format).
     */
    private byte[] data;

    /**
     * Float data (for FLOAT32/FLOAT16).
     */
    @JsonIgnore
    private float[] floatData;

    /**
     * Integer data (for INT8/INT16/INT32/UINT8).
     */
    @JsonIgnore
    private int[] intData;

    /**
     * Long data (for INT64/UINT64).
     */
    @JsonIgnore
    private long[] longData;

    /**
     * Boolean data (for BOOL type).
     */
    @JsonIgnore
    private boolean[] boolData;

    /**
     * String data (for STRING type).
     */
    @JsonIgnore
    private String[] stringData;

    /**
     * Calculate total number of elements.
     */
    public long getElementCount() {
        long count = 1;
        if (shape != null) {
            for (long dim : shape) {
                count *= dim;
            }
        }
        return count;
    }

    /**
     * Get byte size of tensor.
     */
    public long getByteSize() {
        if (data != null) {
            return data.length;
        }
        long elements = getElementCount();
        return elements * dtype.getByteSize();
    }
}