package tech.kayys.gollek.provider.litert;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import java.util.Arrays;
import java.util.Objects;

/**
 * Platform-agnostic tensor representation.
 * 
 * @author Bhangun
 * @since 1.0.0
 */
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

    public TensorData() {
    }

    public TensorData(String name, long[] shape, TensorDataType dtype, byte[] data, float[] floatData, int[] intData,
            long[] longData, boolean[] boolData, String[] stringData) {
        this.name = name;
        this.shape = shape;
        this.dtype = dtype;
        this.data = data;
        this.floatData = floatData;
        this.intData = intData;
        this.longData = longData;
        this.boolData = boolData;
        this.stringData = stringData;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public long[] getShape() {
        return shape;
    }

    public void setShape(long[] shape) {
        this.shape = shape;
    }

    public TensorDataType getDtype() {
        return dtype;
    }

    public void setDtype(TensorDataType dtype) {
        this.dtype = dtype;
    }

    public byte[] getData() {
        return data;
    }

    public void setData(byte[] data) {
        this.data = data;
    }

    public float[] getFloatData() {
        return floatData;
    }

    public void setFloatData(float[] floatData) {
        this.floatData = floatData;
    }

    public int[] getIntData() {
        return intData;
    }

    public void setIntData(int[] intData) {
        this.intData = intData;
    }

    public long[] getLongData() {
        return longData;
    }

    public void setLongData(long[] longData) {
        this.longData = longData;
    }

    public boolean[] getBoolData() {
        return boolData;
    }

    public void setBoolData(boolean[] boolData) {
        this.boolData = boolData;
    }

    public String[] getStringData() {
        return stringData;
    }

    public void setStringData(String[] stringData) {
        this.stringData = stringData;
    }

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
        return elements * (dtype != null ? dtype.getByteSize() : 0);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        TensorData that = (TensorData) o;
        return Objects.equals(name, that.name) && Arrays.equals(shape, that.shape) && dtype == that.dtype
                && Arrays.equals(data, that.data) && Arrays.equals(floatData, that.floatData)
                && Arrays.equals(intData, that.intData) && Arrays.equals(longData, that.longData)
                && Arrays.equals(boolData, that.boolData) && Arrays.equals(stringData, that.stringData);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(name, dtype);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data);
        result = 31 * result + Arrays.hashCode(floatData);
        result = 31 * result + Arrays.hashCode(intData);
        result = 31 * result + Arrays.hashCode(longData);
        result = 31 * result + Arrays.hashCode(boolData);
        result = 31 * result + Arrays.hashCode(stringData);
        return result;
    }

    @Override
    public String toString() {
        return "TensorData{" +
                "name='" + name + '\'' +
                ", shape=" + Arrays.toString(shape) +
                ", dtype=" + dtype +
                ", data=" + Arrays.toString(data) +
                ", floatData=" + Arrays.toString(floatData) +
                ", intData=" + Arrays.toString(intData) +
                ", longData=" + Arrays.toString(longData) +
                ", boolData=" + Arrays.toString(boolData) +
                ", stringData=" + Arrays.toString(stringData) +
                '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String name;
        private long[] shape;
        private TensorDataType dtype;
        private byte[] data;
        private float[] floatData;
        private int[] intData;
        private long[] longData;
        private boolean[] boolData;
        private String[] stringData;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder shape(long[] shape) {
            this.shape = shape;
            return this;
        }

        public Builder dtype(TensorDataType dtype) {
            this.dtype = dtype;
            return this;
        }

        public Builder data(byte[] data) {
            this.data = data;
            return this;
        }

        public Builder floatData(float[] floatData) {
            this.floatData = floatData;
            return this;
        }

        public Builder intData(int[] intData) {
            this.intData = intData;
            return this;
        }

        public Builder longData(long[] longData) {
            this.longData = longData;
            return this;
        }

        public Builder boolData(boolean[] boolData) {
            this.boolData = boolData;
            return this;
        }

        public Builder stringData(String[] stringData) {
            this.stringData = stringData;
            return this;
        }

        public TensorData build() {
            return new TensorData(name, shape, dtype, data, floatData, intData, longData, boolData, stringData);
        }
    }
}
