package tech.kayys.gollek.litert.model;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * LiteRT Model information.
 */
public class LiteRTModelInfo {

    /**
     * Model identifier.
     */
    private String modelId;

    /**
     * Model name.
     */
    private String name;

    /**
     * Model architecture.
     */
    private String architecture;

    /**
     * Model file path or name.
     */
    private String path;

    /**
     * Model file size in bytes.
     */
    private long sizeBytes;

    /**
     * Input tensor information.
     */
    private List<TensorInfo> inputs = List.of();

    /**
     * Output tensor information.
     */
    private List<TensorInfo> outputs = List.of();

    /**
     * Model metadata.
     */
    private Map<String, Object> metadata = Map.of();

    public LiteRTModelInfo() {
    }

    public LiteRTModelInfo(String modelId, String name, String architecture, String path, long sizeBytes,
            List<TensorInfo> inputs, List<TensorInfo> outputs, Map<String, Object> metadata) {
        this.modelId = modelId;
        this.name = name;
        this.architecture = architecture;
        this.path = path;
        this.sizeBytes = sizeBytes;
        this.inputs = inputs;
        this.outputs = outputs;
        this.metadata = metadata;
    }

    public String getModelId() {
        return modelId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getArchitecture() {
        return architecture;
    }

    public void setArchitecture(String architecture) {
        this.architecture = architecture;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public long getSizeBytes() {
        return sizeBytes;
    }

    public void setSizeBytes(long sizeBytes) {
        this.sizeBytes = sizeBytes;
    }

    public List<TensorInfo> getInputs() {
        return inputs;
    }

    public void setInputs(List<TensorInfo> inputs) {
        this.inputs = inputs;
    }

    public List<TensorInfo> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<TensorInfo> outputs) {
        this.outputs = outputs;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        LiteRTModelInfo that = (LiteRTModelInfo) o;
        return sizeBytes == that.sizeBytes && Objects.equals(modelId, that.modelId) && Objects.equals(name, that.name)
                && Objects.equals(architecture, that.architecture) && Objects.equals(path, that.path)
                && Objects.equals(inputs, that.inputs) && Objects.equals(outputs, that.outputs)
                && Objects.equals(metadata, that.metadata);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelId, name, architecture, path, sizeBytes, inputs, outputs, metadata);
    }

    @Override
    public String toString() {
        return "LiteRTModelInfo{" +
                "modelId='" + modelId + '\'' +
                ", name='" + name + '\'' +
                ", architecture='" + architecture + '\'' +
                ", path='" + path + '\'' +
                ", sizeBytes=" + sizeBytes +
                ", inputs=" + inputs +
                ", outputs=" + outputs +
                ", metadata=" + metadata +
                '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelId;
        private String name;
        private String architecture;
        private String path;
        private long sizeBytes;
        private List<TensorInfo> inputs = List.of();
        private List<TensorInfo> outputs = List.of();
        private Map<String, Object> metadata = Map.of();

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder architecture(String architecture) {
            this.architecture = architecture;
            return this;
        }

        public Builder path(String path) {
            this.path = path;
            return this;
        }

        public Builder sizeBytes(long sizeBytes) {
            this.sizeBytes = sizeBytes;
            return this;
        }

        public Builder inputs(List<TensorInfo> inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder outputs(List<TensorInfo> outputs) {
            this.outputs = outputs;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public LiteRTModelInfo build() {
            return new LiteRTModelInfo(modelId, name, architecture, path, sizeBytes, inputs, outputs, metadata);
        }
    }

    /**
     * Tensor information.
     */
    public static class TensorInfo {
        private String name;
        private int[] shape;
        private String dataType;
        private long byteSize;

        public TensorInfo() {
        }

        public TensorInfo(String name, int[] shape, String dataType, long byteSize) {
            this.name = name;
            this.shape = shape;
            this.dataType = dataType;
            this.byteSize = byteSize;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public int[] getShape() {
            return shape;
        }

        public void setShape(int[] shape) {
            this.shape = shape;
        }

        public String getDataType() {
            return dataType;
        }

        public void setDataType(String dataType) {
            this.dataType = dataType;
        }

        public long getByteSize() {
            return byteSize;
        }

        public void setByteSize(long byteSize) {
            this.byteSize = byteSize;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;
            TensorInfo that = (TensorInfo) o;
            return byteSize == that.byteSize && Objects.equals(name, that.name)
                    && java.util.Arrays.equals(shape, that.shape) && Objects.equals(dataType, that.dataType);
        }

        @Override
        public int hashCode() {
            int result = Objects.hash(name, dataType, byteSize);
            result = 31 * result + java.util.Arrays.hashCode(shape);
            return result;
        }

        @Override
        public String toString() {
            return "TensorInfo{" +
                    "name='" + name + '\'' +
                    ", shape=" + java.util.Arrays.toString(shape) +
                    ", dataType='" + dataType + '\'' +
                    ", byteSize=" + byteSize +
                    '}';
        }

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private String name;
            private int[] shape;
            private String dataType;
            private long byteSize;

            public Builder name(String name) {
                this.name = name;
                return this;
            }

            public Builder shape(int[] shape) {
                this.shape = shape;
                return this;
            }

            public Builder dataType(String dataType) {
                this.dataType = dataType;
                return this;
            }

            public Builder byteSize(long byteSize) {
                this.byteSize = byteSize;
                return this;
            }

            public TensorInfo build() {
                return new TensorInfo(name, shape, dataType, byteSize);
            }
        }
    }
}
