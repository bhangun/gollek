package tech.kayys.gollek.sdk.model;

import java.util.Objects;
import tech.kayys.gollek.spi.model.ModelInfo;

/**
 * Result of a model resolution and preparation process.
 */
public final class ModelResolution {
    private final String modelId;
    private final String localPath;
    private final ModelInfo info;

    public ModelResolution(String modelId, String localPath, ModelInfo info) {
        this.modelId = Objects.requireNonNull(modelId, "modelId is required");
        this.localPath = localPath;
        this.info = info;
    }

    public String getModelId() {
        return modelId;
    }

    public String getLocalPath() {
        return localPath;
    }

    public ModelInfo getInfo() {
        return info;
    }
    
    @Override
    public String toString() {
        return "ModelResolution{" +
                "modelId='" + modelId + '\'' +
                ", localPath='" + localPath + '\'' +
                '}';
    }
}
