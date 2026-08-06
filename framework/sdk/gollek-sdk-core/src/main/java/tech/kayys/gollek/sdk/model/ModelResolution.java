package tech.kayys.gollek.sdk.model;

import java.util.Objects;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;

/**
 * Result of a model resolution and preparation process.
 */
public final class ModelResolution {
    private final String modelId;
    private final String localPath;
    private final ModelInfo info;
    private final String providerId;
    private final String notice;

    public ModelResolution(String modelId, String localPath, ModelInfo info) {
        this(modelId, localPath, info, null, null);
    }

    public ModelResolution(String modelId, String localPath, ModelInfo info, String providerId, String notice) {
        this.modelId = Objects.requireNonNull(modelId, "modelId is required");
        this.localPath = localPath;
        this.info = info;
        this.providerId = providerId;
        this.notice = notice;
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

    public String getProviderId() {
        return providerId;
    }

    public String getNotice() {
        return notice;
    }
    
    @Override
    public String toString() {
        return "ModelResolution{" +
                "modelId='" + modelId + '\'' +
                ", localPath='" + localPath + '\'' +
                ", providerId='" + providerId + '\'' +
                '}';
    }
}
