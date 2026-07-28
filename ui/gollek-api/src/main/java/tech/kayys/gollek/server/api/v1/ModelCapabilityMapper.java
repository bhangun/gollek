package tech.kayys.gollek.server.api.v1;

import tech.kayys.gollek.spi.model.ModelInfo;
import java.util.List;
import java.util.Optional;

public class ModelCapabilityMapper {
    public static Object toCapabilityMatrix(String modelId, Optional<ModelInfo> modelInfo, List<Object> providers, Optional<String> preferredProvider) {
        return new java.util.HashMap<>();
    }
}
