package tech.kayys.wayang.tool.service;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.ToolGenerationResult;

public interface ToolGenerationService {
    Uni<ToolGenerationResult> generateTools(GenerateToolsRequest request);
}
