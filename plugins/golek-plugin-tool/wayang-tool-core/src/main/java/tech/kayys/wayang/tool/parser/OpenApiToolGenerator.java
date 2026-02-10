package tech.kayys.wayang.tool.parser;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.ToolGenerationResult;

import java.util.List;

@ApplicationScoped
public class OpenApiToolGenerator {

    public Uni<ToolGenerationResult> generateTools(GenerateToolsRequest request) {
        // Stub implementation
        return Uni.createFrom().item(new ToolGenerationResult(
                "stub-source-id",
                request.namespace(),
                0,
                List.of(),
                List.of("Not implemented yet")));
    }
}
