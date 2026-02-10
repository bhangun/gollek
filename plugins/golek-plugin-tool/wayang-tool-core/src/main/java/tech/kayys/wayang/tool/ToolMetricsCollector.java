package tech.kayys.wayang.tool;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;

/**
 * Tool metrics collector
 */
@ApplicationScoped
public class ToolMetricsCollector {

    public Uni<Void> collect(String toolId, ToolExecutionResult result) {
        // Update tool metrics asynchronously
        return Uni.createFrom().voidItem();
    }
}