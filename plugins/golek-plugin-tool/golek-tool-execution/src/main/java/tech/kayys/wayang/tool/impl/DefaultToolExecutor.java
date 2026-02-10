package tech.kayys.wayang.tool.impl;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.wayang.tool.spi.ToolExecutor;
import tech.kayys.wayang.tool.spi.ToolRegistry;

import java.util.Map;

/**
 * Default implementation of ToolExecutor.
 */
@ApplicationScoped
public class DefaultToolExecutor implements ToolExecutor {

    @Inject
    ToolRegistry toolRegistry;

    @Override
    public Uni<Map<String, Object>> execute(String toolId, Map<String, Object> arguments, Map<String, Object> context) {
        return toolRegistry.getTool(toolId)
                .flatMap(tool -> tool.execute(arguments, context));
    }
}
