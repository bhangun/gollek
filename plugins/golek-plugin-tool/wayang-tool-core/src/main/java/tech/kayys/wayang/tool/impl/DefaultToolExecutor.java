package tech.kayys.wayang.tool.impl;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.wayang.tool.spi.ToolExecutor;
import tech.kayys.wayang.tool.spi.ToolRegistry;
import tech.kayys.golek.tool.dto.InvocationStatus;
import tech.kayys.golek.tool.dto.ToolExecutionResult;
import tech.kayys.golek.tool.validation.ToolArgumentValidator;

import java.util.Map;

/**
 * Default implementation of ToolExecutor.
 */
@ApplicationScoped
public class DefaultToolExecutor implements ToolExecutor {

    @Inject
    ToolRegistry toolRegistry;
    
    private final ToolArgumentValidator validator = new ToolArgumentValidator();

    @Override
    public Uni<ToolExecutionResult> execute(String toolId, Map<String, Object> arguments, Map<String, Object> context) {
        return toolRegistry.getTool(toolId)
                .invoke(tool -> validator.validate(tool, arguments)) // Validate arguments before execution
                .flatMap(tool -> {
                    // Execute the tool and wrap the result in a ToolExecutionResult
                    return tool.execute(arguments, context)
                            .map(output -> ToolExecutionResult.success(
                                generateCallId(toolId),
                                toolId,
                                output,
                                0 // execution time - in a real implementation, measure actual time
                            ))
                            .onFailure().recoverWithItem(throwable -> ToolExecutionResult.failure(
                                generateCallId(toolId),
                                toolId,
                                throwable.getMessage(),
                                0 // execution time
                            ));
                });
    }
    
    private String generateCallId(String toolId) {
        return "call_" + toolId + "_" + System.nanoTime();
    }
}
