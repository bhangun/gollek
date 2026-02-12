package tech.kayys.wayang.tool.impl;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.wayang.tool.spi.ToolExecutor;
import tech.kayys.wayang.tool.spi.ToolRegistry;
import tech.kayys.wayang.tool.dto.InvocationStatus;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;
import tech.kayys.wayang.tool.validation.ToolArgumentValidator;

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
                                toolId,
                                output,
                                0 // execution time
                            ))
                            .onFailure().recoverWithItem(throwable -> ToolExecutionResult.failure(
                                toolId,
                                InvocationStatus.FAILURE,
                                throwable.getMessage(),
                                0 // execution time
                            ));
                });
    }
    
    private String generateCallId(String toolId) {
        return "call_" + toolId + "_" + System.nanoTime();
    }
}
