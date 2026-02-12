package tech.kayys.wayang.tool.spi;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;

import java.util.Map;

/**
 * Strategy for executing tools.
 */
public interface ToolExecutor {

    /**
     * Execute a specific tool.
     *
     * @param toolId    the tool ID to execute
     * @param arguments the arguments
     * @param context   the execution context
     * @return Uni result
     */
    Uni<ToolExecutionResult> execute(String toolId, Map<String, Object> arguments, Map<String, Object> context);
}
