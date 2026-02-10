package tech.kayys.wayang.tool.sandbox;

import io.smallrye.mutiny.Uni;
import java.util.Map;

/**
 * Skeleton for sandboxed tool execution.
 */
public interface SandboxExecutor {

    /**
     * Execute a tool in a sandbox.
     * 
     * @param toolId tool identifier
     * @param args arguments
     * @return result map
     */
    Uni<Map<String, Object>> executeSandboxed(String toolId, Map<String, Object> args);
}
