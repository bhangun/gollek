package tech.kayys.wayang.tool.spi;

import io.smallrye.mutiny.Uni;
import java.util.Map;

/**
 * Interface definition for a generic Tool.
 */
public interface Tool {

    /**
     * Unique identifier for the tool.
     */
    String id();

    /**
     * Human-readable name.
     */
    String name();

    /**
     * Tool description/capabilities.
     */
    String description();

    /**
     * JSON Schema or map describing input parameters.
     */
    Map<String, Object> inputSchema();

    /**
     * Execute the tool with arguments.
     *
     * @param arguments input arguments
     * @param context   execution context
     * @return Uni result map
     */
    Uni<Map<String, Object>> execute(Map<String, Object> arguments, Map<String, Object> context);
}
