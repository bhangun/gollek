package tech.kayys.wayang.tool.utcp;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tool.spi.Tool;

import java.util.Collections;
import java.util.Map;

/**
 * UTCP Tool implementation.
 */
public class UtcpTool implements Tool {

    private final String id;
    private final String name;

    public UtcpTool(String id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String description() {
        return "Universal Tool Call Protocol Tool";
    }

    @Override
    public Map<String, Object> inputSchema() {
        return Collections.emptyMap();
    }

    @Override
    public Uni<Map<String, Object>> execute(Map<String, Object> arguments, Map<String, Object> context) {
        // TODO: Implement UTCP call
        return Uni.createFrom().item(Map.of("status", "executed", "tool", "utcp"));
    }
}
