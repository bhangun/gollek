package tech.kayys.golek.mcp;




import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

public record MCPConfig(
    @JsonProperty("mcpServers") Map<String, MCPServerConfig> mcpServers
) {
    public record MCPServerConfig(
        String command,
        String[] args,
        Map<String, String> env,
        @JsonProperty("transport") String transport,
        @JsonProperty("timeout") Integer timeout,
        @JsonProperty("enabled") Boolean enabled,
        @JsonProperty("description") String description
    ) {
        public MCPServerConfig {
            if (transport == null) transport = "stdio";
            if (timeout == null) timeout = 30000;
            if (enabled == null) enabled = true;
        }
    }
}