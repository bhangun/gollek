package tech.kayys.golek.plugin;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;

import tech.kayys.golek.model.Tool;

public class WeatherPlugin implements ToolPlugin {
    private PluginContext context;
    
    @Override
    public String getId() {
        return "weather";
    }
    
    @Override
    public String getName() {
        return "Weather Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Get weather information";
    }
    
    @Override
    public void initialize(PluginContext context) {
        this.context = context;
    }
    
    @Override
    public void start() {}
    
    @Override
    public void stop() {}
    
    @Override
    public List<Tool> getTools() {
        return List.of(
            Tool.function(
                "get_weather",
                "Get current weather for a location",
                Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "location", Map.of("type", "string"),
                        "units", Map.of("type", "string", "enum", List.of("celsius", "fahrenheit"))
                    ),
                    "required", List.of("location")
                )
            )
        );
    }
    
    @Override
    public String executeTool(String toolName, JsonNode arguments) throws PluginException {
        if (!"get_weather".equals(toolName)) {
            throw new PluginException("Unknown tool: " + toolName);
        }
        
        String location = arguments.get("location").asText();
        String units = arguments.has("units") ? 
            arguments.get("units").asText() : "celsius";
        
        // Call weather API (simplified)
        return String.format("The weather in %s is sunny, 22°%s",
            location, units.equals("celsius") ? "C" : "F");
    }


}
