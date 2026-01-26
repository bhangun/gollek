package tech.kayys.golek.mcp;


import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ToolCallParser {
    private static final Pattern FUNCTION_CALL_PATTERN = 
        Pattern.compile("<function_call>\\s*\\{([^}]+)\\}\\s*</function_call>");
    private static final ObjectMapper mapper = new ObjectMapper();
    
    public static ParsedToolCall parse(String text) {
        Matcher matcher = FUNCTION_CALL_PATTERN.matcher(text);
        if (matcher.find()) {
            try {
                String json = "{" + matcher.group(1) + "}";
                JsonNode node = mapper.readTree(json);
                String name = node.get("name").asText();
                JsonNode args = node.get("arguments");
                return new ParsedToolCall(name, args, true);
            } catch (Exception e) {
                return new ParsedToolCall(null, null, false);
            }
        }
        return new ParsedToolCall(null, null, false);
    }
    
    public record ParsedToolCall(String functionName, JsonNode arguments, boolean found) {}
}
