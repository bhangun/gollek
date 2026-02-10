package tech.kayys.wayang.tool.validation;

import tech.kayys.wayang.tool.spi.Tool;
import tech.kayys.wayang.tool.spi.ToolValidationException;

import java.util.Map;
import java.util.Set;

/**
 * Validates tool arguments against the tool's input schema.
 */
public class ToolArgumentValidator {

    /**
     * Validates the provided arguments against the tool's input schema.
     *
     * @param tool      the tool to validate against
     * @param arguments the arguments to validate
     * @throws ToolValidationException if validation fails
     */
    public void validate(Tool tool, Map<String, Object> arguments) throws ToolValidationException {
        Map<String, Object> inputSchema = tool.inputSchema();
        
        if (inputSchema == null || inputSchema.isEmpty()) {
            // If no schema is defined, accept any arguments
            return;
        }

        // Get required properties from schema
        Set<String> requiredProperties = getRequiredProperties(inputSchema);
        
        // Check for missing required arguments
        for (String requiredProperty : requiredProperties) {
            if (!arguments.containsKey(requiredProperty)) {
                throw new ToolValidationException(
                    String.format("Missing required argument '%s' for tool '%s'", 
                                  requiredProperty, tool.name()));
            }
        }

        // Validate argument types and constraints
        Map<String, Object> properties = getProperties(inputSchema);
        for (Map.Entry<String, Object> entry : arguments.entrySet()) {
            String argName = entry.getKey();
            Object argValue = entry.getValue();
            
            if (properties.containsKey(argName)) {
                validateArgumentType(argName, argValue, (Map<String, Object>) properties.get(argName));
            }
        }
    }

    @SuppressWarnings("unchecked")
    private Set<String> getRequiredProperties(Map<String, Object> schema) {
        Object requiredObj = schema.get("required");
        if (requiredObj instanceof Set) {
            return (Set<String>) requiredObj;
        } else if (requiredObj instanceof java.util.List) {
            return Set.copyOf((java.util.List<String>) requiredObj);
        }
        return Set.of();
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getProperties(Map<String, Object> schema) {
        Object propertiesObj = schema.get("properties");
        if (propertiesObj instanceof Map) {
            return (Map<String, Object>) propertiesObj;
        }
        return Map.of();
    }

    @SuppressWarnings("unchecked")
    private void validateArgumentType(String argName, Object argValue, Map<String, Object> propertySchema) {
        Object typeObj = propertySchema.get("type");
        if (typeObj == null) {
            return; // Skip validation if no type is specified
        }

        String expectedType = typeObj.toString();
        
        switch (expectedType) {
            case "string":
                if (!(argValue instanceof String)) {
                    throw new ToolValidationException(
                        String.format("Argument '%s' must be a string, got %s", 
                                      argName, argValue != null ? argValue.getClass().getSimpleName() : "null"));
                }
                break;
                
            case "integer":
                if (!(argValue instanceof Integer || argValue instanceof Long || 
                      argValue instanceof Short || argValue instanceof Byte)) {
                    throw new ToolValidationException(
                        String.format("Argument '%s' must be an integer, got %s", 
                                      argName, argValue != null ? argValue.getClass().getSimpleName() : "null"));
                }
                break;
                
            case "number":
                if (!(argValue instanceof Number)) {
                    throw new ToolValidationException(
                        String.format("Argument '%s' must be a number, got %s", 
                                      argName, argValue != null ? argValue.getClass().getSimpleName() : "null"));
                }
                break;
                
            case "boolean":
                if (!(argValue instanceof Boolean)) {
                    throw new ToolValidationException(
                        String.format("Argument '%s' must be a boolean, got %s", 
                                      argName, argValue != null ? argValue.getClass().getSimpleName() : "null"));
                }
                break;
                
            case "array":
                if (!(argValue instanceof java.util.List || argValue instanceof Object[])) {
                    throw new ToolValidationException(
                        String.format("Argument '%s' must be an array, got %s", 
                                      argName, argValue != null ? argValue.getClass().getSimpleName() : "null"));
                }
                break;
                
            case "object":
                if (!(argValue instanceof Map)) {
                    throw new ToolValidationException(
                        String.format("Argument '%s' must be an object, got %s", 
                                      argName, argValue != null ? argValue.getClass().getSimpleName() : "null"));
                }
                break;
                
            default:
                // For custom types or unknown types, we'll skip strict validation
                break;
        }
    }
}