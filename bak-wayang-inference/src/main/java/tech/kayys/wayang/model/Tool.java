package tech.kayys.golek.model;


import java.util.Map;

public record Tool(
    String type,
    FunctionDefinition function
) {
    public record FunctionDefinition(
        String name,
        String description,
        Map<String, Object> parameters
    ) {}
    
    public static Tool function(String name, String description, Map<String, Object> parameters) {
        return new Tool("function", new FunctionDefinition(name, description, parameters));
    }
}