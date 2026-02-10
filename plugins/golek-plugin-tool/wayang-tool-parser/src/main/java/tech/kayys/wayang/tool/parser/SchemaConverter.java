package tech.kayys.wayang.mcp.parser;

import io.swagger.v3.oas.models.media.Schema;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.HashMap;
import java.util.Map;

/**
 * Convert OpenAPI schema to JSON Schema
 */
@ApplicationScoped
public class SchemaConverter {

    /**
     * Convert OpenAPI schema to JSON Schema format
     */
    public Map<String, Object> convert(Schema<?> schema) {
        if (schema == null) {
            return Map.of("type", "object");
        }

        Map<String, Object> jsonSchema = new HashMap<>();

        // Type
        if (schema.getType() != null) {
            jsonSchema.put("type", schema.getType());
        }

        // Format
        if (schema.getFormat() != null) {
            jsonSchema.put("format", schema.getFormat());
        }

        // Description
        if (schema.getDescription() != null) {
            jsonSchema.put("description", schema.getDescription());
        }

        // Enum
        if (schema.getEnum() != null) {
            jsonSchema.put("enum", schema.getEnum());
        }

        // Properties (for object types)
        if (schema.getProperties() != null) {
            Map<String, Object> properties = new HashMap<>();
            schema.getProperties().forEach((name, propSchema) -> {
                properties.put(name, convert((Schema<?>) propSchema));
            });
            jsonSchema.put("properties", properties);
        }

        // Required
        if (schema.getRequired() != null) {
            jsonSchema.put("required", schema.getRequired());
        }

        // Items (for array types)
        if (schema.getItems() != null) {
            jsonSchema.put("items", convert(schema.getItems()));
        }

        // Validation constraints
        if (schema.getMinimum() != null) {
            jsonSchema.put("minimum", schema.getMinimum());
        }
        if (schema.getMaximum() != null) {
            jsonSchema.put("maximum", schema.getMaximum());
        }
        if (schema.getMinLength() != null) {
            jsonSchema.put("minLength", schema.getMinLength());
        }
        if (schema.getMaxLength() != null) {
            jsonSchema.put("maxLength", schema.getMaxLength());
        }
        if (schema.getPattern() != null) {
            jsonSchema.put("pattern", schema.getPattern());
        }

        return jsonSchema;
    }
}