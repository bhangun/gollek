package tech.kayys.wayang.tool;

import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.json.JSONObject;
import org.everit.json.schema.Schema;
import org.everit.json.schema.loader.SchemaLoader;
import org.everit.json.schema.ValidationException;
import tech.kayys.wayang.tool.exception.ToolValidationException;
import jakarta.enterprise.context.ApplicationScoped;

/**
 * JSON Schema validator
 */
@ApplicationScoped
public class SchemaValidator {

    private static final Logger LOG = LoggerFactory.getLogger(SchemaValidator.class);

    /**
     * Validate data against JSON schema
     */
    public void validate(Map<String, Object> schema, Map<String, Object> data) {
        if (schema == null || schema.isEmpty()) {
            return;
        }

        try {
            JSONObject schemaJson = new JSONObject(schema);
            Schema jsonSchema = SchemaLoader.load(schemaJson);

            JSONObject dataJson = new JSONObject(data);
            jsonSchema.validate(dataJson);
        } catch (ValidationException e) {
            LOG.error("Schema validation failed: {}", e.getMessage());
            throw new ToolValidationException(
                    "Input validation failed: " + e.getMessage(), e);
        }
    }
}