package tech.kayys.wayang.tool.parser;

import io.swagger.v3.oas.models.media.ArraySchema;
import io.swagger.v3.oas.models.media.IntegerSchema;
import io.swagger.v3.oas.models.media.ObjectSchema;
import io.swagger.v3.oas.models.media.StringSchema;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.tool.parser.SchemaConverter;

import java.math.BigDecimal;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SchemaConverterTest {

    private SchemaConverter schemaConverter;

    @BeforeEach
    void setUp() {
        schemaConverter = new SchemaConverter();
    }

    @Test
    void testConvertNullSchema() {
        Map<String, Object> result = schemaConverter.convert(null);
        
        assertNotNull(result);
        assertEquals("object", result.get("type"));
        assertEquals(1, result.size());
    }

    @Test
    void testConvertStringSchema() {
        StringSchema stringSchema = new StringSchema();
        stringSchema.setDescription("A sample string field");
        stringSchema.setExample("sample value");
        stringSchema.setMinLength(1);
        stringSchema.setMaxLength(100);
        stringSchema.setPattern("^[a-zA-Z]+$");

        Map<String, Object> result = schemaConverter.convert(stringSchema);

        assertNotNull(result);
        assertEquals("string", result.get("type"));
        assertEquals("A sample string field", result.get("description"));
        assertEquals(java.math.BigDecimal.valueOf(1), result.get("minLength"));
        assertEquals(java.math.BigDecimal.valueOf(100), result.get("maxLength"));
        assertEquals("^[a-zA-Z]+$", result.get("pattern"));
    }

    @Test
    void testConvertIntegerSchema() {
        IntegerSchema integerSchema = new IntegerSchema();
        integerSchema.setDescription("A sample integer field");
        integerSchema.setMinimum(BigDecimal.valueOf(0));
        integerSchema.setMaximum(BigDecimal.valueOf(100));

        Map<String, Object> result = schemaConverter.convert(integerSchema);

        assertNotNull(result);
        assertEquals("integer", result.get("type"));
        assertEquals("A sample integer field", result.get("description"));
        assertEquals(java.math.BigDecimal.valueOf(0), result.get("minimum"));
        assertEquals(java.math.BigDecimal.valueOf(100), result.get("maximum"));
    }

    @Test
    void testConvertArraySchema() {
        ArraySchema arraySchema = new ArraySchema();
        arraySchema.setDescription("A sample array field");
        arraySchema.setItems(new StringSchema());

        Map<String, Object> result = schemaConverter.convert(arraySchema);

        assertNotNull(result);
        assertEquals("array", result.get("type"));
        assertEquals("A sample array field", result.get("description"));
        assertTrue(result.containsKey("items"));
        Map<String, Object> items = (Map<String, Object>) result.get("items");
        assertEquals("string", items.get("type"));
    }

    @Test
    void testConvertObjectSchema() {
        ObjectSchema objectSchema = new ObjectSchema();
        objectSchema.addProperty("name", new StringSchema());
        objectSchema.addProperty("age", new IntegerSchema());
        objectSchema.addRequiredItem("name");

        Map<String, Object> result = schemaConverter.convert(objectSchema);

        assertNotNull(result);
        assertEquals("object", result.get("type"));
        assertTrue(result.containsKey("properties"));
        
        Map<String, Object> properties = (Map<String, Object>) result.get("properties");
        assertNotNull(properties.get("name"));
        assertNotNull(properties.get("age"));
        
        assertTrue(result.containsKey("required"));
        assertEquals(1, ((java.util.List<?>) result.get("required")).size());
        assertEquals("name", ((java.util.List<?>) result.get("required")).get(0));
    }

    @Test
    void testConvertSchemaWithEnum() {
        StringSchema stringSchema = new StringSchema();
        stringSchema.addEnumItem("option1");
        stringSchema.addEnumItem("option2");

        Map<String, Object> result = schemaConverter.convert(stringSchema);

        assertNotNull(result);
        assertEquals("string", result.get("type"));
        assertTrue(result.containsKey("enum"));
        java.util.List<?> enumValues = (java.util.List<?>) result.get("enum");
        assertEquals(2, enumValues.size());
        assertTrue(enumValues.contains("option1"));
        assertTrue(enumValues.contains("option2"));
    }
}