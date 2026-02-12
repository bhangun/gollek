package tech.kayys.wayang.tool.parser;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.tool.parser.SpecFormatRegistry;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SpecFormatRegistryTest {

    private SpecFormatRegistry specFormatRegistry;

    @BeforeEach
    void setUp() {
        specFormatRegistry = new SpecFormatRegistry();
    }

    @Test
    void testGetSupportedFormats() {
        Map<String, SpecFormatRegistry.SpecFormatInfo> formats = specFormatRegistry.getSupportedFormats();

        assertNotNull(formats);
        assertFalse(formats.isEmpty());
        
        // Check that some expected formats are present
        assertTrue(formats.containsKey("OPENAPI_3"));
        assertTrue(formats.containsKey("SWAGGER_2"));
        assertTrue(formats.containsKey("POSTMAN"));
        assertTrue(formats.containsKey("ASYNCAPI"));
        assertTrue(formats.containsKey("GRAPHQL"));
        assertTrue(formats.containsKey("WSDL"));
        assertTrue(formats.containsKey("HAR"));
        assertTrue(formats.containsKey("INSOMNIA"));
        assertTrue(formats.containsKey("API_BLUEPRINT"));
        assertTrue(formats.containsKey("RAML"));
    }

    @Test
    void testGetFullySupportedFormats() {
        List<SpecFormatRegistry.SpecFormatInfo> fullySupported = specFormatRegistry.getFullySupportedFormats();

        assertNotNull(fullySupported);
        assertFalse(fullySupported.isEmpty());
        
        // Check that fully supported formats are marked as such
        for (SpecFormatRegistry.SpecFormatInfo format : fullySupported) {
            assertTrue(format.fullySupported());
        }
    }

    @Test
    void testFormatInfoProperties() {
        Map<String, SpecFormatRegistry.SpecFormatInfo> formats = specFormatRegistry.getSupportedFormats();
        
        SpecFormatRegistry.SpecFormatInfo openApi3 = formats.get("OPENAPI_3");
        assertNotNull(openApi3);
        assertEquals("OpenAPI 3.x", openApi3.name());
        assertEquals("Modern REST API specification (3.0.0 - 3.1.0)", openApi3.description());
        assertTrue(openApi3.fileExtensions().contains(".json"));
        assertTrue(openApi3.fileExtensions().contains(".yaml"));
        assertTrue(openApi3.fileExtensions().contains(".yml"));
        assertTrue(openApi3.fullySupported());
        
        SpecFormatRegistry.SpecFormatInfo apiBlueprint = formats.get("API_BLUEPRINT");
        assertNotNull(apiBlueprint);
        assertFalse(apiBlueprint.fullySupported());
    }
}