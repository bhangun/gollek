package tech.kayys.wayang.tool.parser;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.mcp.parser.multispec.SpecTypeDetector;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

class SpecTypeDetectorTest {

    private SpecTypeDetector specTypeDetector;

    @BeforeEach
    void setUp() {
        specTypeDetector = new SpecTypeDetector();
    }

    @Test
    void testDetectOpenApi3Json() {
        String openApi3Json = """
            {
              "openapi": "3.0.0",
              "info": {
                "title": "Sample API",
                "version": "1.0.0"
              },
              "paths": {}
            }
            """;
        
        InputStream inputStream = new ByteArrayInputStream(openApi3Json.getBytes(StandardCharsets.UTF_8));
        String detectedType = specTypeDetector.detectSpecType(inputStream);
        
        assertNotNull(detectedType);
        assertTrue(detectedType.toUpperCase().contains("OPENAPI") || detectedType.toUpperCase().contains("OAS"));
    }

    @Test
    void testDetectOpenApi3Yaml() {
        String openApi3Yaml = """
            openapi: 3.0.0
            info:
              title: Sample API
              version: 1.0.0
            paths: {}
            """;
        
        InputStream inputStream = new ByteArrayInputStream(openApi3Yaml.getBytes(StandardCharsets.UTF_8));
        String detectedType = specTypeDetector.detectSpecType(inputStream);
        
        assertNotNull(detectedType);
        assertTrue(detectedType.toUpperCase().contains("OPENAPI") || detectedType.toUpperCase().contains("OAS"));
    }

    @Test
    void testDetectSwagger2Json() {
        String swagger2Json = """
            {
              "swagger": "2.0",
              "info": {
                "title": "Sample API",
                "version": "1.0.0"
              },
              "paths": {}
            }
            """;
        
        InputStream inputStream = new ByteArrayInputStream(swagger2Json.getBytes(StandardCharsets.UTF_8));
        String detectedType = specTypeDetector.detectSpecType(inputStream);
        
        assertNotNull(detectedType);
        assertTrue(detectedType.toUpperCase().contains("SWAGGER"));
    }

    @Test
    void testDetectInvalidSpec() {
        String invalidSpec = """
            {
              "some": "random",
              "json": "content"
            }
            """;
        
        InputStream inputStream = new ByteArrayInputStream(invalidSpec.getBytes(StandardCharsets.UTF_8));
        String detectedType = specTypeDetector.detectSpecType(inputStream);
        
        // Should return UNKNOWN or null for unrecognized specs
        assertNull(detectedType);
    }

    @Test
    void testDetectEmptySpec() {
        String emptySpec = "";
        
        InputStream inputStream = new ByteArrayInputStream(emptySpec.getBytes(StandardCharsets.UTF_8));
        String detectedType = specTypeDetector.detectSpecType(inputStream);
        
        assertNull(detectedType);
    }

    @Test
    void testDetectWithNullStream() {
        assertThrows(Exception.class, () -> {
            specTypeDetector.detectSpecType(null);
        });
    }
}