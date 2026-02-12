package tech.kayys.wayang.tool.parser;

import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.tool.dto.SourceType;
import tech.kayys.wayang.tool.parser.multispec.SpecTypeDetector;

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

        SourceType detectedType = specTypeDetector.detect(openApi3Json).subscribe().withSubscriber(UniAssertSubscriber.create()).awaitItem().getItem();

        assertNotNull(detectedType);
        assertEquals(SourceType.OPENAPI_3_RAW, detectedType);
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

        SourceType detectedType = specTypeDetector.detect(openApi3Yaml).subscribe().withSubscriber(UniAssertSubscriber.create()).awaitItem().getItem();

        assertNotNull(detectedType);
        assertEquals(SourceType.OPENAPI_3_RAW, detectedType);
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

        SourceType detectedType = specTypeDetector.detect(swagger2Json).subscribe().withSubscriber(UniAssertSubscriber.create()).awaitItem().getItem();

        assertNotNull(detectedType);
        assertEquals(SourceType.SWAGGER_2_RAW, detectedType);
    }

    @Test
    void testDetectInvalidSpec() {
        String invalidSpec = """
            {
              "some": "random",
              "json": "content"
            }
            """;

        Exception exception = assertThrows(RuntimeException.class, () -> {
            specTypeDetector.detect(invalidSpec).subscribe().withSubscriber(UniAssertSubscriber.create()).awaitItem().getItem();
        });

        assertTrue(exception.getMessage().contains("Unknown"));
    }

    @Test
    void testDetectEmptySpec() {
        String emptySpec = "";

        Exception exception = assertThrows(RuntimeException.class, () -> {
            specTypeDetector.detect(emptySpec).subscribe().withSubscriber(UniAssertSubscriber.create()).awaitItem().getItem();
        });

        assertTrue(exception.getMessage().contains("Unknown"));
    }

    @Test
    void testDetectWithNullSpec() {
        Exception exception = assertThrows(RuntimeException.class, () -> {
            specTypeDetector.detect(null).subscribe().withSubscriber(UniAssertSubscriber.create()).awaitItem().getItem();
        });

        assertTrue(exception.getMessage().contains("Unknown"));
    }
}