package tech.kayys.wayang.mcp;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * ============================================================================
 * SPEC FORMAT REGISTRY
 * ============================================================================
 * 
 * Central registry for all supported specification formats
 */
@ApplicationScoped
public class SpecFormatRegistry {

    private static final Map<String, SpecFormatInfo> SUPPORTED_FORMATS = Map.ofEntries(
            Map.entry("OPENAPI_3", new SpecFormatInfo(
                    "OpenAPI 3.x",
                    "Modern REST API specification (3.0.0 - 3.1.0)",
                    List.of(".json", ".yaml", ".yml"),
                    true)),
            Map.entry("SWAGGER_2", new SpecFormatInfo(
                    "Swagger 2.0",
                    "Legacy OpenAPI specification (Swagger 2.0)",
                    List.of(".json", ".yaml", ".yml"),
                    true)),
            Map.entry("POSTMAN", new SpecFormatInfo(
                    "Postman Collection",
                    "Postman API collection (v2.0, v2.1)",
                    List.of(".json"),
                    true)),
            Map.entry("ASYNCAPI", new SpecFormatInfo(
                    "AsyncAPI",
                    "Event-driven API specification (2.x, 3.x)",
                    List.of(".json", ".yaml", ".yml"),
                    true)),
            Map.entry("GRAPHQL", new SpecFormatInfo(
                    "GraphQL Schema",
                    "GraphQL type system definition",
                    List.of(".graphql", ".gql"),
                    true)),
            Map.entry("WSDL", new SpecFormatInfo(
                    "WSDL",
                    "SOAP web service description (1.1, 2.0)",
                    List.of(".wsdl", ".xml"),
                    true)),
            Map.entry("HAR", new SpecFormatInfo(
                    "HTTP Archive",
                    "Recorded HTTP requests and responses",
                    List.of(".har"),
                    true)),
            Map.entry("INSOMNIA", new SpecFormatInfo(
                    "Insomnia Collection",
                    "Insomnia REST client collection",
                    List.of(".json"),
                    true)),
            Map.entry("API_BLUEPRINT", new SpecFormatInfo(
                    "API Blueprint",
                    "Markdown-based API specification",
                    List.of(".apib", ".md"),
                    false // Not fully implemented
            )),
            Map.entry("RAML", new SpecFormatInfo(
                    "RAML",
                    "RESTful API Modeling Language (0.8, 1.0)",
                    List.of(".raml"),
                    false // Not fully implemented
            )));

    public Map<String, SpecFormatInfo> getSupportedFormats() {
        return Collections.unmodifiableMap(SUPPORTED_FORMATS);
    }

    public List<SpecFormatInfo> getFullySupportedFormats() {
        return SUPPORTED_FORMATS.values().stream()
                .filter(SpecFormatInfo::fullySupported)
                .toList();
    }

    public record SpecFormatInfo(
            String name,
            String description,
            List<String> fileExtensions,
            boolean fullySupported) {
    }
}
