package tech.kayys.wayang.mcp.parser.multispec;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.mcp.dto.SourceType;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Automatically detects API specification type
 */
@ApplicationScoped
public class SpecTypeDetector {

    private static final Logger LOG = LoggerFactory.getLogger(SpecTypeDetector.class);

    public Uni<SourceType> detect(String specContent) {
        LOG.debug("Detect: " + specContent);
        return Uni.createFrom().item(() -> {
            // Try to parse as JSON/YAML first
            SpecFormat format = detectFormat(specContent);

            if (format == SpecFormat.JSON) {
                return detectJsonSpec(specContent);
            } else if (format == SpecFormat.YAML) {
                return detectYamlSpec(specContent);
            } else if (format == SpecFormat.XML) {
                return detectXmlSpec(specContent);
            } else if (format == SpecFormat.GRAPHQL) {
                return SourceType.GRAPHQL_RAW;
            }

            throw new IllegalArgumentException("Unknown spec format");
        });
    }

    /**
     * Detect spec format (JSON, YAML, XML, GraphQL)
     */
    private SpecFormat detectFormat(String content) {
        String trimmed = content.trim();

        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            return SpecFormat.JSON;
        } else if (trimmed.startsWith("<?xml") || trimmed.startsWith("<wsdl:") ||
                trimmed.startsWith("<definitions")) {
            return SpecFormat.XML;
        } else if (trimmed.contains("type Query") ||
                trimmed.contains("type Mutation") ||
                trimmed.contains("schema {")) {
            return SpecFormat.GRAPHQL;
        } else {
            return SpecFormat.YAML;
        }
    }

    /**
     * Detect JSON-based spec type
     */
    private SourceType detectJsonSpec(String content) {
        // Check for Postman Collection
        if (content.contains("\"info\"") && content.contains("\"item\"") &&
                (content.contains("\"postman_collection\"") ||
                        content.contains("\"schema\": \"https://schema.getpostman.com/"))) {
            return SourceType.POSTMAN_RAW;
        }

        // Check for AsyncAPI
        if (content.contains("\"asyncapi\"")) {
            return SourceType.ASYNCAPI_RAW;
        }

        // Check for OpenAPI 3.x
        if (content.contains("\"openapi\"") &&
                (content.contains("\"3.0") || content.contains("\"3.1"))) {
            return SourceType.OPENAPI_3_RAW;
        }

        // Check for Swagger 2.0
        if (content.contains("\"swagger\"") && content.contains("\"2.0\"")) {
            return SourceType.SWAGGER_2_RAW;
        }

        throw new IllegalArgumentException("Unknown JSON spec type");
    }

    /**
     * Detect YAML-based spec type
     */
    private SourceType detectYamlSpec(String content) {
        String lower = content.toLowerCase();

        // Check for AsyncAPI
        if (lower.contains("asyncapi:")) {
            return SourceType.ASYNCAPI_RAW;
        }

        // Check for OpenAPI 3.x
        if (lower.contains("openapi:") &&
                (lower.contains("3.0") || lower.contains("3.1"))) {
            return SourceType.OPENAPI_3_RAW;
        }

        // Check for Swagger 2.0
        if (lower.contains("swagger:") && lower.contains("2.0")) {
            return SourceType.SWAGGER_2_RAW;
        }

        throw new IllegalArgumentException("Unknown YAML spec type");
    }

    /**
     * Detect XML-based spec type
     */
    private SourceType detectXmlSpec(String content) {
        if (content.contains("wsdl:") || content.contains("<definitions")) {
            return SourceType.WSDL_URL;
        }

        throw new IllegalArgumentException("Unknown XML spec type");
    }

    enum SpecFormat {
        JSON, YAML, XML, GRAPHQL
    }
}