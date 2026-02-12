package tech.kayys.wayang.tool.parser.multispec;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.OpenApiParseResult;
import tech.kayys.wayang.tool.dto.SourceType;

/**
 * ============================================================================
 * MULTI-SPEC PARSER ORCHESTRATOR
 * ============================================================================
 *
 * Supports multiple API specification formats:
 * - OpenAPI 3.x (3.0.0, 3.0.1, 3.0.2, 3.0.3, 3.1.0)
 * - Swagger 2.0
 * - Postman Collection (v2.0, v2.1)
 * - AsyncAPI (2.x, 3.x)
 * - GraphQL Schema
 * - WSDL (SOAP)
 *
 * All specs are normalized to OpenAPI 3.x internally for tool generation.
 */
@ApplicationScoped
public class MultiSpecParserOrchestrator {

    private static final Logger LOG = LoggerFactory.getLogger(MultiSpecParserOrchestrator.class);

    @Inject
    OpenApiV3Parser openApiV3Parser;

    @Inject
    SwaggerV2Parser swaggerV2Parser;

    @Inject
    PostmanCollectionParser postmanParser;

    @Inject
    AsyncApiParser asyncApiParser;

    @Inject
    GraphQLSchemaParser graphqlParser;

    @Inject
    WsdlParser wsdlParser;

    @Inject
    SpecTypeDetector specTypeDetector;

    /**
     * Parse any supported spec format
     */
    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        LOG.info("Parsing spec: type={}, namespace={}",
                request.sourceType(), request.namespace());

        // Auto-detect if source type is generic
        SourceType detectedType = request.sourceType();
        if (isGenericType(detectedType)) {
            return fetchSpec(request)
                    .flatMap(specContent -> specTypeDetector.detect(specContent)
                            .flatMap(detected -> parseWithDetectedType(detected, specContent, request)));
        }

        // Parse based on explicit type
        return parseByType(detectedType, request);
    }

    /**
     * Check if source type needs auto-detection
     */
    private boolean isGenericType(SourceType type) {
        return type == SourceType.OPENAPI_3_URL ||
                type == SourceType.OPENAPI_3_FILE ||
                type == SourceType.OPENAPI_3_RAW;
    }

    /**
     * Parse with detected spec type
     */
    private Uni<OpenApiParseResult> parseWithDetectedType(
            SourceType detectedType,
            String specContent,
            GenerateToolsRequest request) {

        LOG.info("Auto-detected spec type: {}", detectedType);

        return switch (detectedType) {
            case OPENAPI_3_URL, OPENAPI_3_FILE, OPENAPI_3_RAW ->
                openApiV3Parser.parseFromString(specContent);

            case SWAGGER_2_URL, SWAGGER_2_FILE, SWAGGER_2_RAW ->
                swaggerV2Parser.parseFromString(specContent);

            case POSTMAN_URL, POSTMAN_FILE, POSTMAN_RAW ->
                postmanParser.parseFromString(specContent);

            case ASYNCAPI_URL, ASYNCAPI_FILE, ASYNCAPI_RAW ->
                asyncApiParser.parseFromString(specContent);

            case GRAPHQL_URL, GRAPHQL_FILE, GRAPHQL_RAW ->
                graphqlParser.parseFromString(specContent);

            default -> Uni.createFrom().failure(
                    new UnsupportedOperationException(
                            "Unsupported spec type: " + detectedType));
        };
    }

    /**
     * Parse by explicit type
     */
    private Uni<OpenApiParseResult> parseByType(
            SourceType type,
            GenerateToolsRequest request) {

        return switch (type) {
            case OPENAPI_3_URL, OPENAPI_3_FILE, OPENAPI_3_RAW ->
                openApiV3Parser.parse(request);

            case SWAGGER_2_URL, SWAGGER_2_FILE, SWAGGER_2_RAW ->
                swaggerV2Parser.parse(request);

            case POSTMAN_URL, POSTMAN_FILE, POSTMAN_RAW ->
                postmanParser.parse(request);

            case ASYNCAPI_URL, ASYNCAPI_FILE, ASYNCAPI_RAW ->
                asyncApiParser.parse(request);

            case GRAPHQL_URL, GRAPHQL_FILE, GRAPHQL_RAW ->
                graphqlParser.parse(request);

            case WSDL_URL, WSDL_FILE ->
                wsdlParser.parse(request);

            case GIT -> Uni.createFrom().failure(
                    new UnsupportedOperationException("Git not yet implemented"));

            default -> Uni.createFrom().failure(
                    new UnsupportedOperationException("Unknown type: " + type));
        };
    }

    /**
     * Fetch spec content from source
     */
    private Uni<String> fetchSpec(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case OPENAPI_3_URL, SWAGGER_2_URL, POSTMAN_URL,
                    ASYNCAPI_URL, GRAPHQL_URL, WSDL_URL ->
                fetchFromUrl(request.source());

            case OPENAPI_3_FILE, SWAGGER_2_FILE, POSTMAN_FILE,
                    ASYNCAPI_FILE, GRAPHQL_FILE, WSDL_FILE ->
                fetchFromFile(request.source());

            case OPENAPI_3_RAW, SWAGGER_2_RAW, POSTMAN_RAW,
                    ASYNCAPI_RAW, GRAPHQL_RAW ->
                Uni.createFrom().item(request.source());

            default -> Uni.createFrom().failure(
                    new UnsupportedOperationException("Unknown source type"));
        };
    }

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    private Uni<String> fetchFromUrl(String url) {
        return vertx.createHttpClient()
                .request(io.vertx.core.http.HttpMethod.GET, url)
                .flatMap(req -> req.send()
                        .flatMap(resp -> resp.body()))
                .map(buffer -> buffer.toString());
    }

    private Uni<String> fetchFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString());
    }
}