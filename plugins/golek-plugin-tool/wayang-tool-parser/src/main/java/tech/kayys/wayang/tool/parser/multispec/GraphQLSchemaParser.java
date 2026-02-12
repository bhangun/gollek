package tech.kayys.wayang.tool.parser.multispec;

import io.smallrye.mutiny.Uni;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.Paths;
import io.swagger.v3.oas.models.info.Info;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.OpenApiParseResult;
import java.util.List;

/**
 * GraphQL Schema parser
 * Converts GraphQL schema to OpenAPI 3.x representation
 */
@ApplicationScoped
public class GraphQLSchemaParser {

    private static final Logger LOG = LoggerFactory.getLogger(GraphQLSchemaParser.class);

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case GRAPHQL_URL -> parseFromUrl(request.source());
            case GRAPHQL_FILE -> parseFromFile(request.source());
            case GRAPHQL_RAW -> parseFromString(request.source());
            default -> Uni.createFrom().failure(
                    new IllegalArgumentException("Invalid GraphQL source type"));
        };
    }

    private Uni<OpenApiParseResult> parseFromUrl(String url) {
        return vertx.createHttpClient()
                .request(io.vertx.core.http.HttpMethod.GET, url)
                .flatMap(req -> req.send().flatMap(resp -> resp.body()))
                .map(buffer -> buffer.toString())
                .map(this::convertGraphQLToOpenApi);
    }

    private Uni<OpenApiParseResult> parseFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString())
                .map(this::convertGraphQLToOpenApi);
    }

    public Uni<OpenApiParseResult> parseFromString(String graphQLSchema) {
        return Uni.createFrom().item(convertGraphQLToOpenApi(graphQLSchema));
    }

    /**
     * Convert GraphQL schema to OpenAPI 3.x structure
     * GraphQL queries/mutations become GET/POST endpoints
     */
    private OpenApiParseResult convertGraphQLToOpenApi(String graphQLSchema) {
        try {
            LOG.info("Converting GraphQL schema to OpenAPI 3.x");

            // Basic conversion - in production use graphql-java tools
            OpenAPI openApi = new OpenAPI();
            openApi.setOpenapi("3.0.3");

            // Parse basic info
            Info info = new Info();
            info.setTitle("GraphQL Service");
            info.setDescription("Converted from GraphQL schema");
            info.setVersion("1.0.0");
            openApi.setInfo(info);

            // For now, return basic structure
            // In production, parse GraphQL types and operations
            Paths paths = new Paths();
            openApi.setPaths(paths);

            return new OpenApiParseResult(
                    openApi,
                    graphQLSchema,
                    true,
                    List.of("GraphQL basic conversion - operations not fully supported yet"));

        } catch (Exception e) {
            LOG.error("Failed to parse GraphQL schema", e);
            return new OpenApiParseResult(
                    null,
                    graphQLSchema,
                    false,
                    List.of("GraphQL parse error: " + e.getMessage()));
        }
    }
}