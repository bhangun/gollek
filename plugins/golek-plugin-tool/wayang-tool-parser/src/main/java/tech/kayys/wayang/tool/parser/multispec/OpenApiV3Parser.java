package tech.kayys.wayang.mcp.parser.multispec;

import io.smallrye.mutiny.Uni;
import io.swagger.v3.parser.OpenAPIV3Parser;
import io.swagger.v3.parser.core.models.ParseOptions;
import io.swagger.v3.parser.core.models.SwaggerParseResult;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.mcp.dto.GenerateToolsRequest;
import tech.kayys.wayang.mcp.dto.OpenApiParseResult;
import java.util.List;

/**
 * OpenAPI 3.x parser
 * Parses OpenAPI 3.x specifications
 */
@ApplicationScoped
public class OpenApiV3Parser {

    private static final Logger LOG = LoggerFactory.getLogger(OpenApiV3Parser.class);

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case OPENAPI_3_URL -> parseFromUrl(request.source());
            case OPENAPI_3_FILE -> parseFromFile(request.source());
            case OPENAPI_3_RAW -> parseFromString(request.source());
            default -> Uni.createFrom().failure(
                    new IllegalArgumentException("Invalid OpenAPI 3.x source type"));
        };
    }

    private Uni<OpenApiParseResult> parseFromUrl(String url) {
        return vertx.createHttpClient()
                .request(io.vertx.core.http.HttpMethod.GET, url)
                .flatMap(req -> req.send().flatMap(resp -> resp.body()))
                .map(buffer -> buffer.toString())
                .map(this::parseOpenApi3);
    }

    private Uni<OpenApiParseResult> parseFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString())
                .map(this::parseOpenApi3);
    }

    public Uni<OpenApiParseResult> parseFromString(String openApiSpec) {
        return Uni.createFrom().item(parseOpenApi3(openApiSpec));
    }

    /**
     * Parse OpenAPI 3.x specification
     */
    private OpenApiParseResult parseOpenApi3(String openApiSpec) {
        try {
            LOG.info("Parsing OpenAPI 3.x specification");

            ParseOptions options = new ParseOptions();
            options.setResolve(true);
            options.setResolveFully(true);

            SwaggerParseResult result = new OpenAPIV3Parser()
                    .readContents(openApiSpec, null, options);

            if (result.getOpenAPI() == null) {
                return new OpenApiParseResult(
                        null,
                        openApiSpec,
                        false,
                        result.getMessages());
            }

            LOG.info("Successfully parsed OpenAPI 3.x specification");

            return new OpenApiParseResult(
                    result.getOpenAPI(),
                    openApiSpec,
                    result.getMessages() == null || result.getMessages().isEmpty(),
                    result.getMessages() != null ? result.getMessages() : List.of());

        } catch (Exception e) {
            LOG.error("Failed to parse OpenAPI 3.x", e);
            return new OpenApiParseResult(
                    null,
                    openApiSpec,
                    false,
                    List.of("OpenAPI 3.x parse error: " + e.getMessage()));
        }
    }
}