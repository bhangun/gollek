package tech.kayys.wayang.mcp.parser.multispec;

import io.smallrye.mutiny.Uni;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.parser.converter.SwaggerConverter;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.mcp.domain.*;
import tech.kayys.wayang.mcp.dto.GenerateToolsRequest;
import tech.kayys.wayang.mcp.dto.OpenApiParseResult;
import tech.kayys.wayang.mcp.parser.*;

import java.util.List;

/**
 * Swagger 2.0 parser - converts to OpenAPI 3.x
 */
@ApplicationScoped
public class SwaggerV2Parser {

    private static final Logger LOG = LoggerFactory.getLogger(SwaggerV2Parser.class);

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case SWAGGER_2_URL -> parseFromUrl(request.source());
            case SWAGGER_2_FILE -> parseFromFile(request.source());
            case SWAGGER_2_RAW -> parseFromString(request.source());
            default -> Uni.createFrom().failure(
                    new IllegalArgumentException("Invalid Swagger 2.0 source type"));
        };
    }

    private Uni<OpenApiParseResult> parseFromUrl(String url) {
        return vertx.createHttpClient()
                .request(io.vertx.core.http.HttpMethod.GET, url)
                .flatMap(req -> req.send().flatMap(resp -> resp.body()))
                .map(buffer -> buffer.toString())
                .map(this::convertSwagger2ToOpenApi3);
    }

    private Uni<OpenApiParseResult> parseFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString())
                .map(this::convertSwagger2ToOpenApi3);
    }

    public Uni<OpenApiParseResult> parseFromString(String swagger2Spec) {
        return Uni.createFrom().item(convertSwagger2ToOpenApi3(swagger2Spec));
    }

    /**
     * Convert Swagger 2.0 to OpenAPI 3.x using swagger-parser converter
     */
    private OpenApiParseResult convertSwagger2ToOpenApi3(String swagger2Spec) {
        try {
            LOG.info("Converting Swagger 2.0 to OpenAPI 3.x");

            // Use swagger-parser's built-in converter
            SwaggerConverter converter = new SwaggerConverter();
            OpenAPI openApi = converter.readContents(
                    swagger2Spec,
                    null,
                    null).getOpenAPI();

            if (openApi == null) {
                return new OpenApiParseResult(
                        null,
                        swagger2Spec,
                        false,
                        List.of("Failed to convert Swagger 2.0 to OpenAPI 3.x"));
            }

            LOG.info("Successfully converted Swagger 2.0 to OpenAPI 3.x");

            return new OpenApiParseResult(
                    openApi,
                    swagger2Spec,
                    true,
                    List.of());

        } catch (Exception e) {
            LOG.error("Failed to parse Swagger 2.0", e);
            return new OpenApiParseResult(
                    null,
                    swagger2Spec,
                    false,
                    List.of("Swagger 2.0 parse error: " + e.getMessage()));
        }
    }
}