package tech.kayys.wayang.tool.parser.multispec;

import io.smallrye.mutiny.Uni;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.Paths;
import io.swagger.v3.oas.models.info.Info;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.tool.entity.*;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.OpenApiParseResult;
import tech.kayys.wayang.tool.parser.*;

import java.util.List;

/**
 * AsyncAPI parser (v2.x, v3.x)
 * Converts AsyncAPI to OpenAPI-like structure for event-driven APIs
 */
@ApplicationScoped
public class AsyncApiParser {

    private static final Logger LOG = LoggerFactory.getLogger(AsyncApiParser.class);

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case ASYNCAPI_URL -> parseFromUrl(request.source());
            case ASYNCAPI_FILE -> parseFromFile(request.source());
            case ASYNCAPI_RAW -> parseFromString(request.source());
            default -> Uni.createFrom().failure(
                    new IllegalArgumentException("Invalid AsyncAPI source type"));
        };
    }

    private Uni<OpenApiParseResult> parseFromUrl(String url) {
        return vertx.createHttpClient()
                .request(io.vertx.core.http.HttpMethod.GET, url)
                .flatMap(req -> req.send().flatMap(resp -> resp.body()))
                .map(buffer -> buffer.toString())
                .map(this::convertAsyncApiToOpenApi);
    }

    private Uni<OpenApiParseResult> parseFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString())
                .map(this::convertAsyncApiToOpenApi);
    }

    public Uni<OpenApiParseResult> parseFromString(String asyncApiSpec) {
        return Uni.createFrom().item(convertAsyncApiToOpenApi(asyncApiSpec));
    }

    /**
     * Convert AsyncAPI to OpenAPI structure
     * AsyncAPI channels become webhook endpoints
     */
    private OpenApiParseResult convertAsyncApiToOpenApi(String asyncApiSpec) {
        try {
            LOG.info("Converting AsyncAPI to OpenAPI 3.x (webhook style)");

            // Basic conversion - in production use asyncapi-parser library
            OpenAPI openApi = new OpenAPI();
            openApi.setOpenapi("3.0.3");

            // Parse basic info
            Info info = new Info();
            info.setTitle("AsyncAPI Service");
            info.setDescription("Converted from AsyncAPI specification");
            info.setVersion("1.0.0");
            openApi.setInfo(info);

            // For now, return basic structure
            // In production, parse channels as webhook callbacks
            Paths paths = new Paths();
            openApi.setPaths(paths);

            return new OpenApiParseResult(
                    openApi,
                    asyncApiSpec,
                    true,
                    List.of("AsyncAPI basic conversion - webhooks not fully supported yet"));

        } catch (Exception e) {
            LOG.error("Failed to parse AsyncAPI", e);
            return new OpenApiParseResult(
                    null,
                    asyncApiSpec,
                    false,
                    List.of("AsyncAPI parse error: " + e.getMessage()));
        }
    }
}