package tech.kayys.wayang.tool.parser;

import io.smallrye.mutiny.Uni;
import io.swagger.v3.oas.models.OpenAPI;
import java.util.List;
import io.swagger.v3.parser.OpenAPIV3Parser;
import io.swagger.v3.parser.core.models.ParseOptions;
import io.swagger.v3.parser.core.models.SwaggerParseResult;
import io.vertx.core.http.HttpMethod;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.OpenApiParseException;
import tech.kayys.wayang.tool.dto.OpenApiParseResult;
import tech.kayys.wayang.tool.dto.SourceType;

/**
 * OpenAPI specification parser
 */
@ApplicationScoped
public class OpenApiParser {

    private static final Logger LOG = LoggerFactory.getLogger(OpenApiParser.class);

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    /**
     * Parse OpenAPI from various sources
     */
    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case URL -> parseFromUrl(request.source());
            case FILE -> parseFromFile(request.source());
            case RAW -> parseFromString(request.source());
            case GIT -> parseFromGit(request.source());
            default -> Uni.createFrom().failure(
                    new IllegalArgumentException("Unsupported source type for OpenApiParser: " + request.sourceType()));
        };
    }

    /**
     * Parse from URL
     */
    private Uni<OpenApiParseResult> parseFromUrl(String url) {
        LOG.info("Parsing OpenAPI from URL: {}", url);

        return vertx.createHttpClient()
                .request(HttpMethod.GET, url)
                .flatMap(request -> request.send()
                        .flatMap(response -> {
                            if (response.statusCode() != 200) {
                                return Uni.createFrom().failure(
                                        new OpenApiParseException(
                                                "Failed to fetch OpenAPI spec: HTTP " +
                                                        response.statusCode()));
                            }
                            return response.body();
                        }))
                .map(buffer -> buffer.toString())
                .map(this::parseSpec);
    }

    /**
     * Parse from file
     */
    private Uni<OpenApiParseResult> parseFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString())
                .map(this::parseSpec);
    }

    /**
     * Parse from raw string
     */
    private Uni<OpenApiParseResult> parseFromString(String spec) {
        return Uni.createFrom().item(parseSpec(spec));
    }

    /**
     * Parse from Git repository
     */
    private Uni<OpenApiParseResult> parseFromGit(String gitRef) {
        // TODO: Implement Git integration
        return Uni.createFrom().failure(
                new UnsupportedOperationException("Git source not yet implemented"));
    }

    /**
     * Parse OpenAPI spec string
     */
    private OpenApiParseResult parseSpec(String spec) {
        ParseOptions options = new ParseOptions();
        options.setResolve(true);
        options.setResolveFully(true);

        SwaggerParseResult result = new OpenAPIV3Parser()
                .readContents(spec, null, options);

        if (result.getOpenAPI() == null) {
            return new OpenApiParseResult(
                    null,
                    spec,
                    false,
                    result.getMessages());
        }

        return new OpenApiParseResult(
                result.getOpenAPI(),
                spec,
                result.getMessages() == null || result.getMessages().isEmpty(),
                result.getMessages() != null ? result.getMessages() : List.of());
    }
}