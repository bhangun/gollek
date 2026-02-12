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
 * WSDL (SOAP) parser
 * Converts WSDL definitions to OpenAPI 3.x representation
 */
@ApplicationScoped
public class WsdlParser {

    private static final Logger LOG = LoggerFactory.getLogger(WsdlParser.class);

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case WSDL_URL -> parseFromUrl(request.source());
            case WSDL_FILE -> parseFromFile(request.source());
            default -> Uni.createFrom().failure(
                    new IllegalArgumentException("Invalid WSDL source type"));
        };
    }

    private Uni<OpenApiParseResult> parseFromUrl(String url) {
        return vertx.createHttpClient()
                .request(io.vertx.core.http.HttpMethod.GET, url)
                .flatMap(req -> req.send().flatMap(resp -> resp.body()))
                .map(buffer -> buffer.toString())
                .map(this::convertWsdlToOpenApi);
    }

    private Uni<OpenApiParseResult> parseFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString())
                .map(this::convertWsdlToOpenApi);
    }

    /**
     * Convert WSDL to OpenAPI 3.x structure
     * SOAP operations become REST endpoints
     */
    private OpenApiParseResult convertWsdlToOpenApi(String wsdlContent) {
        try {
            LOG.info("Converting WSDL to OpenAPI 3.x");

            // Basic conversion - in production use wsdl4j or similar
            OpenAPI openApi = new OpenAPI();
            openApi.setOpenapi("3.0.3");

            // Parse basic info
            Info info = new Info();
            info.setTitle("WSDL Service");
            info.setDescription("Converted from WSDL definition");
            info.setVersion("1.0.0");
            openApi.setInfo(info);

            // For now, return basic structure
            // In production, parse WSDL portTypes, bindings, and operations
            Paths paths = new Paths();
            openApi.setPaths(paths);

            return new OpenApiParseResult(
                    openApi,
                    wsdlContent,
                    true,
                    List.of("WSDL basic conversion - operations not fully supported yet"));

        } catch (Exception e) {
            LOG.error("Failed to parse WSDL", e);
            return new OpenApiParseResult(
                    null,
                    wsdlContent,
                    false,
                    List.of("WSDL parse error: " + e.getMessage()));
        }
    }
}