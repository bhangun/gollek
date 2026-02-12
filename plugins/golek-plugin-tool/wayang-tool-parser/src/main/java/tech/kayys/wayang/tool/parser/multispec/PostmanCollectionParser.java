package tech.kayys.wayang.tool.parser.multispec;

import io.smallrye.mutiny.Uni;
import io.swagger.v3.oas.models.*;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.media.Content;
import io.swagger.v3.oas.models.media.MediaType;
import io.swagger.v3.oas.models.media.ObjectSchema;
import io.swagger.v3.oas.models.media.StringSchema;
import io.swagger.v3.oas.models.parameters.Parameter;
import io.swagger.v3.oas.models.responses.ApiResponse;
import io.swagger.v3.oas.models.responses.ApiResponses;
import io.swagger.v3.oas.models.servers.Server;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.OpenApiParseResult;
import java.util.ArrayList;
import java.util.List;

/**
 * Postman Collection parser (v2.0, v2.1)
 * Converts Postman collections to OpenAPI 3.x
 */
@ApplicationScoped
public class PostmanCollectionParser {

    private static final Logger LOG = LoggerFactory.getLogger(PostmanCollectionParser.class);

    @Inject
    io.vertx.mutiny.core.Vertx vertx;

    @Inject
    com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    public Uni<OpenApiParseResult> parse(GenerateToolsRequest request) {
        return switch (request.sourceType()) {
            case POSTMAN_URL -> parseFromUrl(request.source());
            case POSTMAN_FILE -> parseFromFile(request.source());
            case POSTMAN_RAW -> parseFromString(request.source());
            default -> Uni.createFrom().failure(
                    new IllegalArgumentException("Invalid Postman source type"));
        };
    }

    private Uni<OpenApiParseResult> parseFromUrl(String url) {
        return vertx.createHttpClient()
                .request(io.vertx.core.http.HttpMethod.GET, url)
                .flatMap(req -> req.send().flatMap(resp -> resp.body()))
                .map(buffer -> buffer.toString())
                .map(this::convertPostmanToOpenApi);
    }

    private Uni<OpenApiParseResult> parseFromFile(String filePath) {
        return vertx.fileSystem().readFile(filePath)
                .map(buffer -> buffer.toString())
                .map(this::convertPostmanToOpenApi);
    }

    public Uni<OpenApiParseResult> parseFromString(String postmanSpec) {
        return Uni.createFrom().item(convertPostmanToOpenApi(postmanSpec));
    }

    /**
     * Convert Postman Collection to OpenAPI 3.x
     */
    private OpenApiParseResult convertPostmanToOpenApi(String postmanJson) {
        try {
            LOG.info("Converting Postman Collection to OpenAPI 3.x");

            // Parse Postman collection
            PostmanCollection collection = objectMapper.readValue(
                    postmanJson,
                    PostmanCollection.class);

            // Create OpenAPI structure
            OpenAPI openApi = new OpenAPI();
            openApi.setOpenapi("3.0.3");

            // Info
            Info info = new Info();
            info.setTitle(collection.info.name);
            info.setDescription(collection.info.description);
            info.setVersion("1.0.0");
            openApi.setInfo(info);

            // Server
            if (collection.variable != null) {
                String baseUrl = extractBaseUrl(collection.variable);
                if (baseUrl != null) {
                    Server server = new Server();
                    server.setUrl(baseUrl);
                    openApi.setServers(List.of(server));
                }
            }

            // Paths
            Paths paths = new Paths();
            processPostmanItems(collection.item, paths);
            openApi.setPaths(paths);

            LOG.info("Successfully converted Postman Collection to OpenAPI 3.x");

            return new OpenApiParseResult(
                    openApi,
                    postmanJson,
                    true,
                    List.of());

        } catch (Exception e) {
            LOG.error("Failed to parse Postman Collection", e);
            return new OpenApiParseResult(
                    null,
                    postmanJson,
                    false,
                    List.of("Postman parse error: " + e.getMessage()));
        }
    }

    /**
     * Process Postman items recursively
     */
    private void processPostmanItems(
            List<PostmanItem> items,
            Paths paths) {

        if (items == null)
            return;

        for (PostmanItem item : items) {
            if (item.request != null) {
                // Leaf item with request
                addRequestToPath(item, paths);
            } else if (item.item != null) {
                // Folder - recurse
                processPostmanItems(item.item, paths);
            }
        }
    }

    /**
     * Add Postman request to OpenAPI path
     */
    private void addRequestToPath(PostmanItem item, Paths paths) {
        if (item.request == null || item.request.url == null) {
            return;
        }

        String path = extractPath(item.request.url);
        io.swagger.v3.oas.models.PathItem.HttpMethod method = io.swagger.v3.oas.models.PathItem.HttpMethod.valueOf(
                item.request.method.toUpperCase());

        io.swagger.v3.oas.models.PathItem pathItem = paths.get(path);
        if (pathItem == null) {
            pathItem = new io.swagger.v3.oas.models.PathItem();
            paths.addPathItem(path, pathItem);
        }

        io.swagger.v3.oas.models.Operation operation = createOperation(item);
        pathItem.operation(method, operation);
    }

    /**
     * Create OpenAPI operation from Postman request
     */
    private io.swagger.v3.oas.models.Operation createOperation(PostmanItem item) {
        io.swagger.v3.oas.models.Operation operation = new io.swagger.v3.oas.models.Operation();

        // Summary and description
        operation.setSummary(item.name);
        if (item.request.description != null) {
            operation.setDescription(item.request.description);
        }

        // Operation ID
        String operationId = item.name
                .toLowerCase()
                .replaceAll("[^a-z0-9]", "_");
        operation.setOperationId(operationId);

        // Parameters
        if (item.request.url != null && item.request.url.query != null) {
            List<Parameter> parameters = new ArrayList<>();
            for (PostmanQueryParam qp : item.request.url.query) {
                Parameter param = new Parameter();
                param.setName(qp.key);
                param.setIn("query");
                param.setDescription(qp.description);
                param.setRequired(false);

                io.swagger.v3.oas.models.media.Schema<?> schema = new StringSchema();
                param.setSchema(schema);

                parameters.add(param);
            }
            operation.setParameters(parameters);
        }

        // Request body
        if (item.request.body != null &&
                item.request.body.mode != null &&
                item.request.body.mode.equals("raw")) {

            io.swagger.v3.oas.models.parameters.RequestBody requestBody = new io.swagger.v3.oas.models.parameters.RequestBody();

            Content content = new Content();
            MediaType mediaType = new MediaType();

            // Try to infer schema from example
            io.swagger.v3.oas.models.media.Schema<?> schema = new ObjectSchema();
            mediaType.setSchema(schema);

            content.addMediaType("application/json", mediaType);
            requestBody.setContent(content);

            operation.setRequestBody(requestBody);
        }

        // Response
        ApiResponses responses = new ApiResponses();
        ApiResponse response200 = new ApiResponse();
        response200.setDescription("Successful response");
        responses.addApiResponse("200", response200);
        operation.setResponses(responses);

        return operation;
    }

    /**
     * Extract path from Postman URL
     */
    private String extractPath(PostmanUrl url) {
        if (url.raw != null) {
            try {
                java.net.URI uri = new java.net.URI(url.raw);
                String path = uri.getPath();
                return path != null && !path.isEmpty() ? path : "/";
            } catch (Exception e) {
                // Fallback
            }
        }

        if (url.path != null && !url.path.isEmpty()) {
            return "/" + String.join("/", url.path);
        }

        return "/unknown";
    }

    /**
     * Extract base URL from variables
     */
    private String extractBaseUrl(List<PostmanVariable> variables) {
        for (PostmanVariable var : variables) {
            if ("baseUrl".equals(var.key) || "base_url".equals(var.key)) {
                return var.value;
            }
        }
        return null;
    }

    // ==================== POSTMAN DATA STRUCTURES ====================

    static class PostmanCollection {
        public PostmanInfo info;
        public List<PostmanItem> item;
        public List<PostmanVariable> variable;
    }

    static class PostmanInfo {
        public String name;
        public String description;
        public String version;
    }

    static class PostmanItem {
        public String name;
        public PostmanRequest request;
        public List<PostmanItem> item; // For folders
    }

    static class PostmanRequest {
        public String method;
        public PostmanUrl url;
        public PostmanBody body;
        public String description;
        public List<PostmanHeader> header;
    }

    static class PostmanUrl {
        public String raw;
        public List<String> path;
        public List<PostmanQueryParam> query;
    }

    static class PostmanQueryParam {
        public String key;
        public String value;
        public String description;
    }

    static class PostmanBody {
        public String mode;
        public String raw;
    }

    static class PostmanHeader {
        public String key;
        public String value;
    }

    static class PostmanVariable {
        public String key;
        public String value;
    }
}