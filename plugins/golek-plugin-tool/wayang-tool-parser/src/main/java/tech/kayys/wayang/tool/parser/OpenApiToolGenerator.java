package tech.kayys.wayang.tool.parser;

import io.quarkus.hibernate.reactive.panache.Panache;
import io.smallrye.mutiny.Uni;
import io.swagger.v3.oas.models.*;
import io.swagger.v3.oas.models.media.*;
import io.swagger.v3.oas.models.parameters.Parameter;
import io.swagger.v3.oas.models.responses.ApiResponse;
import io.swagger.v3.oas.models.security.SecurityScheme;
import io.swagger.v3.parser.OpenAPIV3Parser;
import io.swagger.v3.parser.core.models.ParseOptions;
import io.swagger.v3.parser.core.models.SwaggerParseResult;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.tool.entity.*;
import tech.kayys.wayang.tool.dto.CapabilityLevel;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.HttpMethod;
import tech.kayys.wayang.tool.dto.OpenApiParseException;
import tech.kayys.wayang.tool.dto.OpenApiParseResult;
import tech.kayys.wayang.tool.dto.ParameterLocation;
import tech.kayys.wayang.tool.dto.ParameterMapping;
import tech.kayys.wayang.tool.dto.RetryConfig;
import tech.kayys.wayang.tool.dto.SourceStatus;
import tech.kayys.wayang.tool.dto.SourceType;
import tech.kayys.wayang.tool.dto.ToolGenerationResult;
import tech.kayys.wayang.tool.repository.ToolRepository;
import tech.kayys.wayang.tool.repository.OpenApiSourceRepository;
import tech.kayys.wayang.tool.ToolCapabilityAnalyzer;
import tech.kayys.wayang.tool.ToolGuardrailGenerator;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * ============================================================================
 * OPENAPI PARSER & TOOL GENERATOR
 * ============================================================================
 *
 * Transforms OpenAPI specifications into executable MCP tools.
 *
 * Pipeline:
 * 1. Parse OpenAPI spec (URL, file, or raw)
 * 2. Validate spec structure
 * 3. Extract operations → MCP tools
 * 4. Generate input/output schemas
 * 5. Configure HTTP execution
 * 6. Map authentication
 * 7. Apply guardrails
 * 8. Persist to registry
 */
@ApplicationScoped
public class OpenApiToolGenerator {

    private static final Logger LOG = LoggerFactory.getLogger(OpenApiToolGenerator.class);

    @Inject
    OpenApiParser parser;

    @Inject
    SchemaConverter schemaConverter;

    @Inject
    ToolGuardrailGenerator guardrailGenerator;

    @Inject
    ToolRepository toolRepository;

    @Inject
    OpenApiSourceRepository openApiSourceRepository;

    @Inject
    ToolCapabilityAnalyzer capabilityAnalyzer;

    /**
     * Generate MCP tools from OpenAPI source
     */
    public Uni<ToolGenerationResult> generateTools(GenerateToolsRequest request) {
        LOG.info("Generating tools from OpenAPI source: {} (tenant: {})",
                request.namespace(), request.requestId());

        return Panache.withTransaction(() ->
        // Parse OpenAPI spec
        parser.parse(request)
                .flatMap(parseResult -> {
                    if (!parseResult.isValid()) {
                        return Uni.createFrom().failure(
                                new OpenApiParseException("Invalid OpenAPI spec: " +
                                        parseResult.errors()));
                    }

                    // Create source record
                    return createOpenApiSource(request, parseResult)
                            .flatMap(source ->
                    // Generate tools from operations
                    generateToolsFromSpec(source, parseResult.openApi(), request)
                            .flatMap(tools ->
                    // Persist tools
                    persistTools(tools)
                            .map(persisted -> new ToolGenerationResult(
                                    source.getSourceId(),
                                    source.getNamespace(),
                                    persisted.size(),
                                    persisted.stream()
                                            .map(McpTool::getToolId)
                                            .collect(Collectors.toList()),
                                    List.of()))));
                }));
    }

    /**
     * Create OpenAPI source record
     */
    private Uni<OpenApiSource> createOpenApiSource(
            GenerateToolsRequest request,
            OpenApiParseResult parseResult) {

        return Uni.createFrom().item(() -> {
            OpenApiSource source = new OpenApiSource();
            source.setSourceId(UUID.randomUUID());
            source.setRequestId(request.requestId());
            source.setNamespace(request.namespace());
            source.setDisplayName(parseResult.openApi().getInfo().getTitle());
            source.setSourceType(request.sourceType());
            source.setSourceLocation(request.source());
            source.setSpecContent(request.sourceType() == SourceType.RAW ? request.source() : null);
            source.setSpecVersion(parseResult.openApi().getOpenapi());
            source.setSpecHash(calculateHash(parseResult.rawSpec()));
            source.setDefaultAuthProfileId(request.authProfileId());
            source.setStatus(SourceStatus.ACTIVE);
            source.setCreatedAt(Instant.now());
            source.setUpdatedAt(Instant.now());
            source.setCreatedBy(request.userId());

            return source;
        }).flatMap(source -> openApiSourceRepository.save(source));
    }

    /**
     * Generate tools from OpenAPI operations
     */
    private Uni<List<McpTool>> generateToolsFromSpec(
            OpenApiSource source,
            OpenAPI openApi,
            GenerateToolsRequest request) {

        return Uni.createFrom().item(() -> {
            List<McpTool> tools = new ArrayList<>();

            // Get base URL
            String baseUrl = extractBaseUrl(openApi);

            // Process each path and operation
            if (openApi.getPaths() != null) {
                openApi.getPaths().forEach((path, pathItem) -> {
                    processPathItem(
                            source,
                            openApi,
                            baseUrl,
                            path,
                            pathItem,
                            request,
                            tools);
                });
            }

            LOG.info("Generated {} tools from OpenAPI spec", tools.size());
            return tools;
        });
    }

    /**
     * Process a single path item (with all HTTP methods)
     */
    private void processPathItem(
            OpenApiSource source,
            OpenAPI openApi,
            String baseUrl,
            String path,
            PathItem pathItem,
            GenerateToolsRequest request,
            List<McpTool> tools) {

        // Process each HTTP method
        Map<PathItem.HttpMethod, Operation> operations = pathItem.readOperationsMap();

        operations.forEach((httpMethod, operation) -> {
            try {
                McpTool tool = createToolFromOperation(
                        source,
                        openApi,
                        baseUrl,
                        path,
                        httpMethod,
                        operation,
                        request);
                tools.add(tool);
            } catch (Exception e) {
                LOG.error("Failed to create tool from operation: {} {}",
                        httpMethod, path, e);
            }
        });
    }

    /**
     * Create MCP tool from OpenAPI operation
     */
    private McpTool createToolFromOperation(
            OpenApiSource source,
            OpenAPI openApi,
            String baseUrl,
            String path,
            PathItem.HttpMethod httpMethod,
            Operation operation,
            GenerateToolsRequest request) {

        McpTool tool = new McpTool();

        // Identity
        String operationId = operation.getOperationId() != null ? operation.getOperationId()
                : generateOperationId(httpMethod, path);

        tool.setToolId(request.namespace() + "." + operationId);
        tool.setRequestId(request.requestId());
        tool.setNamespace(request.namespace());
        tool.setName(operationId);
        tool.setOperationId(operationId);

        // Description - use from OpenAPI or generate
        String description = extractDescription(operation);
        tool.setDescription(description);

        // Capabilities - extract from tags or description
        Set<String> capabilities = extractCapabilities(operation, description);
        tool.setCapabilities(capabilities);

        // Tags
        Set<String> tags = operation.getTags() != null ? new HashSet<>(operation.getTags()) : new HashSet<>();
        tool.setTags(tags);

        // Capability level - analyze operation safety
        CapabilityLevel capabilityLevel = capabilityAnalyzer.analyze(
                httpMethod, operation, path);
        tool.setCapabilityLevel(capabilityLevel);
        tool.setReadOnly(httpMethod == PathItem.HttpMethod.GET);

        // Input schema
        Map<String, Object> inputSchema = generateInputSchema(
                openApi, operation, path);
        tool.setInputSchema(inputSchema);

        // Output schema
        Map<String, Object> outputSchema = generateOutputSchema(
                openApi, operation);
        tool.setOutputSchema(outputSchema);

        // HTTP execution config
        HttpExecutionConfig execConfig = createExecutionConfig(
                baseUrl, path, httpMethod, operation, openApi);
        tool.setExecutionConfig(execConfig);

        // Auth profile
        tool.setAuthProfileId(request.authProfileId());

        // Guardrails
        ToolGuardrails guardrails = guardrailGenerator.generate(
                httpMethod, operation, request.guardrailsConfig());
        tool.setGuardrails(guardrails);

        // Metadata
        tool.setSource(source);
        tool.setCreatedAt(Instant.now());
        tool.setUpdatedAt(Instant.now());
        tool.setCreatedBy(request.userId());
        tool.setEnabled(true);

        // Metrics
        tool.setMetrics(new ToolMetrics());

        return tool;
    }

    /**
     * Generate input schema from parameters and request body
     */
    private Map<String, Object> generateInputSchema(
            OpenAPI openApi,
            Operation operation,
            String path) {

        Map<String, Object> schema = new HashMap<>();
        schema.put("type", "object");

        Map<String, Object> properties = new HashMap<>();
        List<String> required = new ArrayList<>();

        // Path parameters
        if (operation.getParameters() != null) {
            for (Parameter param : operation.getParameters()) {
                if (param.getIn().equals("path") ||
                        param.getIn().equals("query") ||
                        param.getIn().equals("header")) {

                    Map<String, Object> paramSchema = schemaConverter.convert(param.getSchema());
                    paramSchema.put("description", param.getDescription());
                    properties.put(param.getName(), paramSchema);

                    if (Boolean.TRUE.equals(param.getRequired())) {
                        required.add(param.getName());
                    }
                }
            }
        }

        // Request body
        if (operation.getRequestBody() != null) {
            Content content = operation.getRequestBody().getContent();
            if (content != null) {
                MediaType mediaType = content.get("application/json");
                if (mediaType != null && mediaType.getSchema() != null) {
                    Map<String, Object> bodySchema = schemaConverter.convert(mediaType.getSchema());

                    // Merge body schema properties
                    if (bodySchema.get("properties") instanceof Map) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> bodyProps = (Map<String, Object>) bodySchema.get("properties");
                        properties.putAll(bodyProps);

                        // Add required fields from body
                        if (bodySchema.get("required") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<String> bodyRequired = (List<String>) bodySchema.get("required");
                            required.addAll(bodyRequired);
                        }
                    }
                }
            }
        }

        schema.put("properties", properties);
        if (!required.isEmpty()) {
            schema.put("required", required);
        }

        return schema;
    }

    /**
     * Generate output schema from responses
     */
    private Map<String, Object> generateOutputSchema(
            OpenAPI openApi,
            Operation operation) {

        if (operation.getResponses() == null) {
            return Map.of("type", "object");
        }

        // Use 200/201 response schema
        ApiResponse successResponse = operation.getResponses().get("200");
        if (successResponse == null) {
            successResponse = operation.getResponses().get("201");
        }

        if (successResponse != null && successResponse.getContent() != null) {
            MediaType mediaType = successResponse.getContent().get("application/json");
            if (mediaType != null && mediaType.getSchema() != null) {
                return schemaConverter.convert(mediaType.getSchema());
            }
        }

        return Map.of("type", "object");
    }

    /**
     * Create HTTP execution configuration
     */
    private HttpExecutionConfig createExecutionConfig(
            String baseUrl,
            String path,
            PathItem.HttpMethod httpMethod,
            Operation operation,
            OpenAPI openApi) {

        HttpExecutionConfig config = new HttpExecutionConfig();

        // HTTP method
        config.setMethod(HttpMethod.valueOf(httpMethod.name()));
        config.setBaseUrl(baseUrl);
        config.setPath(path);

        // Parameters
        List<ParameterMapping> parameters = new ArrayList<>();
        if (operation.getParameters() != null) {
            for (Parameter param : operation.getParameters()) {
                ParameterMapping mapping = new ParameterMapping();
                mapping.setName(param.getName());
                mapping.setMappedName(param.getName());
                mapping.setLocation(ParameterLocation.valueOf(
                        param.getIn().toUpperCase()));
                mapping.setRequired(Boolean.TRUE.equals(param.getRequired()));
                mapping.setDescription(param.getDescription());
                parameters.add(mapping);
            }
        }
        config.setParameters(parameters);

        // Headers
        Map<String, String> headers = new HashMap<>();
        headers.put("User-Agent", "Gamelan-MCP/1.0");
        config.setHeaders(headers);

        // Content type
        if (operation.getRequestBody() != null &&
                operation.getRequestBody().getContent() != null) {
            String contentType = operation.getRequestBody().getContent()
                    .keySet().stream().findFirst().orElse("application/json");
            config.setContentType(contentType);
        }

        // Retry config - using a simple JSON string representation
        config.setRetryConfig("{\"maxRetries\": 3, \"delay\": 1000, \"jitter\": 0.1}");

        return config;
    }

    /**
     * Extract description from operation (use docs as capabilities)
     */
    private String extractDescription(Operation operation) {
        if (operation.getSummary() != null) {
            return operation.getSummary();
        }
        if (operation.getDescription() != null) {
            return operation.getDescription();
        }
        if (operation.getOperationId() != null) {
            return humanizeOperationId(operation.getOperationId());
        }
        return "No description available";
    }

    /**
     * Extract capabilities from operation
     * Use description/summary as capability if available
     */
    private Set<String> extractCapabilities(Operation operation, String description) {
        Set<String> capabilities = new HashSet<>();

        // Use summary as primary capability
        if (operation.getSummary() != null) {
            capabilities.add(operation.getSummary().toLowerCase());
        }

        // Add tags as capabilities
        if (operation.getTags() != null) {
            capabilities.addAll(operation.getTags().stream()
                    .map(String::toLowerCase)
                    .collect(Collectors.toSet()));
        }

        // If no capabilities found, use operation ID
        if (capabilities.isEmpty() && operation.getOperationId() != null) {
            capabilities.add(humanizeOperationId(operation.getOperationId()));
        }

        return capabilities;
    }

    /**
     * Extract base URL from OpenAPI servers
     */
    private String extractBaseUrl(OpenAPI openApi) {
        if (openApi.getServers() != null && !openApi.getServers().isEmpty()) {
            return openApi.getServers().get(0).getUrl();
        }
        return "https://api.example.com"; // Placeholder
    }

    /**
     * Generate operation ID if not provided
     */
    private String generateOperationId(PathItem.HttpMethod method, String path) {
        String cleaned = path.replaceAll("[^a-zA-Z0-9]", "_");
        return method.name().toLowerCase() + cleaned;
    }

    /**
     * Humanize operation ID for display
     */
    private String humanizeOperationId(String operationId) {
        return operationId
                .replaceAll("([a-z])([A-Z])", "$1 $2")
                .replaceAll("_", " ")
                .toLowerCase();
    }

    /**
     * Calculate hash of spec for change detection
     */
    private String calculateHash(String content) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(content.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(hash);
        } catch (Exception e) {
            return UUID.randomUUID().toString();
        }
    }

    /**
     * Persist generated tools
     */
    private Uni<List<McpTool>> persistTools(List<McpTool> tools) {
        return Uni.join().all(
                tools.stream()
                        .map(tool -> toolRepository.save(tool))
                        .collect(Collectors.toList()))
                .andFailFast();
    }
}