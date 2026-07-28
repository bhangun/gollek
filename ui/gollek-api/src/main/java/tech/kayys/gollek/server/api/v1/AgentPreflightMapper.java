package tech.kayys.gollek.server.api.v1;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.ws.rs.core.HttpHeaders;
import tech.kayys.gollek.client.agent.AgentFeatureNegotiation;
import tech.kayys.gollek.client.agent.AgentReadinessIssueCodes;
import tech.kayys.gollek.client.agent.AgentReadinessMetadata;
import tech.kayys.gollek.client.agent.AgentServingFeatureCatalog;
import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;
import tech.kayys.gollek.client.agent.AgentServingReadinessReport;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.mcp.McpRegistryManager;
import tech.kayys.gollek.sdk.mcp.McpServerSummary;
import tech.kayys.gollek.sdk.mcp.McpToolModel;
import tech.kayys.gollek.spi.model.ModelInfo;
import java.util.HashMap;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

final class AgentPreflightMapper {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String AREA_CAPABILITIES = "capabilities";
    private static final String AREA_CONTRACT = "contract";
    private static final String AREA_FEATURE_NEGOTIATION = AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION;
    private static final String AREA_MODEL_ROUTE = "model_route";
    private static final String AREA_MCP_DISCOVERY = "mcp_discovery";
    private static final String AREA_TOOL_VALIDATION = "tool_validation";
    private static final String AREA_REQUEST_VALIDATION = "request_validation";

    private static final List<String> REQUIRED_CAPABILITY_COMPATIBILITY =
            AgentServingFeatureCatalog.REQUIRED_FEATURES;
    private static final List<String> REQUIRED_CAPABILITY_ENDPOINTS = List.of(
            "openai_chat_completions",
            "openai_responses",
            "openai_embeddings",
            "model_capabilities",
            "agent_contract",
            "agent_readiness_issues",
            "agent_preflight",
            "agent_validation",
            "agent_tool_validation",
            "mcp_tools");
    private static final List<String> REQUIRED_GOLLEK_OWNS = List.of(
            "model_serving",
            "provider_routing",
            "system_prompt_mapping",
            "tool_schema_ingestion",
            "tool_contract_validation",
            "mcp_registry_discovery",
            "embedding_generation",
            "rag_context_injection");
    private static final List<String> REQUIRED_ORCHESTRATOR_OWNS = List.of(
            "planning",
            "memory_policy",
            "retrieval_policy",
            "vector_store_ownership",
            "tool_authorization",
            "tool_execution",
            "tool_result_loop",
            "workflow_state");
    private static final List<String> REQUIRED_CONTRACT_ENDPOINTS = List.of(
            "chat_completions",
            "responses",
            "embeddings",
            "model_capabilities",
            "mcp_tools",
            "agent_capabilities",
            "agent_contract",
            "agent_readiness_issues",
            "agent_preflight",
            "agent_validation",
            "agent_tool_validation");

    private AgentPreflightMapper() {
    }

    static Map<String, Object> preflight(
            HttpHeaders headers,
            JsonNode payload,
            AgentTraceContext trace,
            GollekSdk sdk) {
        Spec spec = Spec.from(payload);
        JsonNode requestPayload = withModel(spec.request(), spec.modelId());
        String surface = normalizeSurface(firstNonBlank(spec.surface(), inferSurface(requestPayload)));
        String modelId = firstNonBlank(spec.modelId(), text(requestPayload, "model"));

        Map<String, Object> capabilities = AgentCapabilitiesMapper.capabilities();
        Map<String, Object> contract = AgentContractMapper.contract();
        Map<String, Object> modelCapabilities = modelCapabilities(sdk, modelId);
        Map<String, Object> mcpDiscovery = spec.discoverMcpTools()
                ? mcpDiscovery(sdk, spec.openAiToolCompatibility(), spec.enabledOnly())
                : notRequestedMcpDiscovery(spec.openAiToolCompatibility(), spec.enabledOnly());
        Map<String, Object> toolValidation = spec.validateTools()
                ? toolValidation(toolsForValidation(requestPayload, mcpDiscovery), trace)
                : notRequestedToolValidation(trace);
        Map<String, Object> requestValidation = spec.validateRequest()
                ? requestValidation(surface, headers, requestPayload, trace)
                : notRequestedRequestValidation(surface, trace);

        List<Issue> issues = new ArrayList<>();
        collectCapabilityIssues(issues, capabilities);
        collectContractIssues(issues, contract);
        collectFeatureNegotiationIssues(issues, capabilities, contract, spec);
        collectModelRouteIssues(issues, modelCapabilities, modelId, surface);
        collectMcpIssues(issues, mcpDiscovery, spec.discoverMcpTools(), spec.mcpDiscoveryRequired());
        collectToolValidationIssues(issues, toolValidation, spec.validateTools(), spec.toolValidationRequired());
        collectRequestValidationIssues(
                issues,
                requestValidation,
                spec.validateRequest(),
                spec.requestValidationRequired());
        List<Issue> blocking = blockingIssues(issues);
        Map<String, Object> checkResults = checkResults(spec, issues, capabilities, contract, requestValidation);

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", "gollek.agent_preflight");
        out.put("contract_version", AgentServingFeatureCatalog.CONTRACT_VERSION);
        out.put("supported_contract_versions", AgentServingFeatureCatalog.SUPPORTED_CONTRACT_VERSIONS);
        out.put("feature_negotiation", AgentServingFeatureCatalog.featureNegotiation());
        out.put("status", blocking.isEmpty() ? "ready" : "blocked");
        out.put("ready", blocking.isEmpty());
        out.put("surface", surface);
        out.put("model", modelId);
        out.put("feature_profile", spec.featureProfile());
        out.put("trace", trace.asMap());
        out.put("boundary", boundary());
        out.put("checks", checks(spec));
        out.put("check_results", checkResults);
        out.putAll(AgentReadinessMetadata.fromChecks(checkResults));
        out.put("readiness_report", readinessReport(surface, modelId, spec.featureProfile(), trace, checkResults, issues));
        out.put("issues", issueViews(issues));
        out.put("issue_hints", issueHints(issues));
        out.put("capabilities", capabilities);
        out.put("contract", contract);
        out.put("model_capabilities", modelCapabilities);
        out.put("mcp_discovery", mcpDiscovery);
        out.put("tool_validation", toolValidation);
        out.put("request_validation", requestValidation);
        return out;
    }

    static JsonNode tracePayload(JsonNode payload) {
        if (payload != null && payload.path("request").isObject()) {
            return payload.path("request");
        }
        return payload;
    }

    private static Map<String, Object> modelCapabilities(GollekSdk sdk, String modelId) {
        if (isBlank(modelId)) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("model_id", null);
            payload.put("known", false);
            payload.put("available", false);
            payload.put("api_contract", Map.of());
            payload.put("openai_compatibility", Map.of());
            payload.put("inference", Map.of());
            payload.put("tooling", Map.of("tool_execution", false));
            payload.put("rag", Map.of("retrieval_policy", false, "vector_store_ownership", false));
            payload.put("embeddings", unavailableEmbeddings());
            payload.put("modalities", Map.of());
            payload.put("provider_candidates", List.of());
            payload.put("metadata", Map.of());
            return payload;
        }
        return new HashMap<>();
    }

    private static Map<String, Object> unavailableEmbeddings() {
        Map<String, Object> embeddings = new LinkedHashMap<>();
        embeddings.put("generation", false);
        embeddings.put("endpoint", "/v1/embeddings");
        embeddings.put("openai_compatible", false);
        embeddings.put("dimensions", null);
        embeddings.put("encoding_formats", List.of("float", "base64"));
        embeddings.put("input_aliases", List.of("input", "inputs"));
        embeddings.put("batch_inputs", true);
        embeddings.put("metadata_passthrough", true);
        embeddings.put("retrieval_policy", false);
        embeddings.put("vector_store_ownership", false);
        return embeddings;
    }

    private static Optional<ModelInfo> safeModelInfo(GollekSdk sdk, String modelId) {
        try {
            return sdk != null ? sdk.getModelInfo(modelId) : Optional.empty();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static List<Object> safeProviders(GollekSdk sdk) {
        try {
            return List.of();
        } catch (Exception ignored) {
            return List.of();
        }
    }

    private static Optional<String> safePreferredProvider(GollekSdk sdk) {
        try {
            return Optional.empty();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static Map<String, Object> mcpDiscovery(GollekSdk sdk, boolean openAiCompat, boolean enabledOnly) {
        Optional<McpRegistryManager> registry = safeMcpRegistry(sdk);
        if (registry.isEmpty()) {
            Map<String, Object> payload = unavailableMcpDiscovery(openAiCompat, enabledOnly);
            payload.put("message", "MCP registry is not available in this SDK runtime.");
            return payload;
        }
        try {
            List<Map<String, Object>> tools = new ArrayList<>();
            for (McpServerSummary server : registry.get().list()) {
                if (enabledOnly && !server.enabled()) {
                    continue;
                }
                for (McpToolModel tool : registry.get().listTools(server.name())) {
                    tools.add(toolView(server.name(), tool, openAiCompat));
                }
            }
            Map<String, Object> payload = baseMcpDiscovery(true, registry.get().registryPath(), openAiCompat, enabledOnly);
            payload.put("tools", tools);
            return payload;
        } catch (Exception e) {
            Map<String, Object> payload = unavailableMcpDiscovery(openAiCompat, enabledOnly);
            payload.put("error", errorMessage(e, "MCP registry request failed."));
            return payload;
        }
    }

    private static Optional<McpRegistryManager> safeMcpRegistry(GollekSdk sdk) {
        try {
            return sdk != null ? Optional.ofNullable(sdk.mcpRegistry()) : Optional.empty();
        } catch (UnsupportedOperationException | IllegalStateException e) {
            return Optional.empty();
        }
    }

    private static Map<String, Object> unavailableMcpDiscovery(boolean openAiCompat, boolean enabledOnly) {
        Map<String, Object> payload = baseMcpDiscovery(false, null, openAiCompat, enabledOnly);
        payload.put("tools", List.of());
        return payload;
    }

    private static Map<String, Object> notRequestedMcpDiscovery(boolean openAiCompat, boolean enabledOnly) {
        Map<String, Object> payload = unavailableMcpDiscovery(openAiCompat, enabledOnly);
        payload.put("requested", false);
        payload.put("message", "MCP discovery was not requested for this preflight.");
        return payload;
    }

    private static Map<String, Object> baseMcpDiscovery(
            boolean available,
            String registryPath,
            boolean openAiCompat,
            boolean enabledOnly) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("available", available);
        payload.put("registry_path", registryPath);
        payload.put("compat", openAiCompat ? "openai" : "mcp");
        payload.put("enabled_only", enabledOnly);
        payload.put("boundary", mcpBoundary());
        return payload;
    }

    private static Map<String, Object> mcpBoundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("role", "discovery_only");
        boundary.put("gollek_exposes", List.of("registered_servers", "tool_schemas"));
        boundary.put("agent_orchestrator_owns",
                List.of("tool_authorization", "tool_execution", "tool_result_loop"));
        boundary.put("tool_execution", false);
        return boundary;
    }

    private static Map<String, Object> toolView(String serverName, McpToolModel tool, boolean openAiCompat) {
        if (openAiCompat) {
            return openAiToolView(serverName, tool);
        }
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("type", "mcp_tool");
        item.put("server", serverName);
        item.put("name", tool.name());
        item.put("description", tool.description());
        item.put("input_schema", tool.inputSchema());
        item.put("execution", false);
        return item;
    }

    private static Map<String, Object> openAiToolView(String serverName, McpToolModel tool) {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", openAiFunctionName(serverName, tool.name()));
        function.put("description", tool.description());
        function.put("parameters", tool.inputSchema());

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("mcp_server", serverName);
        metadata.put("mcp_tool_name", tool.name());
        metadata.put("tool_execution", false);

        Map<String, Object> item = new LinkedHashMap<>();
        item.put("type", "function");
        item.put("function", function);
        item.put("x_gollek", metadata);
        return item;
    }

    private static String openAiFunctionName(String serverName, String toolName) {
        String sourceServer = isBlank(serverName) ? "server" : serverName;
        String sourceTool = isBlank(toolName) ? "tool" : toolName;
        String name = ("mcp_" + sourceServer + "_" + sourceTool)
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9_-]", "_")
                .replaceAll("_+", "_");
        return name.length() <= 64 ? name : name.substring(0, 64);
    }

    private static JsonNode toolsForValidation(JsonNode requestPayload, Map<String, Object> mcpDiscovery) {
        JsonNode requestTools = requestPayload == null ? null : requestPayload.path("tools");
        if (requestTools != null && requestTools.isArray()) {
            return requestTools;
        }
        JsonNode discovered = MAPPER.convertValue(mcpDiscovery, JsonNode.class).path("tools");
        if (discovered.isArray()) {
            return discovered;
        }
        return MAPPER.createArrayNode();
    }

    private static Map<String, Object> toolValidation(JsonNode tools, AgentTraceContext trace) {
        try {
            return AgentToolContractMapper.validatePayload(tools, trace);
        } catch (IllegalArgumentException | IllegalStateException e) {
            Map<String, Object> payload = invalidToolValidation(trace);
            payload.put("error", errorMessage(e, "Tool contract validation failed."));
            return payload;
        }
    }

    private static Map<String, Object> invalidToolValidation(AgentTraceContext trace) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("object", "gollek.tool_contract_validation");
        payload.put("valid", false);
        payload.put("model_invoked", false);
        payload.put("trace", trace.asMap());
        payload.put("tool_count", 0);
        payload.put("normalized", List.of());
        payload.put("warnings", List.of());
        payload.put("boundary", toolValidationBoundary());
        return payload;
    }

    private static Map<String, Object> notRequestedToolValidation(AgentTraceContext trace) {
        Map<String, Object> payload = invalidToolValidation(trace);
        payload.put("requested", false);
        payload.put("error", "Tool validation was not requested for this preflight.");
        return payload;
    }

    private static Map<String, Object> toolValidationBoundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", true);
        boundary.put("tool_execution", false);
        boundary.put("tool_authorization", false);
        return boundary;
    }

    private static Map<String, Object> requestValidation(
            String surface,
            HttpHeaders headers,
            JsonNode requestPayload,
            AgentTraceContext trace) {
        try {
            return AgentValidationMapper.validate(surface, headers, requestPayload, trace);
        } catch (IllegalArgumentException | IllegalStateException e) {
            Map<String, Object> payload = invalidRequestValidation(surface, trace);
            payload.put("error", errorMessage(e, "Agent request validation failed."));
            return payload;
        }
    }

    private static Map<String, Object> invalidRequestValidation(String surface, AgentTraceContext trace) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("object", "gollek.agent_validation");
        payload.put("surface", surface);
        payload.put("valid", false);
        payload.put("model_invoked", false);
        payload.put("trace", trace.asMap());
        payload.put("normalized", Map.of());
        payload.put("boundary", requestValidationBoundary());
        return payload;
    }

    private static Map<String, Object> notRequestedRequestValidation(String surface, AgentTraceContext trace) {
        Map<String, Object> payload = invalidRequestValidation(surface, trace);
        payload.put("requested", false);
        payload.put("error", "Request validation was not requested for this preflight.");
        return payload;
    }

    private static Map<String, Object> requestValidationBoundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", true);
        boundary.put("tool_execution", false);
        boundary.put("retrieval_execution", false);
        return boundary;
    }

    private static void collectCapabilityIssues(List<Issue> issues, Map<String, Object> capabilities) {
        JsonNode raw = MAPPER.convertValue(capabilities, JsonNode.class);
        if (!"inference_serving_engine".equals(text(raw.path("service_role")))) {
            blocking(issues, AREA_CAPABILITIES, "service_role must be inference_serving_engine");
        }
        if (!supportsContractVersion(raw)) {
            blocking(issues, AREA_CAPABILITIES,
                    "missing supported contract version: " + AgentServingFeatureCatalog.CONTRACT_VERSION);
        }
        List<String> compatibility = AgentFeatureNegotiation.from(raw, MAPPER).allFeatures();
        for (String feature : REQUIRED_CAPABILITY_COMPATIBILITY) {
            if (!compatibility.contains(feature)) {
                blocking(issues, AREA_CAPABILITIES, "missing compatibility: " + feature);
            }
        }
        JsonNode endpoints = raw.path("endpoints");
        for (String endpoint : REQUIRED_CAPABILITY_ENDPOINTS) {
            if (isBlank(text(endpoints.path(endpoint)))) {
                blocking(issues, AREA_CAPABILITIES, "missing endpoint: " + endpoint);
            }
        }
        if (!containsNormalized(strings(raw.path("agent_boundary").path("gollek_owns")), "model serving")) {
            blocking(issues, AREA_CAPABILITIES, "agent boundary must keep model serving in Gollek");
        }
        if (!containsNormalized(strings(raw.path("agent_boundary").path("agent_orchestrator_owns")),
                "tool execution loops")) {
            blocking(issues, AREA_CAPABILITIES, "agent boundary must keep tool execution loops in the orchestrator");
        }
        if (!"X-API-Key".equals(text(raw.path("auth").path("x_api_key_header")))) {
            blocking(issues, AREA_CAPABILITIES, "auth.x_api_key_header must be X-API-Key");
        }
        if (!"Bearer token".equals(text(raw.path("auth").path("authorization_header")))) {
            blocking(issues, AREA_CAPABILITIES, "auth.authorization_header must be Bearer token");
        }
        if (!AgentTraceContext.REQUEST_ID_HEADER.equals(text(raw.path("traceability").path("request_id_header")))) {
            blocking(issues, AREA_CAPABILITIES,
                    "traceability.request_id_header must be " + AgentTraceContext.REQUEST_ID_HEADER);
        }
    }

    private static void collectContractIssues(List<Issue> issues, Map<String, Object> contract) {
        JsonNode raw = MAPPER.convertValue(contract, JsonNode.class);
        if (!"inference_serving_engine".equals(text(raw.path("service_role")))) {
            blocking(issues, AREA_CONTRACT, "service_role must be inference_serving_engine");
        }
        if (!supportsContractVersion(raw)) {
            blocking(issues, AREA_CONTRACT,
                    "missing supported contract version: " + AgentServingFeatureCatalog.CONTRACT_VERSION);
        }
        List<String> gollekOwns = strings(raw.path("boundary").path("gollek_owns"));
        for (String required : REQUIRED_GOLLEK_OWNS) {
            if (!gollekOwns.contains(required)) {
                blocking(issues, AREA_CONTRACT, "missing Gollek-owned responsibility: " + required);
            }
        }
        List<String> orchestratorOwns = strings(raw.path("boundary").path("agent_orchestrator_owns"));
        for (String required : REQUIRED_ORCHESTRATOR_OWNS) {
            if (!orchestratorOwns.contains(required)) {
                blocking(issues, AREA_CONTRACT, "missing orchestrator-owned responsibility: " + required);
            }
        }
        JsonNode endpoints = raw.path("endpoints");
        for (String required : REQUIRED_CONTRACT_ENDPOINTS) {
            JsonNode endpoint = endpoints.path(required);
            if (isBlank(text(endpoint.path("method"))) || isBlank(text(endpoint.path("path")))) {
                blocking(issues, AREA_CONTRACT, "missing endpoint: " + required);
            }
        }
        if (raw.path("boundary").path("tool_execution").asBoolean(false)) {
            blocking(issues, AREA_CONTRACT, "Gollek contract must not enable tool execution");
        }
        if (raw.path("boundary").path("retrieval_execution").asBoolean(false)) {
            blocking(issues, AREA_CONTRACT, "Gollek contract must not enable retrieval execution");
        }
        if (!"[DONE]".equals(text(raw.path("streaming").path("done_sentinel")))
                || !raw.path("streaming").path("chat_completions_events").isArray()
                || !raw.path("streaming").path("responses_events").isArray()) {
            blocking(issues, AREA_CONTRACT, "streaming contract must advertise chat and Responses events");
        }
    }

    private static void collectFeatureNegotiationIssues(
            List<Issue> issues,
            Map<String, Object> capabilities,
            Map<String, Object> contract,
            Spec spec) {
        AgentFeatureNegotiation capabilityNegotiation = AgentFeatureNegotiation.from(capabilities, MAPPER);
        AgentFeatureNegotiation contractNegotiation = AgentFeatureNegotiation.from(contract, MAPPER);
        String requiredVersion = spec.requiredContractVersion();
        if (!spec.featureProfileSupported()) {
            blocking(issues, AREA_FEATURE_NEGOTIATION,
                    "feature profile is not supported: " + spec.featureProfile());
        }
        if (!capabilityNegotiation.supportsContractVersion(requiredVersion)
                || !contractNegotiation.supportsContractVersion(requiredVersion)) {
            blocking(issues, AREA_FEATURE_NEGOTIATION,
                    "required contract version is not supported: " + requiredVersion);
        }
        for (String feature : unsupportedFeatures(capabilityNegotiation, contractNegotiation, spec.requiredFeatures())) {
            blocking(issues, AREA_FEATURE_NEGOTIATION, "required agent feature is not supported: " + feature);
        }
        for (String feature : unsupportedFeatures(capabilityNegotiation, contractNegotiation, spec.optionalFeatures())) {
            warning(issues, AREA_FEATURE_NEGOTIATION, "optional agent feature is not supported: " + feature);
        }
    }

    private static Map<String, Object> featureNegotiationDetails(
            Spec spec,
            Map<String, Object> capabilities,
            Map<String, Object> contract) {
        AgentFeatureNegotiation capabilityNegotiation = AgentFeatureNegotiation.from(capabilities, MAPPER);
        AgentFeatureNegotiation contractNegotiation = AgentFeatureNegotiation.from(contract, MAPPER);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("feature_profile", spec.featureProfile());
        details.put("supported_feature_profiles", AgentServingFeatureProfile.supportedProfileNames());
        details.put("feature_profile_supported", spec.featureProfileSupported());
        details.put("required_contract_version", spec.requiredContractVersion());
        details.put("supported_contract_versions", capabilityNegotiation.supportedContractVersions());
        details.put("contract_supported_contract_versions", contractNegotiation.supportedContractVersions());
        details.put("required_features", spec.requiredFeatures());
        details.put("optional_features", spec.optionalFeatures());
        details.put("available_features", capabilityNegotiation.allFeatures());
        details.put("contract_available_features", contractNegotiation.allFeatures());
        details.put("unsupported_required_features",
                unsupportedFeatures(capabilityNegotiation, contractNegotiation, spec.requiredFeatures()));
        details.put("unsupported_optional_features",
                unsupportedFeatures(capabilityNegotiation, contractNegotiation, spec.optionalFeatures()));
        return details;
    }

    private static List<String> unsupportedFeatures(
            AgentFeatureNegotiation capabilities,
            AgentFeatureNegotiation contract,
            List<String> features) {
        if (features == null || features.isEmpty()) {
            return List.of();
        }
        List<String> unsupported = new ArrayList<>();
        for (String feature : features) {
            if (!capabilities.supportsFeature(feature) || !contract.supportsFeature(feature)) {
                unsupported.add(feature);
            }
        }
        return List.copyOf(unsupported);
    }

    private static void collectModelRouteIssues(
            List<Issue> issues,
            Map<String, Object> matrix,
            String modelId,
            String surface) {
        JsonNode raw = MAPPER.convertValue(matrix, JsonNode.class);
        if (isBlank(modelId)) {
            blocking(issues, AREA_MODEL_ROUTE, "model is required for preflight");
            return;
        }
        if (!raw.path("available").asBoolean(false)) {
            blocking(issues, AREA_MODEL_ROUTE, "model is not available through Gollek serving");
        }
        String normalizedSurface = normalizeSurface(surface);
        if ("embeddings".equals(normalizedSurface)) {
            if (!flag(raw, "api_contract", "embeddings_endpoint")) {
                blocking(issues, AREA_MODEL_ROUTE, "model route does not advertise the embeddings endpoint");
            }
            if (!supportsEmbeddingGeneration(raw)) {
                blocking(issues, AREA_MODEL_ROUTE,
                        "model route does not support embedding generation for /v1/embeddings");
            }
            if (flag(raw, "embeddings", "retrieval_policy")) {
                blocking(issues, AREA_MODEL_ROUTE, "embedding route must not own retrieval policy");
            }
            if (flag(raw, "embeddings", "vector_store_ownership")) {
                blocking(issues, AREA_MODEL_ROUTE, "embedding route must not own vector store state");
            }
        } else {
            if (!flag(raw, "inference", "completion")) {
                blocking(issues, AREA_MODEL_ROUTE, "model does not advertise completion serving");
            }
            if ("chat".equals(normalizedSurface) && !supportsChat(raw)) {
                blocking(issues, AREA_MODEL_ROUTE, "model does not advertise Chat Completions compatibility");
            }
            if ("responses".equals(normalizedSurface) && !supportsResponses(raw)) {
                blocking(issues, AREA_MODEL_ROUTE, "model does not advertise Responses compatibility");
            }
            if (!"chat".equals(normalizedSurface) && !"responses".equals(normalizedSurface)) {
                if (!supportsChat(raw)) {
                    blocking(issues, AREA_MODEL_ROUTE, "model does not advertise Chat Completions compatibility");
                }
                if (!supportsResponses(raw)) {
                    blocking(issues, AREA_MODEL_ROUTE, "model does not advertise Responses compatibility");
                }
            }
            if (!flag(raw, "inference", "system_prompt") && !flag(raw, "api_contract", "system_prompt")) {
                blocking(issues, AREA_MODEL_ROUTE, "model does not advertise system prompt mapping");
            }
            if (!flag(raw, "tooling", "tool_definitions")
                    || !flag(raw, "api_contract", "tools_request_schema")) {
                blocking(issues, AREA_MODEL_ROUTE, "model does not advertise tool definition ingestion");
            }
            if (!flag(raw, "rag", "context_injection") || !flag(raw, "api_contract", "rag_context_injection")) {
                blocking(issues, AREA_MODEL_ROUTE, "model does not advertise RAG context injection");
            }
        }
        if (flag(raw, "tooling", "tool_execution")) {
            blocking(issues, AREA_MODEL_ROUTE, "Gollek model capability must not enable tool execution");
        }
        if (flag(raw, "rag", "retrieval_policy")) {
            blocking(issues, AREA_MODEL_ROUTE, "retrieval policy must stay with the agent orchestrator");
        }
        if (flag(raw, "rag", "vector_store_ownership")) {
            blocking(issues, AREA_MODEL_ROUTE, "vector store ownership must stay with the agent orchestrator");
        }
    }

    private static boolean supportsEmbeddingGeneration(JsonNode raw) {
        return flag(raw, "embeddings", "generation")
                || flag(raw, "openai_compatibility", "embeddings")
                || flag(raw, "modalities", "embeddings");
    }

    private static boolean supportsChat(JsonNode raw) {
        return flag(raw, "openai_compatibility", "chat_completions")
                && flag(raw, "api_contract", "chat_completions");
    }

    private static boolean supportsResponses(JsonNode raw) {
        return flag(raw, "openai_compatibility", "responses")
                && flag(raw, "api_contract", "responses");
    }

    private static void collectMcpIssues(
            List<Issue> issues,
            Map<String, Object> discovery,
            boolean requested,
            boolean required) {
        if (!requested) {
            if (required) {
                blocking(issues, AREA_MCP_DISCOVERY, "MCP discovery was not requested");
            }
            return;
        }
        JsonNode raw = MAPPER.convertValue(discovery, JsonNode.class);
        if (!raw.path("available").asBoolean(false)) {
            if (required) {
                blocking(issues, AREA_MCP_DISCOVERY, "MCP discovery is not available");
            } else {
                warning(issues, AREA_MCP_DISCOVERY, "MCP discovery is not available");
            }
        }
        if (!"discovery_only".equals(text(raw.path("boundary").path("role")))) {
            blocking(issues, AREA_MCP_DISCOVERY, "MCP discovery endpoint crossed the serving boundary");
        }
        if (raw.path("boundary").path("tool_execution").asBoolean(false)) {
            blocking(issues, AREA_MCP_DISCOVERY, "MCP discovery must not enable tool execution");
        }
        JsonNode tools = raw.path("tools");
        if (tools.isArray()) {
            for (JsonNode tool : tools) {
                if (tool.path("execution").asBoolean(false)
                        || tool.path("x_gollek").path("tool_execution").asBoolean(false)) {
                    blocking(issues, AREA_MCP_DISCOVERY,
                            "MCP tool must not be executable from Gollek: " + toolName(tool));
                }
            }
        }
    }

    private static void collectToolValidationIssues(
            List<Issue> issues,
            Map<String, Object> validation,
            boolean requested,
            boolean required) {
        if (!requested) {
            if (required) {
                blocking(issues, AREA_TOOL_VALIDATION, "tool validation was not requested");
            }
            return;
        }
        JsonNode raw = MAPPER.convertValue(validation, JsonNode.class);
        if (!raw.path("valid").asBoolean(false)) {
            blocking(issues, AREA_TOOL_VALIDATION, "tool definitions are not valid");
        }
        if (raw.path("model_invoked").asBoolean(false)) {
            blocking(issues, AREA_TOOL_VALIDATION, "tool validation must not invoke a model");
        }
        if (!raw.path("boundary").path("validation_only").asBoolean(false)) {
            blocking(issues, AREA_TOOL_VALIDATION, "tool validation must be validation-only");
        }
        if (raw.path("boundary").path("tool_execution").asBoolean(false)) {
            blocking(issues, AREA_TOOL_VALIDATION, "tool validation must not enable tool execution");
        }
        if (raw.path("boundary").path("tool_authorization").asBoolean(false)) {
            blocking(issues, AREA_TOOL_VALIDATION, "tool validation must not authorize tools");
        }
        JsonNode warnings = raw.path("warnings");
        if (warnings.isArray()) {
            for (JsonNode warning : warnings) {
                warning(
                        issues,
                        AREA_TOOL_VALIDATION,
                        AgentReadinessIssueCodes.toolSchemaWarningCode(text(warning.path("code"))),
                        warningMessage(warning));
            }
        }
    }

    private static void collectRequestValidationIssues(
            List<Issue> issues,
            Map<String, Object> validation,
            boolean requested,
            boolean required) {
        if (!requested) {
            if (required) {
                blocking(issues, AREA_REQUEST_VALIDATION, "request validation was not requested");
            }
            return;
        }
        JsonNode raw = MAPPER.convertValue(validation, JsonNode.class);
        if (!raw.path("valid").asBoolean(false)) {
            blocking(issues, AREA_REQUEST_VALIDATION, "agent request is not valid");
        }
        if (raw.path("model_invoked").asBoolean(false)) {
            blocking(issues, AREA_REQUEST_VALIDATION, "request validation must not invoke a model");
        }
        if (!raw.path("boundary").path("validation_only").asBoolean(false)) {
            blocking(issues, AREA_REQUEST_VALIDATION, "request validation must be validation-only");
        }
        if (raw.path("boundary").path("tool_execution").asBoolean(false)) {
            blocking(issues, AREA_REQUEST_VALIDATION, "request validation must not enable tool execution");
        }
        if (raw.path("boundary").path("retrieval_execution").asBoolean(false)) {
            blocking(issues, AREA_REQUEST_VALIDATION, "request validation must not enable retrieval execution");
        }
        JsonNode toolContract = raw.path("normalized").path("tool_contract");
        if (toolContract.isObject() && !toolContract.path("valid").asBoolean(false)) {
            blocking(issues, AREA_REQUEST_VALIDATION, "embedded tool contract is not valid");
        }
    }

    private static List<Issue> blockingIssues(List<Issue> issues) {
        return issues.stream().filter(issue -> "blocking".equals(issue.severity())).toList();
    }

    private static List<Map<String, Object>> issueViews(List<Issue> issues) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Issue issue : issues) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("area", issue.area());
            item.put("severity", issue.severity());
            item.put("code", issue.code());
            item.put("message", issue.message());
            out.add(item);
        }
        return out;
    }

    private static List<Map<String, Object>> issueHints(List<Issue> issues) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Issue issue : issues) {
            AgentReadinessIssueCodes.CatalogEntry catalog = AgentReadinessIssueCodes.describe(issue.code());
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("area", issue.area());
            item.put("severity", issue.severity());
            item.put("code", issue.code());
            item.put("message", issue.message());
            item.put("default_severity", catalog.defaultSeverity());
            item.put("summary", catalog.summary());
            item.put("remediation", catalog.remediation());
            out.add(item);
        }
        return out;
    }

    private static List<String> messages(List<Issue> issues) {
        return issues.stream().map(Issue::message).toList();
    }

    private static List<String> codes(List<Issue> issues) {
        return issues.stream().map(Issue::code).toList();
    }

    private static Map<String, Object> boundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", true);
        boundary.put("model_invoked", false);
        boundary.put("tool_execution", false);
        boundary.put("retrieval_execution", false);
        boundary.put("tool_authorization", false);
        boundary.put("agent_orchestrator_owns", List.of(
                "planning",
                "memory_policy",
                "retrieval_policy",
                "vector_store_ownership",
                "tool_authorization",
                "tool_execution",
                "tool_result_loop",
                "workflow_state"));
        return boundary;
    }

    private static Map<String, Object> checks(Spec spec) {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put("discover_mcp_tools", spec.discoverMcpTools());
        checks.put("mcp_discovery_required", spec.mcpDiscoveryRequired());
        checks.put("validate_tools", spec.validateTools());
        checks.put("tool_validation_required", spec.toolValidationRequired());
        checks.put("validate_request", spec.validateRequest());
        checks.put("request_validation_required", spec.requestValidationRequired());
        checks.put("openai_tool_compatibility", spec.openAiToolCompatibility());
        checks.put("enabled_only", spec.enabledOnly());
        checks.put("feature_profile", spec.featureProfile());
        checks.put("required_contract_version", spec.requiredContractVersion());
        checks.put("required_features", spec.requiredFeatures());
        checks.put("optional_features", spec.optionalFeatures());
        return checks;
    }

    private static Map<String, Object> checkResults(
            Spec spec,
            List<Issue> issues,
            Map<String, Object> capabilities,
            Map<String, Object> contract,
            Map<String, Object> requestValidation) {
        Map<String, Object> checks = new LinkedHashMap<>();
        addCheckResult(checks, AREA_CAPABILITIES, true, issues);
        addCheckResult(checks, AREA_CONTRACT, true, issues);
        addCheckResult(
                checks,
                AREA_FEATURE_NEGOTIATION,
                true,
                issues,
                featureNegotiationDetails(spec, capabilities, contract));
        addCheckResult(checks, AREA_MODEL_ROUTE, true, issues);
        addCheckResult(checks, AREA_MCP_DISCOVERY, spec.discoverMcpTools() || spec.mcpDiscoveryRequired(), issues);
        addCheckResult(checks, AREA_TOOL_VALIDATION, spec.validateTools() || spec.toolValidationRequired(), issues);
        addCheckResult(
                checks,
                AREA_REQUEST_VALIDATION,
                spec.validateRequest() || spec.requestValidationRequired(),
                issues,
                requestValidationDetails(requestValidation));
        return checks;
    }

    private static void addCheckResult(
            Map<String, Object> checks,
            String area,
            boolean requested,
            List<Issue> issues) {
        addCheckResult(checks, area, requested, issues, Map.of());
    }

    private static void addCheckResult(
            Map<String, Object> checks,
            String area,
            boolean requested,
            List<Issue> issues,
            Map<String, Object> details) {
        List<Issue> areaIssues = issuesForArea(issues, area);
        List<String> blocking = messages(issuesForSeverity(areaIssues, "blocking"));
        List<String> warnings = messages(issuesForSeverity(areaIssues, "warning"));
        List<String> blockingCodes = codes(issuesForSeverity(areaIssues, "blocking"));
        List<String> warningCodes = codes(issuesForSeverity(areaIssues, "warning"));
        List<Map<String, Object>> hints = issueHints(areaIssues);
        Map<String, Object> remediationSource = Map.of(area, Map.of("issue_hints", hints));
        List<Map<String, Object>> remediationPlan = AgentReadinessMetadata.remediationPlan(remediationSource);
        Map<String, Object> check = new LinkedHashMap<>();
        if (!requested) {
            check.put("status", "skipped");
            check.put("ready", true);
        } else {
            check.put("status", blocking.isEmpty() ? "ready" : "blocked");
            check.put("ready", blocking.isEmpty());
        }
        check.put("requested", requested);
        check.put("blocking_messages", blocking);
        check.put("warning_messages", warnings);
        check.put("blocking_codes", blockingCodes);
        check.put("warning_codes", warningCodes);
        check.put("issue_hints", hints);
        check.put("remediation_plan", remediationPlan);
        check.put("blocking_remediation_plan", AgentReadinessMetadata.blockingRemediationPlan(remediationSource));
        check.put("warning_remediation_plan", AgentReadinessMetadata.warningRemediationPlan(remediationSource));
        check.put("remediation_plan_by_code", AgentReadinessMetadata.remediationPlanByCode(remediationSource));
        if (requested && details != null && !details.isEmpty()) {
            check.put("details", details);
        }
        checks.put(area, check);
    }

    private static Map<String, Object> requestValidationDetails(Map<String, Object> validation) {
        if (validation == null || validation.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> details = new LinkedHashMap<>();
        putIfPresent(details, "surface", validation.get("surface"));
        Map<String, Object> normalized = objectMap(validation.get("normalized"));
        putIfPresent(details, "model", normalized.get("model"));
        details.put("boundary", boundaryDetails(validation.get("boundary"), validation.get("model_invoked")));
        if ("embeddings".equals(validation.get("surface"))) {
            details.put("embedding", embeddingValidationDetails(normalized));
        } else if (!normalized.isEmpty()) {
            details.put("request", requestShapeDetails(normalized));
        }
        return details;
    }

    private static Map<String, Object> boundaryDetails(Object rawBoundary, Object modelInvoked) {
        Map<String, Object> boundary = objectMap(rawBoundary);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("validation_only", bool(boundary.get("validation_only"), true));
        details.put("model_invoked", bool(modelInvoked, false));
        details.put("tool_execution", bool(boundary.get("tool_execution"), false));
        details.put("retrieval_execution", bool(boundary.get("retrieval_execution"), false));
        return details;
    }

    private static Map<String, Object> requestShapeDetails(Map<String, Object> normalized) {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("streaming", bool(normalized.get("streaming"), false));
        request.put("message_count", numberOrDefault(normalized.get("message_count"), 0));
        request.put("input_count", numberOrDefault(normalized.get("input_count"), 0));
        request.put("tool_count", numberOrDefault(normalized.get("tool_count"), 0));
        request.put("parameter_keys", listOrEmpty(normalized.get("parameter_keys")));
        Map<String, Object> rag = objectMap(normalized.get("rag"));
        Map<String, Object> ragDetails = new LinkedHashMap<>();
        ragDetails.put("injected", bool(rag.get("injected"), false));
        ragDetails.put("items", numberOrDefault(rag.get("items"), 0));
        putIfPresent(ragDetails, "alias", rag.get("alias"));
        request.put("rag", ragDetails);
        Map<String, Object> toolContract = objectMap(normalized.get("tool_contract"));
        request.put("tool_contract", Map.of(
                "valid", bool(toolContract.get("valid"), true),
                "warning_count", numberOrDefault(toolContract.get("warning_count"), 0)));
        return request;
    }

    private static Map<String, Object> embeddingValidationDetails(Map<String, Object> normalized) {
        Map<String, Object> embedding = new LinkedHashMap<>();
        embedding.put("input_count", numberOrDefault(normalized.get("input_count"), 0));
        embedding.put("input_lengths", listOrEmpty(normalized.get("input_lengths")));
        putIfPresent(embedding, "requested_dimensions", normalized.get("requested_dimensions"));
        putIfPresent(embedding, "encoding_format", normalized.get("encoding_format"));
        embedding.put("parameter_keys", listOrEmpty(normalized.get("parameter_keys")));
        embedding.put("metadata_keys", sortedKeys(objectMap(normalized.get("metadata"))));
        Map<String, Object> rag = objectMap(normalized.get("rag"));
        Map<String, Object> ragDetails = new LinkedHashMap<>();
        ragDetails.put("embedding_generation", bool(rag.get("embedding_generation"), false));
        ragDetails.put("retrieval_execution", bool(rag.get("retrieval_execution"), false));
        putIfPresent(ragDetails, "retrieval_policy_owned_by", rag.get("retrieval_policy_owned_by"));
        putIfPresent(ragDetails, "vector_store_owned_by", rag.get("vector_store_owned_by"));
        ragDetails.put("storage_owned_by_orchestrator",
                "agent_orchestrator".equals(String.valueOf(rag.get("retrieval_policy_owned_by")))
                        && "agent_orchestrator".equals(String.valueOf(rag.get("vector_store_owned_by"))));
        embedding.put("rag", ragDetails);
        return embedding;
    }

    private static Map<String, Object> readinessReport(
            String surface,
            String modelId,
            String featureProfile,
            AgentTraceContext trace,
            Map<String, Object> checkResults,
            List<Issue> issues) {
        Map<String, Object> report = new LinkedHashMap<>();
        List<Issue> blocking = blockingIssues(issues);
        report.put("object", "gollek.agent_readiness_report");
        report.put("contract_version", AgentServingFeatureCatalog.CONTRACT_VERSION);
        report.put("supported_contract_versions", AgentServingFeatureCatalog.SUPPORTED_CONTRACT_VERSIONS);
        report.put("feature_negotiation", AgentServingFeatureCatalog.featureNegotiation());
        report.put("status", blocking.isEmpty() ? "ready" : "blocked");
        report.put("ready", blocking.isEmpty());
        report.put("surface", surface);
        report.put("model", modelId);
        report.put("feature_profile", featureProfile);
        report.put("trace", trace.asMap());
        report.put("boundary", boundary());
        report.put("checks", checkResults);
        report.putAll(AgentReadinessMetadata.fromChecks(checkResults));
        report.put("issues", issueViews(issues));
        report.put("issue_hints", issueHints(issues));
        return report;
    }

    private static List<Issue> issuesForArea(List<Issue> issues, String area) {
        return issues.stream()
                .filter(issue -> area.equals(issue.area()))
                .toList();
    }

    private static List<Issue> issuesForSeverity(List<Issue> issues, String severity) {
        return issues.stream()
                .filter(issue -> severity.equals(issue.severity()))
                .toList();
    }

    private static JsonNode withModel(JsonNode request, String modelId) {
        ObjectNode copy;
        if (request != null && request.isObject()) {
            copy = request.deepCopy();
        } else {
            copy = MAPPER.createObjectNode();
        }
        if (!copy.hasNonNull("model") && !isBlank(modelId)) {
            copy.put("model", modelId);
        }
        return copy;
    }

    private static void blocking(List<Issue> issues, String area, String message) {
        addIssue(issues, new Issue(area, "blocking", null, message));
    }

    private static void warning(List<Issue> issues, String area, String message) {
        warning(issues, area, null, message);
    }

    private static void warning(List<Issue> issues, String area, String code, String message) {
        addIssue(issues, new Issue(area, "warning", code, message));
    }

    private static void addIssue(List<Issue> issues, Issue issue) {
        if (!issues.contains(issue)) {
            issues.add(issue);
        }
    }

    private static boolean flag(JsonNode raw, String section, String field) {
        return raw.path(section).path(field).asBoolean(false);
    }

    private static String toolName(JsonNode tool) {
        String name = text(tool.path("function").path("name"));
        if (!isBlank(name)) {
            return name;
        }
        name = text(tool.path("name"));
        if (!isBlank(name)) {
            return name;
        }
        return firstNonBlank(text(tool.path("x_gollek").path("mcp_tool_name")), "unknown");
    }

    private static String warningMessage(JsonNode warning) {
        StringBuilder message = new StringBuilder("tool schema warning");
        String code = text(warning.path("code"));
        String path = text(warning.path("path"));
        String detail = text(warning.path("message"));
        if (!isBlank(code)) {
            message.append(" (").append(code).append(")");
        }
        if (!isBlank(path)) {
            message.append(" at ").append(path);
        }
        if (!isBlank(detail)) {
            message.append(": ").append(detail);
        }
        return message.toString();
    }

    private static List<String> strings(JsonNode node) {
        if (node == null || !node.isArray()) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        for (JsonNode item : node) {
            String value = text(item);
            if (!isBlank(value)) {
                out.add(value);
            }
        }
        return List.copyOf(out);
    }

    private static boolean containsNormalized(List<String> values, String value) {
        String expected = normalizeToken(value);
        return values.stream().map(AgentPreflightMapper::normalizeToken).anyMatch(expected::equals);
    }

    private static boolean supportsContractVersion(JsonNode raw) {
        return AgentFeatureNegotiation.from(raw, MAPPER)
                .supportsContractVersion(AgentServingFeatureCatalog.CONTRACT_VERSION);
    }

    private static String inferSurface(JsonNode requestPayload) {
        if (requestPayload == null) {
            return "chat";
        }
        if (requestPayload.has("messages")) {
            return "chat";
        }
        if (requestPayload.has("inputs")) {
            return "embeddings";
        }
        if (requestPayload.has("input")) {
            return "responses";
        }
        return "chat";
    }

    private static String normalizeSurface(String surface) {
        String normalized = isBlank(surface) ? "chat" : surface.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "chat", "chat_completions", "chat.completions" -> "chat";
            case "response", "responses" -> "responses";
            case "embedding", "embeddings" -> "embeddings";
            default -> normalized;
        };
    }

    private static String normalizeToken(String value) {
        return value == null
                ? ""
                : value.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "_").replaceAll("^_|_$", "");
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        String value = node.asText();
        return isBlank(value) ? null : value;
    }

    private static String text(JsonNode node, String field) {
        return node == null ? null : text(node.path(field));
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (!isBlank(value)) {
                return value.trim();
            }
        }
        return null;
    }

    private static boolean bool(JsonNode node, boolean fallback, String... names) {
        if (node == null || names == null) {
            return fallback;
        }
        for (String name : names) {
            JsonNode value = node.path(name);
            if (value.isBoolean()) {
                return value.asBoolean();
            }
        }
        return fallback;
    }

    private static boolean bool(Object value, boolean fallback) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value instanceof String text) {
            if ("true".equalsIgnoreCase(text)) {
                return true;
            }
            if ("false".equalsIgnoreCase(text)) {
                return false;
            }
        }
        return fallback;
    }

    private static Object numberOrDefault(Object value, int fallback) {
        return value instanceof Number ? value : fallback;
    }

    private static List<?> listOrEmpty(Object value) {
        return value instanceof List<?> list ? List.copyOf(list) : List.of();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> objectMap(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (entry.getKey() != null) {
                out.put(entry.getKey().toString(), entry.getValue());
            }
        }
        return out;
    }

    private static List<String> sortedKeys(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return List.of();
        }
        return map.keySet().stream().sorted().toList();
    }

    private static void putIfPresent(Map<String, Object> map, String key, Object value) {
        if (value != null) {
            map.put(key, value);
        }
    }

    private static String errorMessage(Exception error, String fallback) {
        return error == null || isBlank(error.getMessage()) ? fallback : error.getMessage();
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private record Issue(String area, String severity, String code, String message) {
        private Issue {
            area = isBlank(area) ? "general" : area;
            severity = isBlank(severity) ? "blocking" : severity;
            code = AgentReadinessIssueCodes.resolve(code, area, severity, message);
            message = isBlank(message) ? "unspecified issue" : message;
        }
    }

    private record Spec(
            String modelId,
            String surface,
            JsonNode request,
            boolean discoverMcpTools,
            boolean mcpDiscoveryRequired,
            boolean validateTools,
            boolean toolValidationRequired,
            boolean validateRequest,
            boolean requestValidationRequired,
            boolean openAiToolCompatibility,
            boolean enabledOnly,
            String featureProfile,
            boolean featureProfileSupported,
            String requiredContractVersion,
            List<String> requiredFeatures,
            List<String> optionalFeatures) {

        private static Spec from(JsonNode payload) {
            JsonNode root = payload == null || payload.isMissingNode() || payload.isNull()
                    ? MAPPER.createObjectNode()
                    : payload;
            JsonNode request = root.path("request").isObject() ? root.path("request") : root;
            String modelId = firstNonBlank(text(root, "model"), text(root, "model_id"), text(request, "model"));
            String surface = firstNonBlank(text(root, "surface"), text(request, "surface"));
            String featureProfile = AgentServingFeatureProfile.normalizeName(firstNonBlank(
                    text(root, "feature_profile"),
                    text(root, "featureProfile"),
                    text(root, "profile"),
                    AgentServingFeatureProfile.DEFAULT_PROFILE));
            Optional<AgentServingFeatureProfile> profile = AgentServingFeatureProfile.find(featureProfile);
            String requiredContractVersion = firstNonBlank(
                    text(root, "required_contract_version"),
                    text(root, "requiredContractVersion"),
                    text(root, "client_contract_version"),
                    text(root, "clientContractVersion"),
                    AgentServingFeatureCatalog.CONTRACT_VERSION);
            List<String> requiredFeatures = firstNonEmptyStrings(
                    root.path("required_features"),
                    root.path("requiredFeatures"),
                    root.path("client_required_features"),
                    root.path("clientRequiredFeatures"));
            if (requiredFeatures.isEmpty()) {
                requiredFeatures = profile.map(AgentServingFeatureProfile::requiredFeatures)
                        .orElse(AgentServingFeatureCatalog.REQUIRED_FEATURES);
            }
            List<String> optionalFeatures = firstNonEmptyStrings(
                    root.path("optional_features"),
                    root.path("optionalFeatures"),
                    root.path("client_optional_features"),
                    root.path("clientOptionalFeatures"));
            if (optionalFeatures.isEmpty()) {
                optionalFeatures = profile.map(AgentServingFeatureProfile::optionalFeatures).orElse(List.of());
            }
            return new Spec(
                    modelId,
                    surface,
                    request,
                    bool(root, true, "discover_mcp_tools", "discoverMcpTools"),
                    bool(root, true, "mcp_discovery_required", "mcpDiscoveryRequired"),
                    bool(root, true, "validate_tools", "validateTools"),
                    bool(root, true, "tool_validation_required", "toolValidationRequired"),
                    bool(root, true, "validate_request", "validateRequest"),
                    bool(root, true, "request_validation_required", "requestValidationRequired"),
                    bool(root, true, "openai_tool_compatibility", "openAiToolCompatibility"),
                    bool(root, true, "enabled_only", "enabledOnly"),
                    featureProfile,
                    profile.isPresent(),
                    requiredContractVersion,
                    requiredFeatures,
                    optionalFeatures);
        }

        private static List<String> firstNonEmptyStrings(JsonNode... nodes) {
            if (nodes == null) {
                return List.of();
            }
            for (JsonNode node : nodes) {
                List<String> values = strings(node);
                if (!values.isEmpty()) {
                    return values;
                }
            }
            return List.of();
        }
    }
}
