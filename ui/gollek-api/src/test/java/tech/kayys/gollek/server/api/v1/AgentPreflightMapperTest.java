package tech.kayys.gollek.server.api.v1;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.sdk.mcp.McpAddRequest;
import tech.kayys.gollek.sdk.mcp.McpDoctorReport;
import tech.kayys.gollek.sdk.mcp.McpEditRequest;
import tech.kayys.gollek.sdk.mcp.McpRegistryManager;
import tech.kayys.gollek.sdk.mcp.McpServerSummary;
import tech.kayys.gollek.sdk.mcp.McpServerView;
import tech.kayys.gollek.sdk.mcp.McpTestReport;
import tech.kayys.gollek.sdk.mcp.McpToolModel;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderInfo;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentPreflightMapperTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    @SuppressWarnings("unchecked")
    void reportsReadyWhenModelMcpToolsAndRequestAreValid() throws Exception {
        JsonNode request = MAPPER.readTree("""
                {
                  "model": "fixture-model",
                  "surface": "chat",
                  "request": {
                    "request_id": "req-preflight-ready",
                    "trace_id": "trace-preflight-ready",
                    "messages": [
                      {"role": "system", "content": "Use context."},
                      {"role": "user", "content": "Preflight this route."}
                    ],
                    "rag_context": [
                      {"source": "docs/agent", "text": "Gollek injects context only."}
                    ]
                  }
                }
                """);

        Map<String, Object> report = AgentPreflightMapper.preflight(
                null,
                request,
                AgentTraceContext.fromPayload(request.path("request")),
                fixtureSdk());

        assertEquals("gollek.agent_preflight", report.get("object"));
        assertEquals("v1", report.get("contract_version"));
        assertEquals(List.of("v1"), report.get("supported_contract_versions"));
        Map<String, Object> featureNegotiation = (Map<String, Object>) report.get("feature_negotiation");
        assertEquals("feature_flags", featureNegotiation.get("mode"));
        assertEquals(AgentServingFeatureProfile.DEFAULT_PROFILE,
                featureNegotiation.get("default_feature_profile"));
        assertTrue(((List<String>) featureNegotiation.get("supported_feature_profiles"))
                .contains(AgentServingFeatureProfile.EMBEDDING_RAG));
        assertTrue(((List<String>) featureNegotiation.get("required_features"))
                .contains("agent_feature_negotiation"));
        assertEquals("ready", report.get("status"));
        assertEquals(true, report.get("ready"));
        assertEquals(0, report.get("blocking_issue_count"));
        assertEquals("fixture-model", report.get("model"));
        assertEquals(7L, report.get("ready_check_count"));
        assertEquals(0L, report.get("blocked_check_count"));
        assertEquals(0L, report.get("skipped_check_count"));
        assertEquals(List.of(), report.get("blocking_codes"));
        assertEquals(List.of(), report.get("warning_codes"));
        assertEquals(List.of(), report.get("issue_hints"));
        assertEquals(Map.of(), report.get("issue_hints_by_area"));
        assertEquals(Map.of(), report.get("issue_hints_by_code"));
        assertEquals(List.of(), report.get("remediation_plan"));
        assertEquals(List.of(), report.get("blocking_remediation_plan"));
        assertEquals(List.of(), report.get("warning_remediation_plan"));
        assertEquals(Map.of(), report.get("remediation_plan_by_area"));
        assertEquals(Map.of(), report.get("remediation_plan_by_code"));

        Map<String, Object> boundary = (Map<String, Object>) report.get("boundary");
        assertEquals(true, boundary.get("validation_only"));
        assertEquals(false, boundary.get("model_invoked"));
        assertEquals(false, boundary.get("tool_execution"));
        assertEquals(false, boundary.get("retrieval_execution"));

        Map<String, Object> checkResults = (Map<String, Object>) report.get("check_results");
        assertEquals("ready", ((Map<String, Object>) checkResults.get("capabilities")).get("status"));
        assertEquals(List.of(), ((Map<String, Object>) checkResults.get("capabilities")).get("blocking_codes"));
        assertEquals(List.of(), ((Map<String, Object>) checkResults.get("capabilities")).get("issue_hints"));
        assertEquals(List.of(), ((Map<String, Object>) checkResults.get("capabilities")).get("remediation_plan"));
        assertEquals(List.of(), ((Map<String, Object>) checkResults.get("capabilities"))
                .get("blocking_remediation_plan"));
        assertEquals(List.of(), ((Map<String, Object>) checkResults.get("capabilities"))
                .get("warning_remediation_plan"));
        assertEquals(Map.of(), ((Map<String, Object>) checkResults.get("capabilities")).get("remediation_plan_by_code"));
        assertEquals("ready", ((Map<String, Object>) checkResults.get("contract")).get("status"));
        Map<String, Object> featureNegotiationCheck =
                (Map<String, Object>) checkResults.get("feature_negotiation");
        assertEquals("ready", featureNegotiationCheck.get("status"));
        assertEquals(true, featureNegotiationCheck.get("ready"));
        Map<String, Object> featureDetails = (Map<String, Object>) featureNegotiationCheck.get("details");
        assertEquals(AgentServingFeatureProfile.DEFAULT_PROFILE, featureDetails.get("feature_profile"));
        assertEquals(true, featureDetails.get("feature_profile_supported"));
        assertEquals("v1", featureDetails.get("required_contract_version"));
        assertEquals(List.of(), featureDetails.get("unsupported_required_features"));
        assertEquals(List.of(), featureDetails.get("unsupported_optional_features"));
        assertEquals("ready", ((Map<String, Object>) checkResults.get("model_route")).get("status"));
        assertEquals("ready", ((Map<String, Object>) checkResults.get("mcp_discovery")).get("status"));
        assertEquals("ready", ((Map<String, Object>) checkResults.get("tool_validation")).get("status"));
        assertEquals("ready", ((Map<String, Object>) checkResults.get("request_validation")).get("status"));
        assertEquals(List.copyOf(checkResults.keySet()), report.get("checked_areas"));

        Map<String, Object> readinessReport = (Map<String, Object>) report.get("readiness_report");
        assertEquals("gollek.agent_readiness_report", readinessReport.get("object"));
        assertEquals("v1", readinessReport.get("contract_version"));
        assertEquals(List.of("v1"), readinessReport.get("supported_contract_versions"));
        assertEquals(featureNegotiation, readinessReport.get("feature_negotiation"));
        assertEquals("ready", readinessReport.get("status"));
        assertEquals(checkResults, readinessReport.get("checks"));
        assertEquals(List.copyOf(checkResults.keySet()), readinessReport.get("checked_areas"));
        assertEquals(List.of(), readinessReport.get("issue_hints"));
        assertEquals(Map.of(), readinessReport.get("issue_hints_by_area"));
        assertEquals(Map.of(), readinessReport.get("issue_hints_by_code"));
        assertEquals(List.of(), readinessReport.get("remediation_plan"));
        assertEquals(List.of(), readinessReport.get("blocking_remediation_plan"));
        assertEquals(List.of(), readinessReport.get("warning_remediation_plan"));
        assertEquals(Map.of(), readinessReport.get("remediation_plan_by_area"));
        assertEquals(Map.of(), readinessReport.get("remediation_plan_by_code"));

        Map<String, Object> mcp = (Map<String, Object>) report.get("mcp_discovery");
        assertEquals(true, mcp.get("available"));
        assertEquals("openai", mcp.get("compat"));
        assertEquals(1, ((List<Map<String, Object>>) mcp.get("tools")).size());

        Map<String, Object> toolValidation = (Map<String, Object>) report.get("tool_validation");
        assertEquals(true, toolValidation.get("valid"));
        assertEquals(false, toolValidation.get("model_invoked"));
        assertEquals(1, toolValidation.get("tool_count"));

        Map<String, Object> requestValidation = (Map<String, Object>) report.get("request_validation");
        assertEquals(true, requestValidation.get("valid"));
        assertEquals(false, requestValidation.get("model_invoked"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void blocksWhenClientRequiresUnsupportedAgentFeature() throws Exception {
        JsonNode request = MAPPER.readTree("""
                {
                  "model": "fixture-model",
                  "surface": "chat",
                  "required_contract_version": "v2",
                  "required_features": ["rag_context", "wayang_planner"],
                  "optional_features": ["wayang_low_code_designer"],
                  "discover_mcp_tools": false,
                  "mcp_discovery_required": false,
                  "validate_tools": false,
                  "tool_validation_required": false,
                  "request": {
                    "messages": [
                      {"role": "user", "content": "Check negotiation."}
                    ]
                  }
                }
                """);

        Map<String, Object> report = AgentPreflightMapper.preflight(
                null,
                request,
                AgentTraceContext.fromPayload(request.path("request")),
                fixtureSdk());

        assertEquals("blocked", report.get("status"));
        assertEquals(false, report.get("ready"));
        assertTrue(((List<String>) report.get("blocking_codes")).contains("CONTRACT_VERSION_UNSUPPORTED"));
        assertTrue(((List<String>) report.get("blocking_codes")).contains("REQUIRED_AGENT_FEATURE_UNSUPPORTED"));
        assertTrue(((List<String>) report.get("warning_codes")).contains("OPTIONAL_AGENT_FEATURE_UNSUPPORTED"));

        Map<String, Object> checkResults = (Map<String, Object>) report.get("check_results");
        Map<String, Object> negotiation = (Map<String, Object>) checkResults.get("feature_negotiation");
        assertEquals("blocked", negotiation.get("status"));
        assertEquals(false, negotiation.get("ready"));
        assertTrue(((List<String>) negotiation.get("blocking_messages"))
                .contains("required contract version is not supported: v2"));
        assertTrue(((List<String>) negotiation.get("blocking_messages"))
                .contains("required agent feature is not supported: wayang_planner"));
        assertTrue(((List<String>) negotiation.get("warning_messages"))
                .contains("optional agent feature is not supported: wayang_low_code_designer"));
        Map<String, Object> details = (Map<String, Object>) negotiation.get("details");
        assertEquals("v2", details.get("required_contract_version"));
        assertEquals(List.of("wayang_planner"), details.get("unsupported_required_features"));
        assertEquals(List.of("wayang_low_code_designer"), details.get("unsupported_optional_features"));
        List<Map<String, Object>> blockingRemediation =
                (List<Map<String, Object>>) negotiation.get("blocking_remediation_plan");
        assertTrue(blockingRemediation.stream()
                .anyMatch(remediation -> "CONTRACT_VERSION_UNSUPPORTED".equals(remediation.get("code"))));
        assertTrue(blockingRemediation.stream()
                .anyMatch(remediation -> "REQUIRED_AGENT_FEATURE_UNSUPPORTED".equals(remediation.get("code"))));
        List<Map<String, Object>> warningRemediation =
                (List<Map<String, Object>>) negotiation.get("warning_remediation_plan");
        assertTrue(warningRemediation.stream()
                .anyMatch(remediation -> "OPTIONAL_AGENT_FEATURE_UNSUPPORTED".equals(remediation.get("code"))));
    }

    @Test
    @SuppressWarnings("unchecked")
    void blocksUnknownServingFeatureProfile() throws Exception {
        JsonNode request = MAPPER.readTree("""
                {
                  "model": "fixture-model",
                  "surface": "chat",
                  "feature_profile": "wayang_planner",
                  "discover_mcp_tools": false,
                  "mcp_discovery_required": false,
                  "validate_tools": false,
                  "tool_validation_required": false,
                  "request": {
                    "messages": [
                      {"role": "user", "content": "Check profile negotiation."}
                    ]
                  }
                }
                """);

        Map<String, Object> report = AgentPreflightMapper.preflight(
                null,
                request,
                AgentTraceContext.fromPayload(request.path("request")),
                fixtureSdk());

        assertEquals("blocked", report.get("status"));
        assertTrue(((List<String>) report.get("blocking_codes")).contains("FEATURE_PROFILE_UNSUPPORTED"));

        Map<String, Object> checkResults = (Map<String, Object>) report.get("check_results");
        Map<String, Object> negotiation = (Map<String, Object>) checkResults.get("feature_negotiation");
        assertEquals("blocked", negotiation.get("status"));
        assertTrue(((List<String>) negotiation.get("blocking_messages"))
                .contains("feature profile is not supported: wayang_planner"));
        Map<String, Object> details = (Map<String, Object>) negotiation.get("details");
        assertEquals("wayang_planner", details.get("feature_profile"));
        assertEquals(false, details.get("feature_profile_supported"));
        assertTrue(((List<String>) details.get("supported_feature_profiles"))
                .contains(AgentServingFeatureProfile.CHAT_AGENT));
    }

    @Test
    @SuppressWarnings("unchecked")
    void reportsInvalidToolsAsBlockedWithoutCrossingServingBoundary() throws Exception {
        JsonNode request = MAPPER.readTree("""
                {
                  "model": "fixture-model",
                  "surface": "chat",
                  "mcp_discovery_required": false,
                  "request": {
                    "messages": [
                      {"role": "user", "content": "Validate bad tool."}
                    ],
                    "tools": [
                      {
                        "type": "function",
                        "function": {
                          "name": "bad tool",
                          "parameters": {"type": "object"}
                        }
                      }
                    ]
                  }
                }
                """);

        Map<String, Object> report = AgentPreflightMapper.preflight(
                null,
                request,
                AgentTraceContext.fromPayload(request.path("request")),
                fixtureSdk());

        assertEquals("blocked", report.get("status"));
        assertEquals(false, report.get("ready"));
        assertTrue((Integer) report.get("blocking_issue_count") >= 1);
        assertTrue((Long) report.get("blocked_check_count") >= 1L);

        Map<String, Object> boundary = (Map<String, Object>) report.get("boundary");
        assertEquals(false, boundary.get("model_invoked"));
        assertEquals(false, boundary.get("tool_execution"));

        Map<String, List<String>> issuesByArea = (Map<String, List<String>>) report.get("issues_by_area");
        assertTrue(issuesByArea.get("tool_validation").contains("tool definitions are not valid"));
        Map<String, List<String>> issueCodesByArea =
                (Map<String, List<String>>) report.get("issue_codes_by_area");
        assertTrue(issueCodesByArea.get("tool_validation").contains("TOOL_DEFINITIONS_INVALID"));
        assertTrue(((List<String>) report.get("blocking_codes")).contains("TOOL_DEFINITIONS_INVALID"));
        List<Map<String, Object>> issues = (List<Map<String, Object>>) report.get("issues");
        assertTrue(issues.stream()
                .anyMatch(issue -> "TOOL_DEFINITIONS_INVALID".equals(issue.get("code"))));
        List<Map<String, Object>> issueHints = (List<Map<String, Object>>) report.get("issue_hints");
        assertTrue(issueHints.stream()
                .anyMatch(hint -> "TOOL_DEFINITIONS_INVALID".equals(hint.get("code"))
                        && "tool_validation".equals(hint.get("area"))
                        && ((String) hint.get("remediation")).contains("tool schema")));
        Map<String, List<Map<String, Object>>> hintsByArea =
                (Map<String, List<Map<String, Object>>>) report.get("issue_hints_by_area");
        assertTrue(hintsByArea.get("tool_validation").stream()
                .anyMatch(hint -> "TOOL_DEFINITIONS_INVALID".equals(hint.get("code"))));
        Map<String, List<Map<String, Object>>> hintsByCode =
                (Map<String, List<Map<String, Object>>>) report.get("issue_hints_by_code");
        assertTrue(hintsByCode.get("TOOL_DEFINITIONS_INVALID").stream()
                .anyMatch(hint -> "tool_validation".equals(hint.get("area"))));
        List<Map<String, Object>> remediationPlan =
                (List<Map<String, Object>>) report.get("remediation_plan");
        assertTrue(remediationPlan.stream()
                .anyMatch(remediation -> "TOOL_DEFINITIONS_INVALID".equals(remediation.get("code"))
                        && "tool_validation".equals(remediation.get("area"))
                        && ((String) remediation.get("remediation")).contains("tool schema")
                        && ((List<String>) remediation.get("messages"))
                                .contains("tool definitions are not valid")));
        List<Map<String, Object>> blockingRemediationPlan =
                (List<Map<String, Object>>) report.get("blocking_remediation_plan");
        assertTrue(blockingRemediationPlan.stream()
                .anyMatch(remediation -> "TOOL_DEFINITIONS_INVALID".equals(remediation.get("code"))));
        assertEquals(List.of(), report.get("warning_remediation_plan"));
        Map<String, List<Map<String, Object>>> remediationByArea =
                (Map<String, List<Map<String, Object>>>) report.get("remediation_plan_by_area");
        assertTrue(remediationByArea.get("tool_validation").stream()
                .anyMatch(remediation -> "TOOL_DEFINITIONS_INVALID".equals(remediation.get("code"))));
        Map<String, List<Map<String, Object>>> remediationByCode =
                (Map<String, List<Map<String, Object>>>) report.get("remediation_plan_by_code");
        assertTrue(remediationByCode.get("TOOL_DEFINITIONS_INVALID").stream()
                .anyMatch(remediation -> "tool_validation".equals(remediation.get("area"))));

        Map<String, Object> checkResults = (Map<String, Object>) report.get("check_results");
        Map<String, Object> toolValidationCheck = (Map<String, Object>) checkResults.get("tool_validation");
        assertEquals("blocked", toolValidationCheck.get("status"));
        assertEquals(false, toolValidationCheck.get("ready"));
        assertTrue(((List<String>) toolValidationCheck.get("blocking_messages"))
                .contains("tool definitions are not valid"));
        assertTrue(((List<String>) toolValidationCheck.get("blocking_codes"))
                .contains("TOOL_DEFINITIONS_INVALID"));
        List<Map<String, Object>> toolValidationHints =
                (List<Map<String, Object>>) toolValidationCheck.get("issue_hints");
        assertTrue(toolValidationHints.stream()
                .anyMatch(hint -> "TOOL_DEFINITIONS_INVALID".equals(hint.get("code"))));
        List<Map<String, Object>> toolValidationRemediations =
                (List<Map<String, Object>>) toolValidationCheck.get("remediation_plan");
        assertTrue(toolValidationRemediations.stream()
                .anyMatch(remediation -> "TOOL_DEFINITIONS_INVALID".equals(remediation.get("code"))
                        && ((List<String>) remediation.get("messages"))
                                .contains("tool definitions are not valid")));
        assertEquals(toolValidationRemediations, toolValidationCheck.get("blocking_remediation_plan"));
        assertEquals(List.of(), toolValidationCheck.get("warning_remediation_plan"));
        Map<String, List<Map<String, Object>>> toolValidationRemediationsByCode =
                (Map<String, List<Map<String, Object>>>) toolValidationCheck.get("remediation_plan_by_code");
        assertTrue(toolValidationRemediationsByCode.get("TOOL_DEFINITIONS_INVALID").stream()
                .anyMatch(remediation -> "tool_validation".equals(remediation.get("area"))));
        assertEquals(List.copyOf(checkResults.keySet()), report.get("checked_areas"));

        Map<String, Object> readinessReport = (Map<String, Object>) report.get("readiness_report");
        assertEquals("gollek.agent_readiness_report", readinessReport.get("object"));
        assertEquals("blocked", readinessReport.get("status"));
        assertEquals(checkResults, readinessReport.get("checks"));
        assertEquals(List.copyOf(checkResults.keySet()), readinessReport.get("checked_areas"));
        assertEquals(issueHints, readinessReport.get("issue_hints"));
        assertEquals(hintsByArea, readinessReport.get("issue_hints_by_area"));
        assertEquals(hintsByCode, readinessReport.get("issue_hints_by_code"));
        assertEquals(remediationPlan, readinessReport.get("remediation_plan"));
        assertEquals(blockingRemediationPlan, readinessReport.get("blocking_remediation_plan"));
        assertEquals(List.of(), readinessReport.get("warning_remediation_plan"));
        assertEquals(remediationByArea, readinessReport.get("remediation_plan_by_area"));
        assertEquals(remediationByCode, readinessReport.get("remediation_plan_by_code"));

        Map<String, Object> toolValidation = (Map<String, Object>) report.get("tool_validation");
        assertEquals(false, toolValidation.get("valid"));
        assertEquals(false, toolValidation.get("model_invoked"));
        assertTrue(((String) toolValidation.get("error")).contains("function.name"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void reportsEmbeddingRouteReadyWithoutToolingChecks() throws Exception {
        JsonNode request = MAPPER.readTree("""
                {
                  "model": "fixture-embed",
                  "surface": "embeddings",
                  "feature_profile": "embedding_rag",
                  "discover_mcp_tools": false,
                  "mcp_discovery_required": false,
                  "validate_tools": false,
                  "tool_validation_required": false,
                  "request": {
                    "request_id": "req-embed-ready",
                    "trace_id": "trace-embed-ready",
                    "input": ["alpha", "beta"],
                    "dimensions": 768,
                    "encoding_format": "float",
                    "metadata": {"workflow": "rag-indexing"}
                  }
                }
                """);

        Map<String, Object> report = AgentPreflightMapper.preflight(
                null,
                request,
                AgentTraceContext.fromPayload(request.path("request")),
                fixtureSdk());

        assertEquals("ready", report.get("status"));
        assertEquals(true, report.get("ready"));
        assertEquals("embeddings", report.get("surface"));
        assertEquals(5L, report.get("ready_check_count"));
        assertEquals(0L, report.get("blocked_check_count"));
        assertEquals(2L, report.get("skipped_check_count"));

        Map<String, Object> checkResults = (Map<String, Object>) report.get("check_results");
        Map<String, Object> featureNegotiation =
                (Map<String, Object>) checkResults.get("feature_negotiation");
        assertEquals("ready", featureNegotiation.get("status"));
        Map<String, Object> featureDetails = (Map<String, Object>) featureNegotiation.get("details");
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG, featureDetails.get("feature_profile"));
        assertTrue(((List<String>) featureDetails.get("required_features")).contains("openai_embeddings"));
        assertTrue(((List<String>) featureDetails.get("required_features")).contains("rag_context"));
        assertTrue(!((List<String>) featureDetails.get("required_features")).contains("mcp_tool_discovery"));
        assertEquals("ready", ((Map<String, Object>) checkResults.get("model_route")).get("status"));
        assertEquals("skipped", ((Map<String, Object>) checkResults.get("mcp_discovery")).get("status"));
        assertEquals(false, ((Map<String, Object>) checkResults.get("mcp_discovery")).get("requested"));
        assertEquals("skipped", ((Map<String, Object>) checkResults.get("tool_validation")).get("status"));
        assertEquals(false, ((Map<String, Object>) checkResults.get("tool_validation")).get("requested"));
        Map<String, Object> requestValidationCheck =
                (Map<String, Object>) checkResults.get("request_validation");
        Map<String, Object> details = (Map<String, Object>) requestValidationCheck.get("details");
        Map<String, Object> embeddingDetails = (Map<String, Object>) details.get("embedding");
        Map<String, Object> embeddingRag = (Map<String, Object>) embeddingDetails.get("rag");
        assertEquals("embeddings", details.get("surface"));
        assertEquals("fixture-embed", details.get("model"));
        assertEquals(2, embeddingDetails.get("input_count"));
        assertEquals(List.of(5, 4), embeddingDetails.get("input_lengths"));
        assertEquals(768, embeddingDetails.get("requested_dimensions"));
        assertEquals("float", embeddingDetails.get("encoding_format"));
        assertEquals(List.of("workflow"), embeddingDetails.get("metadata_keys"));
        assertEquals(true, embeddingRag.get("embedding_generation"));
        assertEquals(false, embeddingRag.get("retrieval_execution"));
        assertEquals("agent_orchestrator", embeddingRag.get("retrieval_policy_owned_by"));
        assertEquals("agent_orchestrator", embeddingRag.get("vector_store_owned_by"));
        assertEquals(true, embeddingRag.get("storage_owned_by_orchestrator"));
        assertEquals(List.copyOf(checkResults.keySet()), report.get("checked_areas"));

        Map<String, Object> readinessReport = (Map<String, Object>) report.get("readiness_report");
        assertEquals(checkResults, readinessReport.get("checks"));
        assertEquals(List.copyOf(checkResults.keySet()), readinessReport.get("checked_areas"));

        Map<String, Object> modelCapabilities = (Map<String, Object>) report.get("model_capabilities");
        Map<String, Object> embeddings = (Map<String, Object>) modelCapabilities.get("embeddings");
        assertEquals(true, embeddings.get("generation"));
        assertEquals(768, embeddings.get("dimensions"));
        assertEquals(false, embeddings.get("retrieval_policy"));
        assertEquals(false, embeddings.get("vector_store_ownership"));

        Map<String, Object> requestValidation = (Map<String, Object>) report.get("request_validation");
        Map<String, Object> normalized = (Map<String, Object>) requestValidation.get("normalized");
        assertEquals(2, normalized.get("input_count"));
        assertEquals(768, normalized.get("requested_dimensions"));
        assertEquals("float", normalized.get("encoding_format"));
        Map<String, Object> rag = (Map<String, Object>) normalized.get("rag");
        assertEquals(true, rag.get("embedding_generation"));
        assertEquals(false, rag.get("retrieval_execution"));
        assertEquals("agent_orchestrator", rag.get("vector_store_owned_by"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void blocksEmbeddingRouteWhenModelDoesNotSupportEmbeddingGeneration() throws Exception {
        JsonNode request = MAPPER.readTree("""
                {
                  "model": "fixture-model",
                  "surface": "embeddings",
                  "discover_mcp_tools": false,
                  "mcp_discovery_required": false,
                  "validate_tools": false,
                  "tool_validation_required": false,
                  "request": {
                    "request_id": "req-embed-blocked",
                    "trace_id": "trace-embed-blocked",
                    "input": ["alpha"]
                  }
                }
                """);

        Map<String, Object> report = AgentPreflightMapper.preflight(
                null,
                request,
                AgentTraceContext.fromPayload(request.path("request")),
                fixtureSdk());

        assertEquals("blocked", report.get("status"));
        assertEquals(false, report.get("ready"));

        Map<String, List<String>> issuesByArea = (Map<String, List<String>>) report.get("issues_by_area");
        assertTrue(issuesByArea.get("model_route")
                .contains("model route does not support embedding generation for /v1/embeddings"));
        Map<String, List<String>> issueCodesByArea =
                (Map<String, List<String>>) report.get("issue_codes_by_area");
        assertTrue(issueCodesByArea.get("model_route").contains("EMBEDDING_GENERATION_UNSUPPORTED"));
        assertTrue(((List<String>) report.get("blocking_codes")).contains("EMBEDDING_GENERATION_UNSUPPORTED"));
        List<Map<String, Object>> issues = (List<Map<String, Object>>) report.get("issues");
        assertTrue(issues.stream()
                .anyMatch(issue -> "EMBEDDING_GENERATION_UNSUPPORTED".equals(issue.get("code"))));
        List<Map<String, Object>> issueHints = (List<Map<String, Object>>) report.get("issue_hints");
        assertTrue(issueHints.stream()
                .anyMatch(hint -> "EMBEDDING_GENERATION_UNSUPPORTED".equals(hint.get("code"))
                        && ((String) hint.get("remediation")).contains("embedding model")));
        Map<String, List<Map<String, Object>>> hintsByCode =
                (Map<String, List<Map<String, Object>>>) report.get("issue_hints_by_code");
        assertTrue(hintsByCode.get("EMBEDDING_GENERATION_UNSUPPORTED").stream()
                .anyMatch(hint -> "model_route".equals(hint.get("area"))));

        Map<String, Object> checkResults = (Map<String, Object>) report.get("check_results");
        Map<String, Object> modelRoute = (Map<String, Object>) checkResults.get("model_route");
        assertEquals("blocked", modelRoute.get("status"));
        assertEquals(false, modelRoute.get("ready"));
        assertTrue(((List<String>) modelRoute.get("blocking_messages"))
                .contains("model route does not support embedding generation for /v1/embeddings"));
        assertTrue(((List<String>) modelRoute.get("blocking_codes"))
                .contains("EMBEDDING_GENERATION_UNSUPPORTED"));
        List<Map<String, Object>> modelRouteHints = (List<Map<String, Object>>) modelRoute.get("issue_hints");
        assertTrue(modelRouteHints.stream()
                .anyMatch(hint -> "EMBEDDING_GENERATION_UNSUPPORTED".equals(hint.get("code"))));
    }

    private static GollekSdk fixtureSdk() {
        return (GollekSdk) Proxy.newProxyInstance(
                GollekSdk.class.getClassLoader(),
                new Class<?>[] { GollekSdk.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "getModelInfo" -> Optional.of(modelInfo((String) args[0]));
                    case "listAvailableProviders" -> List.of(providerInfo());
                    case "getPreferredProvider" -> Optional.of("fixture-provider");
                    case "mcpRegistry" -> new FixtureMcpRegistry();
                    case "toString" -> "FixtureGollekSdk";
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }

    private static ModelInfo modelInfo(String modelId) {
        ModelInfo.Builder builder = ModelInfo.builder()
                .modelId("fixture-model")
                .name("Fixture Model")
                .architecture("fixture-transformer")
                .format("SAFETENSORS")
                .contextLength(8192L)
                .outputTokenLimit(1024)
                .metadata(Map.of("tool_calling", true));
        if ("fixture-embed".equals(modelId)) {
            builder.modelId("fixture-embed")
                    .name("Fixture Embedding Model")
                    .embeddingSize(768)
                    .metadata(Map.of("embeddings", true));
        }
        return builder.build();
    }

    private static ProviderInfo providerInfo() {
        return ProviderInfo.builder()
                .id("fixture-provider")
                .name("Fixture Provider")
                .healthStatus(ProviderHealth.Status.HEALTHY)
                .supportedModels(Set.of("fixture-model"))
                .capabilities(ProviderCapabilities.builder()
                        .streaming(true)
                        .functionCalling(true)
                        .toolCalling(true)
                        .maxContextTokens(8192)
                        .maxOutputTokens(1024)
                        .build())
                .build();
    }

    private static class FixtureMcpRegistry implements McpRegistryManager {
        @Override
        public String registryPath() {
            return "/fixture/gollek/mcp.json";
        }

        @Override
        public List<McpServerSummary> list() {
            return List.of(new McpServerSummary("knowledge", true));
        }

        @Override
        public McpServerView show(String name) throws SdkException {
            return new McpServerView(name, true, "stdio", "npx", 1, 0, null, "{}");
        }

        @Override
        public List<McpToolModel> listTools(String name) {
            return List.of(new McpToolModel(
                    "search",
                    "Search indexed knowledge",
                    Map.of(
                            "type", "object",
                            "required", List.of("query"),
                            "properties", Map.of(
                                    "query", Map.of("type", "string")))));
        }

        @Override
        public List<String> add(McpAddRequest request) {
            throw unsupportedMutation();
        }

        @Override
        public void remove(String name) {
            throw unsupportedMutation();
        }

        @Override
        public void rename(String oldName, String newName) {
            throw unsupportedMutation();
        }

        @Override
        public void edit(McpEditRequest request) {
            throw unsupportedMutation();
        }

        @Override
        public void setEnabled(String name, boolean enabled) {
            throw unsupportedMutation();
        }

        @Override
        public int importFromFile(String filePath, boolean replace) {
            throw unsupportedMutation();
        }

        @Override
        public int exportToFile(String filePath, String name) {
            throw unsupportedMutation();
        }

        @Override
        public McpDoctorReport doctor() {
            throw unsupportedMutation();
        }

        @Override
        public McpTestReport test(String name, boolean all, long timeoutMs) {
            throw unsupportedMutation();
        }

        private UnsupportedOperationException unsupportedMutation() {
            return new UnsupportedOperationException("fixture registry is read-only");
        }
    }
}
