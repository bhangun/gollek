package tech.kayys.gollek.server.api.v1;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import com.fasterxml.jackson.databind.JsonNode;
import jakarta.ws.rs.core.HttpHeaders;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

final class AgentValidationMapper {

    private AgentValidationMapper() {
    }

    static Map<String, Object> validate(String requestedSurface, HttpHeaders headers, JsonNode payload) {
        return validate(requestedSurface, headers, payload, AgentTraceContext.from(headers, payload));
    }

    static Map<String, Object> validate(
            String requestedSurface, HttpHeaders headers, JsonNode payload, AgentTraceContext traceContext) {
        AgentTraceContext trace = traceContext != null ? traceContext : AgentTraceContext.from(headers, payload);
        String surface = surface(requestedSurface, payload);
        return switch (surface) {
            case "chat" -> inferencePayload(
                    "chat",
                    AgenticApiMapper.toInferenceRequest(payload, AgenticApiMapper.apiKey(headers), trace),
                    trace,
                    payload);
            case "responses" -> inferencePayload(
                    "responses",
                    AgenticApiMapper.toResponsesInferenceRequest(payload, AgenticApiMapper.apiKey(headers), trace),
                    trace,
                    payload);
            case "embedding", "embeddings" -> embeddingPayload(AgenticApiMapper.toEmbeddingRequest(payload, trace), trace);
            default -> throw new IllegalArgumentException(
                    "surface must be one of chat, responses, or embeddings");
        };
    }

    private static Map<String, Object> inferencePayload(
            String surface, InferenceRequest request, AgentTraceContext trace, JsonNode sourcePayload) {
        Map<String, Object> payload = basePayload(surface, trace);
        Map<String, Object> normalized = new LinkedHashMap<>();
        normalized.put("request_id", request.getRequestId());
        request.getTraceId().ifPresent(traceId -> normalized.put("trace_id", traceId));
        normalized.put("model", request.getModel());
        normalized.put("streaming", request.isStreaming());
        normalized.put("stream_options", AgentStreamOptions.from(sourcePayload).summary(surface));
        normalized.put("message_count", request.getMessages().size());
        normalized.put("messages", messages(request.getMessages()));
        normalized.put("tool_count", request.getTools() != null ? request.getTools().size() : 0);
        normalized.put("tools", tools(request.getTools()));
        normalized.put("tool_contract", AgentToolContractMapper.summary(
                sourcePayload != null ? sourcePayload.path("tools") : null));
        normalized.put("tool_choice", request.getToolChoice());
        normalized.put("parameter_keys", sortedKeys(request.getParameters()));
        normalized.put("metadata", request.getMetadata());
        normalized.put("rag", ragSummary(request));
        request.getSessionId().ifPresent(sessionId -> normalized.put("session_id", sessionId));
        request.getUserId().ifPresent(userId -> normalized.put("user_id", userId));
        payload.put("normalized", normalized);
        return payload;
    }

    private static Map<String, Object> embeddingPayload(EmbeddingRequest request, AgentTraceContext trace) {
        Map<String, Object> payload = basePayload("embeddings", trace);
        Map<String, Object> normalized = new LinkedHashMap<>();
        normalized.put("request_id", request.requestId());
        normalized.put("trace_id", trace.asMap().get("trace_id"));
        normalized.put("model", request.model());
        normalized.put("input_count", request.inputs().size());
        normalized.put("input_lengths", request.inputs().stream()
                .map(input -> input != null ? input.length() : 0)
                .toList());
        if (request.parameters().containsKey("dimensions")) {
            normalized.put("requested_dimensions", request.parameters().get("dimensions"));
        }
        normalized.put("encoding_format", request.parameters().getOrDefault("encoding_format", "float"));
        normalized.put("parameter_keys", sortedKeys(request.parameters()));
        normalized.put("metadata", request.parameters().getOrDefault("metadata", Map.of()));
        normalized.put("rag", embeddingRagSummary());
        payload.put("normalized", normalized);
        return payload;
    }

    private static Map<String, Object> basePayload(String surface, AgentTraceContext trace) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("object", "gollek.agent_validation");
        payload.put("surface", surface);
        payload.put("valid", true);
        payload.put("model_invoked", false);
        payload.put("trace", trace.asMap());
        payload.put("boundary", boundary());
        return payload;
    }

    private static Map<String, Object> boundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", true);
        boundary.put("tool_execution", false);
        boundary.put("retrieval_execution", false);
        return boundary;
    }

    private static List<Map<String, Object>> messages(List<Message> messages) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Message message : messages) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("role", message.getRole().name().toLowerCase(Locale.ROOT));
            item.put("content_length", message.getContent() != null ? message.getContent().length() : 0);
            if (message.getName() != null) {
                item.put("name", message.getName());
            }
            if (message.getToolCallId() != null) {
                item.put("tool_call_id", message.getToolCallId());
            }
            if (message.getToolCalls() != null && !message.getToolCalls().isEmpty()) {
                item.put("tool_call_count", message.getToolCalls().size());
            }
            out.add(item);
        }
        return out;
    }

    private static List<Map<String, Object>> tools(List<ToolDefinition> tools) {
        if (tools == null || tools.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (ToolDefinition tool : tools) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("name", tool.getName());
            item.put("type", tool.getType().name().toLowerCase(Locale.ROOT));
            item.put("strict", tool.isStrict());
            item.put("parameter_keys", sortedKeys(tool.getParameters()));
            if (!tool.getMetadata().isEmpty()) {
                item.put("metadata", tool.getMetadata());
            }
            out.add(item);
        }
        return out;
    }

    private static Map<String, Object> ragSummary(InferenceRequest request) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("injected", Boolean.TRUE.equals(request.getMetadata().get("rag_context_injected")));
        summary.put("context_parameter", request.getParameters().containsKey("rag_context"));
        summary.put("source_parameter", request.getParameters().containsKey("rag_sources"));
        Object items = request.getMetadata().get("rag_context_items");
        if (items != null) {
            summary.put("items", items);
        }
        Object alias = request.getMetadata().get("rag_context_alias");
        if (alias != null) {
            summary.put("alias", alias);
        }
        return summary;
    }

    private static Map<String, Object> embeddingRagSummary() {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("embedding_generation", true);
        summary.put("retrieval_execution", false);
        summary.put("retrieval_policy_owned_by", "agent_orchestrator");
        summary.put("vector_store_owned_by", "agent_orchestrator");
        return summary;
    }

    private static List<String> sortedKeys(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return List.of();
        }
        return map.keySet().stream().sorted().toList();
    }

    private static String surface(String requestedSurface, JsonNode payload) {
        if (!isBlank(requestedSurface)) {
            return normalizeSurface(requestedSurface);
        }
        if (payload != null && payload.has("surface")) {
            return normalizeSurface(payload.path("surface").asText());
        }
        if (payload != null && payload.has("messages")) {
            return "chat";
        }
        if (payload != null && payload.has("input")) {
            return "responses";
        }
        throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "surface is required when it cannot be inferred");
    }

    private static String normalizeSurface(String surface) {
        String normalized = surface == null ? "" : surface.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "chat", "chat_completions", "chat.completions" -> "chat";
            case "response", "responses" -> "responses";
            case "embedding", "embeddings" -> "embeddings";
            default -> normalized;
        };
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}
