package tech.kayys.gollek.server.api.v1;


import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.spi.tool.ToolCall;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

final class AgenticApiMapper {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private AgenticApiMapper() {
    }

    static InferenceRequest toInferenceRequest(JsonNode root, String apiKey) {
        return toInferenceRequest(root, apiKey, AgentTraceContext.fromPayload(root));
    }

    static InferenceRequest toInferenceRequest(JsonNode root, String apiKey, AgentTraceContext traceContext) {
        AgentTraceContext trace = traceContext != null ? traceContext : AgentTraceContext.fromPayload(root);
        InferenceRequest.Builder builder = InferenceRequest.builder()
                .requestId(trace.requestId())
                .model(requiredText(root, "model"));

        if (!isBlank(apiKey)) {
            builder.apiKey(apiKey);
        }
        boolean ragContextChecked = false;
        for (JsonNode message : iterable(root.path("messages"))) {
            Message mapped = toMessage(message);
            if (!ragContextChecked && mapped.getRole() != Message.Role.SYSTEM) {
                RagContextMapper.apply(builder, root);
                ragContextChecked = true;
            }
            builder.message(mapped);
        }
        if (!ragContextChecked) {
            RagContextMapper.apply(builder, root);
        }
        if (root.has("prompt") && !isBlank(root.path("prompt").asText())) {
            builder.message(Message.user(root.path("prompt").asText()));
        }
        for (ToolDefinition tool : toTools(root.path("tools"))) {
            builder.tool(tool);
        }
        if (root.has("tool_choice")) {
            builder.toolChoice(toJava(root.path("tool_choice")));
        }
        applyCommonParameters(builder, root);
        applyMetadata(builder, root.path("metadata"));
        if (!isBlank(text(root, "user", null))) {
            builder.userId(root.path("user").asText());
        }
        if (!isBlank(text(root, "session_id", null))) {
            builder.sessionId(root.path("session_id").asText());
        }
        trace.apply(builder);
        if (root.path("stream").asBoolean(false)) {
            builder.streaming(true);
        }
        return builder.build();
    }

    static InferenceRequest toResponsesInferenceRequest(JsonNode root, String apiKey) {
        return toResponsesInferenceRequest(root, apiKey, AgentTraceContext.fromPayload(root));
    }

    static InferenceRequest toResponsesInferenceRequest(JsonNode root, String apiKey, AgentTraceContext traceContext) {
        AgentTraceContext trace = traceContext != null ? traceContext : AgentTraceContext.fromPayload(root);
        InferenceRequest.Builder builder = InferenceRequest.builder()
                .requestId(trace.requestId())
                .model(requiredText(root, "model"));

        if (!isBlank(apiKey)) {
            builder.apiKey(apiKey);
        }
        if (!isBlank(text(root, "instructions", null))) {
            builder.message(new Message(Message.Role.SYSTEM, root.path("instructions").asText(), null, null, null, null));
        }
        RagContextMapper.apply(builder, root);
        appendResponsesInput(builder, root.path("input"));
        for (ToolDefinition tool : toTools(root.path("tools"))) {
            builder.tool(tool);
        }
        if (root.has("tool_choice")) {
            builder.toolChoice(toJava(root.path("tool_choice")));
        }
        applyCommonParameters(builder, root);
        if (root.path("text").path("format").isObject()) {
            builder.parameter("response_format", toJava(root.path("text").path("format")));
        }
        applyMetadata(builder, root.path("metadata"));
        if (!isBlank(text(root, "user", null))) {
            builder.userId(root.path("user").asText());
        }
        if (!isBlank(text(root, "conversation", null))) {
            builder.sessionId(root.path("conversation").asText());
        } else if (!isBlank(text(root, "previous_response_id", null))) {
            builder.sessionId(root.path("previous_response_id").asText());
        }
        trace.apply(builder);
        if (root.path("stream").asBoolean(false)) {
            builder.streaming(true);
        }
        return builder.build();
    }

    static EmbeddingRequest toEmbeddingRequest(JsonNode root) {
        return toEmbeddingRequest(root, AgentTraceContext.fromPayload(root));
    }

    static EmbeddingRequest toEmbeddingRequest(JsonNode root, AgentTraceContext traceContext) {
        AgentTraceContext trace = traceContext != null ? traceContext : AgentTraceContext.fromPayload(root);
        Map<String, Object> parameters = new LinkedHashMap<>();
        copyIfPresent(parameters, root, "dimensions");
        copyIfPresent(parameters, root, "encoding_format");
        copyIfPresent(parameters, root, "user");
        if (root.path("metadata").isObject()) {
            parameters.put("metadata", toJava(root.path("metadata")));
        }
        trace.applyToParameters(parameters);
        return EmbeddingRequest.builder()
                .requestId(trace.requestId())
                .model(requiredText(root, "model"))
                .inputs(embeddingInputs(root.has("input") ? root.path("input") : root.path("inputs")))
                .parameters(parameters)
                .build();
    }

    static boolean isOpenAiEmbeddingRequest(JsonNode root) {
        return root != null && root.has("input");
    }

    static Map<String, Object> toOpenAiChatResponse(InferenceResponse response, String fallbackModel) {
        return toOpenAiChatResponse(response, fallbackModel, null);
    }

    static Map<String, Object> toOpenAiChatResponse(
            InferenceResponse response, String fallbackModel, AgentTraceContext traceContext) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("id", "chatcmpl-" + response.getRequestId());
        payload.put("object", "chat.completion");
        payload.put("created", response.getTimestamp().getEpochSecond());
        payload.put("model", response.getModel() != null ? response.getModel() : fallbackModel);
        payload.put("choices", List.of(choice(response)));
        payload.put("usage", usage(response));
        payload.put("metadata", traceContext != null
                ? traceContext.withTraceMetadata(response.getMetadata())
                : response.getMetadata());
        return payload;
    }

    static String toOpenAiChatStreamJson(StreamingInferenceChunk chunk, String fallbackModel) {
        return toOpenAiChatStreamJson(chunk, fallbackModel, null);
    }

    static String toOpenAiChatStreamJson(
            StreamingInferenceChunk chunk, String fallbackModel, AgentTraceContext traceContext) {
        return toOpenAiChatStreamJson(chunk, fallbackModel, traceContext, AgentStreamOptions.DEFAULT);
    }

    static String toOpenAiChatStreamJson(
            StreamingInferenceChunk chunk,
            String fallbackModel,
            AgentTraceContext traceContext,
            AgentStreamOptions streamOptions) {
        AgentStreamOptions options = streamOptions != null ? streamOptions : AgentStreamOptions.DEFAULT;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("id", "chatcmpl-" + chunk.requestId());
        payload.put("object", "chat.completion.chunk");
        payload.put("created", chunk.emittedAt() != null
                ? chunk.emittedAt().getEpochSecond()
                : java.time.Instant.now().getEpochSecond());
        payload.put("model", fallbackModel);
        payload.put("choices", List.of(streamChoice(chunk)));
        Map<String, Object> metadata = streamMetadata(
                chunk.metadata(),
                traceContext,
                options,
                "chat.completions",
                chunk.index(),
                chunk.finished(),
                chunk.finishReason());
        if (!metadata.isEmpty()) {
            payload.put("metadata", metadata);
        }
        if (options.includeUsage() && chunk.finished() && chunk.usage() != null) {
            payload.put("usage", streamUsage(chunk.usage()));
        }
        return json(payload);
    }

    static String toOpenAiChatStreamErrorJson(Throwable error) {
        return toOpenAiChatStreamErrorJson(error, null);
    }

    static String toOpenAiChatStreamErrorJson(Throwable error, AgentTraceContext traceContext) {
        String message = error != null && error.getMessage() != null
                ? error.getMessage()
                : "Streaming inference failed";
        return json(AgentErrorMapper.error(message, AgentErrorMapper.SERVER_ERROR, traceContext));
    }

    static Map<String, Object> toOpenAiModelsResponse(List<ModelInfo> models) {
        List<Map<String, Object>> data = new ArrayList<>();
        for (ModelInfo model : models) {
            data.add(openAiModel(model));
        }
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("object", "list");
        payload.put("data", data);
        return payload;
    }

    static Map<String, Object> toOpenAiResponse(InferenceResponse response, String fallbackModel) {
        return toOpenAiResponse(response, fallbackModel, null);
    }

    static Map<String, Object> toOpenAiResponse(
            InferenceResponse response, String fallbackModel, AgentTraceContext traceContext) {
        String id = "resp_" + response.getRequestId();
        String messageId = "msg_" + response.getRequestId();
        Map<String, Object> outputMessage = new LinkedHashMap<>();
        outputMessage.put("id", messageId);
        outputMessage.put("type", "message");
        outputMessage.put("status", "completed");
        outputMessage.put("role", "assistant");
        outputMessage.put("content", List.of(Map.of(
                "type", "output_text",
                "text", response.getContent() != null ? response.getContent() : "",
                "annotations", List.of())));

        List<Map<String, Object>> output = new ArrayList<>();
        output.add(outputMessage);
        output.addAll(responseToolOutputItems(response));

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("id", id);
        payload.put("object", "response");
        payload.put("created_at", response.getTimestamp().getEpochSecond());
        payload.put("status", "completed");
        payload.put("model", response.getModel() != null ? response.getModel() : fallbackModel);
        payload.put("output", output);
        payload.put("output_text", response.getContent() != null ? response.getContent() : "");
        payload.put("usage", usage(response));
        payload.put("metadata", traceContext != null
                ? traceContext.withTraceMetadata(response.getMetadata())
                : response.getMetadata());
        return payload;
    }

    static String toOpenAiResponseStreamCreatedJson(InferenceRequest request) {
        return toOpenAiResponseStreamCreatedJson(request, null);
    }

    static String toOpenAiResponseStreamCreatedJson(InferenceRequest request, AgentTraceContext traceContext) {
        return toOpenAiResponseStreamCreatedJson(request, traceContext, AgentStreamOptions.DEFAULT);
    }

    static String toOpenAiResponseStreamCreatedJson(
            InferenceRequest request, AgentTraceContext traceContext, AgentStreamOptions streamOptions) {
        AgentStreamOptions options = streamOptions != null ? streamOptions : AgentStreamOptions.DEFAULT;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "response.created");
        payload.put("sequence_number", 0);
        payload.put("response", responseStreamObject(
                request.getRequestId(),
                request.getModel(),
                java.time.Instant.now().getEpochSecond(),
                "in_progress",
                "",
                null,
                traceContext,
                options,
                0,
                false,
                null));
        return json(payload);
    }

    static List<String> toOpenAiResponseStreamJson(StreamingInferenceChunk chunk, String fallbackModel, String outputText) {
        return toOpenAiResponseStreamJson(chunk, fallbackModel, outputText, null);
    }

    static List<String> toOpenAiResponseStreamJson(
            StreamingInferenceChunk chunk, String fallbackModel, String outputText, AgentTraceContext traceContext) {
        return toOpenAiResponseStreamJson(
                chunk,
                fallbackModel,
                outputText,
                traceContext,
                AgentStreamOptions.DEFAULT);
    }

    static List<String> toOpenAiResponseStreamJson(
            StreamingInferenceChunk chunk,
            String fallbackModel,
            String outputText,
            AgentTraceContext traceContext,
            AgentStreamOptions streamOptions) {
        AgentStreamOptions options = streamOptions != null ? streamOptions : AgentStreamOptions.DEFAULT;
        List<String> events = new ArrayList<>();
        String responseId = "resp_" + chunk.requestId();
        String messageId = "msg_" + chunk.requestId();
        if (!isBlank(chunk.delta())) {
            Map<String, Object> delta = new LinkedHashMap<>();
            delta.put("type", "response.output_text.delta");
            delta.put("sequence_number", chunk.index() + 1);
            delta.put("response_id", responseId);
            delta.put("item_id", messageId);
            delta.put("output_index", 0);
            delta.put("content_index", 0);
            delta.put("delta", chunk.delta());
            if (options.includeTrace() && traceContext != null) {
                delta.put("trace", traceContext.asMap());
            }
            putStreamMetadata(delta, null, traceContext, options, "responses",
                    chunk.index() + 1, false, null);
            events.add(json(delta));
        }
        if (chunk.finished()) {
            Map<String, Object> done = new LinkedHashMap<>();
            done.put("type", "response.output_text.done");
            done.put("sequence_number", chunk.index() + 2);
            done.put("item_id", messageId);
            done.put("output_index", 0);
            done.put("content_index", 0);
            done.put("text", outputText);
            if (options.includeTrace() && traceContext != null) {
                done.put("trace", traceContext.asMap());
            }
            putStreamMetadata(done, null, traceContext, options, "responses",
                    chunk.index() + 2, true, chunk.finishReason());
            events.add(json(done));

            Map<String, Object> completed = new LinkedHashMap<>();
            completed.put("type", "response.completed");
            completed.put("sequence_number", chunk.index() + 3);
            completed.put("response", responseStreamObject(
                    chunk.requestId(),
                    fallbackModel,
                    chunk.emittedAt() != null
                            ? chunk.emittedAt().getEpochSecond()
                            : java.time.Instant.now().getEpochSecond(),
                    "completed",
                    outputText,
                    chunk.usage(),
                    traceContext,
                    options,
                    chunk.index() + 3,
                    true,
                    chunk.finishReason()));
            events.add(json(completed));
        }
        return events;
    }

    static String toOpenAiResponseStreamErrorJson(Throwable error) {
        return toOpenAiResponseStreamErrorJson(error, null);
    }

    static String toOpenAiResponseStreamErrorJson(Throwable error, AgentTraceContext traceContext) {
        String message = error != null && error.getMessage() != null
                ? error.getMessage()
                : "Streaming response failed";
        return json(Map.of(
                "type", "error",
                "sequence_number", 0,
                "error", AgentErrorMapper.errorObject(message, AgentErrorMapper.SERVER_ERROR, traceContext)));
    }

    static Map<String, Object> toOpenAiEmbeddingResponse(EmbeddingResponse response) {
        return toOpenAiEmbeddingResponse(response, null);
    }

    static Map<String, Object> toOpenAiEmbeddingResponse(
            EmbeddingResponse response, AgentTraceContext traceContext) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("object", "list");
        payload.put("model", response.model());
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < response.embeddings().size(); i++) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("object", "embedding");
            item.put("index", i);
            item.put("embedding", response.embeddings().get(i));
            data.add(item);
        }
        payload.put("data", data);
        payload.put("usage", Map.of("prompt_tokens", 0, "total_tokens", 0));
        payload.put("metadata", traceContext != null
                ? traceContext.withTraceMetadata(response.metadata())
                : response.metadata());
        return payload;
    }

    static String apiKey(jakarta.ws.rs.core.HttpHeaders headers) {
        if (headers == null) {
            return null;
        }
        String key = headers.getHeaderString("X-API-Key");
        if (!isBlank(key)) {
            return key.trim();
        }
        String auth = headers.getHeaderString("Authorization");
        if (auth == null) {
            return null;
        }
        String trimmed = auth.trim();
        return trimmed.regionMatches(true, 0, "Bearer ", 0, "Bearer ".length())
                ? trimmed.substring("Bearer ".length()).trim()
                : null;
    }

    private static Map<String, Object> openAiModel(ModelInfo model) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("id", model.getModelId());
        item.put("object", "model");
        item.put("created", model.getCreatedAt() != null ? model.getCreatedAt().getEpochSecond() : 0);
        item.put("owned_by", owner(model));
        Map<String, Object> metadata = new LinkedHashMap<>(model.getMetadata());
        if (model.getFormat() != null) {
            metadata.put("format", model.getFormat());
        }
        if (model.getDescription() != null) {
            metadata.put("description", model.getDescription());
        }
        if (model.getSizeBytes() != null) {
            metadata.put("size_bytes", model.getSizeBytes());
        }
        if (!metadata.isEmpty()) {
            item.put("metadata", metadata);
        }
        return item;
    }

    private static Map<String, Object> responseStreamObject(String requestId, String model, long createdAt,
            String status, String outputText, StreamingInferenceChunk.ChunkUsage usage,
            AgentTraceContext traceContext) {
        return responseStreamObject(
                requestId,
                model,
                createdAt,
                status,
                outputText,
                usage,
                traceContext,
                AgentStreamOptions.DEFAULT,
                0,
                "completed".equals(status),
                null);
    }

    private static Map<String, Object> responseStreamObject(String requestId, String model, long createdAt,
            String status, String outputText, StreamingInferenceChunk.ChunkUsage usage,
            AgentTraceContext traceContext, AgentStreamOptions streamOptions, int sequenceNumber,
            boolean finished, String finishReason) {
        AgentStreamOptions options = streamOptions != null ? streamOptions : AgentStreamOptions.DEFAULT;
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", "resp_" + requestId);
        response.put("object", "response");
        response.put("created_at", createdAt);
        response.put("status", status);
        response.put("model", model);
        response.put("output", isBlank(outputText) ? List.of() : List.of(responseOutputMessage(requestId, outputText)));
        response.put("output_text", outputText);
        Map<String, Object> metadata = streamMetadata(
                null,
                traceContext,
                options,
                "responses",
                sequenceNumber,
                finished,
                finishReason);
        if (!metadata.isEmpty()) {
            response.put("metadata", metadata);
        }
        if (options.includeUsage() && usage != null) {
            response.put("usage", streamUsage(usage));
        }
        return response;
    }

    private static Map<String, Object> responseOutputMessage(String requestId, String outputText) {
        Map<String, Object> outputMessage = new LinkedHashMap<>();
        outputMessage.put("id", "msg_" + requestId);
        outputMessage.put("type", "message");
        outputMessage.put("status", "completed");
        outputMessage.put("role", "assistant");
        outputMessage.put("content", List.of(Map.of(
                "type", "output_text",
                "text", outputText,
                "annotations", List.of())));
        return outputMessage;
    }

    private static void putStreamMetadata(
            Map<String, Object> event,
            Map<String, Object> sourceMetadata,
            AgentTraceContext traceContext,
            AgentStreamOptions options,
            String surface,
            int sequenceNumber,
            boolean finished,
            String finishReason) {
        Map<String, Object> metadata = streamMetadata(
                sourceMetadata,
                traceContext,
                options,
                surface,
                sequenceNumber,
                finished,
                finishReason);
        if (!metadata.isEmpty()) {
            event.put("metadata", metadata);
        }
    }

    private static Map<String, Object> streamMetadata(
            Map<String, Object> sourceMetadata,
            AgentTraceContext traceContext,
            AgentStreamOptions options,
            String surface,
            int sequenceNumber,
            boolean finished,
            String finishReason) {
        AgentStreamOptions streamOptions = options != null ? options : AgentStreamOptions.DEFAULT;
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (sourceMetadata != null) {
            metadata.putAll(sourceMetadata);
        }
        if (streamOptions.includeTrace() && traceContext != null) {
            metadata.put("gollek_trace", traceContext.asMap());
        }
        if (streamOptions.includeStreamMetadata()) {
            Map<String, Object> stream = new LinkedHashMap<>();
            stream.put("surface", surface);
            stream.put("sequence_number", sequenceNumber);
            stream.put("final", finished);
            stream.put("include_usage", streamOptions.includeUsage());
            if (!isBlank(finishReason)) {
                stream.put("finish_reason", finishReason(finishReason));
            }
            metadata.put("gollek_stream", stream);
        }
        return metadata;
    }

    private static String owner(ModelInfo model) {
        Object owner = model.getMetadata().get("owned_by");
        return owner != null && !isBlank(owner.toString()) ? owner.toString() : "gollek";
    }

    private static void appendResponsesInput(InferenceRequest.Builder builder, JsonNode input) {
        if (input == null || input.isMissingNode() || input.isNull()) {
            return;
        }
        if (input.isTextual()) {
            builder.message(Message.user(input.asText()));
            return;
        }
        if (input.isArray()) {
            for (JsonNode item : input) {
                appendResponsesInputItem(builder, item);
            }
            return;
        }
        appendResponsesInputItem(builder, input);
    }

    private static void appendResponsesInputItem(InferenceRequest.Builder builder, JsonNode item) {
        if (item == null || item.isNull() || item.isMissingNode()) {
            return;
        }
        if (item.isTextual()) {
            builder.message(Message.user(item.asText()));
            return;
        }
        String type = text(item, "type", "").trim().toLowerCase(Locale.ROOT);
        switch (type) {
            case "function_call", "custom_tool_call" -> {
                builder.message(responsesFunctionCallMessage(item));
                return;
            }
            case "function_call_output", "custom_tool_call_output" -> {
                builder.message(responsesToolOutputMessage(item));
                return;
            }
            case "reasoning", "item_reference" -> {
                return;
            }
            default -> {
            }
        }
        String role = text(item, "role", "user");
        String content = responseContentText(item.has("content") ? item.path("content") : item);
        builder.message(message(role, content, text(item, "name", null), text(item, "tool_call_id", null),
                toToolCalls(item.path("tool_calls"))));
    }

    private static Message toMessage(JsonNode node) {
        String role = text(node, "role", "user").toLowerCase(Locale.ROOT);
        String content = responseContentText(node.path("content"));
        String name = text(node, "name", null);
        String toolCallId = text(node, "tool_call_id", null);
        return message(role, content, name, toolCallId, toToolCalls(node.path("tool_calls")));
    }

    private static Message responsesFunctionCallMessage(JsonNode item) {
        String callId = text(item, "call_id", text(item, "id", "call_" + UUID.randomUUID()));
        JsonNode args = item.has("arguments") ? item.path("arguments") : item.path("input");
        ToolCall call = ToolCall.builder()
                .id(callId)
                .name(text(item, "name", "tool"))
                .type("function")
                .arguments(arguments(args))
                .build();
        return new Message(Message.Role.ASSISTANT, "", null, null, List.of(call), null);
    }

    private static Message responsesToolOutputMessage(JsonNode item) {
        String callId = text(item, "call_id", text(item, "tool_call_id", "tool_call"));
        JsonNode output = item.has("output") ? item.path("output") : item.path("content");
        return Message.tool(callId, responseContentText(output));
    }

    private static Message message(String role, String content, String name, String toolCallId, List<ToolCall> toolCalls) {
        String normalized = role == null ? "user" : role.toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "system", "developer" -> new Message(Message.Role.SYSTEM, content, null, name, null, null);
            case "assistant" -> new Message(Message.Role.ASSISTANT, content, null, name, toolCalls, null);
            case "tool" -> Message.tool(toolCallId != null ? toolCallId : "tool_call", content);
            case "function" -> new Message(Message.Role.FUNCTION, content, null, name, null, null);
            default -> new Message(Message.Role.USER, content, null, name, null, null);
        };
    }

    private static List<ToolCall> toToolCalls(JsonNode nodes) {
        if (!nodes.isArray()) {
            return null;
        }
        List<ToolCall> calls = new ArrayList<>();
        for (JsonNode node : nodes) {
            JsonNode function = node.path("function");
            String name = text(function, "name", text(node, "name", "tool"));
            calls.add(ToolCall.builder()
                    .id(text(node, "id", "call_" + UUID.randomUUID()))
                    .name(name)
                    .type(text(node, "type", "function"))
                    .arguments(arguments(function.path("arguments")))
                    .build());
        }
        return calls;
    }

    private static List<ToolDefinition> toTools(JsonNode nodes) {
        return AgentToolContractMapper.toToolDefinitions(nodes);
    }

    private static void applyCommonParameters(InferenceRequest.Builder builder, JsonNode root) {
        copyParameter(builder, root, "temperature");
        copyParameter(builder, root, "top_p");
        copyParameter(builder, root, "top_k");
        copyParameter(builder, root, "max_tokens");
        if (root.has("max_output_tokens") && !root.has("max_tokens")) {
            builder.parameter("max_tokens", toJava(root.path("max_output_tokens")));
        }
        copyParameter(builder, root, "presence_penalty");
        copyParameter(builder, root, "frequency_penalty");
        copyParameter(builder, root, "stop");
        copyParameter(builder, root, "response_format");
        copyParameter(builder, root, "rag_context");
        copyParameter(builder, root, "rag_sources");
        copyParameter(builder, root, "embedding_model");
        if (root.path("rag_enabled").isBoolean()) {
            builder.parameter("rag_enabled", root.path("rag_enabled").asBoolean());
            builder.metadata("rag_enabled", root.path("rag_enabled").asBoolean());
        }
    }

    private static void copyParameter(InferenceRequest.Builder builder, JsonNode root, String field) {
        if (root.has(field)) {
            builder.parameter(field, toJava(root.path(field)));
        }
    }

    private static void applyMetadata(InferenceRequest.Builder builder, JsonNode node) {
        for (Map.Entry<String, Object> entry : map(node).entrySet()) {
            builder.metadata(entry.getKey(), entry.getValue());
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Map.of();
        }
        return MAPPER.convertValue(node, Map.class);
    }

    private static Map<String, Object> arguments(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return Map.of();
        }
        if (node.isTextual()) {
            try {
                JsonNode parsed = MAPPER.readTree(node.asText());
                return map(parsed);
            } catch (JsonProcessingException ignored) {
                return Map.of("value", node.asText());
            }
        }
        return map(node);
    }

    private static List<String> embeddingInputs(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "input or inputs is required");
        }
        if (node.isArray()) {
            List<String> inputs = new ArrayList<>();
            for (JsonNode item : node) {
                inputs.add(item.isTextual() ? item.asText() : item.toString());
            }
            return inputs;
        }
        return List.of(node.isTextual() ? node.asText() : node.toString());
    }

    private static Map<String, Object> choice(InferenceResponse response) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", assistantMessage(response));
        choice.put("finish_reason", finishReason(response.getFinishReason()));
        return choice;
    }

    private static Map<String, Object> streamChoice(StreamingInferenceChunk chunk) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", streamDelta(chunk));
        choice.put("finish_reason", chunk.finished() ? finishReason(chunk.finishReason()) : null);
        return choice;
    }

    private static Map<String, Object> streamDelta(StreamingInferenceChunk chunk) {
        Map<String, Object> delta = new LinkedHashMap<>();
        if (chunk.index() == 0) {
            delta.put("role", "assistant");
        }
        if (!isBlank(chunk.delta())) {
            delta.put("content", chunk.delta());
        }
        return delta;
    }

    private static Map<String, Object> assistantMessage(InferenceResponse response) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", response.getContent());
        if (response.hasToolCalls()) {
            message.put("tool_calls", openAiToolCalls(response.getToolCalls()));
        }
        return message;
    }

    private static List<Map<String, Object>> openAiToolCalls(List<InferenceResponse.ToolCall> calls) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (int i = 0; i < calls.size(); i++) {
            InferenceResponse.ToolCall call = calls.get(i);
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", call.name());
            function.put("arguments", json(call.arguments()));
            Map<String, Object> toolCall = new LinkedHashMap<>();
            toolCall.put("id", "call_" + i + "_" + Math.abs(call.name().hashCode()));
            toolCall.put("type", "function");
            toolCall.put("function", function);
            out.add(toolCall);
        }
        return out;
    }

    private static List<Map<String, Object>> responseToolOutputItems(InferenceResponse response) {
        if (!response.hasToolCalls()) {
            return List.of();
        }
        List<Map<String, Object>> items = new ArrayList<>();
        for (int i = 0; i < response.getToolCalls().size(); i++) {
            InferenceResponse.ToolCall call = response.getToolCalls().get(i);
            String stableId = i + "_" + Math.abs(call.name().hashCode());
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id", "fc_" + stableId);
            item.put("type", "function_call");
            item.put("name", call.name());
            item.put("arguments", json(call.arguments()));
            item.put("call_id", "call_" + stableId);
            item.put("status", "completed");
            items.add(item);
        }
        return items;
    }

    private static String responseContentText(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return "";
        }
        if (node.isTextual()) {
            return node.asText();
        }
        if (node.isArray()) {
            List<String> parts = new ArrayList<>();
            for (JsonNode item : node) {
                String text = responseContentText(item);
                if (!isBlank(text)) {
                    parts.add(text);
                }
            }
            return String.join("\n", parts);
        }
        if (node.isObject()) {
            if (node.has("text")) {
                return node.path("text").asText();
            }
            if (node.has("input_text")) {
                return node.path("input_text").asText();
            }
            if (node.has("output_text")) {
                return node.path("output_text").asText();
            }
            if (node.has("content")) {
                return responseContentText(node.path("content"));
            }
        }
        return node.toString();
    }

    private static Map<String, Object> usage(InferenceResponse response) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", response.getInputTokens());
        usage.put("completion_tokens", response.getOutputTokens());
        usage.put("total_tokens", response.getTokensUsed());
        return usage;
    }

    private static Map<String, Object> streamUsage(StreamingInferenceChunk.ChunkUsage source) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", source.inputTokens());
        usage.put("completion_tokens", source.outputTokens());
        usage.put("total_tokens", source.inputTokens() + source.outputTokens());
        return usage;
    }

    private static String finishReason(InferenceResponse.FinishReason reason) {
        if (reason == null) {
            return "stop";
        }
        return switch (reason) {
            case TOOL_CALLS -> "tool_calls";
            case LENGTH -> "length";
            case ERROR -> "error";
            default -> "stop";
        };
    }

    private static String finishReason(String reason) {
        if (isBlank(reason)) {
            return "stop";
        }
        return switch (reason.trim().toLowerCase(Locale.ROOT)) {
            case "tool_calls" -> "tool_calls";
            case "length" -> "length";
            case "error" -> "error";
            case "content_filter" -> "content_filter";
            default -> "stop";
        };
    }

    private static void copyIfPresent(Map<String, Object> parameters, JsonNode root, String field) {
        if (root.has(field)) {
            parameters.put(field, toJava(root.path(field)));
        }
    }

    private static Object toJava(JsonNode node) {
        return MAPPER.convertValue(node, Object.class);
    }

    private static Iterable<JsonNode> iterable(JsonNode node) {
        if (node != null && node.isArray()) {
            return node;
        }
        return List.of();
    }

    private static String requiredText(JsonNode node, String field) {
        String value = text(node, field, null);
        if (isBlank(value)) {
            throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, field + " is required");
        }
        return value;
    }

    private static String text(JsonNode node, String field, String fallback) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || value.isNull() ? fallback : value.asText();
    }

    private static String json(Object value) {
        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return "{}";
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}
