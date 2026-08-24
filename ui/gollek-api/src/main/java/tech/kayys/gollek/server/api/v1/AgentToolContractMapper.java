package tech.kayys.gollek.server.api.v1;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

final class AgentToolContractMapper {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern TOOL_NAME = Pattern.compile("[A-Za-z0-9_-]{1,64}");
    private static final List<String> UNSUPPORTED_SCHEMA_KEYWORDS = List.of(
            "$ref",
            "$defs",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "patternProperties",
            "dependentSchemas",
            "dependencies",
            "unevaluatedProperties");

    private AgentToolContractMapper() {
    }

    static Map<String, Object> validatePayload(JsonNode payload, AgentTraceContext trace) {
        ToolValidationResult result = validateTools(toolArray(payload));
        result.throwIfInvalid();

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("object", "gollek.tool_contract_validation");
        body.put("valid", true);
        body.put("model_invoked", false);
        body.put("trace", trace.asMap());
        body.put("tool_count", result.tools().size());
        body.put("normalized", result.normalized());
        body.put("warnings", result.warnings());
        body.put("boundary", boundary());
        return body;
    }

    static List<ToolDefinition> toToolDefinitions(JsonNode toolsNode) {
        ToolValidationResult result = validateTools(toolsNode);
        result.throwIfInvalid();
        return result.tools();
    }

    static Map<String, Object> summary(JsonNode toolsNode) {
        ToolValidationResult result = validateTools(toolsNode);
        result.throwIfInvalid();
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("valid", true);
        summary.put("tool_count", result.tools().size());
        summary.put("warning_count", result.warnings().size());
        summary.put("warnings", result.warnings());
        summary.put("tool_execution", false);
        return summary;
    }

    private static ToolValidationResult validateTools(JsonNode toolsNode) {
        if (toolsNode == null || toolsNode.isMissingNode() || toolsNode.isNull()) {
            return new ToolValidationResult(List.of(), List.of(), List.of(), List.of());
        }
        if (!toolsNode.isArray()) {
            return new ToolValidationResult(
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of("tools must be an array"));
        }

        List<ToolDefinition> tools = new ArrayList<>();
        List<Map<String, Object>> normalized = new ArrayList<>();
        List<Map<String, Object>> warnings = new ArrayList<>();
        List<String> errors = new ArrayList<>();

        for (int i = 0; i < toolsNode.size(); i++) {
            JsonNode node = toolsNode.get(i);
            ToolParseResult parsed = parseTool(i, node);
            errors.addAll(parsed.errors());
            warnings.addAll(parsed.warnings());
            if (parsed.tool() != null) {
                tools.add(parsed.tool());
                normalized.add(parsed.normalized());
            }
        }
        return new ToolValidationResult(tools, normalized, warnings, errors);
    }

    private static ToolParseResult parseTool(int index, JsonNode node) {
        List<Map<String, Object>> warnings = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        if (node == null || !node.isObject()) {
            errors.add("tools[" + index + "] must be an object");
            return new ToolParseResult(null, Map.of(), warnings, errors);
        }

        String rawType = text(node, "type", "function");
        ToolDefinition.Type type = toolType(rawType);
        if (type == null) {
            errors.add("tools[" + index + "].type must be one of function, mcp_tool, code_interpreter, or file_search");
            return new ToolParseResult(null, Map.of(), warnings, errors);
        }

        JsonNode function = node.has("function") ? node.path("function") : node;
        if (node.has("function") && !function.isObject()) {
            errors.add("tools[" + index + "].function must be an object");
            return new ToolParseResult(null, Map.of(), warnings, errors);
        }

        String name = text(function, "name", text(node, "name", defaultName(type)));
        if (isBlank(name)) {
            errors.add("tools[" + index + "].function.name is required");
        } else if (!TOOL_NAME.matcher(name).matches()) {
            errors.add("tools[" + index + "].function.name must match [A-Za-z0-9_-]{1,64}");
        }

        boolean strict = strict(index, node, function, errors);
        JsonNode parametersNode = function.has("parameters") ? function.path("parameters") : null;
        Map<String, Object> parameters = parameters(index, parametersNode, strict, warnings, errors);
        if (!errors.isEmpty()) {
            return new ToolParseResult(null, Map.of(), warnings, errors);
        }

        Map<String, Object> metadata = metadata(node, function);
        ToolDefinition.Builder builder = ToolDefinition.builder()
                .name(name)
                .type(type)
                .description(text(function, "description", null))
                .parameters(parameters)
                .strict(strict);
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            builder.metadata(entry.getKey(), entry.getValue());
        }
        ToolDefinition tool = builder.build();

        Map<String, Object> normalized = normalized(index, tool, parameters, warnings);
        return new ToolParseResult(tool, normalized, warnings, errors);
    }

    private static Map<String, Object> normalized(
            int index, ToolDefinition tool, Map<String, Object> parameters, List<Map<String, Object>> warnings) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("index", index);
        out.put("name", tool.getName());
        out.put("type", tool.getType().name().toLowerCase(Locale.ROOT));
        out.put("strict", tool.isStrict());
        out.put("description_present", tool.getDescription().isPresent());
        out.put("parameter_keys", sortedKeys(parameters));
        out.put("parameter_schema", parameterSchemaSummary(parameters));
        out.put("metadata", tool.getMetadata());
        out.put("warning_count", warnings.stream()
                .filter(warning -> Integer.valueOf(index).equals(warning.get("index")))
                .count());
        return out;
    }

    private static Map<String, Object> parameterSchemaSummary(Map<String, Object> parameters) {
        Map<String, Object> schema = new LinkedHashMap<>();
        Object type = parameters.get("type");
        if (type != null) {
            schema.put("type", type);
        }
        Object properties = parameters.get("properties");
        if (properties instanceof Map<?, ?> map) {
            schema.put("property_count", map.size());
        }
        Object required = parameters.get("required");
        if (required instanceof List<?> list) {
            schema.put("required", list);
        }
        List<String> unsupported = UNSUPPORTED_SCHEMA_KEYWORDS.stream()
                .filter(parameters::containsKey)
                .toList();
        if (!unsupported.isEmpty()) {
            schema.put("unsupported_keywords", unsupported);
        }
        return schema;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parameters(int index, JsonNode parametersNode, boolean strict,
            List<Map<String, Object>> warnings, List<String> errors) {
        if (parametersNode == null || parametersNode.isMissingNode() || parametersNode.isNull()) {
            warn(warnings, index, null, "parameters_missing",
                    "Tool parameters are omitted; model callers may send unconstrained arguments.");
            return Map.of();
        }
        if (!parametersNode.isObject()) {
            errors.add("tools[" + index + "].function.parameters must be an object");
            return Map.of();
        }

        Map<String, Object> parameters = MAPPER.convertValue(parametersNode, Map.class);
        Object schemaType = parameters.get("type");
        if (schemaType != null && !"object".equals(schemaType)) {
            warn(warnings, index, null, "schema_type_not_object",
                    "Tool parameter schema type is not object; some agent clients may reject it.");
        }
        if (strict && !"object".equals(schemaType)) {
            warn(warnings, index, null, "strict_schema_without_object_type",
                    "Strict tools should use a top-level object schema.");
        }

        JsonNode required = parametersNode.path("required");
        if (!required.isMissingNode()) {
            if (!required.isArray()) {
                errors.add("tools[" + index + "].function.parameters.required must be an array of strings");
            } else {
                for (JsonNode item : required) {
                    if (!item.isTextual()) {
                        errors.add("tools[" + index + "].function.parameters.required must be an array of strings");
                        break;
                    }
                }
            }
        }
        JsonNode properties = parametersNode.path("properties");
        if (!properties.isMissingNode() && !properties.isObject()) {
            errors.add("tools[" + index + "].function.parameters.properties must be an object");
        }
        JsonNode additionalProperties = parametersNode.path("additionalProperties");
        if (!additionalProperties.isMissingNode()
                && !additionalProperties.isBoolean()
                && !additionalProperties.isObject()) {
            errors.add("tools[" + index + "].function.parameters.additionalProperties must be a boolean or object");
        }

        collectSchemaWarnings(index, null, parametersNode, warnings);
        return parameters;
    }

    private static void collectSchemaWarnings(
            int index, String path, JsonNode node, List<Map<String, Object>> warnings) {
        if (node == null || !node.isObject()) {
            return;
        }
        for (String keyword : UNSUPPORTED_SCHEMA_KEYWORDS) {
            if (node.has(keyword)) {
                String fullPath = isBlank(path) ? keyword : path + "." + keyword;
                warn(warnings, index, fullPath, "schema_feature_may_be_ignored",
                        "JSON Schema keyword '" + keyword + "' may not be supported by every agent client.");
            }
        }
        node.fields().forEachRemaining(entry -> {
            String childPath = isBlank(path) ? entry.getKey() : path + "." + entry.getKey();
            JsonNode child = entry.getValue();
            if (child.isObject()) {
                collectSchemaWarnings(index, childPath, child, warnings);
            } else if (child.isArray()) {
                for (int i = 0; i < child.size(); i++) {
                    collectSchemaWarnings(index, childPath + "[" + i + "]", child.get(i), warnings);
                }
            }
        });
    }

    private static boolean strict(int index, JsonNode node, JsonNode function, List<String> errors) {
        JsonNode strict = function.has("strict") ? function.path("strict") : node.path("strict");
        if (strict.isMissingNode() || strict.isNull()) {
            return false;
        }
        if (!strict.isBoolean()) {
            errors.add("tools[" + index + "].function.strict must be a boolean");
            return false;
        }
        return strict.asBoolean();
    }

    private static Map<String, Object> metadata(JsonNode node, JsonNode function) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.putAll(map(node.path("metadata")));
        metadata.putAll(map(function.path("metadata")));
        metadata.putAll(map(node.path("x_gollek")));
        metadata.putAll(map(function.path("x_gollek")));
        return metadata;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Map.of();
        }
        return MAPPER.convertValue(node, Map.class);
    }

    private static JsonNode toolArray(JsonNode payload) {
        if (payload == null || payload.isMissingNode() || payload.isNull()) {
            throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "tools array is required");
        }
        if (payload.isArray()) {
            return payload;
        }
        if (payload.has("tools")) {
            return payload.path("tools");
        }
        throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "tools array is required");
    }

    private static ToolDefinition.Type toolType(String raw) {
        String normalized = raw == null ? "" : raw.trim().toLowerCase(Locale.ROOT).replace('-', '_');
        return switch (normalized) {
            case "function" -> ToolDefinition.Type.FUNCTION;
            case "mcp", "mcp_tool" -> ToolDefinition.Type.MCP_TOOL;
            case "code_interpreter" -> ToolDefinition.Type.CODE_INTERPRETER;
            case "file_search" -> ToolDefinition.Type.FILE_SEARCH;
            default -> null;
        };
    }

    private static String defaultName(ToolDefinition.Type type) {
        return switch (type) {
            case CODE_INTERPRETER -> "code_interpreter";
            case FILE_SEARCH -> "file_search";
            default -> null;
        };
    }

    private static Map<String, Object> boundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", true);
        boundary.put("tool_execution", false);
        boundary.put("tool_authorization", false);
        return boundary;
    }

    private static void warn(List<Map<String, Object>> warnings, int index, String path, String code, String message) {
        Map<String, Object> warning = new LinkedHashMap<>();
        warning.put("index", index);
        if (!isBlank(path)) {
            warning.put("path", "tools[" + index + "].function.parameters." + path);
        }
        warning.put("code", code);
        warning.put("message", message);
        warnings.add(warning);
    }

    private static List<String> sortedKeys(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return List.of();
        }
        return map.keySet().stream().sorted().toList();
    }

    private static String text(JsonNode node, String field, String fallback) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || value.isNull() ? fallback : value.asText();
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    record ToolValidationResult(
            List<ToolDefinition> tools,
            List<Map<String, Object>> normalized,
            List<Map<String, Object>> warnings,
            List<String> errors) {
        void throwIfInvalid() {
            if (!errors.isEmpty()) {
                throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, errors.get(0));
            }
        }
    }

    private record ToolParseResult(
            ToolDefinition tool,
            Map<String, Object> normalized,
            List<Map<String, Object>> warnings,
            List<String> errors) {
    }
}
