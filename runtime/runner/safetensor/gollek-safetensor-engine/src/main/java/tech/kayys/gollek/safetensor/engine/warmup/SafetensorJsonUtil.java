package tech.kayys.gollek.safetensor.engine.warmup;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.*;

/**
 * Utility class for parsing SafeTensor configuration JSON nodes.
 */
public final class SafetensorJsonUtil {

    private SafetensorJsonUtil() {
        // Utility class
    }

    public static Optional<Integer> extractInt(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isInt()) {
            return Optional.of(value.asInt());
        }
        return Optional.empty();
    }

    public static Optional<Float> extractFloat(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isNumber()) {
            return Optional.of((float) value.asDouble());
        }
        return Optional.empty();
    }

    public static Optional<Boolean> extractBoolean(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isBoolean()) {
            return Optional.of(value.asBoolean());
        }
        return Optional.empty();
    }

    public static Optional<String> extractString(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isTextual()) {
            return Optional.of(value.asText());
        }
        return Optional.empty();
    }

    public static List<String> extractStringArray(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isArray()) {
            List<String> result = new ArrayList<>();
            for (JsonNode element : value) {
                if (element.isTextual()) {
                    result.add(element.asText());
                }
            }
            return result;
        }
        return List.of();
    }

    public static Map<String, Object> extractAdditionalProperties(JsonNode root) {
        Map<String, Object> properties = new HashMap<>();
        Set<String> knownFields = Set.of(
                "r", "rank", "lora_alpha", "lora_dropout", "bias",
                "target_modules", "task_type", "inference_mode",
                "base_model_name_or_path", "peft_type");

        for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String key = entry.getKey();
            if (!knownFields.contains(key)) {
                properties.put(key, toJsonValue(entry.getValue()));
            }
        }
        return properties;
    }

    public static Object toJsonValue(JsonNode node) {
        if (node.isTextual()) return node.asText();
        if (node.isInt()) return node.asInt();
        if (node.isLong()) return node.asLong();
        if (node.isFloatingPointNumber()) return (float) node.asDouble();
        if (node.isDouble()) return node.asDouble();
        if (node.isBoolean()) return node.asBoolean();
        if (node.isArray()) {
            List<Object> list = new ArrayList<>();
            for (JsonNode element : node) {
                list.add(toJsonValue(element));
            }
            return list;
        }
        if (node.isObject()) {
            Map<String, Object> map = new HashMap<>();
            for (Iterator<Map.Entry<String, JsonNode>> it = node.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> entry = it.next();
                map.put(entry.getKey(), toJsonValue(entry.getValue()));
            }
            return map;
        }
        return null;
    }
}
