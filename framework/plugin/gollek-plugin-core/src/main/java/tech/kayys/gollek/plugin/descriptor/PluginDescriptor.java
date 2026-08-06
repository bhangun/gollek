/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.descriptor;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Plugin descriptor containing metadata and configuration.
 *
 * @param id Plugin ID
 * @param name Plugin name
 * @param version Plugin version
 * @param description Plugin description
 * @param vendor Plugin vendor
 * @param mainClass Main plugin class
 * @param dependencies Plugin dependencies
 * @param optionalDependencies Optional dependencies
 * @param properties Plugin properties
 *
 * @since 2.1.0
 */
public record PluginDescriptor(
        String id,
        String name,
        String version,
        String description,
        String vendor,
        String mainClass,
        List<String> dependencies,
        List<String> optionalDependencies,
        Map<String, Object> properties
) {

    /**
     * Create descriptor with minimal information.
     */
    public PluginDescriptor {
        if (dependencies == null) {
            dependencies = Collections.emptyList();
        }
        if (optionalDependencies == null) {
            optionalDependencies = Collections.emptyList();
        }
        if (properties == null) {
            properties = Collections.emptyMap();
        }
    }

    /**
     * Parse descriptor from JSON.
     *
     * @param json JSON string
     * @return Plugin descriptor
     */
    public static PluginDescriptor fromJson(String json) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(json);
            return new PluginDescriptor(
                    text(root, "id", "unknown"),
                    text(root, "name", "Unknown Plugin"),
                    text(root, "version", "1.0.0"),
                    text(root, "description", "Plugin without descriptor"),
                    text(root, "vendor", "Unknown"),
                    text(root, "mainClass", "tech.kayys.gollek.plugin.UnknownPlugin"),
                    strings(mapper, root.get("dependencies")),
                    strings(mapper, root.get("optionalDependencies")),
                    properties(mapper, root.get("properties")));
        } catch (Exception e) {
            return new PluginDescriptor(
                    "unknown",
                    "Unknown Plugin",
                    "1.0.0",
                    "Plugin without descriptor",
                    "Unknown",
                    "tech.kayys.gollek.plugin.UnknownPlugin",
                    Collections.emptyList(),
                    Collections.emptyList(),
                    Collections.emptyMap());
        }
    }

    private static String text(JsonNode root, String field, String fallback) {
        JsonNode value = root == null ? null : root.get(field);
        return value != null && value.isTextual() && !value.asText().isBlank()
                ? value.asText()
                : fallback;
    }

    private static List<String> strings(ObjectMapper mapper, JsonNode node) {
        if (node == null || !node.isArray()) {
            return Collections.emptyList();
        }
        return mapper.convertValue(node, new TypeReference<List<String>>() {
        });
    }

    private static Map<String, Object> properties(ObjectMapper mapper, JsonNode node) {
        if (node == null || !node.isObject()) {
            return Collections.emptyMap();
        }
        return mapper.convertValue(node, new TypeReference<Map<String, Object>>() {
        });
    }
}
