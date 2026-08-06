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
 *
 * @author Bhangun
 */

package tech.kayys.gollek.spi.node;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Complete visual and behavioral descriptor for a node type.
 *
 * <p>This record is serialized to JSON and sent to the Flutter UI to populate
 * the node palette, render node cards, and validate connections at design time.
 * It bundles port definitions, category, and UI metadata into a single
 * immutable snapshot.</p>
 *
 * <h2>Serialization</h2>
 * <p>This record is Jackson-friendly. The Flutter client deserializes it directly
 * to build the visual node registry.</p>
 *
 * @param nodeType        Unique type identifier (e.g., "gollek/llm-inference")
 * @param displayName     Human-readable label shown on the node header
 * @param description     Tooltip or detail panel description
 * @param category        Visual grouping category
 * @param iconHint        Material icon name override (null falls back to category icon)
 * @param colorHint       Hex color override (null falls back to category color)
 * @param inputs          Ordered list of input port descriptors
 * @param outputs         Ordered list of output port descriptors
 * @param defaultConfig   Default configuration key-value pairs for this node
 * @param tags            Searchable tags for the Flutter palette filter
 * @param version         Node type version (for schema evolution)
 * @param deprecated      Whether this node type is deprecated
 * @param deprecationNote Migration hint if deprecated
 *
 * @since 3.0.0
 */
public record NodeDescriptor(
        String nodeType,
        String displayName,
        String description,
        NodeCategory category,
        String iconHint,
        String colorHint,
        List<NodePort> inputs,
        List<NodePort> outputs,
        Map<String, Object> defaultConfig,
        List<String> tags,
        String version,
        boolean deprecated,
        String deprecationNote
) {

    /**
     * Canonical constructor with validation.
     */
    public NodeDescriptor {
        Objects.requireNonNull(nodeType, "nodeType is required");
        Objects.requireNonNull(displayName, "displayName is required");
        Objects.requireNonNull(category, "category is required");
        inputs = inputs != null ? List.copyOf(inputs) : List.of();
        outputs = outputs != null ? List.copyOf(outputs) : List.of();
        defaultConfig = defaultConfig != null ? Map.copyOf(defaultConfig) : Map.of();
        tags = tags != null ? List.copyOf(tags) : List.of();
        version = version != null ? version : "1.0.0";
        description = description != null ? description : "";
        deprecationNote = deprecationNote != null ? deprecationNote : "";
    }

    /**
     * Effective icon — falls back to the category's icon hint.
     */
    public String effectiveIcon() {
        return iconHint != null ? iconHint : category.iconHint();
    }

    /**
     * Effective color — falls back to the category's color hint.
     */
    public String effectiveColor() {
        return colorHint != null ? colorHint : category.colorHint();
    }

    /**
     * Get all ports (inputs + outputs) in a single list.
     */
    public List<NodePort> allPorts() {
        return java.util.stream.Stream.concat(inputs.stream(), outputs.stream()).toList();
    }

    /**
     * Create a builder pre-populated with required fields.
     */
    public static Builder builder(String nodeType, String displayName, NodeCategory category) {
        return new Builder(nodeType, displayName, category);
    }

    /**
     * Fluent builder for {@link NodeDescriptor}.
     */
    public static final class Builder {
        private final String nodeType;
        private final String displayName;
        private final NodeCategory category;
        private String description;
        private String iconHint;
        private String colorHint;
        private List<NodePort> inputs;
        private List<NodePort> outputs;
        private Map<String, Object> defaultConfig;
        private List<String> tags;
        private String version;
        private boolean deprecated = false;
        private String deprecationNote;

        private Builder(String nodeType, String displayName, NodeCategory category) {
            this.nodeType = nodeType;
            this.displayName = displayName;
            this.category = category;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder iconHint(String iconHint) {
            this.iconHint = iconHint;
            return this;
        }

        public Builder colorHint(String colorHint) {
            this.colorHint = colorHint;
            return this;
        }

        public Builder inputs(List<NodePort> inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder inputs(NodePort... inputs) {
            this.inputs = List.of(inputs);
            return this;
        }

        public Builder outputs(List<NodePort> outputs) {
            this.outputs = outputs;
            return this;
        }

        public Builder outputs(NodePort... outputs) {
            this.outputs = List.of(outputs);
            return this;
        }

        public Builder defaultConfig(Map<String, Object> defaultConfig) {
            this.defaultConfig = defaultConfig;
            return this;
        }

        public Builder tags(List<String> tags) {
            this.tags = tags;
            return this;
        }

        public Builder tags(String... tags) {
            this.tags = List.of(tags);
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder deprecated(boolean deprecated, String note) {
            this.deprecated = deprecated;
            this.deprecationNote = note;
            return this;
        }

        public NodeDescriptor build() {
            return new NodeDescriptor(
                    nodeType, displayName, description, category,
                    iconHint, colorHint, inputs, outputs,
                    defaultConfig, tags, version, deprecated, deprecationNote
            );
        }
    }
}
