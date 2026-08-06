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
import java.util.Objects;

/**
 * Describes a single input or output port on a {@link NodePlugin}.
 *
 * <p>
 * Ports are the connection points in a visual graph editor. Each port has a
 * unique name within its node, a data type for type-safe wiring, and optional
 * constraints that the Flutter UI uses for validation and rendering.
 * </p>
 *
 * <h2>Example</h2>
 * 
 * <pre>{@code
 * NodePort promptInput = NodePort.builder("prompt", Direction.INPUT, DataType.STRING)
 *         .displayName("User Prompt")
 *         .description("The user's natural language input")
 *         .required(true)
 *         .build();
 * }</pre>
 *
 * @since 3.0.0
 */
public final class NodePort {

    /**
     * Port direction relative to the node.
     */
    public enum Direction {
        INPUT,
        OUTPUT
    }

    /**
     * Supported data types that flow through ports.
     * Used by the UI for type-safe connection validation.
     */
    public enum DataType {
        /** Plain text / prompt string */
        STRING,
        /** Structured JSON object */
        JSON,
        /** Chat message list (conversation history) */
        MESSAGE_LIST,
        /** Full inference request */
        INFERENCE_REQUEST,
        /** Full inference response */
        INFERENCE_RESPONSE,
        /** Streaming chunk */
        STREAMING_CHUNK,
        /** Embedding vector */
        EMBEDDING,
        /** Binary blob (images, audio, etc.) */
        BINARY,
        /** Boolean flag / signal */
        BOOLEAN,
        /** Numeric value (int or float) */
        NUMBER,
        /** Arbitrary object (for advanced plugins) */
        ANY
    }

    private final String name;
    private final Direction direction;
    private final DataType dataType;
    private final String displayName;
    private final String description;
    private final boolean required;
    private final boolean multiConnect;
    private final Object defaultValue;
    private final List<String> acceptedTypes;

    private NodePort(Builder builder) {
        this.name = Objects.requireNonNull(builder.name, "Port name is required");
        this.direction = Objects.requireNonNull(builder.direction, "Port direction is required");
        this.dataType = Objects.requireNonNull(builder.dataType, "Port data type is required");
        this.displayName = builder.displayName != null ? builder.displayName : builder.name;
        this.description = builder.description != null ? builder.description : "";
        this.required = builder.required;
        this.multiConnect = builder.multiConnect;
        this.defaultValue = builder.defaultValue;
        this.acceptedTypes = builder.acceptedTypes != null
                ? List.copyOf(builder.acceptedTypes)
                : List.of();
    }

    // --- Accessors ---

    /** Unique port name within the node (used as wire key). */
    public String name() {
        return name;
    }

    /** Port direction (INPUT or OUTPUT). */
    public Direction direction() {
        return direction;
    }

    /** Primary data type carried by this port. */
    public DataType dataType() {
        return dataType;
    }

    /** Human-readable label for the Flutter UI. */
    public String displayName() {
        return displayName;
    }

    /** Tooltip description for the Flutter UI. */
    public String description() {
        return description;
    }

    /** Whether this port must be connected for the node to execute. */
    public boolean required() {
        return required;
    }

    /** Whether multiple connections can attach to this port. */
    public boolean multiConnect() {
        return multiConnect;
    }

    /** Default value used when no wire is connected (nullable). */
    public Object defaultValue() {
        return defaultValue;
    }

    /**
     * Additional data type names this input port accepts beyond its primary type.
     * Empty list means only the primary {@link #dataType()} is accepted.
     */
    public List<String> acceptedTypes() {
        return acceptedTypes;
    }

    /**
     * Check if this port can accept a connection from the given source port.
     */
    public boolean isCompatibleWith(NodePort source) {
        if (this.direction == source.direction) {
            return false; // Cannot connect same-direction ports
        }
        if (this.dataType == DataType.ANY || source.dataType == DataType.ANY) {
            return true;
        }
        if (this.dataType == source.dataType) {
            return true;
        }
        return acceptedTypes.contains(source.dataType.name());
    }

    // --- Builder ---

    public static Builder builder(String name, Direction direction, DataType dataType) {
        return new Builder(name, direction, dataType);
    }

    /** Convenience factory for a required input port. */
    public static NodePort requiredInput(String name, DataType dataType) {
        return builder(name, Direction.INPUT, dataType).required(true).build();
    }

    /** Convenience factory for an output port. */
    public static NodePort output(String name, DataType dataType) {
        return builder(name, Direction.OUTPUT, dataType).build();
    }

    public static final class Builder {
        private final String name;
        private final Direction direction;
        private final DataType dataType;
        private String displayName;
        private String description;
        private boolean required = false;
        private boolean multiConnect = false;
        private Object defaultValue;
        private List<String> acceptedTypes;

        private Builder(String name, Direction direction, DataType dataType) {
            this.name = name;
            this.direction = direction;
            this.dataType = dataType;
        }

        public Builder displayName(String displayName) {
            this.displayName = displayName;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder required(boolean required) {
            this.required = required;
            return this;
        }

        public Builder multiConnect(boolean multiConnect) {
            this.multiConnect = multiConnect;
            return this;
        }

        public Builder defaultValue(Object defaultValue) {
            this.defaultValue = defaultValue;
            return this;
        }

        public Builder acceptedTypes(List<String> acceptedTypes) {
            this.acceptedTypes = acceptedTypes;
            return this;
        }

        public NodePort build() {
            return new NodePort(this);
        }
    }

    @Override
    public String toString() {
        return "NodePort{" +
                "name='" + name + '\'' +
                ", direction=" + direction +
                ", dataType=" + dataType +
                ", required=" + required +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof NodePort that))
            return false;
        return name.equals(that.name) && direction == that.direction;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, direction);
    }
}
