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

/**
 * Categories for visual grouping of nodes in the Flutter graph editor.
 *
 * <p>Each category maps to a distinct visual section in the node palette,
 * with a display name, icon hint, and color hint that the Flutter UI
 * uses for consistent visual theming.</p>
 *
 * @since 3.0.0
 */
public enum NodeCategory {

    /**
     * Input/source nodes: user prompts, file readers, API triggers.
     */
    INPUT("Input", "input", "#4CAF50"),

    /**
     * Processing/transformation nodes: prompt templates, formatters, parsers.
     */
    PROCESSING("Processing", "settings", "#2196F3"),

    /**
     * Inference nodes: LLM invocation, embedding generation.
     */
    INFERENCE("Inference", "psychology", "#9C27B0"),

    /**
     * Routing/control flow: conditionals, switches, loops, fan-out.
     */
    CONTROL_FLOW("Control Flow", "call_split", "#FF9800"),

    /**
     * Memory/state: context windows, KV cache, conversation history.
     */
    MEMORY("Memory", "memory", "#00BCD4"),

    /**
     * Tool/function calling: external API calls, code execution.
     */
    TOOL("Tool", "build", "#795548"),

    /**
     * Validation/safety: content moderation, schema validation.
     */
    VALIDATION("Validation", "verified", "#F44336"),

    /**
     * Output/sink: response formatting, streaming output, file writers.
     */
    OUTPUT("Output", "output", "#607D8B"),

    /**
     * Optimization: caching, batching, speculative decoding.
     */
    OPTIMIZATION("Optimization", "speed", "#FF5722"),

    /**
     * Observability: logging, metrics, tracing.
     */
    OBSERVABILITY("Observability", "monitoring", "#8BC34A"),

    /**
     * Custom/third-party extensions.
     */
    CUSTOM("Custom", "extension", "#9E9E9E");

    private final String displayName;
    private final String iconHint;
    private final String colorHint;

    NodeCategory(String displayName, String iconHint, String colorHint) {
        this.displayName = displayName;
        this.iconHint = iconHint;
        this.colorHint = colorHint;
    }

    /**
     * Human-readable label for the Flutter node palette.
     */
    public String displayName() {
        return displayName;
    }

    /**
     * Material icon name hint for the Flutter UI.
     */
    public String iconHint() {
        return iconHint;
    }

    /**
     * Hex color hint for the Flutter UI node header.
     */
    public String colorHint() {
        return colorHint;
    }
}
