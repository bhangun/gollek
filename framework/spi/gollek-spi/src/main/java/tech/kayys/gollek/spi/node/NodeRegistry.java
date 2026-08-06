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
import java.util.Optional;

/**
 * Registry of available {@link NodePlugin} types.
 *
 * <p>The graph executor and the Flutter REST API both query this registry to:
 * <ul>
 *   <li>Discover all available node types for the palette</li>
 *   <li>Resolve a node type by ID for graph execution</li>
 *   <li>Filter nodes by category for UI grouping</li>
 * </ul>
 *
 * <p>Implementations should be CDI beans (e.g., {@code @ApplicationScoped})
 * and auto-discover {@code NodePlugin} instances via CDI injection or
 * ServiceLoader.</p>
 *
 * @since 3.0.0
 */
public interface NodeRegistry {

    /**
     * Register a node plugin.
     *
     * @param node the node plugin to register
     * @throws IllegalArgumentException if a node with the same ID is already registered
     */
    void register(NodePlugin node);

    /**
     * Unregister a node plugin by ID.
     *
     * @param nodeId the node plugin ID
     * @return the removed node, or empty if not found
     */
    Optional<NodePlugin> unregister(String nodeId);

    /**
     * Look up a node plugin by its unique ID.
     *
     * @param nodeId the node type ID (e.g., "gollek/llm-inference")
     * @return the node plugin, or empty if not found
     */
    Optional<NodePlugin> get(String nodeId);

    /**
     * Get all registered node plugins.
     *
     * @return unmodifiable list of all registered nodes
     */
    List<NodePlugin> all();

    /**
     * Get all node descriptors (for sending to the Flutter UI).
     *
     * @return list of descriptors for all registered nodes
     */
    default List<NodeDescriptor> allDescriptors() {
        return all().stream()
                .map(NodePlugin::descriptor)
                .toList();
    }

    /**
     * Get nodes filtered by category.
     *
     * @param category the category to filter by
     * @return list of matching node plugins
     */
    default List<NodePlugin> byCategory(NodeCategory category) {
        return all().stream()
                .filter(n -> n.category() == category)
                .toList();
    }

    /**
     * Search nodes by tag.
     *
     * @param tag the tag to search for
     * @return list of matching node plugins
     */
    default List<NodePlugin> byTag(String tag) {
        return all().stream()
                .filter(n -> n.descriptor().tags().contains(tag))
                .toList();
    }

    /**
     * Check if a node type is registered.
     *
     * @param nodeId the node type ID
     * @return true if registered
     */
    default boolean contains(String nodeId) {
        return get(nodeId).isPresent();
    }

    /**
     * Get the total number of registered nodes.
     */
    default int size() {
        return all().size();
    }
}
