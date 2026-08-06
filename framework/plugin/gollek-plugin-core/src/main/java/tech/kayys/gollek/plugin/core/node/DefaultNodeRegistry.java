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

package tech.kayys.gollek.plugin.core.node;

import jakarta.inject.Singleton;
import tech.kayys.gollek.spi.node.NodePlugin;
import tech.kayys.gollek.spi.node.NodeRegistry;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Default in-memory implementation of {@link NodeRegistry}.
 *
 * <p>Thread-safe via {@link ConcurrentHashMap}. Discovered {@link NodePlugin}
 * instances are registered during application startup, and the registry is
 * queried by the graph executor at runtime and the REST API for palette data.</p>
 *
 * @since 3.0.0
 */
@Singleton
public class DefaultNodeRegistry implements NodeRegistry {

    private final Map<String, NodePlugin> nodes = new ConcurrentHashMap<>();

    @Override
    public void register(NodePlugin node) {
        if (node == null) {
            throw new IllegalArgumentException("NodePlugin must not be null");
        }
        NodePlugin existing = nodes.putIfAbsent(node.id(), node);
        if (existing != null) {
            throw new IllegalArgumentException(
                    "Node with ID '" + node.id() + "' is already registered: " +
                            existing.getClass().getName());
        }
    }

    @Override
    public Optional<NodePlugin> unregister(String nodeId) {
        return Optional.ofNullable(nodes.remove(nodeId));
    }

    @Override
    public Optional<NodePlugin> get(String nodeId) {
        return Optional.ofNullable(nodes.get(nodeId));
    }

    @Override
    public List<NodePlugin> all() {
        return List.copyOf(nodes.values());
    }

    @Override
    public String toString() {
        return "DefaultNodeRegistry{registered=" + nodes.size() + "}";
    }
}
