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
package tech.kayys.golek.model.core;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.model.ModelArtifact;
import tech.kayys.golek.spi.model.ModelDescriptor;
import tech.kayys.golek.spi.model.ModelRef;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.Pageable;

import java.nio.file.Path;
import java.util.List;

public interface ModelRepository {
    Uni<ModelManifest> findById(String modelId, String requestId);

    Uni<List<ModelManifest>> list(String requestId, Pageable pageable);

    Uni<ModelManifest> save(ModelManifest manifest);

    Uni<Void> delete(String modelId, String requestId);

    ModelDescriptor resolve(ModelRef ref);

    ModelArtifact fetch(ModelDescriptor descriptor);

    boolean supports(ModelRef ref);

    Path downloadArtifact(ModelManifest manifest, ModelFormat format);

    boolean isCached(String modelId, ModelFormat format);

    void evictCache(String modelId, ModelFormat format);
}