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
 * @author bhangun
 */
package tech.kayys.golek.model.core;

import tech.kayys.golek.spi.model.ModelManifest;

import io.smallrye.mutiny.Uni;

import java.util.List;

public interface ModelRepository {
    Uni<ModelManifest> findById(String modelId, String tenantId);

    Uni<List<ModelManifest>> list(String tenantId, Pageable pageable);

    Uni<ModelManifest> save(ModelManifest manifest);

    Uni<Void> delete(String modelId, String tenantId);
}