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

package tech.kayys.gollek.model.core;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.model.download.DownloadProgressListener;

import java.nio.file.Path;
import java.util.List;

/**
 * Interface for remote model repositories (e.g., HuggingFace, S3)
 */
public interface RemoteModelRepository {

    String type();

    Uni<ModelManifest> fetchMetadata(String modelId, String requestId);

    Uni<List<ModelManifest>> search(String query, String requestId);

    Uni<Path> downloadArtifact(ModelManifest manifest, String artifactId, Path targetDir,
            DownloadProgressListener listener);
}
