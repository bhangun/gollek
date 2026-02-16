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
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.model.download.DownloadProgressListener;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.model.exception.InferenceException;

import java.util.Optional;

/**
 * Service to synchronize models between remote and local repositories
 */
@ApplicationScoped
public class ModelSyncService {

    private static final Logger LOG = Logger.getLogger(ModelSyncService.class);

    @Inject
    LocalModelRepository localRepo;

    @Inject
    Instance<RemoteModelRepository> remoteRepos;

    /**
     * Sync model from remote to local
     */
    public Uni<ModelManifest> sync(String modelId, String requestId, String remoteType,
            DownloadProgressListener listener) {
        RemoteModelRepository remoteRepo = findRemoteRepo(remoteType)
                .orElseThrow(() -> new InferenceException(
                        ErrorCode.CONFIG_UNSUPPORTED,
                        "Unsupported remote repository type: " + remoteType)
                        .addContext("remoteType", remoteType));

        LOG.infof("Starting sync for model %s from %s", modelId, remoteType);

        return remoteRepo.fetchMetadata(modelId, requestId)
                .flatMap(manifest -> {
                    // Start download
                    // For now, we assume we download all artifacts in the manifest
                    // In a more advanced impl, we might filter by format/device
                    return downloadAllArtifacts(remoteRepo, manifest, listener)
                            .replaceWith(manifest);
                })
                .flatMap(manifest -> localRepo.save(manifest))
                .onItem().invoke(m -> LOG.infof("Successfully synced model %s", m.modelId()));
    }

    private Uni<Void> downloadAllArtifacts(RemoteModelRepository remoteRepo, ModelManifest manifest,
            DownloadProgressListener listener) {
        // Simple implementation: download sequentially for now
        // Advanced: parallel downloads
        Uni<Void> chain = Uni.createFrom().nullItem();

        for (String artifactId : manifest.artifacts().keySet().stream().map(Enum::name).toList()) {
            chain = chain
                    .flatMap(v -> remoteRepo.downloadArtifact(manifest, artifactId, null, listener).replaceWithVoid());
        }

        return chain;
    }

    private Optional<RemoteModelRepository> findRemoteRepo(String type) {
        return remoteRepos.stream()
                .filter(r -> r.type().equalsIgnoreCase(type))
                .findFirst();
    }
}
