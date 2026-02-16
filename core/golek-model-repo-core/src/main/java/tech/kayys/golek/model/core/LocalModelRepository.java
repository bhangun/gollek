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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ModelRef;
import tech.kayys.golek.spi.model.ModelDescriptor;
import tech.kayys.golek.spi.model.ModelArtifact;
import tech.kayys.golek.spi.model.Pageable;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.exception.InferenceException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Disk-based implementation of ModelRepository
 */
@ApplicationScoped
public class LocalModelRepository implements ModelRepository {

    private static final Logger LOG = Logger.getLogger(LocalModelRepository.class);
    private static final String MANIFEST_FILE = "manifest.json";

    private final Path rootPath;
    private final ObjectMapper objectMapper;

    @Inject
    public LocalModelRepository(
            @ConfigProperty(name = "golek.model.repo.path", defaultValue = "models") String repoPath) {
        this.rootPath = Path.of(repoPath);
        this.objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
        ensureDirectory(rootPath);
    }

    @Override
    public Uni<ModelManifest> findById(String modelId, String path) {
        return Uni.createFrom().item(() -> {
            Path manifestPath = getManifestPath(modelId, path);
            if (!Files.exists(manifestPath)) {
                return null;
            }
            try {
                return objectMapper.readValue(manifestPath.toFile(), ModelManifest.class);
            } catch (IOException e) {
                LOG.errorf(e, "Failed to read manifest for model %s", modelId);
                throw new InferenceException(ErrorCode.STORAGE_READ_FAILED,
                        "Failed to read manifest for model " + modelId, e)
                        .addContext("modelId", modelId)
                        .addContext("path", path)
                        .addContext("path", manifestPath.toString());
            }
        });
    }

    @Override
    public Uni<List<ModelManifest>> list(String path, Pageable pageable) {
        return Uni.createFrom().item(() -> {
            Path tenantPath = rootPath.resolve(path);
            if (!Files.exists(tenantPath)) {
                return List.of();
            }

            List<ModelManifest> manifests = new ArrayList<>();
            try (Stream<Path> modelDirs = Files.list(tenantPath)) {
                modelDirs.filter(Files::isDirectory)
                        .forEach(dir -> {
                            Path manifestPath = dir.resolve(MANIFEST_FILE);
                            if (Files.exists(manifestPath)) {
                                try {
                                    manifests.add(objectMapper.readValue(manifestPath.toFile(), ModelManifest.class));
                                } catch (IOException e) {
                                    LOG.warnf("Failed to read manifest in %s: %s", dir, e.getMessage());
                                }
                            }
                        });
            } catch (IOException e) {
                LOG.errorf(e, "Failed to list models for tenant %s", path);
                throw new InferenceException(ErrorCode.STORAGE_READ_FAILED,
                        "Failed to list models for tenant " + path, e)
                        .addContext("path", path)
                        .addContext("path", tenantPath.toString());
            }

            // Apply pagination
            int start = Math.min(pageable.offset(), manifests.size());
            int end = Math.min(start + pageable.size(), manifests.size());
            return manifests.subList(start, end);
        });
    }

    @Override
    public Uni<ModelManifest> save(ModelManifest manifest) {
        return Uni.createFrom().item(() -> {
            Path modelDir = rootPath.resolve(manifest.path()).resolve(manifest.modelId());
            ensureDirectory(modelDir);

            Path manifestPath = modelDir.resolve(MANIFEST_FILE);
            Path tempManifestPath = modelDir.resolve(MANIFEST_FILE + ".tmp");
            Path lockPath = modelDir.resolve(".lock");

            try {
                // Try to acquire lock
                if (!acquireLock(lockPath)) {
                    throw new InferenceException(ErrorCode.STORAGE_WRITE_FAILED,
                            "Could not acquire lock for model " + manifest.modelId())
                            .addContext("modelId", manifest.modelId())
                            .addContext("path", manifest.path())
                            .addContext("path", lockPath.toString());
                }

                try {
                    // Atomic write: Write to .tmp then move
                    objectMapper.writerWithDefaultPrettyPrinter().writeValue(tempManifestPath.toFile(), manifest);
                    Files.move(tempManifestPath, manifestPath, StandardCopyOption.REPLACE_EXISTING,
                            StandardCopyOption.ATOMIC_MOVE);
                    LOG.infof("Saved manifest for model %s (atomically)", manifest.modelId());
                    return manifest;
                } finally {
                    releaseLock(lockPath);
                }
            } catch (IOException e) {
                LOG.errorf(e, "Failed to save manifest for model %s", manifest.modelId());
                throw new InferenceException(ErrorCode.STORAGE_WRITE_FAILED,
                        "Failed to save model manifest", e)
                        .addContext("modelId", manifest.modelId())
                        .addContext("path", manifest.path())
                        .addContext("path", manifestPath.toString());
            }
        });
    }

    private boolean acquireLock(Path lockPath) throws IOException {
        int retries = 5;
        while (retries > 0) {
            try {
                Files.createFile(lockPath);
                return true;
            } catch (IOException e) {
                retries--;
                try {
                    Thread.sleep(200);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IOException(ie);
                }
            }
        }
        return false;
    }

    private void releaseLock(Path lockPath) {
        try {
            Files.deleteIfExists(lockPath);
        } catch (IOException e) {
            LOG.warnf("Failed to release lock: %s", lockPath);
        }
    }

    @Override
    public Uni<Void> delete(String modelId, String path) {
        return Uni.createFrom().item(() -> {
            Path modelDir = rootPath.resolve(path).resolve(modelId);
            if (Files.exists(modelDir)) {
                try (Stream<Path> files = Files.walk(modelDir)) {
                    files.sorted((a, b) -> b.compareTo(a)) // Delete files before directories
                            .forEach(p -> {
                                try {
                                    Files.delete(p);
                                } catch (IOException e) {
                                    LOG.warnf("Failed to delete %s: %s", p, e.getMessage());
                                }
                            });
                } catch (IOException e) {
                    LOG.errorf(e, "Failed to delete model directory %s", modelDir);
                    throw new InferenceException(ErrorCode.STORAGE_WRITE_FAILED,
                            "Failed to delete model", e)
                            .addContext("modelId", modelId)
                            .addContext("path", path)
                            .addContext("path", modelDir.toString());
                }
            }
            return null;
        });
    }

    @Override
    public ModelDescriptor resolve(ModelRef ref) {
        return new ModelDescriptor(
                ref.name(),
                ref.parameters().getOrDefault("format", "gguf"),
                java.net.URI.create("file://" + rootPath.resolve(ref.name()).toString()),
                ref.parameters());
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        Path modelPath = rootPath.resolve(descriptor.id());
        return new ModelArtifact(modelPath, null, descriptor.metadata());
    }

    @Override
    public boolean supports(ModelRef ref) {
        return "local".equalsIgnoreCase(ref.scheme()) || ref.scheme() == null;
    }

    @Override
    public Path downloadArtifact(ModelManifest manifest, ModelFormat format) {
        Path modelDir = rootPath.resolve(manifest.path()).resolve(manifest.modelId());
        Path artifactPath = modelDir.resolve(format.toString().toLowerCase());
        if (Files.exists(artifactPath)) {
            return artifactPath;
        }
        throw new InferenceException(ErrorCode.MODEL_NOT_FOUND, "Artifact not found for format: " + format)
                .addContext("modelId", manifest.modelId())
                .addContext("format", format.toString());
    }

    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        // For local repo, "cached" means the file exists.
        // We check across all tenants if it's a shared repository or specific ones.
        // Implementation simplified: check in any tenant dir.
        try (Stream<Path> tenants = Files.list(rootPath)) {
            return tenants.filter(Files::isDirectory)
                    .anyMatch(tenantDir -> Files
                            .exists(tenantDir.resolve(modelId).resolve(format.toString().toLowerCase())));
        } catch (IOException e) {
            return false;
        }
    }

    @Override
    public void evictCache(String modelId, ModelFormat format) {
        // Find and delete the artifact from any tenant directories
        try (Stream<Path> tenants = Files.list(rootPath)) {
            tenants.filter(Files::isDirectory)
                    .forEach(tenantDir -> {
                        try {
                            Files.deleteIfExists(tenantDir.resolve(modelId).resolve(format.toString().toLowerCase()));
                        } catch (IOException e) {
                            LOG.warnf("Failed to evict cache for model %s: %s", modelId, e.getMessage());
                        }
                    });
        } catch (IOException e) {
            LOG.errorf(e, "Failed to list tenants for cache eviction");
        }
    }

    private Path getManifestPath(String modelId, String path) {
        return rootPath.resolve(path).resolve(modelId).resolve(MANIFEST_FILE);
    }

    private void ensureDirectory(Path path) {
        try {
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            throw new InferenceException(ErrorCode.STORAGE_WRITE_FAILED,
                    "Failed to create directory: " + path, e)
                    .addContext("path", path.toString());
        }
    }
}
