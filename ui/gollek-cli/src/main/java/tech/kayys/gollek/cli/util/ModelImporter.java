package tech.kayys.gollek.cli.util;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.model.repo.local.GollekManifest;
import tech.kayys.gollek.model.repo.local.ManifestStore;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Instant;
import java.util.List;

/**
 * Utility to import (move) or copy model files/directories into the
 * gollek model repository at {@code ~/.gollek/models/blobs/}.
 * <p>
 * Registers models using the manifest mechanism.
 */
@ApplicationScoped
public class ModelImporter {

    @Inject
    ManifestStore manifestStore;

    public ModelImporter() {}

    /**
     * Import a model file or directory into the gollek repository.
     *
     * @param source      the source path (file or directory)
     * @param move        if true, the source is moved (deleted after copy); if false, it is copied
     * @param isDirectory if true, source is treated as a directory
     * @return the destination path inside the blobs directory
     */
    public Path importModel(Path source, boolean move, boolean isDirectory) {
        try {
            String fileName = source.getFileName().toString();
            // Generate a manifest name/id base
            String nameBase = fileName.contains(".") ? fileName.substring(0, fileName.lastIndexOf('.')) : fileName;
            String manifestId = "imported__" + nameBase.replaceAll("[^a-zA-Z0-9._-]", "_");

            Path blobDir = ManifestStore.getBlobsDir().resolve(manifestId);
            Files.createDirectories(blobDir);

            Path destination = blobDir.resolve(fileName);

            if (isDirectory) {
                if (move) {
                    moveDirectory(source, destination);
                } else {
                    copyDirectory(source, destination);
                }
            } else {
                if (move) {
                    Files.move(source, destination, StandardCopyOption.REPLACE_EXISTING);
                } else {
                    Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
                }
            }

            // Register via ManifestStore
            GollekManifest manifest = new GollekManifest();
            manifest.setId(manifestId);
            manifest.setName(manifestId);
            manifest.setModelId(fileName);
            manifest.setSource("local");
            manifest.setFormat(ManifestStore.detectFormat(destination));
            manifest.setPipeline(ManifestStore.isPipeline(destination));
            manifest.setBlobPath(destination.toAbsolutePath().toString());
            manifest.setFiles(ManifestStore.listBlobFiles(destination));
            manifest.setCreatedAt(Instant.now());
            
            try {
                if (isDirectory) {
                    try (var walk = Files.walk(destination)) {
                        manifest.setSizeBytes(walk.filter(Files::isRegularFile)
                            .mapToLong(p -> { try { return Files.size(p); } catch (IOException e) { return 0; } })
                            .sum());
                    }
                } else {
                    manifest.setSizeBytes(Files.size(destination));
                }
            } catch (IOException ignored) {}

            manifestStore.save(manifest);

            return destination;
        } catch (IOException e) {
            throw new RuntimeException("Failed to " + (move ? "import" : "copy") + " model: " + e.getMessage(), e);
        }
    }

    private void copyDirectory(Path source, Path target) throws IOException {
        Files.walkFileTree(source, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                Files.createDirectories(target.resolve(source.relativize(dir)));
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.copy(file, target.resolve(source.relativize(file)), StandardCopyOption.REPLACE_EXISTING);
                return FileVisitResult.CONTINUE;
            }
        });
    }

    private void moveDirectory(Path source, Path target) throws IOException {
        copyDirectory(source, target);
        // Delete source after successful copy
        Files.walkFileTree(source, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.delete(dir);
                return FileVisitResult.CONTINUE;
            }
        });
    }
}
