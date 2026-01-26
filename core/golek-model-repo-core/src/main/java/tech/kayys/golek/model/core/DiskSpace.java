package tech.kayys.golek.model.core;

/**
 * Disk space requirements
 */
public record DiskSpace(
        long sizeBytes,
        String path) {
}
