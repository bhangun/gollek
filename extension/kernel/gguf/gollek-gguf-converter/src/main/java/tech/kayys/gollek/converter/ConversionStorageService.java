package tech.kayys.gollek.converter;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Storage service for multi-tenant model file management.
 * 
 * <p>
 * Provides tenant-isolated storage with quota management and
 * security controls.
 * 
 * @author Bhangun
 * @version 1.0.0
 */
@ApplicationScoped
public class ConversionStorageService {

    private static final Logger log = LoggerFactory.getLogger(ConversionStorageService.class);

    @ConfigProperty(name = "converter.storage.base-path", defaultValue = "/var/lib/inference/models")
    String baseStoragePath;

    @ConfigProperty(name = "converter.storage.tenant-quota-gb", defaultValue = "100")
    long tenantQuotaGb;

    /**
     * Get base path for tenant storage.
     * 
     * @param requestId tenant identifier
     * @return tenant base path
     */
    public Path getTenantBasePath(String requestId) {
        if (requestId == null || requestId.isEmpty()) {
            throw new IllegalArgumentException("Tenant ID is required");
        }

        // Sanitize tenant ID to prevent path traversal
        String sanitizedRequestId = sanitizeRequestId(requestId);

        Path tenantPath = Paths.get(baseStoragePath, sanitizedRequestId);

        // Ensure directory exists
        try {
            if (!Files.exists(tenantPath)) {
                Files.createDirectories(tenantPath);
                log.info("Created storage directory for tenant: {}", sanitizedRequestId);
            }
        } catch (IOException e) {
            log.error("Failed to create tenant storage directory", e);
            throw new RuntimeException("Failed to initialize tenant storage", e);
        }

        return tenantPath;
    }

    /**
     * Get path for model conversions.
     * 
     * @param requestId tenant identifier
     * @return conversions directory path
     */
    public Path getConversionsPath(String requestId) {
        Path basePath = getTenantBasePath(requestId);
        Path conversionsPath = basePath.resolve("conversions");

        try {
            if (!Files.exists(conversionsPath)) {
                Files.createDirectories(conversionsPath);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create conversions directory", e);
        }

        return conversionsPath;
    }

    /**
     * Get path for temporary files.
     * 
     * @param requestId tenant identifier
     * @return temp directory path
     */
    public Path getTempPath(String requestId) {
        Path basePath = getTenantBasePath(requestId);
        Path tempPath = basePath.resolve("temp");

        try {
            if (!Files.exists(tempPath)) {
                Files.createDirectories(tempPath);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create temp directory", e);
        }

        return tempPath;
    }

    /**
     * Check tenant storage quota.
     * 
     * @param requestId tenant identifier
     * @return storage info
     */
    public StorageInfo getStorageInfo(String requestId) {
        Path basePath = getTenantBasePath(requestId);

        try {
            long usedBytes = calculateDirectorySize(basePath);
            long quotaBytes = tenantQuotaGb * 1024 * 1024 * 1024;

            return new StorageInfo(
                    usedBytes,
                    quotaBytes,
                    quotaBytes - usedBytes,
                    (double) usedBytes / quotaBytes * 100);
        } catch (IOException e) {
            log.error("Failed to calculate storage for tenant {}", requestId, e);
            return new StorageInfo(0, tenantQuotaGb * 1024 * 1024 * 1024,
                    tenantQuotaGb * 1024 * 1024 * 1024, 0);
        }
    }

    /**
     * Check if tenant has sufficient quota.
     * 
     * @param requestId      tenant identifier
     * @param requiredBytes required space in bytes
     * @return true if sufficient quota
     */
    public boolean checkQuota(String requestId, long requiredBytes) {
        StorageInfo info = getStorageInfo(requestId);
        return info.availableBytes >= requiredBytes;
    }

    /**
     * Validate path is within tenant boundaries.
     * 
     * @param requestId tenant identifier
     * @param path     path to validate
     * @return true if valid
     */
    public boolean validatePath(String requestId, Path path) {
        Path tenantBase = getTenantBasePath(requestId);
        Path normalized = path.normalize().toAbsolutePath();

        return normalized.startsWith(tenantBase);
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    private String sanitizeRequestId(String requestId) {
        // Remove any path traversal attempts and special characters
        return requestId.replaceAll("[^a-zA-Z0-9_-]", "_");
    }

    private long calculateDirectorySize(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return 0;
        }

        return Files.walk(directory)
                .filter(Files::isRegularFile)
                .mapToLong(path -> {
                    try {
                        return Files.size(path);
                    } catch (IOException e) {
                        log.warn("Failed to get size of {}", path, e);
                        return 0;
                    }
                })
                .sum();
    }

    /**
     * Storage information record.
     */
    public record StorageInfo(
            long usedBytes,
            long quotaBytes,
            long availableBytes,
            double usagePercent) {
        public String getUsedFormatted() {
            return formatBytes(usedBytes);
        }

        public String getQuotaFormatted() {
            return formatBytes(quotaBytes);
        }

        public String getAvailableFormatted() {
            return formatBytes(availableBytes);
        }

        private String formatBytes(long bytes) {
            double gb = bytes / (1024.0 * 1024.0 * 1024.0);
            if (gb >= 1.0) {
                return String.format("%.2f GB", gb);
            }
            double mb = bytes / (1024.0 * 1024.0);
            return String.format("%.2f MB", mb);
        }
    }
}
