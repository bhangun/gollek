package tech.kayys.golek.sdk.core.model;

import java.util.Map;
import java.util.Objects;

/**
 * System information DTO representing runtime environment details.
 */
public final class SystemInfo {

    private final String cliVersion;
    private final String javaVersion;
    private final String osName;
    private final String osVersion;
    private final String osArch;
    private final String userName;
    private final String userHome;
    private final long totalMemory;
    private final long freeMemory;
    private final long maxMemory;
    private final int availableProcessors;
    private final Map<String, Object> metadata;

    public SystemInfo(
            String cliVersion,
            String javaVersion,
            String osName,
            String osVersion,
            String osArch,
            String userName,
            String userHome,
            long totalMemory,
            long freeMemory,
            long maxMemory,
            int availableProcessors,
            Map<String, Object> metadata) {
        this.cliVersion = Objects.requireNonNull(cliVersion, "cliVersion is required");
        this.javaVersion = javaVersion;
        this.osName = osName;
        this.osVersion = osVersion;
        this.osArch = osArch;
        this.userName = userName;
        this.userHome = userHome;
        this.totalMemory = totalMemory;
        this.freeMemory = freeMemory;
        this.maxMemory = maxMemory;
        this.availableProcessors = availableProcessors;
        this.metadata = metadata != null ? Map.copyOf(metadata) : Map.of();
    }

    public String getCliVersion() {
        return cliVersion;
    }

    public String getJavaVersion() {
        return javaVersion;
    }

    public String getOsName() {
        return osName;
    }

    public String getOsVersion() {
        return osVersion;
    }

    public String getOsArch() {
        return osArch;
    }

    public String getUserName() {
        return userName;
    }

    public String getUserHome() {
        return userHome;
    }

    public long getTotalMemory() {
        return totalMemory;
    }

    public long getFreeMemory() {
        return freeMemory;
    }

    public long getMaxMemory() {
        return maxMemory;
    }

    public int getAvailableProcessors() {
        return availableProcessors;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Returns a human-readable memory string.
     */
    public String getMemoryFormatted() {
        return String.format("Total: %s, Free: %s, Max: %s",
                formatBytes(totalMemory),
                formatBytes(freeMemory),
                formatBytes(maxMemory));
    }

    private String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        } else if (bytes < 1024 * 1024) {
            return String.format("%.1f KB", bytes / 1024.0);
        } else if (bytes < 1024 * 1024 * 1024) {
            return String.format("%.1f MB", bytes / (1024.0 * 1024));
        } else {
            return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String cliVersion;
        private String javaVersion;
        private String osName;
        private String osVersion;
        private String osArch;
        private String userName;
        private String userHome;
        private long totalMemory;
        private long freeMemory;
        private long maxMemory;
        private int availableProcessors;
        private Map<String, Object> metadata;

        public Builder cliVersion(String cliVersion) {
            this.cliVersion = cliVersion;
            return this;
        }

        public Builder javaVersion(String javaVersion) {
            this.javaVersion = javaVersion;
            return this;
        }

        public Builder osName(String osName) {
            this.osName = osName;
            return this;
        }

        public Builder osVersion(String osVersion) {
            this.osVersion = osVersion;
            return this;
        }

        public Builder osArch(String osArch) {
            this.osArch = osArch;
            return this;
        }

        public Builder userName(String userName) {
            this.userName = userName;
            return this;
        }

        public Builder userHome(String userHome) {
            this.userHome = userHome;
            return this;
        }

        public Builder totalMemory(long totalMemory) {
            this.totalMemory = totalMemory;
            return this;
        }

        public Builder freeMemory(long freeMemory) {
            this.freeMemory = freeMemory;
            return this;
        }

        public Builder maxMemory(long maxMemory) {
            this.maxMemory = maxMemory;
            return this;
        }

        public Builder availableProcessors(int availableProcessors) {
            this.availableProcessors = availableProcessors;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public SystemInfo build() {
            return new SystemInfo(cliVersion, javaVersion, osName, osVersion, osArch,
                    userName, userHome, totalMemory, freeMemory, maxMemory, availableProcessors, metadata);
        }
    }

    @Override
    public String toString() {
        return "SystemInfo{" +
                "cliVersion='" + cliVersion + '\'' +
                ", javaVersion='" + javaVersion + '\'' +
                ", osName='" + osName + '\'' +
                ", osArch='" + osArch + '\'' +
                ", availableProcessors=" + availableProcessors +
                ", memory=" + getMemoryFormatted() +
                '}';
    }
}