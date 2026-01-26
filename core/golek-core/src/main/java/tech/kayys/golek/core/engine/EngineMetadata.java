package tech.kayys.golek.core.engine;

import java.util.List;

/**
 * Engine metadata containing version and capability information.
 */
public record EngineMetadata(
        String version,
        List<String> supportedPhases,
        String buildTime,
        String gitCommit
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String version = "unknown";
        private List<String> supportedPhases = List.of();
        private String buildTime = "unknown";
        private String gitCommit = "unknown";

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder supportedPhases(List<String> phases) {
            this.supportedPhases = phases;
            return this;
        }

        public Builder buildTime(String buildTime) {
            this.buildTime = buildTime;
            return this;
        }

        public Builder gitCommit(String gitCommit) {
            this.gitCommit = gitCommit;
            return this;
        }

        public EngineMetadata build() {
            return new EngineMetadata(version, supportedPhases, buildTime, gitCommit);
        }
    }
}
