package tech.kayys.wayang.memory.model;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

public class SecurityScanResult {
    private final String sessionId;
    private final List<SecurityViolation> violations;
    private final Instant scannedAt;
    private final boolean passed;

    private SecurityScanResult(String sessionId, List<SecurityViolation> violations, Instant scannedAt) {
        this.sessionId = sessionId;
        this.violations = violations;
        this.scannedAt = scannedAt;
        this.passed = violations.isEmpty();
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getSessionId() { return sessionId; }
    public List<SecurityViolation> getViolations() { return violations; }
    public Instant getScannedAt() { return scannedAt; }
    public boolean isPassed() { return passed; }

    public static class Builder {
        private String sessionId;
        private final List<SecurityViolation> violations = new ArrayList<>();

        public Builder sessionId(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public Builder addViolation(String type, String memoryId, String description) {
            violations.add(new SecurityViolation(type, memoryId, description));
            return this;
        }

        public SecurityScanResult build() {
            return new SecurityScanResult(sessionId, new ArrayList<>(violations), Instant.now());
        }
    }

    public static class SecurityViolation {
        private final String type;
        private final String memoryId;
        private final String description;

        public SecurityViolation(String type, String memoryId, String description) {
            this.type = type;
            this.memoryId = memoryId;
            this.description = description;
        }

        public String getType() { return type; }
        public String getMemoryId() { return memoryId; }
        public String getDescription() { return description; }
    }
}