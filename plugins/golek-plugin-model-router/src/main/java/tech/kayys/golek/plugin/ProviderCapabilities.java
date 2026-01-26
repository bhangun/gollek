package tech.kayys.golek.plugin;

/**
 * Represents the capabilities and characteristics of a provider.
 */
public class ProviderCapabilities {
    
    private final String providerId;
    private final double reliability; // Percentage (0-100)
    private final double avgLatencyMs; // Average latency in milliseconds
    private final double costPerThousand; // Cost per thousand tokens/units
    private final String performanceTier; // "low", "medium", "high"

    public ProviderCapabilities(String providerId, double reliability, 
                              double avgLatencyMs, double costPerThousand, 
                              String performanceTier) {
        this.providerId = providerId;
        this.reliability = reliability;
        this.avgLatencyMs = avgLatencyMs;
        this.costPerThousand = costPerThousand;
        this.performanceTier = performanceTier;
    }

    public String providerId() {
        return providerId;
    }

    public double reliability() {
        return reliability;
    }

    public double avgLatencyMs() {
        return avgLatencyMs;
    }

    public double costPerThousand() {
        return costPerThousand;
    }

    public String performanceTier() {
        return performanceTier;
    }

    public boolean isHighPerformance() {
        return "high".equalsIgnoreCase(performanceTier);
    }

    public boolean isLowCost() {
        return "low".equalsIgnoreCase(performanceTier);
    }

    public boolean isHighReliability() {
        return reliability >= 99.0;
    }

    public static ProviderCapabilitiesBuilder builder() {
        return new ProviderCapabilitiesBuilder();
    }

    public static class ProviderCapabilitiesBuilder {
        private String providerId;
        private double reliability = 99.0;
        private double avgLatencyMs = 100.0;
        private double costPerThousand = 0.05;
        private String performanceTier = "medium";

        public ProviderCapabilitiesBuilder providerId(String providerId) {
            this.providerId = providerId;
            return this;
        }

        public ProviderCapabilitiesBuilder reliability(double reliability) {
            this.reliability = reliability;
            return this;
        }

        public ProviderCapabilitiesBuilder avgLatencyMs(double avgLatencyMs) {
            this.avgLatencyMs = avgLatencyMs;
            return this;
        }

        public ProviderCapabilitiesBuilder costPerThousand(double costPerThousand) {
            this.costPerThousand = costPerThousand;
            return this;
        }

        public ProviderCapabilitiesBuilder performanceTier(String performanceTier) {
            this.performanceTier = performanceTier;
            return this;
        }

        public ProviderCapabilities build() {
            return new ProviderCapabilities(providerId, reliability, avgLatencyMs, 
                                         costPerThousand, performanceTier);
        }
    }
}