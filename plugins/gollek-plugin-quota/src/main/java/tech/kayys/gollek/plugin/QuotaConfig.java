package tech.kayys.gollek.plugin;

/**
 * Configuration for tenant quota limits.
 */
public class QuotaConfig {
    
    private final long limit;
    private final long periodMs;
    private final String unit;
    private final boolean enabled;

    public QuotaConfig(long limit, long periodMs, String unit, boolean enabled) {
        this.limit = limit;
        this.periodMs = periodMs;
        this.unit = unit;
        this.enabled = enabled;
    }

    public long getLimit() {
        return limit;
    }

    public long getPeriodMs() {
        return periodMs;
    }

    public String getUnit() {
        return unit;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public static QuotaConfigBuilder builder() {
        return new QuotaConfigBuilder();
    }

    public static class QuotaConfigBuilder {
        private long limit = 1000; // default 1000 requests
        private long periodMs = 3600000; // default 1 hour
        private String unit = "requests";
        private boolean enabled = true;

        public QuotaConfigBuilder limit(long limit) {
            this.limit = limit;
            return this;
        }

        public QuotaConfigBuilder periodMs(long periodMs) {
            this.periodMs = periodMs;
            return this;
        }

        public QuotaConfigBuilder unit(String unit) {
            this.unit = unit;
            return this;
        }

        public QuotaConfigBuilder enabled(boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        public QuotaConfig build() {
            return new QuotaConfig(limit, periodMs, unit, enabled);
        }
    }
}