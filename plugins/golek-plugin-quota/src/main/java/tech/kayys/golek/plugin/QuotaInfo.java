package tech.kayys.golek.plugin;

/**
 * Information about a tenant's current quota status.
 */
public class QuotaInfo {
    
    private final String id;
    private final long used;
    private final long limit;
    private final long remaining;
    private final long resetTime;

    public QuotaInfo(String id, long used, long limit, long remaining, long resetTime) {
        this.id = id;
        this.used = used;
        this.limit = limit;
        this.remaining = remaining;
        this.resetTime = resetTime;
    }

    public String getId() {
        return id;
    }

    public long getUsed() {
        return used;
    }

    public long getLimit() {
        return limit;
    }

    public long getRemaining() {
        return remaining;
    }

    public long getResetTime() {
        return resetTime;
    }

    public boolean hasCapacity() {
        return remaining > 0;
    }

    public double getUtilizationPercentage() {
        if (limit <= 0) {
            return used > 0 ? Double.POSITIVE_INFINITY : 0.0;
        }
        return Math.min(100.0, (double) used / limit * 100.0);
    }
}