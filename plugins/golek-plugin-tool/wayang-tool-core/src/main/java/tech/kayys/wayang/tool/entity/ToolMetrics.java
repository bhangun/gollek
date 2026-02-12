package tech.kayys.wayang.tool.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;

@Embeddable
public class ToolMetrics {
    @Column(name = "total_invocations")
    private Long totalInvocations = 0L;
    
    @Column(name = "success_count")
    private Long successCount = 0L;
    
    @Column(name = "error_count")
    private Long errorCount = 0L;
    
    @Column(name = "avg_response_time_ms")
    private Double avgResponseTimeMs = 0.0;
    
    @Column(name = "last_invoked_at")
    private java.time.Instant lastInvokedAt;
    
    // Getters and setters
    public Long getTotalInvocations() {
        return totalInvocations;
    }
    
    public void setTotalInvocations(Long totalInvocations) {
        this.totalInvocations = totalInvocations;
    }
    
    public Long getSuccessCount() {
        return successCount;
    }
    
    public void setSuccessCount(Long successCount) {
        this.successCount = successCount;
    }
    
    public Long getErrorCount() {
        return errorCount;
    }
    
    public void setErrorCount(Long errorCount) {
        this.errorCount = errorCount;
    }
    
    public Double getAvgResponseTimeMs() {
        return avgResponseTimeMs;
    }
    
    public void setAvgResponseTimeMs(Double avgResponseTimeMs) {
        this.avgResponseTimeMs = avgResponseTimeMs;
    }
    
    public java.time.Instant getLastInvokedAt() {
        return lastInvokedAt;
    }
    
    public void setLastInvokedAt(java.time.Instant lastInvokedAt) {
        this.lastInvokedAt = lastInvokedAt;
    }
}