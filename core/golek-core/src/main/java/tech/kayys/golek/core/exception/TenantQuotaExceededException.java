package tech.kayys.golek.core.exception;

import tech.kayys.golek.spi.error.ErrorCode;

public class TenantQuotaExceededException extends InferenceException {
    
    private final String tenantId;
    private final String resourceType;
    
    public TenantQuotaExceededException(String tenantId, String resourceType, String message) {
        super(ErrorCode.QUOTA_EXCEEDED, message);
        this.tenantId = tenantId;
        this.resourceType = resourceType;
    }
    
    public TenantQuotaExceededException(String tenantId, String resourceType, String message, Throwable cause) {
        super(ErrorCode.QUOTA_EXCEEDED, message, cause);
        this.tenantId = tenantId;
        this.resourceType = resourceType;
    }
    
    public String getTenantId() {
        return tenantId;
    }
    
    public String getResourceType() {
        return resourceType;
    }
}
