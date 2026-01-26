package tech.kayys.golek.core.exception;

public class TenantQuotaExceededException extends InferenceException {
    
    private final String tenantId;
    private final String resourceType;
    
    public TenantQuotaExceededException(String tenantId, String resourceType, String message) {
        super(message);
        this.tenantId = tenantId;
        this.resourceType = resourceType;
    }
    
    public TenantQuotaExceededException(String tenantId, String resourceType, String message, Throwable cause) {
        super(message, cause);
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