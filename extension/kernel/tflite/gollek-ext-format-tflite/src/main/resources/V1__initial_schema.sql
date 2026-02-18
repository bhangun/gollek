-- Inference Engine Platform - Initial Database Schema
-- Version: 1.0.0
-- Description: Core tables for model registry, inference tracking, and multi-tenancy

-- ===== Tenants =====
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    tier VARCHAR(50) NOT NULL DEFAULT 'FREE',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    
    CONSTRAINT chk_tenant_status CHECK (status IN ('ACTIVE', 'SUSPENDED', 'DELETED')),
    CONSTRAINT chk_tenant_tier CHECK (tier IN ('FREE', 'BASIC', 'PRO', 'ENTERPRISE'))
);

CREATE INDEX idx_tenants_status ON tenants(status);
CREATE INDEX idx_tenants_tier ON tenants(tier);

-- ===== Tenant Quotas =====
CREATE TABLE tenant_quotas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    resource_type VARCHAR(100) NOT NULL,
    quota_limit BIGINT NOT NULL,
    quota_used BIGINT NOT NULL DEFAULT 0,
    reset_period VARCHAR(50) NOT NULL DEFAULT 'MONTHLY',
    last_reset_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(tenant_id, resource_type),
    CONSTRAINT chk_quota_reset CHECK (reset_period IN ('HOURLY', 'DAILY', 'MONTHLY', 'NEVER'))
);

CREATE INDEX idx_tenant_quotas_tenant ON tenant_quotas(tenant_id);

-- ===== Models =====
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    model_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    framework VARCHAR(50) NOT NULL,
    stage VARCHAR(50) NOT NULL DEFAULT 'DEVELOPMENT',
    tags TEXT[],
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    
    UNIQUE(tenant_id, model_id),
    CONSTRAINT chk_model_framework CHECK (framework IN ('LITERT', 'ONNX', 'TENSORFLOW', 'PYTORCH', 'JAX', 'TRITON')),
    CONSTRAINT chk_model_stage CHECK (stage IN ('DEVELOPMENT', 'STAGING', 'PRODUCTION', 'DEPRECATED', 'ARCHIVED'))
);

CREATE INDEX idx_models_tenant ON models(tenant_id);
CREATE INDEX idx_models_stage ON models(stage);
CREATE INDEX idx_models_framework ON models(framework);
CREATE INDEX idx_models_tags ON models USING GIN(tags);

-- ===== Model Versions =====
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL,
    storage_uri TEXT NOT NULL,
    format VARCHAR(50) NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    size_bytes BIGINT NOT NULL,
    manifest JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_id, version),
    CONSTRAINT chk_version_status CHECK (status IN ('ACTIVE', 'DEPRECATED', 'DELETED'))
);

CREATE INDEX idx_model_versions_model ON model_versions(model_id);
CREATE INDEX idx_model_versions_status ON model_versions(status);

-- ===== Inference Requests (Audit Log) =====
CREATE TABLE inference_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    model_version_id UUID REFERENCES model_versions(id) ON DELETE SET NULL,
    runner_name VARCHAR(100),
    device_type VARCHAR(50),
    status VARCHAR(50) NOT NULL,
    latency_ms BIGINT,
    input_size_bytes BIGINT,
    output_size_bytes BIGINT,
    error_code VARCHAR(50),
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    user_id VARCHAR(255),
    
    CONSTRAINT chk_request_status CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'TIMEOUT'))
);

CREATE INDEX idx_inference_requests_tenant_created ON inference_requests(tenant_id, created_at DESC);
CREATE INDEX idx_inference_requests_model ON inference_requests(model_id);
CREATE INDEX idx_inference_requests_status ON inference_requests(status);
CREATE INDEX idx_inference_requests_request_id ON inference_requests(request_id);
CREATE INDEX idx_inference_requests_completed ON inference_requests(completed_at) WHERE completed_at IS NOT NULL;

-- ===== Model Conversion Jobs =====
CREATE TABLE conversion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    source_format VARCHAR(50) NOT NULL,
    target_format VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    progress_percent INTEGER DEFAULT 0,
    result_storage_uri TEXT,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_conversion_status CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    CONSTRAINT chk_progress CHECK (progress_percent BETWEEN 0 AND 100)
);

CREATE INDEX idx_conversion_jobs_tenant ON conversion_jobs(tenant_id);
CREATE INDEX idx_conversion_jobs_status ON conversion_jobs(status);
CREATE INDEX idx_conversion_jobs_created ON conversion_jobs(created_at DESC);

-- ===== Runner Health Status =====
CREATE TABLE runner_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    runner_name VARCHAR(100) NOT NULL,
    device_type VARCHAR(50) NOT NULL,
    is_healthy BOOLEAN NOT NULL,
    last_heartbeat TIMESTAMP NOT NULL,
    error_message TEXT,
    metadata JSONB,
    
    UNIQUE(runner_name, device_type)
);

CREATE INDEX idx_runner_health_healthy ON runner_health(is_healthy);
CREATE INDEX idx_runner_health_heartbeat ON runner_health(last_heartbeat DESC);

-- ===== Model Metrics Aggregation =====
CREATE TABLE model_metrics_hourly (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    hour_bucket TIMESTAMP NOT NULL,
    total_requests BIGINT NOT NULL DEFAULT 0,
    successful_requests BIGINT NOT NULL DEFAULT 0,
    failed_requests BIGINT NOT NULL DEFAULT 0,
    avg_latency_ms BIGINT,
    p50_latency_ms BIGINT,
    p95_latency_ms BIGINT,
    p99_latency_ms BIGINT,
    total_input_bytes BIGINT NOT NULL DEFAULT 0,
    total_output_bytes BIGINT NOT NULL DEFAULT 0,
    
    UNIQUE(tenant_id, model_id, hour_bucket)
);

CREATE INDEX idx_model_metrics_tenant_bucket ON model_metrics_hourly(tenant_id, hour_bucket DESC);
CREATE INDEX idx_model_metrics_model ON model_metrics_hourly(model_id);

-- ===== API Keys (for programmatic access) =====
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    key_prefix VARCHAR(10) NOT NULL,
    name VARCHAR(255) NOT NULL,
    scopes TEXT[] NOT NULL,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    revoked_at TIMESTAMP,
    
    CONSTRAINT chk_not_revoked_with_expiry CHECK (revoked_at IS NULL OR expires_at IS NULL OR revoked_at >= expires_at)
);

CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(tenant_id) WHERE revoked_at IS NULL AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP);

-- ===== Audit Log =====
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_log_tenant_created ON audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_log_action ON audit_log(action);

-- ===== Functions & Triggers =====

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for models table
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for tenants table
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to increment quota usage
CREATE OR REPLACE FUNCTION increment_quota_usage(
    p_tenant_id UUID,
    p_resource_type VARCHAR,
    p_increment BIGINT
) RETURNS BOOLEAN AS $$
DECLARE
    v_current_used BIGINT;
    v_limit BIGINT;
BEGIN
    -- Get current usage and limit
    SELECT quota_used, quota_limit INTO v_current_used, v_limit
    FROM tenant_quotas
    WHERE tenant_id = p_tenant_id AND resource_type = p_resource_type
    FOR UPDATE;
    
    -- Check if quota would be exceeded
    IF v_current_used + p_increment > v_limit THEN
        RETURN FALSE;
    END IF;
    
    -- Update usage
    UPDATE tenant_quotas
    SET quota_used = quota_used + p_increment
    WHERE tenant_id = p_tenant_id AND resource_type = p_resource_type;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to reset quotas based on reset period
CREATE OR REPLACE FUNCTION reset_expired_quotas() RETURNS void AS $$
BEGIN
    UPDATE tenant_quotas
    SET quota_used = 0,
        last_reset_at = CURRENT_TIMESTAMP
    WHERE 
        (reset_period = 'HOURLY' AND last_reset_at < CURRENT_TIMESTAMP - INTERVAL '1 hour') OR
        (reset_period = 'DAILY' AND last_reset_at < CURRENT_TIMESTAMP - INTERVAL '1 day') OR
        (reset_period = 'MONTHLY' AND last_reset_at < CURRENT_TIMESTAMP - INTERVAL '1 month');
END;
$$ LANGUAGE plpgsql;

-- ===== Initial Data =====

-- Create default tenant for development
INSERT INTO tenants (tenant_id, name, status, tier)
VALUES ('default', 'Default Tenant', 'ACTIVE', 'ENTERPRISE');

-- Set default quotas for default tenant
INSERT INTO tenant_quotas (tenant_id, resource_type, quota_limit, reset_period)
SELECT 
    id,
    resource_type,
    quota_limit,
    'MONTHLY'
FROM tenants, (
    VALUES 
        ('requests', 1000000),
        ('storage_gb', 100),
        ('models', 50),
        ('concurrent_requests', 100)
) AS quotas(resource_type, quota_limit)
WHERE tenant_id = 'default';

-- ===== Comments =====
COMMENT ON TABLE tenants IS 'Multi-tenant isolation and management';
COMMENT ON TABLE tenant_quotas IS 'Resource quotas and usage tracking per tenant';
COMMENT ON TABLE models IS 'Model registry with versioning';
COMMENT ON TABLE model_versions IS 'Individual model versions with storage metadata';
COMMENT ON TABLE inference_requests IS 'Audit log of all inference requests';
COMMENT ON TABLE conversion_jobs IS 'Model format conversion job tracking';
COMMENT ON TABLE runner_health IS 'Health status of inference runners';
COMMENT ON TABLE model_metrics_hourly IS 'Aggregated metrics for analytics';
COMMENT ON TABLE api_keys IS 'API keys for programmatic access';
COMMENT ON TABLE audit_log IS 'Complete audit trail of all changes';
