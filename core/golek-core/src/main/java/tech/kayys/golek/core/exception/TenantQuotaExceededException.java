/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author bhangun
 */

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
