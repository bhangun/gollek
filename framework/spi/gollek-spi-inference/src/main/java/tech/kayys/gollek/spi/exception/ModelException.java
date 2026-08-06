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
 */

package tech.kayys.gollek.spi.exception;

import tech.kayys.gollek.error.ErrorCode;

/**
 * Base exception for model-related errors.
 *
 * @since 2.1.0
 */
public class ModelException extends InferenceException {

    private final String modelId;

    public ModelException(ErrorCode errorCode, String message, String modelId) {
        super(errorCode, message);
        this.modelId = modelId;
    }

    public ModelException(ErrorCode errorCode, String message, String modelId, Throwable cause) {
        super(errorCode, message, cause);
        this.modelId = modelId;
    }

    public String getModelId() {
        return modelId;
    }
}
