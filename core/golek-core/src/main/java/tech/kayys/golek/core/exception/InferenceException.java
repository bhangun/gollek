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
 * @author Bhangun
 */

package tech.kayys.golek.core.exception;

import tech.kayys.golek.spi.error.ErrorCode;

/**
 * Exception thrown when an inference error occurs.
 */
public class InferenceException extends tech.kayys.golek.spi.exception.InferenceException {

    public InferenceException(String message) {
        super(ErrorCode.INTERNAL_ERROR, message);
    }

    public InferenceException(String message, Throwable cause) {
        super(ErrorCode.INTERNAL_ERROR, message, cause);
    }

    public InferenceException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}
