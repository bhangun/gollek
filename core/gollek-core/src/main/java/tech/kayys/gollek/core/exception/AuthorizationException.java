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

package tech.kayys.gollek.core.exception;

import tech.kayys.gollek.spi.error.ErrorCode;

public class AuthorizationException extends InferenceException {

    public AuthorizationException(String message) {
        super(ErrorCode.AUTH_PERMISSION_DENIED, message);
    }

    public AuthorizationException(String message, Throwable cause) {
        super(ErrorCode.AUTH_PERMISSION_DENIED, message, cause);
    }
}
