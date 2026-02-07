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

public class NoCompatibleProviderException extends InferenceException {

    public NoCompatibleProviderException(String message) {
        super(ErrorCode.ROUTING_NO_COMPATIBLE_PROVIDER, message);
    }

    public NoCompatibleProviderException(String message, Throwable cause) {
        super(ErrorCode.ROUTING_NO_COMPATIBLE_PROVIDER, message, cause);
    }
}
