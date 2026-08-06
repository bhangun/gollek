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

package tech.kayys.gollek.spi.plugin;

import tech.kayys.gollek.spi.exception.PluginException;

/**
 * Exception thrown when phase plugin execution fails.
 *
 * @since 2.1.0
 */
public class PhasePluginException extends PluginException {

    private final boolean retryable;

    public PhasePluginException(String message) {
        this(message, null, false);
    }

    public PhasePluginException(String message, Throwable cause) {
        this(message, cause, false);
    }

    public PhasePluginException(String message, Throwable cause, boolean retryable) {
        super(message, cause);
        this.retryable = retryable;
    }

    public boolean isRetryable() {
        return retryable;
    }
}
