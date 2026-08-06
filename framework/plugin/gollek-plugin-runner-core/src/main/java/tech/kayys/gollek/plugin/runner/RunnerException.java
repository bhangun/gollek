/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.runner;

/**
 * Base exception for runner plugin operations (unchecked).
 *
 * @since 2.0.0
 */
public class RunnerException extends RuntimeException {

    private final String errorCode;
    private final String format;
    private final String operation;

    public RunnerException(String message) {
        super(message);
        this.errorCode = "RUNNER_ERROR";
        this.format = null;
        this.operation = null;
    }

    public RunnerException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "RUNNER_ERROR";
        this.format = null;
        this.operation = null;
    }

    public RunnerException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
        this.format = null;
        this.operation = null;
    }

    public RunnerException(String errorCode, String format, String operation, String message) {
        super(message);
        this.errorCode = errorCode;
        this.format = format;
        this.operation = operation;
    }

    public RunnerException(String errorCode, String format, String operation, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.format = format;
        this.operation = operation;
    }

    /**
     * Get error code.
     */
    public String getErrorCode() {
        return errorCode;
    }

    /**
     * Get format.
     */
    public String getFormat() {
        return format;
    }

    /**
     * Get operation.
     */
    public String getOperation() {
        return operation;
    }
}
