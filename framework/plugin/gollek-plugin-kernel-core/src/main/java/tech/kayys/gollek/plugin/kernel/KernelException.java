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

package tech.kayys.gollek.plugin.kernel;

/**
 * Base exception for kernel plugin operations.
 *
 * @since 2.0.0
 */
public class KernelException extends Exception {

    private final String errorCode;
    private final String platform;
    private final String operation;

    /**
     * Create kernel exception with message.
     *
     * @param message error message
     */
    public KernelException(String message) {
        super(message);
        this.errorCode = "KERNEL_ERROR";
        this.platform = null;
        this.operation = null;
    }

    /**
     * Create kernel exception with message and cause.
     *
     * @param message error message
     * @param cause root cause
     */
    public KernelException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "KERNEL_ERROR";
        this.platform = null;
        this.operation = null;
    }

    /**
     * Create kernel exception with error code and message.
     *
     * @param errorCode error code
     * @param message error message
     */
    public KernelException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
        this.platform = null;
        this.operation = null;
    }

    /**
     * Create kernel exception with full details.
     *
     * @param errorCode error code
     * @param platform platform where error occurred
     * @param operation operation that failed
     * @param message error message
     */
    public KernelException(String errorCode, String platform, String operation, String message) {
        super(message);
        this.errorCode = errorCode;
        this.platform = platform;
        this.operation = operation;
    }

    /**
     * Create kernel exception with full details and cause.
     *
     * @param errorCode error code
     * @param platform platform where error occurred
     * @param operation operation that failed
     * @param message error message
     * @param cause root cause
     */
    public KernelException(String errorCode, String platform, String operation, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.platform = platform;
        this.operation = operation;
    }

    /**
     * Get error code.
     *
     * @return error code
     */
    public String getErrorCode() {
        return errorCode;
    }

    /**
     * Get platform.
     *
     * @return platform (null if not specified)
     */
    public String getPlatform() {
        return platform;
    }

    /**
     * Get operation.
     *
     * @return operation (null if not specified)
     */
    public String getOperation() {
        return operation;
    }

    /**
     * Get error details as map.
     *
     * @return error details
     */
    public java.util.Map<String, Object> toMap() {
        java.util.Map<String, Object> details = new java.util.HashMap<>();
        details.put("errorCode", errorCode);
        details.put("message", getMessage());
        if (platform != null) {
            details.put("platform", platform);
        }
        if (operation != null) {
            details.put("operation", operation);
        }
        if (getCause() != null) {
            details.put("cause", getCause().getClass().getSimpleName());
        }
        return details;
    }
}
