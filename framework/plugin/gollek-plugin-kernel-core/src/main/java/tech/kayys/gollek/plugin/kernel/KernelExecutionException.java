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
 * Exception thrown when kernel operation execution fails.
 * This is an unchecked exception (extends RuntimeException).
 *
 * @since 2.0.0
 */
public class KernelExecutionException extends RuntimeException {

    private final String errorCode;
    private final String platform;
    private final String operation;

    public KernelExecutionException(String message) {
        super(message);
        this.errorCode = "KERNEL_EXEC_ERROR";
        this.platform = null;
        this.operation = null;
    }

    public KernelExecutionException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "KERNEL_EXEC_ERROR";
        this.platform = null;
        this.operation = null;
    }

    public KernelExecutionException(String operation, String message) {
        super(message);
        this.errorCode = "KERNEL_EXEC_ERROR";
        this.platform = null;
        this.operation = operation;
    }

    public KernelExecutionException(String platform, String operation, String message) {
        super(message);
        this.errorCode = "KERNEL_EXEC_ERROR";
        this.platform = platform;
        this.operation = operation;
    }

    public KernelExecutionException(String platform, String operation, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "KERNEL_EXEC_ERROR";
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
}
