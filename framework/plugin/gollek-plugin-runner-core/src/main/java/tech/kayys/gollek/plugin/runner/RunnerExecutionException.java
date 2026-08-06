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
 * Exception thrown when runner execution fails (unchecked).
 *
 * @since 2.0.0
 */
public class RunnerExecutionException extends RunnerException {

    public RunnerExecutionException(String message) {
        super("RUNNER_EXEC_ERROR", null, null, message);
    }

    public RunnerExecutionException(String message, Throwable cause) {
        super("RUNNER_EXEC_ERROR", null, null, message, cause);
    }

    public RunnerExecutionException(String operation, String message) {
        super("RUNNER_EXEC_ERROR", null, operation, message);
    }

    public RunnerExecutionException(String format, String operation, String message) {
        super("RUNNER_EXEC_ERROR", format, operation, message);
    }

    public RunnerExecutionException(String format, String operation, String message, Throwable cause) {
        super("RUNNER_EXEC_ERROR", format, operation, message, cause);
    }
}
