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
 * Exception thrown when model loading fails (unchecked).
 *
 * @since 2.0.0
 */
public class ModelLoadException extends RunnerException {

    private final String modelPath;

    public ModelLoadException(String modelPath, String message) {
        super("MODEL_LOAD_ERROR", null, null, "Failed to load model: " + modelPath + " - " + message);
        this.modelPath = modelPath;
    }

    public ModelLoadException(String modelPath, String message, Throwable cause) {
        super("MODEL_LOAD_ERROR", null, null, "Failed to load model: " + modelPath + " - " + message, cause);
        this.modelPath = modelPath;
    }

    /**
     * Get model path.
     */
    public String getModelPath() {
        return modelPath;
    }
}
