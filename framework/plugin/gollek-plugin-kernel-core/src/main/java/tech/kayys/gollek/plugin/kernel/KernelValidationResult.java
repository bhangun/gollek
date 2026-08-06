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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Validation result for kernel plugin with status and error messages.
 *
 * @since 2.0.0
 */
public final class KernelValidationResult {

    private final boolean valid;
    private final List<String> errors;
    private final List<String> warnings;

    private KernelValidationResult(boolean valid, List<String> errors, List<String> warnings) {
        this.valid = valid;
        this.errors = Collections.unmodifiableList(errors);
        this.warnings = Collections.unmodifiableList(warnings);
    }

    /**
     * Check if validation passed.
     *
     * @return true if valid
     */
    public boolean isValid() {
        return valid;
    }

    /**
     * Get validation errors.
     *
     * @return immutable list of errors
     */
    public List<String> getErrors() {
        return errors;
    }

    /**
     * Get validation warnings.
     *
     * @return immutable list of warnings
     */
    public List<String> getWarnings() {
        return warnings;
    }

    /**
     * Get first error message.
     *
     * @return optional first error
     */
    public java.util.Optional<String> getFirstError() {
        return errors.isEmpty() ?
            java.util.Optional.empty() :
            java.util.Optional.of(errors.get(0));
    }

    /**
     * Create valid result.
     *
     * @return valid result
     */
    public static KernelValidationResult valid() {
        return new KernelValidationResult(true, List.of(), List.of());
    }

    /**
     * Create invalid result with error.
     *
     * @param error error message
     * @return invalid result
     */
    public static KernelValidationResult invalid(String error) {
        return new KernelValidationResult(false, List.of(error), List.of());
    }

    /**
     * Create invalid result with errors.
     *
     * @param errors error messages
     * @return invalid result
     */
    public static KernelValidationResult invalid(List<String> errors) {
        return new KernelValidationResult(false, errors, List.of());
    }

    /**
     * Create builder.
     *
     * @return new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for validation results.
     */
    public static class Builder {
        private boolean valid = true;
        private final List<String> errors = new ArrayList<>();
        private final List<String> warnings = new ArrayList<>();

        public Builder valid(boolean valid) {
            this.valid = valid;
            return this;
        }

        public Builder error(String error) {
            this.errors.add(Objects.requireNonNull(error));
            this.valid = false;
            return this;
        }

        public Builder errors(List<String> errors) {
            this.errors.addAll(errors);
            this.valid = false;
            return this;
        }

        public Builder warning(String warning) {
            this.warnings.add(Objects.requireNonNull(warning));
            return this;
        }

        public Builder warnings(List<String> warnings) {
            this.warnings.addAll(warnings);
            return this;
        }

        public KernelValidationResult build() {
            return new KernelValidationResult(valid, errors, warnings);
        }
    }
}
