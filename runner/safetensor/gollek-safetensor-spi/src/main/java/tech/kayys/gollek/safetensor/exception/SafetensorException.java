/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.exception;

import java.nio.file.Path;

/**
 * Base class for all SafeTensors loader and inference exceptions.
 *
 * <p>Carries the {@link Path} of the file that triggered the error so callers
 * can include it in error messages without re-parsing the exception message string.
 *
 * <p>Specialised subclasses cover the most common failure modes:
 * <ul>
 *   <li>{@link HeaderParseException} — malformed JSON header</li>
 *   <li>{@link ValidationException} — tensor metadata fails invariant checks</li>
 *   <li>{@link TensorNotFoundException} — requested tensor name absent from file</li>
 *   <li>{@link DTypeConversionException} — unrecognised dtype string</li>
 *   <li>{@link IoException} — underlying I/O failure</li>
 *   <li>{@link ShardException} — error in a specific shard of a multi-file model</li>
 *   <li>{@link DTypeMismatchException} — operation requires a different dtype</li>
 * </ul>
 */
public class SafetensorException extends RuntimeException {

    private final Path path;

    /**
     * Constructs an exception with a message and the offending file path.
     *
     * @param message description of the failure
     * @param path    path of the SafeTensors file, or {@code null} if unknown
     */
    public SafetensorException(String message, Path path) {
        super(buildMessage(message, path));
        this.path = path;
    }

    /**
     * Constructs an exception with a message, file path, and root cause.
     *
     * @param message description of the failure
     * @param path    path of the SafeTensors file, or {@code null} if unknown
     * @param cause   the underlying exception
     */
    public SafetensorException(String message, Path path, Throwable cause) {
        super(buildMessage(message, path), cause);
        this.path = path;
    }

    /**
     * Returns the path of the SafeTensors file that triggered this exception.
     *
     * @return file path, or {@code null} if not available
     */
    public Path path() {
        return path;
    }

    private static String buildMessage(String message, Path path) {
        return path != null ? "[" + path + "] " + message : message;
    }

    // ── Nested exception types ────────────────────────────────────────────────

    /**
     * Thrown when the SafeTensors JSON header cannot be parsed.
     */
    public static final class HeaderParseException extends SafetensorException {
        private final long failureOffset;

        /**
         * @param message       description of the parse failure
         * @param path          file path
         * @param failureOffset byte offset within the file where parsing failed,
         *                      or {@code -1} if unknown
         * @param cause         the underlying parse exception
         */
        public HeaderParseException(String message, Path path, long failureOffset, Throwable cause) {
            super(message + " (at byte offset " + failureOffset + ")", path, cause);
            this.failureOffset = failureOffset;
        }

        /**
         * @param message description of the parse failure
         * @param path    file path
         * @param cause   the underlying parse exception
         */
        public HeaderParseException(String message, Path path, Throwable cause) {
            this(message, path, -1L, cause);
        }

        /**
         * Returns the byte offset where parsing failed, or {@code -1} if unknown.
         *
         * @return failure offset in bytes
         */
        public long failureOffset() {
            return failureOffset;
        }
    }

    /**
     * Thrown when a tensor's metadata fails an invariant check (e.g. shape/dtype mismatch).
     */
    public static final class ValidationException extends SafetensorException {
        private final String tensorName;

        /**
         * @param message    description of the validation failure
         * @param path       file path
         * @param tensorName name of the offending tensor, or {@code null}
         */
        public ValidationException(String message, Path path, String tensorName) {
            super(message, path);
            this.tensorName = tensorName;
        }

        /** @param message description; @param path file path */
        public ValidationException(String message, Path path) {
            this(message, path, null);
        }

        /**
         * Returns the name of the tensor that failed validation, or {@code null}.
         *
         * @return tensor name, or {@code null}
         */
        public String tensorName() {
            return tensorName;
        }
    }

    /**
     * Thrown when a requested tensor name is not present in the SafeTensors file.
     */
    public static final class TensorNotFoundException extends SafetensorException {
        private final String tensorName;

        /**
         * @param tensorName the name that was not found
         * @param path       file path
         */
        public TensorNotFoundException(String tensorName, Path path) {
            super("AccelTensor '" + tensorName + "' not found in SafeTensors file", path);
            this.tensorName = tensorName;
        }

        /**
         * Returns the tensor name that was not found.
         *
         * @return tensor name
         */
        public String tensorName() {
            return tensorName;
        }
    }

    /**
     * Thrown when a dtype string from the JSON header cannot be mapped to a known type.
     */
    public static final class DTypeConversionException extends SafetensorException {
        private final String rawDType;

        /**
         * @param rawDType the unrecognised dtype string
         * @param path     file path
         * @param cause    the underlying exception
         */
        public DTypeConversionException(String rawDType, Path path, Throwable cause) {
            super("Unrecognised dtype '" + rawDType + "'", path, cause);
            this.rawDType = rawDType;
        }

        /**
         * Returns the raw dtype string that could not be converted.
         *
         * @return raw dtype string
         */
        public String rawDType() {
            return rawDType;
        }
    }

    /**
     * Thrown when an I/O error occurs while reading a SafeTensors file.
     */
    public static final class IoException extends SafetensorException {
        /**
         * @param message description of the I/O failure
         * @param path    file path
         * @param cause   the underlying I/O exception
         */
        public IoException(String message, Path path, Throwable cause) {
            super(message, path, cause);
        }

        /**
         * @param path  file path
         * @param cause the underlying I/O exception
         */
        public IoException(Path path, Throwable cause) {
            super("I/O error reading SafeTensors file", path, cause);
        }
    }

    /**
     * Thrown when loading a specific shard of a multi-file model fails.
     */
    public static final class ShardException extends SafetensorException {
        private final Path shardPath;

        /**
         * @param message   description of the shard failure
         * @param indexPath path to the shard index file
         * @param shardPath path to the failing shard, or {@code null}
         * @param cause     the underlying exception
         */
        public ShardException(String message, Path indexPath, Path shardPath, Throwable cause) {
            super(message + (shardPath != null ? " [shard: " + shardPath + "]" : ""),
                    indexPath, cause);
            this.shardPath = shardPath;
        }

        /** @param message description; @param indexPath index path; @param shardPath shard path */
        public ShardException(String message, Path indexPath, Path shardPath) {
            this(message, indexPath, shardPath, null);
        }

        /**
         * Returns the path of the failing shard, or {@code null} if unknown.
         *
         * @return shard path, or {@code null}
         */
        public Path shardPath() {
            return shardPath;
        }
    }

    /**
     * Thrown when a tensor's actual dtype does not match what an operation requires.
     */
    public static final class DTypeMismatchException extends SafetensorException {
        private final String tensorName;
        private final String actualDType;
        private final String expectedDType;

        /**
         * @param tensorName   name of the tensor
         * @param actualDType  dtype found in the file
         * @param expectedDType dtype required by the operation
         * @param path         file path
         */
        public DTypeMismatchException(String tensorName, String actualDType,
                String expectedDType, Path path) {
            super("AccelTensor '" + tensorName + "' has dtype=" + actualDType
                    + " but operation requires " + expectedDType, path);
            this.tensorName = tensorName;
            this.actualDType = actualDType;
            this.expectedDType = expectedDType;
        }

        /** @return tensor name */
        public String tensorName() { return tensorName; }

        /** @return actual dtype string found in the file */
        public String actualDType() { return actualDType; }

        /** @return dtype string required by the operation */
        public String expectedDType() { return expectedDType; }
    }
}
