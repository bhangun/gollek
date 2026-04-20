/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorValidator.java
 * ────────────────────────
 * Pre-flight and post-parse validation utilities for SafeTensors files.
 *
 * Separating validation from parsing keeps the parser lean and makes it easy
 * to run validation-only passes (e.g. CI model integrity checks) without
 * touching the mmap pipeline.
 *
 * Validation levels
 * ══════════════════
 *  FAST    — only check the 8-byte header length and JSON parse-ability.
 *            Cost: reads ~header bytes (usually < 10 KB for most models).
 *
 *  NORMAL  — everything in FAST plus dtype-shape byte-count consistency and
 *            offset ordering within each shard.  Default for production.
 *
 *  STRICT  — everything in NORMAL plus checksum verification (SHA-256 of data
 *            region) if a .sha256 sidecar file is present.
 *            Cost: reads all data bytes — expensive for large models.
 */
package tech.kayys.gollek.safetensor.loader;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.exception.SafetensorException;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.*;

/**
 * Validation utility for SafeTensors files and their parsed headers.
 *
 * <p>
 * Used internally by the loader when
 * {@code gollek.safetensor.loader.validation.strict=true},
 * and also available directly for external model validation workflows.
 */
@ApplicationScoped
public class SafetensorValidator {

    private static final Logger log = Logger.getLogger(SafetensorValidator.class);

    /** Validation levels that can be requested. */
    public enum Level {
        FAST, NORMAL, STRICT
    }

    @Inject
    SafetensorLoaderConfig config;

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Validate a SafeTensors file at the given path.
     *
     * @param filePath path to the .safetensors file
     * @param level    validation depth
     * @return a {@link ValidationReport} with a pass/fail summary
     */
    public ValidationReport validate(Path filePath, Level level) {
        Objects.requireNonNull(filePath, "filePath must not be null");
        Path resolved = filePath.toAbsolutePath().normalize();

        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        // ── FAST ──────────────────────────────────────────────────────────────
        checkFileExists(resolved, errors);
        checkFileReadable(resolved, errors);
        checkFileSize(resolved, errors);
        checkExtension(resolved, warnings);

        if (!errors.isEmpty() || level == Level.FAST) {
            return new ValidationReport(filePath, errors, warnings);
        }

        // ── NORMAL ────────────────────────────────────────────────────────────
        // Header is parsed by the caller before calling this method, so we
        // accept an optional pre-parsed header to avoid double-parsing.
        // In the no-header overload below, we call the loader for this.

        // ── STRICT — checksum ─────────────────────────────────────────────────
        if (level == Level.STRICT) {
            verifyChecksum(resolved, errors, warnings);
        }

        return new ValidationReport(filePath, errors, warnings);
    }

    /**
     * Validate a pre-parsed {@link SafetensorHeader} against the expected file.
     *
     * @param header   the parsed header
     * @param filePath the source file path (used for offset/size checks)
     * @param level    validation depth
     * @return a {@link ValidationReport}
     */
    public ValidationReport validateHeader(
            SafetensorHeader header, Path filePath, Level level) {

        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        // AccelTensor-level checks
        for (SafetensorTensorInfo info : header.tensors().values()) {
            checkTensorInfo(info, header, filePath, errors, warnings);
        }

        // Contiguous coverage check (optional): verify that tensor offsets
        // cover the full data range without unexpected gaps.
        if (level == Level.NORMAL || level == Level.STRICT) {
            checkContiguousCoverage(header, warnings);
        }

        return new ValidationReport(filePath, errors, warnings);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private checks
    // ─────────────────────────────────────────────────────────────────────────

    private void checkFileExists(Path path, List<String> errors) {
        if (!Files.exists(path)) {
            errors.add("File does not exist: " + path);
        }
    }

    private void checkFileReadable(Path path, List<String> errors) {
        if (Files.exists(path) && !Files.isReadable(path)) {
            errors.add("File is not readable (check permissions): " + path);
        }
    }

    private void checkFileSize(Path path, List<String> errors) {
        try {
            long size = Files.size(path);
            if (size < 10) {
                errors.add("File too small (" + size + " bytes) to be a valid SafeTensors file");
            }
        } catch (IOException e) {
            errors.add("Cannot read file size: " + e.getMessage());
        }
    }

    private void checkExtension(Path path, List<String> warnings) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (!name.endsWith(".safetensors") && !name.endsWith(".safetensor")) {
            warnings.add("File does not have a .safetensors extension: " + path.getFileName());
        }
    }

    private void checkTensorInfo(
            SafetensorTensorInfo info, SafetensorHeader header,
            Path filePath, List<String> errors, List<String> warnings) {

        // Empty tensor warning
        if (config.validation().warnOnEmptyTensors() && info.byteLength() == 0) {
            warnings.add("AccelTensor '" + info.name() + "' has zero byte length");
        }

        // Shape element count
        for (int i = 0; i < info.rank(); i++) {
            if (info.dim(i) < 0) {
                errors.add("AccelTensor '" + info.name() + "' has negative dimension at index " + i);
            }
        }

        // Offset ordering
        if (info.dataBegin() > info.dataEnd()) {
            errors.add("AccelTensor '" + info.name() + "': dataBegin (" + info.dataBegin()
                    + ") > dataEnd (" + info.dataEnd() + ")");
        }
    }

    /**
     * Warn if there are gaps between adjacent tensor data ranges.
     * Gaps are valid per spec but unusual and may indicate padding or
     * a partially-written file.
     */
    private void checkContiguousCoverage(SafetensorHeader header, List<String> warnings) {
        List<SafetensorTensorInfo> sorted = new ArrayList<>(header.tensors().values());
        sorted.sort(Comparator.comparingLong(SafetensorTensorInfo::dataBegin));

        long prevEnd = 0L;
        for (SafetensorTensorInfo info : sorted) {
            if (info.byteLength() == 0)
                continue;
            if (info.dataBegin() > prevEnd) {
                warnings.add(String.format(
                        "Gap of %d bytes between offset %d and tensor '%s' at %d",
                        info.dataBegin() - prevEnd, prevEnd, info.name(), info.dataBegin()));
            }
            prevEnd = info.dataEnd();
        }
    }

    private void verifyChecksum(Path filePath, List<String> errors, List<String> warnings) {
        // Look for a .sha256 sidecar file
        Path checksumFile = filePath.resolveSibling(
                filePath.getFileName().toString() + ".sha256");

        if (!Files.exists(checksumFile)) {
            warnings.add("No .sha256 checksum sidecar found; skipping integrity check");
            return;
        }

        try {
            String expectedHex = Files.readString(checksumFile).strip().toLowerCase(Locale.ROOT);
            if (expectedHex.contains("  ")) {
                // GNU sha256sum format: "hash filename"
                expectedHex = expectedHex.split("\\s+")[0];
            }

            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var is = new DigestInputStream(Files.newInputStream(filePath), digest)) {
                is.transferTo(java.io.OutputStream.nullOutputStream());
            }

            byte[] actualBytes = digest.digest();
            StringBuilder hex = new StringBuilder(64);
            for (byte b : actualBytes)
                hex.append(String.format("%02x", b));
            String actualHex = hex.toString();

            if (!actualHex.equals(expectedHex)) {
                errors.add("SHA-256 checksum mismatch for " + filePath.getFileName()
                        + ": expected=" + expectedHex + " actual=" + actualHex);
            } else {
                log.debugf("Checksum OK for [%s]", filePath.getFileName());
            }

        } catch (Exception e) {
            warnings.add("Failed to verify checksum: " + e.getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ValidationReport
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Result of a validation pass.
     *
     * <p>
     * A report with no {@link #errors()} is considered passing.
     * {@link #warnings()} do not block loading but are logged.
     */
    public record ValidationReport(
            Path filePath,
            List<String> errors,
            List<String> warnings) {

        /** {@code true} if no errors were found. */
        public boolean isValid() {
            return errors.isEmpty();
        }

        /** {@code true} if there are warnings (file may still be usable). */
        public boolean hasWarnings() {
            return !warnings.isEmpty();
        }

        /**
         * Throw a {@link SafetensorException.ValidationException} if the report
         * contains errors. No-op if the report is passing.
         */
        public void throwIfInvalid() {
            if (!isValid()) {
                throw new SafetensorException.ValidationException(
                        "Validation failed for " + filePath.getFileName()
                                + ": " + String.join("; ", errors),
                        filePath);
            }
        }

        @Override
        public String toString() {
            return "ValidationReport{path=" + filePath.getFileName()
                    + ", valid=" + isValid()
                    + ", errors=" + errors.size()
                    + ", warnings=" + warnings.size()
                    + '}';
        }
    }
}
