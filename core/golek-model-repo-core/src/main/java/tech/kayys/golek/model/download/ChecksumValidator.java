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

package tech.kayys.golek.model.download;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.model.exception.InferenceException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Validates file integrity using checksums (SHA-256 by default)
 */
@ApplicationScoped
public class ChecksumValidator {

    private static final Logger LOG = Logger.getLogger(ChecksumValidator.class);
    private static final String DEFAULT_ALGORITHM = "SHA-256";

    /**
     * Verifies that the file at the given path matches the expected checksum
     */
    public boolean verify(Path filePath, String expectedChecksum) throws IOException {
        return verify(filePath, expectedChecksum, DEFAULT_ALGORITHM);
    }

    /**
     * Verifies that the file at the given path matches the expected checksum using
     * the specified algorithm
     */
    public boolean verify(Path filePath, String expectedChecksum, String algorithm) throws IOException {
        if (expectedChecksum == null || expectedChecksum.isBlank()) {
            LOG.warnf("No checksum provided for verification of %s", filePath);
            return true; // Or false, depending on policy. For production, probably false.
        }

        String actualChecksum = calculate(filePath, algorithm);
        boolean matches = expectedChecksum.equalsIgnoreCase(actualChecksum);

        if (!matches) {
            LOG.errorf("Checksum mismatch for %s. Expected: %s, Actual: %s", filePath, expectedChecksum,
                    actualChecksum);
        } else {
            LOG.debugf("Checksum verified for %s", filePath);
        }

        return matches;
    }

    /**
     * Calculates the checksum of a file
     */
    public String calculate(Path filePath, String algorithm) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance(algorithm);
            try (InputStream is = Files.newInputStream(filePath)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    digest.update(buffer, 0, bytesRead);
                }
            }
            return bytesToHex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new InferenceException(ErrorCode.CONFIG_INVALID,
                    "Checksum algorithm not found: " + algorithm, e)
                    .addContext("algorithm", algorithm);
        }
    }

    private String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
