/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantStatsTest.java
 * ───────────────────────
 * Tests for QuantStats.
 */
package tech.kayys.gollek.safetensor.quantization;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for QuantStats.
 */
class QuantStatsTest {

    @Test
    void testCompressionRatioCalculation() {
        QuantStats stats = QuantStats.builder()
                .originalSizeBytes(1000)
                .quantizedSizeBytes(250)
                .build();

        assertEquals(4.0, stats.getCompressionRatio());
    }

    @Test
    void testDurationCalculation() {
        Instant start = Instant.now();
        Instant end = start.plusSeconds(30);

        QuantStats stats = QuantStats.builder()
                .startTime(start)
                .endTime(end)
                .build();

        assertEquals(30, stats.getDuration().getSeconds());
    }

    @Test
    void testFormatSize() {
        assertEquals("512 B", QuantStats.formatSize(512));
        assertEquals("1.50 KB", QuantStats.formatSize(1536));
        assertEquals("2.00 MB", QuantStats.formatSize(2 * 1024 * 1024));
        assertEquals("3.50 GB", QuantStats.formatSize((long) (3.5 * 1024 * 1024 * 1024)));
    }

    @Test
    void testBuilderWithAllFields() {
        Instant start = Instant.now();
        Instant end = start.plusSeconds(60);

        QuantStats stats = QuantStats.builder()
                .originalSizeBytes(8_000_000_000L)
                .quantizedSizeBytes(2_000_000_000L)
                .peakMemoryBytes(4_000_000_000L)
                .tensorCount(256)
                .paramCount(7_000_000_000L)
                .avgQuantErrorMse(0.0001)
                .maxQuantError(0.01)
                .startTime(start)
                .endTime(end)
                .status(QuantStats.Status.COMPLETED)
                .build();

        assertEquals(8_000_000_000L, stats.getOriginalSizeBytes());
        assertEquals(2_000_000_000L, stats.getQuantizedSizeBytes());
        assertEquals(4.0, stats.getCompressionRatio());
        assertEquals(60, stats.getDuration().getSeconds());
        assertEquals(4_000_000_000L, stats.getPeakMemoryBytes());
        assertEquals(256, stats.getTensorCount());
        assertEquals(7_000_000_000L, stats.getParamCount());
        assertEquals(0.0001, stats.getAvgQuantErrorMse());
        assertEquals(0.01, stats.getMaxQuantError());
        assertEquals(QuantStats.Status.COMPLETED, stats.getStatus());
        assertNull(stats.getErrorMessage());
    }

    @Test
    void testFailedStatus() {
        QuantStats stats = QuantStats.builder()
                .status(QuantStats.Status.FAILED)
                .errorMessage("Quantization failed due to OOM")
                .build();

        assertEquals(QuantStats.Status.FAILED, stats.getStatus());
        assertEquals("Quantization failed due to OOM", stats.getErrorMessage());
    }

    @Test
    void testZeroCompressionRatio() {
        QuantStats stats = QuantStats.builder()
                .originalSizeBytes(1000)
                .quantizedSizeBytes(0)
                .build();

        assertEquals(0.0, stats.getCompressionRatio());
    }
}
