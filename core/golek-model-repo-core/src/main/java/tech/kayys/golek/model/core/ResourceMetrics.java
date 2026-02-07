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

package tech.kayys.golek.model.core;

/**
 * Resource metrics - MOVED to golek-spi
 * 
 * @deprecated Use {@link tech.kayys.golek.spi.model.ResourceMetrics} instead
 */
@Deprecated(since = "1.1.0", forRemoval = true)
public record ResourceMetrics(
                long cpuUsagePercent,
                long memoryUsageBytes,
                long gpuUsagePercent,
                long vramUsageBytes,
                int activeRequests) {

        // Delegate to API version
        public tech.kayys.golek.spi.model.ResourceMetrics toApi() {
                return new tech.kayys.golek.spi.model.ResourceMetrics(
                                cpuUsagePercent, memoryUsageBytes, gpuUsagePercent, vramUsageBytes, activeRequests);
        }
}
