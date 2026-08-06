/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;

record HostLoadSnapshot(
        int availableProcessors,
        double systemLoadAverage,
        double systemCpuLoad,
        double processCpuLoad,
        long freeMemoryBytes,
        long totalMemoryBytes) {

    static HostLoadSnapshot capture() {
        java.lang.management.OperatingSystemMXBean bean = ManagementFactory.getOperatingSystemMXBean();
        int processors = Math.max(1, bean.getAvailableProcessors());
        double loadAverage = bean.getSystemLoadAverage();
        double cpuLoad = invokeDouble(bean, "getCpuLoad", "getSystemCpuLoad");
        double processCpuLoad = invokeDouble(bean, "getProcessCpuLoad");
        long freeMemory = invokeLong(bean, "getFreeMemorySize", "getFreePhysicalMemorySize");
        long totalMemory = invokeLong(bean, "getTotalMemorySize", "getTotalPhysicalMemorySize");
        return new HostLoadSnapshot(processors, loadAverage, cpuLoad, processCpuLoad, freeMemory, totalMemory);
    }

    boolean isAvailable() {
        return availableProcessors > 0
                || isFinite(systemLoadAverage)
                || isFinite(systemCpuLoad)
                || isFinite(processCpuLoad);
    }

    double normalizedLoad() {
        if (!isFinite(systemLoadAverage) || availableProcessors <= 0) {
            return 0.0;
        }
        return Math.max(0.0, systemLoadAverage / availableProcessors);
    }

    double freeMemoryPercent() {
        if (freeMemoryBytes <= 0L || totalMemoryBytes <= 0L) {
            return Double.NaN;
        }
        return Math.max(0.0, Math.min(1.0, (double) freeMemoryBytes / totalMemoryBytes));
    }

    String pressureLabel() {
        double normalizedLoad = normalizedLoad();
        double cpuLoad = isFinite(systemCpuLoad) ? systemCpuLoad : 0.0;
        double freeMemory = freeMemoryPercent();
        if (normalizedLoad >= 1.25 || cpuLoad >= 0.90 || (isFinite(freeMemory) && freeMemory <= 0.08)) {
            return "high";
        }
        if (normalizedLoad >= 0.85 || cpuLoad >= 0.70 || (isFinite(freeMemory) && freeMemory <= 0.15)) {
            return "elevated";
        }
        return "normal";
    }

    boolean hasPressure() {
        return !"normal".equals(pressureLabel());
    }

    double percentOrZero(double value) {
        return isFinite(value) ? value * 100.0 : 0.0;
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static double invokeDouble(Object target, String... methodNames) {
        Object value = invokeNumber(target, methodNames);
        if (value instanceof Number number) {
            double result = number.doubleValue();
            return result >= 0.0 ? result : Double.NaN;
        }
        return Double.NaN;
    }

    private static long invokeLong(Object target, String... methodNames) {
        Object value = invokeNumber(target, methodNames);
        return value instanceof Number number ? number.longValue() : -1L;
    }

    private static Object invokeNumber(Object target, String... methodNames) {
        for (String methodName : methodNames) {
            try {
                Method method = target.getClass().getMethod(methodName);
                return method.invoke(target);
            } catch (ReflectiveOperationException | RuntimeException ignored) {
                // Some JVMs expose only a subset of the com.sun.management metrics.
            }
        }
        return null;
    }
}
