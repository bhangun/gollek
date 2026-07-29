package tech.kayys.gollek.cli.runtime;

import java.lang.reflect.Method;
import java.util.Locale;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;

/**
 * Small CLI boundary for Metal runtime probing.
 *
 * <p>
 * The CLI should not directly depend on the optional Metal module at
 * compile-time, because CPU-only and non-macOS builds still need to start.
 */
public final class CliMetalRuntime {
    private CliMetalRuntime() {
    }

    public record NativeStatus(boolean active, String deviceName, String errorMessage) {
    }

    public static boolean isMetal(KernelPlatform platform) {
        String displayName = platform == null ? null : platform.getDisplayName();
        return displayName != null && displayName.toLowerCase(Locale.ROOT).contains("metal");
    }

    public static void initializeIfMetal(KernelPlatform platform) {
        if (isMetal(platform)) {
            initialize();
        }
    }

    public static void initialize() {
        try {
            Class<?> metalBindingClass = Class.forName("tech.kayys.alkhawarizm.metal.binding.MetalBinding");
            Method initializeMethod = metalBindingClass.getMethod("initialize");
            initializeMethod.invoke(null);
            Method getInstanceMethod = metalBindingClass.getMethod("getInstance");
            Object binding = getInstanceMethod.invoke(null);
            Method initMethod = binding.getClass().getMethod("init");
            initMethod.invoke(binding);
        } catch (Throwable ignored) {
            // Callers validate status separately and decide whether CPU fallback is
            // acceptable.
        }
    }

    public static boolean isNativeActive() {
        return status().active();
    }

    public static NativeStatus status() {
        try {
            Class<?> metalBindingClass = Class.forName("tech.kayys.alkhawarizm.metal.binding.MetalBinding");
            Method getInstanceMethod = metalBindingClass.getMethod("getInstance");
            Object binding = getInstanceMethod.invoke(null);

            String device = "unknown";
            try {
                Method deviceMethod = binding.getClass().getMethod("deviceName");
                Object deviceValue = deviceMethod.invoke(binding);
                if (deviceValue != null) {
                    device = String.valueOf(deviceValue);
                }
            } catch (Throwable ignored) {
                // Device name is diagnostic only.
            }

            boolean nativeActive = false;
            try {
                Method nativeMethod = binding.getClass().getMethod("isRuntimeActive");
                Object nativeValue = nativeMethod.invoke(binding);
                if (nativeValue instanceof Boolean b) {
                    nativeActive = b;
                }
            } catch (NoSuchMethodException ignored) {
                try {
                    Method nativeMethod = binding.getClass().getMethod("isNativeAvailable");
                    Object nativeValue = nativeMethod.invoke(binding);
                    if (nativeValue instanceof Boolean b) {
                        nativeActive = b;
                    }
                } catch (NoSuchMethodException ignoredAgain) {
                    nativeActive = false;
                }
            }

            return new NativeStatus(nativeActive, device, null);
        } catch (Throwable t) {
            String message = t.getMessage();
            return new NativeStatus(false, "unknown", message == null ? t.getClass().getSimpleName() : message);
        }
    }

    public static boolean allowCpuFallbackWhenMetalRequested() {
        String env = System.getenv("GOLLEK_ALLOW_CPU_FALLBACK");
        if (env == null || env.isBlank()) {
            return false;
        }
        return "1".equals(env) || "true".equalsIgnoreCase(env) || "yes".equalsIgnoreCase(env);
    }
}
