package tech.kayys.gollek.spi.tensor;

import tech.kayys.gollek.spi.model.DeviceType;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Registry for compute kernels with automatic device detection and selection.
 * <p>
 * Provides a centralized way to discover, register, and select the best available
 * compute kernel based on hardware capabilities and preferences.
 * <p>
 * <b>Usage:</b>
 * <pre>
 * // Automatic selection (best available device)
 * ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
 * 
 * // Specific device selection
 * ComputeKernel cuda = ComputeKernelRegistry.get().getByDeviceType(DeviceType.CUDA);
 * 
 * // Register custom kernel
 * ComputeKernelRegistry.get().register(myCustomKernel);
 * </pre>
 * 
 * @since 0.1.0
 */
public class ComputeKernelRegistry {

    private static final Logger LOG = Logger.getLogger(ComputeKernelRegistry.class);
    private static volatile ComputeKernelRegistry instance;

    /** Registered kernel suppliers (for lazy instantiation) */
    private final Map<DeviceType, Supplier<ComputeKernel>> kernelSuppliers = new ConcurrentHashMap<>();

    /** Instantiated kernels */
    private final Map<DeviceType, ComputeKernel> kernelInstances = new ConcurrentHashMap<>();

    /** Device priority order (first available wins) */
    private final List<DeviceType> devicePriority = new ArrayList<>(List.of(
        DeviceType.CUDA,       // Standard NVIDIA
        DeviceType.METAL,      // Apple Silicon
        DeviceType.ROCM,       // AMD
        DeviceType.CPU         // Fallback
    ));

    private ComputeKernelRegistry() {
        // Register default kernel suppliers
        registerDefaults();
    }

    /**
     * Gets the singleton registry instance.
     */
    public static ComputeKernelRegistry get() {
        if (instance == null) {
            synchronized (ComputeKernelRegistry.class) {
                if (instance == null) {
                    instance = new ComputeKernelRegistry();
                }
            }
        }
        return instance;
    }

    /**
     * Registers a kernel supplier for a device type.
     *
     * @param deviceType the device type
     * @param supplier supplier to create kernel instances
     */
    public void register(DeviceType deviceType, Supplier<ComputeKernel> supplier) {
        kernelSuppliers.put(deviceType, supplier);
        LOG.infof("Registered kernel supplier: %s", deviceType);
    }

    /**
     * Registers a pre-instantiated kernel.
     *
     * @param kernel the kernel instance
     */
    public void register(ComputeKernel kernel) {
        kernelInstances.put(kernel.deviceType(), kernel);
        LOG.infof("Registered kernel instance: %s (%s)", 
            kernel.deviceType(), kernel.deviceName());
    }

    /**
     * Gets the best available kernel based on device priority.
     *
     * @return best available kernel, or CPU fallback if nothing else is available
     */
    public ComputeKernel getBestAvailable() {
        // Try priority list
        for (DeviceType deviceType : devicePriority) {
            try {
                ComputeKernel kernel = getByDeviceType(deviceType);
                if (kernel != null && kernel.isAvailable()) {
                    LOG.infof("Selected compute kernel: %s (%s)", 
                        kernel.deviceType(), kernel.deviceName());
                    return kernel;
                }
            } catch (Exception e) {
                LOG.debugf(e, "Device %s not available: %s", deviceType, e.getMessage());
            }
        }

        // Ultimate fallback: try to get or create CPU kernel
        ComputeKernel cpuKernel = getByDeviceType(DeviceType.CPU);
        if (cpuKernel == null) {
            // Register CPU kernel on the fly
            register(DeviceType.CPU, () -> new CpuKernel());
            cpuKernel = getByDeviceType(DeviceType.CPU);
        }
        return cpuKernel;
    }

    /**
     * Gets a kernel by device type.
     *
     * @param deviceType the device type
     * @return kernel instance, or null if not available
     */
    public ComputeKernel getByDeviceType(DeviceType deviceType) {
        // Return cached instance if available
        ComputeKernel cached = kernelInstances.get(deviceType);
        if (cached != null) {
            return cached;
        }

        // Instantiate from supplier
        Supplier<ComputeKernel> supplier = kernelSuppliers.get(deviceType);
        if (supplier != null) {
            try {
                ComputeKernel kernel = supplier.get();
                if (kernel.isAvailable()) {
                    kernelInstances.put(deviceType, kernel);
                    LOG.infof("Instantiated kernel: %s (%s)", 
                        deviceType, kernel.deviceName());
                    return kernel;
                }
            } catch (Exception e) {
                LOG.debugf(e, "Failed to instantiate %s kernel", deviceType);
            }
        }

        return null;
    }

    /**
     * Gets all available kernels.
     */
    public List<ComputeKernel> getAllAvailable() {
        List<ComputeKernel> available = new ArrayList<>();
        for (DeviceType deviceType : kernelSuppliers.keySet()) {
            ComputeKernel kernel = getByDeviceType(deviceType);
            if (kernel != null && kernel.isAvailable()) {
                available.add(kernel);
            }
        }
        return available;
    }

    /**
     * Checks if a specific device type is available.
     */
    public boolean isAvailable(DeviceType deviceType) {
        return getByDeviceType(deviceType) != null;
    }

    /**
     * Sets device priority order.
     *
     * @param priority ordered list of device types (first available wins)
     */
    public void setDevicePriority(List<DeviceType> priority) {
        devicePriority.clear();
        devicePriority.addAll(priority);
        // Always ensure CPU is last
        if (!devicePriority.contains(DeviceType.CPU)) {
            devicePriority.add(DeviceType.CPU);
        }
    }

    /**
     * Clears all registered kernels (useful for testing).
     */
    public void clear() {
        kernelInstances.clear();
    }

    // ── Private Helpers ─────────────────────────────────────────────────

    private void registerDefaults() {
        // CUDA
        register(DeviceType.CUDA, () -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.cuda.binding.CudaBinding");
                return (ComputeKernel) clazz.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                throw new IllegalStateException("CUDA not available", e);
            }
        });

        // Metal
        register(DeviceType.METAL, () -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.metal.binding.MetalBinding");
                return (ComputeKernel) clazz.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                throw new IllegalStateException("Metal not available", e);
            }
        });

        // ROCm
        register(DeviceType.ROCM, () -> {
            try {
                Class<?> clazz = Class.forName("tech.kayys.gollek.rocm.binding.RocmHipBinding");
                return (ComputeKernel) clazz.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                throw new IllegalStateException("ROCm not available", e);
            }
        });

        // CPU (always available)
        register(DeviceType.CPU, () -> new CpuKernel());
    }
}
