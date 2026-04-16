package tech.kayys.gollek.cuda.detection;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

/**
 * Detects NVIDIA CUDA device capabilities on this host using the CUDA Driver
 * API
 * via Java's Foreign Function & Memory (FFM) API.
 *
 * <p>
 * This implementation uses the CUDA Driver API (libcuda.so) directly without
 * requiring
 * the CUDA Runtime library. It provides accurate device information including:
 * - Device name
 * - Compute capability
 * - Memory size and available memory
 * - Number of streaming multiprocessors
 * - Estimated CUDA core count
 * </p>
 */
@ApplicationScoped
public class CudaDetector {

    private static final Logger LOG = Logger.getLogger(CudaDetector.class);

    // CUDA Driver API function names
    private static final String CU_INIT = "cuInit";
    private static final String CU_DEVICE_GET_COUNT = "cuDeviceGetCount";
    private static final String CU_DEVICE_GET = "cuDeviceGet";
    private static final String CU_DEVICE_GET_NAME = "cuDeviceGetName";
    private static final String CU_DEVICE_GET_ATTRIBUTE = "cuDeviceGetAttribute";
    private static final String CU_MEM_GET_INFO = "cuMemGetInfo";
    private static final String CU_CTX_CREATE = "cuCtxCreate";
    private static final String CU_CTX_PUSH_CURRENT = "cuCtxPushCurrent";
    private static final String CU_CTX_POP_CURRENT = "cuCtxPopCurrent";

    // CUDA device attributes (from cuda.h)
    private static final int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    private static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    private static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    // Symbol lookup and method handles
    private final Optional<SymbolLookup> cudaLookup;
    private MethodHandle cuInitHandle;
    private MethodHandle cuDeviceGetCountHandle;
    private MethodHandle cuDeviceGetHandle;
    private MethodHandle cuDeviceGetNameHandle;
    private MethodHandle cuDeviceGetAttributeHandle;
    private MethodHandle cuMemGetInfoHandle;
    private MethodHandle cuCtxCreateHandle;
    private MethodHandle cuCtxPushCurrentHandle;
    private MethodHandle cuCtxPopCurrentHandle;

    // Arena for memory allocation
    private final Arena arena = Arena.ofAuto();

    public CudaDetector() {
        this.cudaLookup = findCudaLibrary();
        if (cudaLookup.isPresent()) {
            initializeMethodHandles();
        }
    }

    /**
     * Find the CUDA library in standard locations.
     */
    private Optional<SymbolLookup> findCudaLibrary() {
        String[] libraryPaths = {
                "libcuda.so",
                "libcuda.so.1",
                "/usr/lib/x86_64-linux-gnu/libcuda.so",
                "/usr/lib64/libcuda.so",
                "/usr/local/cuda/lib64/libcuda.so",
                "cuda.dll", // Windows
                "cuda" // macOS (unlikely but included)
        };

        // Check CUDA_PATH environment variable
        String cudaPath = System.getenv("CUDA_PATH");
        if (cudaPath != null && !cudaPath.isBlank()) {
            Path path = Paths.get(cudaPath, "lib", "libcuda.so");
            if (Files.exists(path)) {
                try {
                    return Optional.of(SymbolLookup.libraryLookup(path.toString(), arena));
                } catch (Exception e) {
                    LOG.debugf("Failed to load CUDA from %s: %s", path, e.getMessage());
                }
            }
        }

        // Try standard library names
        for (String libName : libraryPaths) {
            try {
                SymbolLookup lookup = SymbolLookup.libraryLookup(libName, arena);
                // Verify we can find cuInit to confirm it's the right library
                if (lookup.find(CU_INIT).isPresent()) {
                    LOG.infof("Found CUDA library: %s", libName);
                    return Optional.of(lookup);
                }
            } catch (Exception e) {
                LOG.debugf("Failed to load %s: %s", libName, e.getMessage());
            }
        }

        LOG.debug("No CUDA library found");
        return Optional.empty();
    }

    /**
     * Initialize method handles for CUDA functions.
     */
    private void initializeMethodHandles() {
        if (cudaLookup.isEmpty()) {
            return;
        }

        SymbolLookup lookup = cudaLookup.get();
        Linker linker = Linker.nativeLinker();

        try {
            // cuInit(unsigned int flags) -> CUresult
            cuInitHandle = linker.downcallHandle(
                    lookup.find(CU_INIT).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

            // cuDeviceGetCount(int* count) -> CUresult
            cuDeviceGetCountHandle = linker.downcallHandle(
                    lookup.find(CU_DEVICE_GET_COUNT).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            // cuDeviceGet(CUdevice* device, int ordinal) -> CUresult
            cuDeviceGetHandle = linker.downcallHandle(
                    lookup.find(CU_DEVICE_GET).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            // cuDeviceGetName(char* name, int len, CUdevice dev) -> CUresult
            cuDeviceGetNameHandle = linker.downcallHandle(
                    lookup.find(CU_DEVICE_GET_NAME).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT));

            // cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) -> CUresult
            cuDeviceGetAttributeHandle = linker.downcallHandle(
                    lookup.find(CU_DEVICE_GET_ATTRIBUTE).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT));

            // cuMemGetInfo(size_t* free, size_t* total) -> CUresult
            cuMemGetInfoHandle = linker.downcallHandle(
                    lookup.find(CU_MEM_GET_INFO).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

            // cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) -> CUresult
            cuCtxCreateHandle = linker.downcallHandle(
                    lookup.find(CU_CTX_CREATE).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT));

            // cuCtxPushCurrent(CUcontext ctx) -> CUresult
            cuCtxPushCurrentHandle = linker.downcallHandle(
                    lookup.find(CU_CTX_PUSH_CURRENT).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            // cuCtxPopCurrent(CUcontext* pctx) -> CUresult
            cuCtxPopCurrentHandle = linker.downcallHandle(
                    lookup.find(CU_CTX_POP_CURRENT).orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        } catch (Exception e) {
            LOG.errorf("Failed to initialize CUDA method handles: %s", e.getMessage());
        }
    }

    /**
     * Detect CUDA capabilities on this host.
     *
     * @return CudaCapabilities describing the primary CUDA device
     */
    public CudaCapabilities detect() {
        // Check if CUDA_VISIBLE_DEVICES is set to "none"
        String visibleDevices = System.getenv("CUDA_VISIBLE_DEVICES");
        if ("none".equalsIgnoreCase(visibleDevices)) {
            LOG.debug("CUDA disabled via CUDA_VISIBLE_DEVICES=none");
            return CudaCapabilities.unavailable("CUDA_VISIBLE_DEVICES=none");
        }

        // Check if CUDA library is available
        if (cudaLookup.isEmpty() || cuInitHandle == null) {
            LOG.debug("CUDA library not available");
            return CudaCapabilities.unavailable("CUDA library not found");
        }

        // Try to detect CUDA via FFM
        try {
            return detectViaFfm();
        } catch (Throwable e) {
            LOG.warnf("CUDA detection failed: %s", e.getMessage());
            return CudaCapabilities.unavailable("Detection failed: " + e.getMessage());
        }
    }

    /**
     * Detect CUDA capabilities using FFM bindings.
     */
    private CudaCapabilities detectViaFfm() throws Throwable {
        // Initialize CUDA driver API
        int result = (int) cuInitHandle.invokeExact(0);
        if (result != 0) {
            throw new RuntimeException("cuInit failed with code: " + result);
        }

        // Get device count
        MemorySegment countPtr = arena.allocate(ValueLayout.JAVA_INT);
        result = (int) cuDeviceGetCountHandle.invokeExact(countPtr);
        if (result != 0) {
            throw new RuntimeException("cuDeviceGetCount failed with code: " + result);
        }
        int deviceCount = countPtr.get(ValueLayout.JAVA_INT, 0);

        if (deviceCount == 0) {
            return CudaCapabilities.unavailable("No CUDA-capable devices found");
        }

        // Check CUDA_VISIBLE_DEVICES filtering
        String visibleDevices = System.getenv("CUDA_VISIBLE_DEVICES");
        if (visibleDevices != null && !visibleDevices.isBlank()) {
            String[] devices = visibleDevices.split(",");
            deviceCount = Math.min(deviceCount, devices.length);
        }

        // Use first available device
        int deviceId = 0;

        // Get device handle
        MemorySegment devicePtr = arena.allocate(ValueLayout.JAVA_INT);
        result = (int) cuDeviceGetHandle.invokeExact(devicePtr, deviceId);
        if (result != 0) {
            throw new RuntimeException("cuDeviceGet failed with code: " + result);
        }
        int device = devicePtr.get(ValueLayout.JAVA_INT, 0);

        // Create context for memory queries
        MemorySegment contextPtr = arena.allocate(ValueLayout.ADDRESS);
        result = (int) cuCtxCreateHandle.invokeExact(contextPtr, 0, device);
        if (result != 0) {
            throw new RuntimeException("cuCtxCreate failed with code: " + result);
        }
        MemorySegment context = contextPtr.get(ValueLayout.ADDRESS, 0);

        // Push context
        result = (int) cuCtxPushCurrentHandle.invokeExact(context);
        if (result != 0) {
            throw new RuntimeException("cuCtxPushCurrent failed with code: " + result);
        }

        try {
            // Get device name - allocate 256 bytes for the name
            MemorySegment nameSegment = arena.allocate(256);
            result = (int) cuDeviceGetNameHandle.invokeExact(nameSegment, 256, device);
            if (result != 0) {
                throw new RuntimeException("cuDeviceGetName failed with code: " + result);
            }

            // Read the C string from memory segment
            String deviceName = readCString(nameSegment);

            // Get compute capability
            int major = getDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
            int minor = getDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
            int computeCap = major * 10 + minor;

            // Get multiprocessor count
            int smCount = getDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);

            // Get memory info
            MemorySegment freePtr = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment totalPtr = arena.allocate(ValueLayout.JAVA_LONG);
            result = (int) cuMemGetInfoHandle.invokeExact(freePtr, totalPtr);
            if (result != 0) {
                throw new RuntimeException("cuMemGetInfo failed with code: " + result);
            }
            long freeMem = freePtr.get(ValueLayout.JAVA_LONG, 0);
            long totalMem = totalPtr.get(ValueLayout.JAVA_LONG, 0);

            // Estimate CUDA cores
            int cudaCores = estimateCudaCores(computeCap, smCount);

            return new CudaCapabilities(
                    true,
                    computeCap,
                    deviceName,
                    cudaCores,
                    totalMem,
                    freeMem,
                    smCount,
                    null);

        } finally {
            // Pop context
            MemorySegment oldContextPtr = arena.allocate(ValueLayout.ADDRESS);
            result = (int) cuCtxPopCurrentHandle.invokeExact(oldContextPtr);
            if (result != 0) {
                LOG.warnf("cuCtxPopCurrent failed with code: %d", result);
            }
        }
    }

    /**
     * Read a C-style null-terminated string from a MemorySegment.
     */
    private String readCString(MemorySegment segment) {
        StringBuilder sb = new StringBuilder();
        byte b;
        long offset = 0;

        while ((b = segment.get(ValueLayout.JAVA_BYTE, offset)) != 0) {
            sb.append((char) b);
            offset++;
        }

        return sb.toString();
    }

    /**
     * Get a device attribute value.
     */
    private int getDeviceAttribute(int device, int attribute) throws Throwable {
        MemorySegment valuePtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) cuDeviceGetAttributeHandle.invokeExact(valuePtr, attribute, device);
        if (result != 0) {
            throw new RuntimeException("cuDeviceGetAttribute failed with code: " + result);
        }
        return valuePtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Estimate CUDA core count from compute capability and SM count.
     * Based on NVIDIA's architecture specifications.
     */
    private int estimateCudaCores(int computeCap, int smCount) {
        int major = computeCap / 10;
        int coresPerSm;

        switch (major) {
            case 2: // Fermi
                coresPerSm = 32;
                break;
            case 3: // Kepler
                coresPerSm = 192;
                break;
            case 5: // Maxwell
                coresPerSm = 128;
                break;
            case 6: // Pascal
                coresPerSm = 64;
                break;
            case 7: // Volta, Turing
                coresPerSm = 64;
                break;
            case 8: // Ampere
                coresPerSm = 128;
                break;
            case 9: // Hopper
                coresPerSm = 128;
                break;
            case 10: // Blackwell (future)
                coresPerSm = 128;
                break;
            default:
                LOG.warnf("Unknown compute capability %d, assuming 128 cores/SM", major);
                coresPerSm = 128;
        }

        return smCount * coresPerSm;
    }

    /**
     * Clean up resources.
     */
    public void close() {
        arena.close();
    }
}