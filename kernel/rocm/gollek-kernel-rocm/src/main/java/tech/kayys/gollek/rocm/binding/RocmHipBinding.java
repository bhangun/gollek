package tech.kayys.gollek.rocm.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM binding to the HIP runtime C API ({@code hip_runtime_api.h}).
 *
 * <p>Loads {@code libamdhip64.so} at runtime and binds the key HIP runtime
 * functions for device management, memory management, kernel execution, and
 * HIP module loading. Mirrors the singleton pattern of
 * {@link tech.kayys.gollek.kernel.fa3.FlashAttention3Binding} exactly.
 *
 * <h2>ROCm / HIP</h2>
 * <p>HIP (Heterogeneous-Compute Interface for Portability) is AMD's runtime
 * API that mirrors the CUDA runtime API almost 1:1. The same CUDA kernel code
 * can be ported to HIP with minimal changes using {@code hipify-clang}. The
 * Gollek extension module's CUDA kernels (FA4, GDN, QLoRA, EP, Offload,
 * EVICPRESS) all have direct HIP equivalents compiled from the same sources
 * with {@code hipcc} instead of {@code nvcc}.
 *
 * <h2>Key differences from CUDA</h2>
 * <table border="1" cellpadding="3">
 * <tr><th>CUDA</th><th>HIP</th></tr>
 * <tr><td>cudaError_t</td><td>hipError_t (same int codes)</td></tr>
 * <tr><td>cudaMalloc</td><td>hipMalloc</td></tr>
 * <tr><td>cudaMemcpy</td><td>hipMemcpy (same enum values)</td></tr>
 * <tr><td>cudaStream_t</td><td>hipStream_t</td></tr>
 * <tr><td>cudaDeviceProp</td><td>hipDeviceProp_t</td></tr>
 * <tr><td>cuModuleLoad</td><td>hipModuleLoad (.hsaco file)</td></tr>
 * <tr><td>cuLaunchKernel</td><td>hipModuleLaunchKernel</td></tr>
 * </table>
 *
 * <h2>MI300X specifics</h2>
 * <p>The AMD MI300X has 192 GB HBM3 unified CPU+GPU memory (XDNA architecture),
 * making it comparable to Apple Silicon's UMA but at data-centre scale. On
 * MI300X the {@code hipMemcpyHostToDevice} calls in this binding are near-zero
 * cost because the CPU and GPU share physical memory.
 *
 * <h2>C functions bound</h2>
 * <pre>
 * hipError_t hipGetDeviceCount(int*)
 * hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int)
 * hipError_t hipSetDevice(int)
 * hipError_t hipDeviceSynchronize()
 * hipError_t hipMalloc(void**, size_t)
 * hipError_t hipMallocManaged(void**, size_t, unsigned int)
 * hipError_t hipFree(void*)
 * hipError_t hipMemcpy(void*, void*, size_t, hipMemcpyKind)
 * hipError_t hipMemcpyAsync(void*, void*, size_t, hipMemcpyKind, hipStream_t)
 * hipError_t hipMemset(void*, int, size_t)
 * hipError_t hipStreamCreate(hipStream_t*)
 * hipError_t hipStreamSynchronize(hipStream_t)
 * hipError_t hipStreamDestroy(hipStream_t)
 * hipError_t hipModuleLoad(hipModule_t*, const char*)
 * hipError_t hipModuleGetFunction(hipFunction_t*, hipModule_t, const char*)
 * hipError_t hipModuleLaunchKernel(hipFunction_t,
 *              uint, uint, uint, uint, uint, uint,
 *              uint, hipStream_t, void**, void**)
 * const char* hipGetErrorString(hipError_t)
 * </pre>
 *
 * <h2>hipMemcpyKind values</h2>
 * <pre>
 *   hipMemcpyHostToHost     = 0
 *   hipMemcpyHostToDevice   = 1
 *   hipMemcpyDeviceToHost   = 2
 *   hipMemcpyDeviceToDevice = 3
 * </pre>
 *
 * <h2>Build / install ROCm</h2>
 * <pre>
 * # Ubuntu 22.04:
 * wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*amd64.deb
 * sudo apt install ./amdgpu-install*.deb
 * sudo amdgpu-install --usecase=rocm
 * # Library path: /opt/rocm/lib/libamdhip64.so
 * </pre>
 */
public class RocmHipBinding {

    private static final Logger LOG = Logger.getLogger(RocmHipBinding.class);
    private static volatile RocmHipBinding instance;

    // ── hipMemcpyKind enum values ─────────────────────────────────────────────
    public static final int HIP_MEMCPY_H2H = 0;
    public static final int HIP_MEMCPY_H2D = 1;
    public static final int HIP_MEMCPY_D2H = 2;
    public static final int HIP_MEMCPY_D2D = 3;

    // hipMallocManaged flags
    public static final int HIP_MEM_ATTACH_GLOBAL = 1;

    // ── C function names ──────────────────────────────────────────────────────
    private static final String FN_GET_DEVICE_COUNT   = "hipGetDeviceCount";
    private static final String FN_GET_DEVICE_PROPS   = "hipGetDeviceProperties";
    private static final String FN_SET_DEVICE         = "hipSetDevice";
    private static final String FN_DEVICE_SYNC        = "hipDeviceSynchronize";
    private static final String FN_MALLOC             = "hipMalloc";
    private static final String FN_MALLOC_MANAGED     = "hipMallocManaged";
    private static final String FN_FREE               = "hipFree";
    private static final String FN_MEMCPY             = "hipMemcpy";
    private static final String FN_MEMCPY_ASYNC       = "hipMemcpyAsync";
    private static final String FN_MEMSET             = "hipMemset";
    private static final String FN_STREAM_CREATE      = "hipStreamCreate";
    private static final String FN_STREAM_SYNC        = "hipStreamSynchronize";
    private static final String FN_STREAM_DESTROY     = "hipStreamDestroy";
    private static final String FN_MODULE_LOAD        = "hipModuleLoad";
    private static final String FN_MODULE_GET_FN      = "hipModuleGetFunction";
    private static final String FN_MODULE_LAUNCH      = "hipModuleLaunchKernel";
    private static final String FN_GET_ERROR_STRING   = "hipGetErrorString";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private RocmHipBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new RocmHipBinding(lk);
            LOG.infof("RocmHipBinding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load ROCm HIP from %s: %s. CPU fallback active.",
                    libraryPath, e.getMessage());
            instance = new RocmHipBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new RocmHipBinding(null);
        LOG.info("RocmHipBinding: CPU fallback mode");
    }

    public static RocmHipBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("RocmHipBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Device management ─────────────────────────────────────────────────────

    /** Number of ROCm GPU devices visible to the process. */
    public int getDeviceCount() {
        if (!nativeAvailable) return 0;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment n = a.allocate(ValueLayout.JAVA_INT);
            int err = (int) invoke(FN_GET_DEVICE_COUNT, n);
            return err == 0 ? n.get(ValueLayout.JAVA_INT, 0) : 0;
        } catch (Throwable t) { return 0; }
    }

    /** Set the current device for the calling thread. */
    public int setDevice(int deviceId) {
        if (!nativeAvailable) return 0;
        try { return (int) invoke(FN_SET_DEVICE, deviceId); }
        catch (Throwable t) { return -1; }
    }

    /** Block the calling thread until all operations on the device complete. */
    public int deviceSynchronize() {
        if (!nativeAvailable) return 0;
        try { return (int) invoke(FN_DEVICE_SYNC); }
        catch (Throwable t) { return -1; }
    }

    /**
     * Get device name from hipDeviceProp_t.
     * hipDeviceProp_t is a large struct; we read only the first 256 bytes
     * (the {@code name} field is a {@code char[256]} at offset 0).
     */
    public String getDeviceName(int deviceId) {
        if (!nativeAvailable) return "CPU (no ROCm)";
        try (Arena a = Arena.ofConfined()) {
            MemorySegment props = a.allocate(1024L, 8); // hipDeviceProp_t
            int err = (int) invoke(FN_GET_DEVICE_PROPS, props, deviceId);
            if (err != 0) return "ROCm device " + deviceId;
            return props.reinterpret(256).getString(0);
        } catch (Throwable t) { return "ROCm device " + deviceId; }
    }

    // ── Memory management ─────────────────────────────────────────────────────

    /**
     * Allocate device memory.
     *
     * @param bytes number of bytes
     * @return device pointer as MemorySegment, or NULL on failure
     */
    public MemorySegment malloc(long bytes) {
        if (!nativeAvailable) return MemorySegment.NULL;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment ptrPtr = a.allocate(ValueLayout.ADDRESS);
            int err = (int) invoke(FN_MALLOC, ptrPtr, bytes);
            if (err != 0) {
                LOG.warnf("hipMalloc(%d bytes) failed: %d", bytes, err);
                return MemorySegment.NULL;
            }
            return ptrPtr.get(ValueLayout.ADDRESS, 0).reinterpret(bytes);
        } catch (Throwable t) { return MemorySegment.NULL; }
    }

    /**
     * Allocate managed (unified) memory — accessible from both CPU and GPU.
     * On MI300X this is essentially always the same physical DRAM.
     *
     * @param bytes  number of bytes
     * @param flags  {@link #HIP_MEM_ATTACH_GLOBAL}
     * @return unified memory pointer
     */
    public MemorySegment mallocManaged(long bytes, int flags) {
        if (!nativeAvailable) return Arena.ofShared().allocate(bytes, 64);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment ptrPtr = a.allocate(ValueLayout.ADDRESS);
            int err = (int) invoke(FN_MALLOC_MANAGED, ptrPtr, bytes, flags);
            if (err != 0) {
                LOG.warnf("hipMallocManaged(%d bytes) failed: %d — using Java heap", bytes, err);
                return Arena.ofShared().allocate(bytes, 64);
            }
            return ptrPtr.get(ValueLayout.ADDRESS, 0).reinterpret(bytes);
        } catch (Throwable t) { return Arena.ofShared().allocate(bytes, 64); }
    }

    /** Free device memory previously allocated by {@link #malloc}. */
    public int free(MemorySegment ptr) {
        if (!nativeAvailable || isNull(ptr)) return 0;
        try { return (int) invoke(FN_FREE, ptr); }
        catch (Throwable t) { return -1; }
    }

    /**
     * Synchronous host→device copy.
     *
     * @param dst    device pointer
     * @param src    host pointer
     * @param bytes  number of bytes
     * @return hipError_t (0 = success)
     */
    public int memcpyH2D(MemorySegment dst, MemorySegment src, long bytes) {
        if (!nativeAvailable) {
            MemorySegment.copy(src, 0L, dst, 0L, bytes);
            return 0;
        }
        try { return (int) invoke(FN_MEMCPY, dst, src, bytes, HIP_MEMCPY_H2D); }
        catch (Throwable t) { throw new RuntimeException("hipMemcpy H2D failed", t); }
    }

    /**
     * Synchronous device→host copy.
     */
    public int memcpyD2H(MemorySegment dst, MemorySegment src, long bytes) {
        if (!nativeAvailable) {
            MemorySegment.copy(src, 0L, dst, 0L, bytes);
            return 0;
        }
        try { return (int) invoke(FN_MEMCPY, dst, src, bytes, HIP_MEMCPY_D2H); }
        catch (Throwable t) { throw new RuntimeException("hipMemcpy D2H failed", t); }
    }

    /**
     * Asynchronous host→device copy on a HIP stream.
     */
    public int memcpyAsyncH2D(MemorySegment dst, MemorySegment src,
                               long bytes, MemorySegment stream) {
        if (!nativeAvailable) {
            MemorySegment.copy(src, 0L, dst, 0L, bytes);
            return 0;
        }
        try { return (int) invoke(FN_MEMCPY_ASYNC, dst, src, bytes, HIP_MEMCPY_H2D, stream); }
        catch (Throwable t) { throw new RuntimeException("hipMemcpyAsync H2D failed", t); }
    }

    public int memset(MemorySegment ptr, int value, long bytes) {
        if (!nativeAvailable) { ptr.fill((byte) value); return 0; }
        try { return (int) invoke(FN_MEMSET, ptr, value, bytes); }
        catch (Throwable t) { return -1; }
    }

    // ── Stream management ─────────────────────────────────────────────────────

    public MemorySegment streamCreate() {
        if (!nativeAvailable) return MemorySegment.NULL;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment streamPtr = a.allocate(ValueLayout.ADDRESS);
            int err = (int) invoke(FN_STREAM_CREATE, streamPtr);
            return err == 0 ? streamPtr.get(ValueLayout.ADDRESS, 0) : MemorySegment.NULL;
        } catch (Throwable t) { return MemorySegment.NULL; }
    }

    public int streamSynchronize(MemorySegment stream) {
        if (!nativeAvailable || isNull(stream)) return 0;
        try { return (int) invoke(FN_STREAM_SYNC, stream); }
        catch (Throwable t) { return -1; }
    }

    public int streamDestroy(MemorySegment stream) {
        if (!nativeAvailable || isNull(stream)) return 0;
        try { return (int) invoke(FN_STREAM_DESTROY, stream); }
        catch (Throwable t) { return -1; }
    }

    // ── Module / kernel loading ───────────────────────────────────────────────

    /**
     * Load a pre-compiled HIP kernel module ({@code .hsaco} file).
     *
     * <p>The {@code .hsaco} file is compiled from CUDA source using
     * {@code hipcc -c --amdgpu-target=gfx942 kernel.cu -o kernel.hsaco}
     * (gfx942 = MI300X).
     *
     * @param hsacoPath path to the {@code .hsaco} file
     * @return hipModule_t opaque handle
     */
    public MemorySegment moduleLoad(String hsacoPath) {
        if (!nativeAvailable) return MemorySegment.NULL;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment modPtr  = a.allocate(ValueLayout.ADDRESS);
            MemorySegment pathSeg = a.allocateFrom(hsacoPath);
            int err = (int) invoke(FN_MODULE_LOAD, modPtr, pathSeg);
            if (err != 0) {
                LOG.warnf("hipModuleLoad(%s) failed: %d", hsacoPath, err);
                return MemorySegment.NULL;
            }
            return modPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            LOG.warnf("hipModuleLoad failed: %s", t.getMessage());
            return MemorySegment.NULL;
        }
    }

    /**
     * Get a function handle from a loaded module.
     *
     * @param module   hipModule_t from {@link #moduleLoad}
     * @param fnName   kernel function name (must match {@code __global__} function)
     * @return hipFunction_t opaque handle
     */
    public MemorySegment moduleGetFunction(MemorySegment module, String fnName) {
        if (!nativeAvailable || isNull(module)) return MemorySegment.NULL;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment fnPtr  = a.allocate(ValueLayout.ADDRESS);
            MemorySegment nameSeg = a.allocateFrom(fnName);
            int err = (int) invoke(FN_MODULE_GET_FN, fnPtr, module, nameSeg);
            if (err != 0) {
                LOG.warnf("hipModuleGetFunction(%s) failed: %d", fnName, err);
                return MemorySegment.NULL;
            }
            return fnPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) { return MemorySegment.NULL; }
    }

    /**
     * Launch a HIP kernel.
     *
     * @param function    hipFunction_t from {@link #moduleGetFunction}
     * @param gridX/Y/Z   grid dimensions
     * @param blockX/Y/Z  block dimensions
     * @param sharedMem   shared memory bytes per block
     * @param stream      hipStream_t (NULL for default)
     * @param kernelArgs  kernel argument pointer array — each element is a void*
     *                    pointing to the argument value
     * @return hipError_t
     */
    public int moduleLaunchKernel(MemorySegment function,
                                   int gridX,  int gridY,  int gridZ,
                                   int blockX, int blockY, int blockZ,
                                   int sharedMem,
                                   MemorySegment stream,
                                   MemorySegment kernelArgs) {
        if (!nativeAvailable) return 0;
        try {
            return (int) invoke(FN_MODULE_LAUNCH,
                    function,
                    gridX, gridY, gridZ,
                    blockX, blockY, blockZ,
                    sharedMem,
                    isNull(stream) ? MemorySegment.NULL : stream,
                    kernelArgs,
                    MemorySegment.NULL   // extra (unused)
            );
        } catch (Throwable t) {
            throw new RuntimeException("hipModuleLaunchKernel failed", t);
        }
    }

    /** Translate a hipError_t code to a human-readable string. */
    public String getErrorString(int errorCode) {
        if (!nativeAvailable) return "hipError_t(" + errorCode + ")";
        try {
            MemorySegment msgPtr = (MemorySegment) invoke(FN_GET_ERROR_STRING, errorCode);
            return isNull(msgPtr) ? "hipError_t(" + errorCode + ")"
                    : msgPtr.reinterpret(256).getString(0);
        } catch (Throwable t) { return "hipError_t(" + errorCode + ")"; }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // hipError_t hipGetDeviceCount(int*)
        bind(FN_GET_DEVICE_COUNT, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        // hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int)
        bind(FN_GET_DEVICE_PROPS, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        // hipError_t hipSetDevice(int)
        bind(FN_SET_DEVICE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        // hipError_t hipDeviceSynchronize()
        bind(FN_DEVICE_SYNC, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // hipError_t hipMalloc(void**, size_t)
        bind(FN_MALLOC, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        // hipError_t hipMallocManaged(void**, size_t, unsigned int)
        bind(FN_MALLOC_MANAGED, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // ptr*
                ValueLayout.JAVA_LONG, // size
                ValueLayout.JAVA_INT   // flags
        ));

        // hipError_t hipFree(void*)
        bind(FN_FREE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        // hipError_t hipMemcpy(void*, void*, size_t, hipMemcpyKind)
        bind(FN_MEMCPY, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // dst
                ValueLayout.ADDRESS,   // src
                ValueLayout.JAVA_LONG, // bytes
                ValueLayout.JAVA_INT   // kind
        ));

        // hipError_t hipMemcpyAsync(void*, void*, size_t, hipMemcpyKind, hipStream_t)
        bind(FN_MEMCPY_ASYNC, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // dst
                ValueLayout.ADDRESS,   // src
                ValueLayout.JAVA_LONG, // bytes
                ValueLayout.JAVA_INT,  // kind
                ValueLayout.ADDRESS    // stream
        ));

        // hipError_t hipMemset(void*, int, size_t)
        bind(FN_MEMSET, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG
        ));

        // hipError_t hipStreamCreate(hipStream_t*)
        bind(FN_STREAM_CREATE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        // hipError_t hipStreamSynchronize(hipStream_t)
        bind(FN_STREAM_SYNC, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        // hipError_t hipStreamDestroy(hipStream_t)
        bind(FN_STREAM_DESTROY, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        // hipError_t hipModuleLoad(hipModule_t*, const char*)
        bind(FN_MODULE_LOAD, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        // hipError_t hipModuleGetFunction(hipFunction_t*, hipModule_t, const char*)
        bind(FN_MODULE_GET_FN, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, // fn_out*
                ValueLayout.ADDRESS, // module
                ValueLayout.ADDRESS  // name
        ));

        // hipError_t hipModuleLaunchKernel(hipFunction_t,
        //   uint, uint, uint, uint, uint, uint, uint, hipStream_t, void**, void**)
        bind(FN_MODULE_LAUNCH, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // function
                ValueLayout.JAVA_INT,  // gridX
                ValueLayout.JAVA_INT,  // gridY
                ValueLayout.JAVA_INT,  // gridZ
                ValueLayout.JAVA_INT,  // blockX
                ValueLayout.JAVA_INT,  // blockY
                ValueLayout.JAVA_INT,  // blockZ
                ValueLayout.JAVA_INT,  // sharedMemBytes
                ValueLayout.ADDRESS,   // stream
                ValueLayout.ADDRESS,   // kernelParams
                ValueLayout.ADDRESS    // extra
        ));

        // const char* hipGetErrorString(hipError_t) -> pointer
        bind(FN_GET_ERROR_STRING, FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            methodHandles.put(name,
                    Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
            LOG.debugf("RocmHipBinding: bound %s", name);
        } else {
            LOG.warnf("RocmHipBinding: symbol not found — %s", name);
        }
    }

    private Object invoke(String fn, Object... args) {
        MethodHandle mh = methodHandles.get(fn);
        if (mh == null) throw new IllegalStateException("HIP symbol not bound: " + fn);
        try { return mh.invokeWithArguments(args); }
        catch (Throwable t) { throw new RuntimeException("HIP " + fn + " failed", t); }
    }

    private static boolean isNull(MemorySegment s) {
        return s == null || s.equals(MemorySegment.NULL) || s.address() == 0;
    }

    public static void reset() { instance = null; }
}
