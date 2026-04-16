/**
 * GollekNativeBridge.kt
 * Kotlin wrapper around the Gollek C++ core via JNI.
 *
 * Replaces LiteRTNativeBindings.java + LiteRTCpuRunner.java on Android.
 * The native library (libgollek_core.so) is built by CMake via the
 * Android NDK and packaged in the APK's jniLibs automatically.
 *
 * Usage:
 *   val engine = GollekEngine(GollekConfig(numThreads = 4, delegate = Delegate.AUTO))
 *   engine.loadModelFromAssets(context, "models/mobilenet_v2.litertlm")
 *   val result = engine.infer(inputBuffer, outputBuffer)
 *   engine.close()
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

package tech.kayys.gollek.native

import android.content.Context
import android.content.res.AssetManager
import java.io.Closeable
import java.nio.ByteBuffer

/* ── Data classes ──────────────────────────────────────────────────────── */

enum class GollekDelegate(val value: Int) {
    NONE    (0),
    CPU     (1),
    GPU     (2),
    NNAPI   (3),
    HEXAGON (4),
    COREML  (5),
    AUTO    (6);
}

data class GollekConfig(
    val numThreads:    Int           = 4,
    val delegate:      GollekDelegate = GollekDelegate.AUTO,
    val enableXnnpack: Boolean       = true,
    val useMemoryPool: Boolean       = true,
    val poolSizeBytes: Long          = 0L,   // 0 = use engine default (16 MB)
)

data class TensorInfo(
    val type:     Int,
    val dims:     IntArray,
    val byteSize: Long,
)

enum class GollekStatus(val code: Int) {
    OK                   (0),
    ERROR                (1),
    ERROR_MODEL_LOAD     (2),
    ERROR_ALLOC_TENSORS  (3),
    ERROR_INVOKE         (4),
    ERROR_INVALID_ARG    (5),
    ERROR_NOT_INITIALIZED(6),
    ERROR_DELEGATE_FAILED(7);

    companion object {
        fun from(code: Int) = values().firstOrNull { it.code == code } ?: ERROR
    }
}

class GollekException(status: GollekStatus, detail: String)
    : RuntimeException("GollekEngine error [$status]: $detail")

/* ── Native bridge (internal) ─────────────────────────────────────────── */

internal object GollekNativeBridge {
    init {
        System.loadLibrary("gollek_core")
    }

    @JvmStatic external fun nativeCreate           (config: GollekConfig?): Long
    @JvmStatic external fun nativeDestroy          (handle: Long)
    @JvmStatic external fun nativeLoadModelFromFile  (handle: Long, path: String): Int
    @JvmStatic external fun nativeLoadModelFromBuffer(handle: Long, buffer: ByteBuffer): Int
    @JvmStatic external fun nativeLoadModelFromAsset (handle: Long, assetManager: AssetManager, assetPath: String): Int
    @JvmStatic external fun nativeGetInputCount    (handle: Long): Int
    @JvmStatic external fun nativeGetOutputCount   (handle: Long): Int
    @JvmStatic external fun nativeGetInputInfo     (handle: Long, index: Int): LongArray?
    @JvmStatic external fun nativeSetInput         (handle: Long, index: Int, buffer: ByteBuffer, bytes: Int): Int
    @JvmStatic external fun nativeInvoke           (handle: Long): Int
    @JvmStatic external fun nativeGetOutput        (handle: Long, index: Int, buffer: ByteBuffer, bytes: Int): Int
    @JvmStatic external fun nativeLastError        (handle: Long): String
    @JvmStatic external fun nativeVersion          (): String
}

/* ── Public GollekEngine ──────────────────────────────────────────────── */

/**
 * High-level, resource-safe wrapper around the Gollek native inference engine.
 * Implements [Closeable] so it works with Kotlin's `use { }` block.
 */
class GollekEngine(config: GollekConfig = GollekConfig()) : Closeable {

    private val handle: Long = GollekNativeBridge.nativeCreate(config)
        .also { check(it != 0L) { "Failed to create native Gollek engine" } }

    /* ── Model loading ──────────────────────────────────────────────── */

    fun loadModelFromFile(path: String): GollekStatus {
        return check(GollekNativeBridge.nativeLoadModelFromFile(handle, path))
    }

    fun loadModelFromBuffer(buffer: ByteBuffer): GollekStatus {
        require(buffer.isDirect) { "Buffer must be a direct ByteBuffer" }
        return check(GollekNativeBridge.nativeLoadModelFromBuffer(handle, buffer))
    }

    fun loadModelFromAssets(context: Context, assetPath: String): GollekStatus {
        return check(GollekNativeBridge.nativeLoadModelFromAsset(
            handle, context.assets, assetPath))
    }

    /* ── Introspection ──────────────────────────────────────────────── */

    val inputCount:  Int get() = GollekNativeBridge.nativeGetInputCount(handle)
    val outputCount: Int get() = GollekNativeBridge.nativeGetOutputCount(handle)

    fun getInputInfo(index: Int = 0): TensorInfo? {
        val raw = GollekNativeBridge.nativeGetInputInfo(handle, index) ?: return null
        val type     = raw[0].toInt()
        val numDims  = raw[1].toInt()
        val dims     = IntArray(numDims) { raw[2 + it].toInt() }
        val byteSize = raw[2 + numDims]
        return TensorInfo(type, dims, byteSize)
    }

    /* ── Inference ──────────────────────────────────────────────────── */

    fun setInput(index: Int, buffer: ByteBuffer, bytes: Int = buffer.remaining()): GollekStatus {
        require(buffer.isDirect) { "Buffer must be direct" }
        return check(GollekNativeBridge.nativeSetInput(handle, index, buffer, bytes))
    }

    fun invoke(): GollekStatus =
        check(GollekNativeBridge.nativeInvoke(handle))

    fun getOutput(index: Int, buffer: ByteBuffer, bytes: Int = buffer.remaining()): GollekStatus {
        require(buffer.isDirect) { "Buffer must be direct" }
        return check(GollekNativeBridge.nativeGetOutput(handle, index, buffer, bytes))
    }

    /**
     * Convenience: single-tensor inference in one call.
     * @return GOLLEK_OK or throws [GollekException] if throwOnError is true.
     */
    fun infer(
        input:        ByteBuffer,
        output:       ByteBuffer,
        throwOnError: Boolean = true,
    ): GollekStatus {
        var s = setInput(0, input)
        if (s != GollekStatus.OK) return handleError(s, throwOnError)
        s = invoke()
        if (s != GollekStatus.OK) return handleError(s, throwOnError)
        s = getOutput(0, output)
        if (s != GollekStatus.OK) return handleError(s, throwOnError)
        return GollekStatus.OK
    }

    /* ── Diagnostics ────────────────────────────────────────────────── */

    val lastError: String get() = GollekNativeBridge.nativeLastError(handle)
    val version:   String get() = GollekNativeBridge.nativeVersion()

    /* ── Lifecycle ──────────────────────────────────────────────────── */

    override fun close() {
        GollekNativeBridge.nativeDestroy(handle)
    }

    /* ── Private helpers ────────────────────────────────────────────── */

    private fun check(code: Int): GollekStatus = GollekStatus.from(code)

    private fun handleError(s: GollekStatus, throwOnError: Boolean): GollekStatus {
        if (throwOnError) throw GollekException(s, lastError)
        return s
    }
}
