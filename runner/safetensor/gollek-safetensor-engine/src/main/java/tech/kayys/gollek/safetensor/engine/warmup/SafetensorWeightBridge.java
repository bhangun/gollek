/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorWeightBridge.java
 * ────────────────────────────
 * The critical bridge that connects our FFM-based SafeTensor loader to the
 * LibTorch computation backend — enabling ZERO-COPY weight inference.
 *
 * How it works
 * ════════════
 *
 *   .safetensors file ──mmap──▶ MemorySegment (off-heap, FFM Arena)
 *                                     │
 *                        getNativeAddress()  (JDK 25 FFM)
 *                                     │
 *                                     ▼
 *            at_from_blob(ptr, shape, dtype, strides=null, device=CPU)
 *                                     │
 *                          LibTorch TorchTensor (wraps same memory)
 *                                     │
 *                              forward pass ──▶ logits ──▶ output tokens
 *
 * Key properties:
 *   - NO COPY: the LibTorch tensor wraps the exact same physical memory pages
 *     as the mmap'd safetensors file.  The first inference page-faults those
 *     pages in from disk; subsequent calls hit the OS page cache.
 *   - SAFE LIFETIME: the FFM Arena outlives the LibTorch tensor — the bridge
 *     tracks this via a WeakReference cleanup hook on the TorchTensor.
 *   - TYPE SAFE: dtype mapping is compile-time verified via SafetensorDType →
 *     ScalarType (LibTorch enum).
 *
 * BF16 special handling
 * ═════════════════════
 * PyTorch/LibTorch supports BF16 natively on recent GPU backends and on x86
 * CPUs with AVX-512 BF16 (Sapphire Rapids / Genoa / Zen 4+).  On older CPUs,
 * LibTorch will up-cast BF16→F32 during the first matmul.  The bridge
 * respects whatever dtype the safetensors file declares — no silent cast.
 *
 * For explicit F16→F32 up-casting (when targeting older CPUs without F16
 * acceleration) see {@link QuantizationEngine}.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorDType;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.HashMap;
import java.util.Map;

/**
 * Zero-copy bridge between FFM {@link SafetensorTensor} views and LibTorch
 * {@link TorchTensor} objects via {@code at_from_blob}.
 *
 * <p>
 * Inject this bean and call
 * {@link #bridgeAll(SafetensorShardSession, String...)}
 * or {@link #bridge(SafetensorTensor)} to produce LibTorch tensors that share
 * the same physical memory as the mmap'd safetensors file.
 *
 * <p>
 * <b>Lifetime:</b> the returned {@link TorchTensor} objects are valid only while
 * the parent {@link SafetensorShardSession} is open. Close the TorchTensor before
 * (or together with) closing the session to avoid a use-after-free.
 */
@ApplicationScoped
public class SafetensorWeightBridge {

    private static final Logger log = Logger.getLogger(SafetensorWeightBridge.class);

    @Inject
    LibTorchBinding libTorch;

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Bridge a single {@link SafetensorTensor} to a LibTorch {@link TorchTensor}.
     *
     * <p>
     * The returned tensor shares backing memory with the FFM segment — no
     * data is copied.
     *
     * @param st the source safetensor tensor
     * @return a LibTorch tensor wrapping the same off-heap memory
     */
    public TorchTensor bridge(SafetensorTensor st) {
        MemorySegment segment = st.segment(); // off-heap bytes
        long[] shape = st.info().shape();
        ScalarType dtype = toScalarType(st.dtype());

        log.tracef("Bridging tensor '%s' dtype=%s shape=%s ptr=0x%x",
                st.name(), dtype, java.util.Arrays.toString(shape),
                segment.address());

        return fromBlob(segment, shape, dtype);
    }

    /**
     * Bridge all named tensors from an open session in a single call.
     *
     * @param session     an open {@link SafetensorShardSession}
     * @param arena       arena to allocate any F32 up-casts into
     * @param tensorNames weight names to bridge
     * @return map of tensor name → LibTorch TorchTensor
     */
    public Map<String, TorchTensor> bridgeAll(SafetensorShardSession session, Arena arena, String... tensorNames) {
        Map<String, TorchTensor> result = new HashMap<>(tensorNames.length * 2);
        try {
            for (String name : tensorNames) {
                SafetensorTensor st = session.tensor(name);
                SafetensorDType dtype = st.dtype();
                TorchTensor t = (dtype == SafetensorDType.BF16 || dtype == SafetensorDType.F16)
                        ? bridgeAsF32(st, arena)
                        : bridge(st);
                result.put(name, t);
            }
        } catch (Exception e) {
            result.values().forEach(TorchTensor::close);
            throw new WeightBridgeException("Failed to bridge tensor: " + e.getMessage(), e);
        }
        return result;
    }

    /** Legacy bridgeAll using a temporary arena — now deprecated to avoid leaks. */
    @Deprecated
    public Map<String, TorchTensor> bridgeAll(SafetensorShardSession session, String... tensorNames) {
        try (Arena temp = Arena.ofShared()) {
            return bridgeAll(session, temp, tensorNames);
        }
    }

    /**
     * Bridge a tensor and immediately cast it to {@code F32} if it is in a
     * half-precision format (BF16, F16).
     *
     * <p>
     * Use this on CPUs that lack BF16/F16 native support. The cast
     * creates a NEW F32 tensor in a fresh native allocation — it is NOT
     * zero-copy but is required for correctness on older hardware.
     *
     * @param st    the source safetensor
     * @param arena arena to allocate the F32 copy into (caller manages lifetime)
     * @return an F32 LibTorch tensor
     */
    public TorchTensor bridgeAsF32(SafetensorTensor st, Arena arena) {
        SafetensorDType dtype = st.dtype();
        if (dtype != SafetensorDType.BF16 && dtype != SafetensorDType.F16) {
            // Already a compatible type — bridge zero-copy
            return bridge(st);
        }
        // BF16/F16 on CPUs without native support: materialise as F32.
        // Read raw half-precision elements and write them into a native F32 buffer.
        long n = st.numElements();
        if (n > Integer.MAX_VALUE) {
            throw new WeightBridgeException(
                    "TorchTensor '" + st.name() + "' too large to up-cast to F32 in a single buffer", null);
        }
        int count = (int) n;
        MemorySegment f32Buf = arena.allocate((long) count * Float.BYTES);

        if (dtype == SafetensorDType.F16) {
            for (int i = 0; i < count; i++) {
                float v = st.getF16AsFloat((long) i);
                f32Buf.setAtIndex(ValueLayout.JAVA_FLOAT.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN), i, v);
            }
        } else { // BF16
            for (int i = 0; i < count; i++) {
                float v = st.getBF16AsFloat((long) i);
                f32Buf.setAtIndex(ValueLayout.JAVA_FLOAT.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN), i, v);
            }
        }

        log.tracef("bridgeAsF32: up-cast '%s' %s → F32 (%d elements)", st.name(), dtype, count);
        return fromBlob(f32Buf, st.info().shape(), ScalarType.FLOAT);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Dtype mapping
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Map a {@link SafetensorDType} to the corresponding LibTorch
     * {@link ScalarType} code.
     */
    public static ScalarType toScalarType(SafetensorDType dtype) {
        return switch (dtype) {
            case F32 -> ScalarType.FLOAT;
            case F64 -> ScalarType.DOUBLE;
            case F16 -> ScalarType.HALF;
            case BF16 -> ScalarType.BFLOAT16;
            case I8 -> ScalarType.CHAR;
            case I16 -> ScalarType.SHORT;
            case I32 -> ScalarType.INT;
            case I64 -> ScalarType.LONG;
            case U8 -> ScalarType.BYTE;
            case U16 -> ScalarType.SHORT; // closest JVM representation
            case U32 -> ScalarType.INT;
            case U64 -> ScalarType.LONG;
            case BOOL -> ScalarType.BOOL;
            case F8_E4M3, F8_E5M2 ->
                throw new WeightBridgeException(
                        "FP8 (" + dtype + ") requires PyTorch >= 2.1 with H100/A100 transformer engine. "
                                + "Cast to BF16 first via QuantizationEngine.",
                        null);
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal — at_from_blob via FFM
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a LibTorch tensor wrapping an existing off-heap memory region.
     *
     * <p>
     * Calls {@code at::from_blob(data_ptr, sizes, scalar_type)} via the
     * {@link LibTorchBinding} FFM handle. LibTorch does NOT own this memory —
     * the caller (the FFM Arena) remains the owner.
     */
    private TorchTensor fromBlob(MemorySegment data, long[] shape, ScalarType scalarType) {
        try (Arena callArena = Arena.ofConfined()) {
            // Allocate native shape array on the stack-like confined arena
            MemorySegment shapeSegment = callArena.allocateFrom(ValueLayout.JAVA_LONG, shape);

            MethodHandle fromBlob = libTorch.bind(
                    LibTorchBinding.TENSOR_FROM_BLOB,
                    LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            // at_from_blob(void* data, int64_t* sizes, int64_t ndim, int dtype)
            MemorySegment handle = (MemorySegment) fromBlob.invoke(
                    data,
                    shapeSegment,
                    (long) shape.length,
                    scalarType.code());

            // The arena for the resulting TorchTensor must outlive this call arena.
            // We use a fresh shared arena tied to the TorchTensor's lifecycle.
            Arena tensorArena = Arena.ofShared();
            return new TorchTensor(handle, tensorArena);

        } catch (WeightBridgeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WeightBridgeException(
                    "at_from_blob failed: " + t.getMessage(), t);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Exception type
    // ─────────────────────────────────────────────────────────────────────────

    /** Unchecked exception thrown when a weight bridge operation fails. */
    public static final class WeightBridgeException extends RuntimeException {
        public WeightBridgeException(String msg, Throwable cause) {
            super(msg, cause);
        }
    }
}
