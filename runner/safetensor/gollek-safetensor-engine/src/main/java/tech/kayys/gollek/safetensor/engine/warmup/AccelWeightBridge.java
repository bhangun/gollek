/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AccelWeightBridge.java
 * ──────────────────────
 * Zero-copy bridge between FFM-mmap'd safetensor files and AccelTensor.
 * Replaces SafetensorWeightBridge — no LibTorch dependency.
 *
 * BF16/F16 weights are upcast to F32 at load time using pure Java.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorDType;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Zero-copy bridge from mmap'd safetensor files to {@link AccelTensor}.
 * <p>
 * F32 weights are wrapped directly (zero-copy).
 * BF16/F16 weights are upcast to F32 (copy).
 */
@ApplicationScoped
public class AccelWeightBridge {

    private static final Logger log = Logger.getLogger(AccelWeightBridge.class);

    /**
     * Bridge a single SafetensorTensor to an AccelTensor.
     * <p>
     * F32 tensors are zero-copy wrapped.
     * BF16/F16 tensors are upcast to F32.
     */
    public AccelTensor bridge(SafetensorTensor st) {
        SafetensorDType dtype = st.dtype();
        long[] shape = st.info().shape();
        String name = st.name();

        AccelTensor t;
        if (dtype == SafetensorDType.F32) {
            // Zero-copy: wrap the mmap'd segment directly
            log.tracef("Bridge zero-copy: '%s' F32 shape=%s", name, java.util.Arrays.toString(shape));
            t = AccelTensor.wrapSegment(st.segment(), shape);
        } else if (dtype == SafetensorDType.BF16 || dtype == SafetensorDType.F16) {
            log.debugf("Bridge upcast: '%s' %s → F32 shape=%s", name, dtype, java.util.Arrays.toString(shape));
            t = upcastToF32(st, dtype);
        } else {
            // For integer types, convert to float
            log.debugf("Bridge convert: '%s' %s → F32", name, dtype);
            t = convertToF32(st, dtype);
        }

        // We no longer pre-transpose here. AccelOps.linear handles transposition 
        // via CblasTrans during execution, which is more efficient and avoids 
        // dimension mismatch confusion.

        return t;
    }

    /**
     * Bridge all tensors from an open session.
     */
    public Map<String, AccelTensor> bridgeAll(SafetensorShardSession session) {
        Set<String> names = session.tensorNames();
        Map<String, AccelTensor> result = new HashMap<>(names.size() * 2);
        try {
            for (String name : names) {
                SafetensorTensor st = session.tensor(name);
                result.put(name, bridge(st));
            }
        } catch (Exception e) {
            // Clean up on failure
            result.values().forEach(AccelTensor::close);
            throw new RuntimeException("Failed to bridge weights: " + e.getMessage(), e);
        }
        return result;
    }

    // ── Internal ──────────────────────────────────────────────────────

    private AccelTensor upcastToF32(SafetensorTensor st, SafetensorDType dtype) {
        log.debugf("Upcasting '%s' (%s) to F32 using toFloatArray() fallback", st.name(), dtype);
        
        float[] values = st.toFloatArray();
        AccelTensor t = AccelTensor.fromFloatArray(values, st.info().shape());

        // Diagnostic parity check
        if (Math.abs(t.min()) > 50.0 || Math.abs(t.max()) > 50.0) {
            log.warnf("Suspicious weight magnitude in '%s': min=%.4f, max=%.4f.",
                    st.name(), t.min(), t.max());
            log.warnf("First 5 upcast values for '%s': %s", st.name(), getFirstValues(t, 5));
        }
        return t;
    }

    private AccelTensor convertToF32(SafetensorTensor st, SafetensorDType dtype) {
        long n = st.numElements();
        MemorySegment seg = st.segment();
        float[] f32 = new float[(int) n];

        switch (dtype) {
            case I8 -> {
                for (int i = 0; i < n; i++) f32[i] = seg.get(ValueLayout.JAVA_BYTE, i);
            }
            case I16, U16 -> {
                for (int i = 0; i < n; i++) f32[i] = seg.getAtIndex(ValueLayout.JAVA_SHORT, i);
            }
            case I32, U32 -> {
                for (int i = 0; i < n; i++) f32[i] = seg.getAtIndex(ValueLayout.JAVA_INT, i);
            }
            case I64, U64 -> {
                for (int i = 0; i < n; i++) f32[i] = seg.getAtIndex(ValueLayout.JAVA_LONG, i);
            }
            case F64 -> {
                for (int i = 0; i < n; i++) f32[i] = (float) seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            }
            case U8 -> {
                for (int i = 0; i < n; i++) f32[i] = Byte.toUnsignedInt(seg.get(ValueLayout.JAVA_BYTE, i));
            }
            case BOOL -> {
                for (int i = 0; i < n; i++) f32[i] = seg.get(ValueLayout.JAVA_BYTE, i) != 0 ? 1.0f : 0.0f;
            }
            default -> throw new UnsupportedOperationException("Cannot convert " + dtype + " to F32");
        }

        return AccelTensor.fromFloatArray(f32, st.info().shape());
    }

    private String getRawHex(MemorySegment seg, int maxBytes) {
        int n = (int) Math.min(seg.byteSize(), maxBytes);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append(String.format("%02x", seg.get(ValueLayout.JAVA_BYTE, i)));
        }
        return sb.toString();
    }

    private String getFirstValues(AccelTensor t, int count) {
        int n = (int) Math.min(t.numel(), count);
        float[] vals = new float[n];
        for (int i = 0; i < n; i++) {
            vals[i] = t.getFlat(i);
        }
        return java.util.Arrays.toString(vals);
    }
}
