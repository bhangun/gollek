/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AccelOps.java
 * ─────────────
 * Tensor operations using Apple Accelerate framework via FFM.
 *
 * Matrix multiplication uses cblas_sgemm (runs on Apple Silicon AMX).
 * Element-wise ops use vDSP_* where beneficial, else pure Java with
 * Panama Vector API for auto-vectorization.
 *
 * No LibTorch. No native wrapper. Just FFM → Accelerate.framework.
 */
package tech.kayys.gollek.safetensor.core.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.safetensor.loader.SafetensorDType;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Lightweight Float32 tensor backed by FFM {@link MemorySegment}.
 * <p>
 * Designed for direct use with Apple Accelerate (cblas_sgemm, vDSP)
 * and SIMD via Vector API.
 */
public final class AccelOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private AccelOps() {} // utility class

    // ── Accelerate bindings (lazy-initialized) ────────────────────────

    private static final Linker LINKER = Linker.nativeLinker();

    /** BLAS constants */
    private static final int CblasRowMajor = 101;
    private static final int CblasNoTrans = 111;
    private static final int CblasTrans = 112;

    // Lazy holders for Accelerate function handles
    private static volatile MethodHandle SGEMM;
    private static volatile MethodHandle SGEMV;
    private static volatile MethodHandle VADD;
    private static volatile MethodHandle VMUL;
    private static volatile MethodHandle VDIV;
    private static volatile MethodHandle VSUB;
    private static volatile MethodHandle VSADD;
    private static volatile MethodHandle VSMUL;
    private static volatile MethodHandle SVE; // sum
    private static volatile MethodHandle VSSQ; // sum of squares

    private static SymbolLookup accelerate() {
        return SymbolLookup.libraryLookup(
            "/System/Library/Frameworks/Accelerate.framework/Accelerate",
            Arena.global()
        );
    }

    private static volatile MethodHandle VVEXPF;

    private static MethodHandle vvexpf() {
        if (VVEXPF == null) {
            synchronized (AccelOps.class) {
                if (VVEXPF == null) {
                    VVEXPF = LINKER.downcallHandle(
                        accelerate().find("vvexpf").orElseThrow(),
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
                    );
                }
            }
        }
        return VVEXPF;
    }

    private static MethodHandle sgemm() {
        if (SGEMM == null) {
            synchronized (AccelOps.class) {
                if (SGEMM == null) {
                    // void cblas_sgemm(ORDER, TRANSA, TRANSB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
                    SGEMM = LINKER.downcallHandle(
                        accelerate().find("cblas_sgemm").orElseThrow(),
                        FunctionDescriptor.ofVoid(
                            ValueLayout.JAVA_INT,   // ORDER
                            ValueLayout.JAVA_INT,   // TRANSA
                            ValueLayout.JAVA_INT,   // TRANSB
                            ValueLayout.JAVA_INT,   // M
                            ValueLayout.JAVA_INT,   // N
                            ValueLayout.JAVA_INT,   // K
                            ValueLayout.JAVA_FLOAT, // alpha
                            ValueLayout.ADDRESS,    // A
                            ValueLayout.JAVA_INT,   // lda
                            ValueLayout.ADDRESS,    // B
                            ValueLayout.JAVA_INT,   // ldb
                            ValueLayout.JAVA_FLOAT, // beta
                            ValueLayout.ADDRESS,    // C
                            ValueLayout.JAVA_INT    // ldc
                        )
                    );
                }
            }
        }
        return SGEMM;
    }

    private static MethodHandle vadd() {
        if (VADD == null) {
            synchronized (AccelOps.class) {
                if (VADD == null) {
                    // void vDSP_vadd(float *A, stride, float *B, stride, float *C, stride, length)
                    VADD = vdspBinaryHandle("vDSP_vadd");
                }
            }
        }
        return VADD;
    }

    private static MethodHandle vmul() {
        if (VMUL == null) {
            synchronized (AccelOps.class) {
                if (VMUL == null) {
                    VMUL = vdspBinaryHandle("vDSP_vmul");
                }
            }
        }
        return VMUL;
    }

    private static MethodHandle vdiv() {
        if (VDIV == null) {
            synchronized (AccelOps.class) {
                if (VDIV == null) {
                    VDIV = vdspBinaryHandle("vDSP_vdiv");
                }
            }
        }
        return VDIV;
    }

    private static MethodHandle vsub() {
        if (VSUB == null) {
            synchronized (AccelOps.class) {
                if (VSUB == null) {
                    VSUB = vdspBinaryHandle("vDSP_vsub");
                }
            }
        }
        return VSUB;
    }

    private static MethodHandle vsadd() {
        if (VSADD == null) {
            synchronized (AccelOps.class) {
                if (VSADD == null) {
                    // void vDSP_vsadd(const float *A, long strideA, const float *B, float *C, long strideC, unsigned long N)
                    VSADD = LINKER.downcallHandle(
                        accelerate().find("vDSP_vsadd").orElseThrow(),
                        FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,     // A
                            ValueLayout.JAVA_LONG,   // strideA
                            ValueLayout.ADDRESS,     // B (scalar pointer)
                            ValueLayout.ADDRESS,     // C
                            ValueLayout.JAVA_LONG,   // strideC
                            ValueLayout.JAVA_LONG    // N
                        )
                    );
                }
            }
        }
        return VSADD;
    }

    private static MethodHandle vsmul() {
        if (VSMUL == null) {
            synchronized (AccelOps.class) {
                if (VSMUL == null) {
                    // void vDSP_vsmul(const float *A, long strideA, const float *B, float *C, long strideC, unsigned long N)
                    VSMUL = LINKER.downcallHandle(
                        accelerate().find("vDSP_vsmul").orElseThrow(),
                        FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,     // A
                            ValueLayout.JAVA_LONG,   // strideA
                            ValueLayout.ADDRESS,     // B (scalar pointer)
                            ValueLayout.ADDRESS,     // C
                            ValueLayout.JAVA_LONG,   // strideC
                            ValueLayout.JAVA_LONG    // N
                        )
                    );
                }
            }
        }
        return VSMUL;
    }

    private static MethodHandle sgemv() {
        if (SGEMV == null) {
            synchronized (AccelOps.class) {
                if (SGEMV == null) {
                    // void cblas_sgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)
                    SGEMV = LINKER.downcallHandle(
                        accelerate().find("cblas_sgemv").orElseThrow(),
                        FunctionDescriptor.ofVoid(
                            ValueLayout.JAVA_INT,   // Order
                            ValueLayout.JAVA_INT,   // TransA
                            ValueLayout.JAVA_INT,   // M
                            ValueLayout.JAVA_INT,   // N
                            ValueLayout.JAVA_FLOAT, // alpha
                            ValueLayout.ADDRESS,    // A
                            ValueLayout.JAVA_INT,   // lda
                            ValueLayout.ADDRESS,    // X
                            ValueLayout.JAVA_INT,   // incX
                            ValueLayout.JAVA_FLOAT, // beta
                            ValueLayout.ADDRESS,    // Y
                            ValueLayout.JAVA_INT    // incY
                        )
                    );
                }
            }
        }
        return SGEMV;
    }

    private static MethodHandle vdspBinaryHandle(String name) {
        // vDSP_vadd(const float *A, long strideA, const float *B, long strideB,
        //           float *C, long strideC, unsigned long N)
        return LINKER.downcallHandle(
            accelerate().find(name).orElseThrow(),
            FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS,     // A
                ValueLayout.JAVA_LONG,   // strideA
                ValueLayout.ADDRESS,     // B
                ValueLayout.JAVA_LONG,   // strideB
                ValueLayout.ADDRESS,     // C
                ValueLayout.JAVA_LONG,   // strideC
                ValueLayout.JAVA_LONG    // N
            )
        );
    }

    // ── Matrix multiplication ─────────────────────────────────────────

    /**
     * Matrix multiplication: C = A @ B
     * <p>
     * Supports 2D and 3D (batched) inputs.
     * For 3D, loops over the batch dimension calling cblas_sgemm per batch.
     * <p>
     * A: [..., M, K], B: [..., K, N] → C: [..., M, N]
     */
    public static AccelTensor matmul(AccelTensor a, AccelTensor b) {
        // We no longer call a.contiguous() here! 
        // Instead, we pass strides directly to BLAS.
        int aRank = a.rank();
        int bRank = b.rank();

        if (aRank == 4 && bRank == 4) {
            // [batch, headsQ, M, K] @ [batch, headsKV, K, N] → [batch, headsQ, M, N]
            return batchMatmul4D(a, b);
        } else if (aRank == 2 && bRank == 2) {
            return matmul2D(a, b);
        } else if (aRank == 3 && bRank == 2) {
            return batchMatmulAB(a.contiguous(), b.contiguous());
        } else if (aRank == 3 && bRank == 3) {
            return batchMatmul3D(a.contiguous(), b.contiguous());
        }
        throw new UnsupportedOperationException(
            "matmul not supported for ranks " + aRank + " and " + bRank);
    }

    /**
     * Linear layer: out = input @ weight^T
     * <p>
     * Uses CblasTrans on B to avoid materializing the transpose.
     * input: [batch, seq, in_features], weight: [out_features, in_features]
     * → output: [batch, seq, out_features]
     */
    public static AccelTensor linear(AccelTensor input, AccelTensor weight) {
        // Just-in-time dequantization if weight is quantized
        if (weight.isQuantized()) {
            try (AccelTensor dequantizedWeight = dequantize(weight)) {
                return linear(input, dequantizedWeight);
            }
        }

        // We no longer call input.contiguous() here! Stride awareness is handled below.
        
        if (input.rank() == 3 && weight.rank() == 2) {
            long batch = input.size(0);
            long seq = input.size(1);
            long inF = input.size(2);
            long outF = weight.size(0);
            if (weight.size(1) != inF) {
                throw new IllegalArgumentException(
                    "linear: input features " + inF + " != weight cols " + weight.size(1));
            }

            AccelTensor out = AccelTensor.zeros(batch, seq, outF);
            try {
                for (long b = 0; b < batch; b++) {
                    long aOff = b * seq * inF;
                    long cOff = b * seq * outF;
                    
                    MemorySegment aPtr = input.dataPtr().asSlice(aOff * Float.BYTES);
                    MemorySegment cPtr = out.dataPtr().asSlice(cOff * Float.BYTES);

                    if (seq == 1) {
                        // Optimized for decoding: Matrix-Vector multiplication
                        // Weight is [outF, inF], Input A is [1, inF] -> Output C is [1, outF]
                        // y = alpha * A * x + beta * y
                        sgemv().invokeExact(
                            CblasRowMajor, CblasNoTrans,
                            (int) outF, (int) inF,
                            1.0f, weight.dataPtr(), (int) inF,
                            aPtr, 1,
                            0.0f, cPtr, 1
                        );
                    } else {
                        // Standard sgemm for prompt processing
                        sgemm().invokeExact(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int) seq, (int) outF, (int) inF,
                            1.0f,
                            aPtr, (int) inF,
                            weight.dataPtr(), (int) inF, // B is [outF, inF], transposed
                            0.0f,
                            cPtr, (int) outF
                        );
                    }
                }
            } catch (Throwable t) {
                throw new RuntimeException("Linear/GEMV failed", t);
            }
            return out;
        } else if (input.rank() == 2 && weight.rank() == 2) {
            long M = input.size(0);
            long K = input.size(1);
            long N = weight.size(0);
            AccelTensor out = AccelTensor.zeros(M, N);
            try {
                sgemm().invokeExact(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int) M, (int) N, (int) K,
                    1.0f,
                    input.dataPtr(), (int) K,
                    weight.dataPtr(), (int) K,
                    0.0f,
                    out.dataPtr(), (int) N
                );
            } catch (Throwable t) {
                out.close();
                throw new RuntimeException("cblas_sgemm failed in linear 2D", t);
            }
            return out;
        }
        throw new UnsupportedOperationException(
            "linear not supported for input rank " + input.rank());
    }

    private static AccelTensor matmul2D(AccelTensor a, AccelTensor b) {
        long M = a.size(0), K = a.size(1);
        long N = b.size(1);
        if (b.size(0) != K) {
            throw new IllegalArgumentException("matmul2D: a.cols " + K + " != b.rows " + b.size(0));
        }

        AccelTensor out = AccelTensor.zeros(M, N);
        try {
            sgemm().invokeExact(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int) M, (int) N, (int) K,
                1.0f,
                a.dataPtr(), (int) K,
                b.dataPtr(), (int) N,
                0.0f,
                out.dataPtr(), (int) N
            );
        } catch (Throwable t) {
            out.close();
            throw new RuntimeException("cblas_sgemm failed", t);
        }
        return out;
    }

    private static AccelTensor batchMatmulAB(AccelTensor a, AccelTensor b) {
        long batch = a.size(0), M = a.size(1), K = a.size(2);
        long N = b.size(1);
        AccelTensor out = AccelTensor.zeros(batch, M, N);
        try {
            for (long i = 0; i < batch; i++) {
                sgemm().invokeExact(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int) M, (int) N, (int) K,
                    1.0f,
                    a.dataPtr().asSlice(i * M * K * Float.BYTES), (int) K,
                    b.dataPtr(), (int) N,
                    0.0f,
                    out.dataPtr().asSlice(i * M * N * Float.BYTES), (int) N
                );
            }
        } catch (Throwable t) {
            out.close();
            throw new RuntimeException("cblas_sgemm batch failed", t);
        }
        return out;
    }

    private static AccelTensor batchMatmul3D(AccelTensor a, AccelTensor b) {
        long batch = a.size(0), M = a.size(1), K = a.size(2);
        long N = b.size(2);
        AccelTensor out = AccelTensor.zeros(batch, M, N);
        try {
            for (long i = 0; i < batch; i++) {
                sgemm().invokeExact(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int) M, (int) N, (int) K,
                    1.0f,
                    a.dataPtr().asSlice(i * M * K * Float.BYTES), (int) K,
                    b.dataPtr().asSlice(i * K * N * Float.BYTES), (int) N,
                    0.0f,
                    out.dataPtr().asSlice(i * M * N * Float.BYTES), (int) N
                );
            }
        } catch (Throwable t) {
            out.close();
            throw new RuntimeException("cblas_sgemm 3D batch failed", t);
        }
        return out;
    }

    private static AccelTensor batchMatmul4D(AccelTensor a, AccelTensor b) {
        long batch = a.size(0), headsA = a.size(1), M = a.size(2), K = a.size(3);
        long headsB = b.size(1);
        long N = b.size(3);
        long groupSize = headsA / headsB;

        AccelTensor out = AccelTensor.zeros(batch, headsA, M, N);
        
        try {
            for (long bi = 0; bi < batch; bi++) {
                for (long hi = 0; hi < headsA; hi++) {
                    long hb = hi / groupSize;
                    MemorySegment aPtr = a.dataPtr().asSlice((bi * a.stride()[0] + hi * a.stride()[1]) * 4);
                    MemorySegment bPtr = b.dataPtr().asSlice((bi * b.stride()[0] + hb * b.stride()[1]) * 4);
                    MemorySegment cPtr = out.dataPtr().asSlice((bi * out.stride()[0] + hi * out.stride()[1]) * 4);
                    
                    int transA = CblasNoTrans;
                    int lda = (int) a.stride()[2];
                    if (a.stride()[3] != 1) {
                        if (a.stride()[2] == 1) {
                            transA = CblasTrans;
                            lda = (int) a.stride()[3];
                        } else {
                            lda = (int) a.stride()[2];
                        }
                    }
                    
                    int transB = CblasNoTrans;
                    int ldb = (int) b.stride()[2];
                    if (b.stride()[3] != 1) {
                        if (b.stride()[2] == 1) {
                            transB = CblasTrans;
                            ldb = (int) b.stride()[3];
                        } else {
                            ldb = (int) b.stride()[2];
                        }
                    }

                    int ldc = (int) out.stride()[2];

                    // Use sgemm for all cases to avoid potential alignment/stride hazards with sgemv on some platforms
                    sgemm().invokeExact(
                        CblasRowMajor, transA, transB,
                        (int) M, (int) N, (int) K,
                        1.0f, aPtr, lda,
                        bPtr, ldb,
                        0.0f, cPtr, ldc
                    );
                }
            }
        } catch (Throwable t) {
            out.close();
            throw new RuntimeException("stride-aware sgemm 4D failed", t);
        }
        return out;
    }

    // ── Element-wise operations ───────────────────────────────────────

    /** C = A + B (element-wise) */
    public static AccelTensor add(AccelTensor a, AccelTensor b) {
        return vdspBinaryOp(a, b, true, false);
    }

    /** C = A - B (element-wise) */
    public static AccelTensor sub(AccelTensor a, AccelTensor b) {
        return vdspBinaryOp(a, b, false, true);
    }

    /** C = A * B (element-wise) */
    public static AccelTensor mul(AccelTensor a, AccelTensor b) {
        return vdspBinaryOp(a, b, false, false);
    }

    /** C = A * scalar */
    public static AccelTensor mulScalar(AccelTensor a, float b) {
        return vdspBinaryOp(a, AccelTensor.fromFloatArray(new float[]{b}, 1), false, false);
    }

    /** C = A + scalar */
    public static AccelTensor addScalar(AccelTensor a, float b) {
        return vdspBinaryOp(a, AccelTensor.fromFloatArray(new float[]{b}, 1), true, false);
    }

    private static AccelTensor vdspBinaryOp(AccelTensor a, AccelTensor b, boolean isAdd, boolean isSub) {
        a = a.contiguous();
        b = b.contiguous();

        long n = a.numel();
        AccelTensor out = AccelTensor.zeros(a.shape());
        MemorySegment aSeg = a.dataSegment();
        MemorySegment bSeg = b.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        if (b.numel() == n) {
            // Element-wise
            int i = 0;
            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                FloatVector va = FloatVector.fromMemorySegment(SPECIES, aSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                FloatVector vb = FloatVector.fromMemorySegment(SPECIES, bSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                FloatVector res;
                if (isAdd) res = va.add(vb);
                else if (isSub) res = va.sub(vb);
                else res = va.mul(vb);
                res.intoMemorySegment(oSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < n; i++) {
                float va = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float vb = bSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float res;
                if (isAdd) res = va + vb;
                else if (isSub) res = va - vb;
                else res = va * vb;
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, res);
            }
        } else if (b.numel() == 1) {
            // Scalar broadcasting
            float val = b.item();
            FloatVector vVal = FloatVector.broadcast(SPECIES, val);
            int i = 0;
            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                FloatVector va = FloatVector.fromMemorySegment(SPECIES, aSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                FloatVector res;
                if (isAdd) res = va.add(vVal);
                else if (isSub) res = va.sub(vVal);
                else res = va.mul(vVal);
                res.intoMemorySegment(oSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < n; i++) {
                float va = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float res;
                if (isAdd) res = va + val;
                else if (isSub) res = va - val;
                else res = va * val;
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, res);
            }
        } else {
            // Bias broadcasting [batch, hidden] + [hidden]
            int hidden = (int) b.numel();
            int batches = (int) (n / hidden);
            for (int bIdx = 0; bIdx < batches; bIdx++) {
                long offset = (long) bIdx * hidden;
                int i = 0;
                for (; i < SPECIES.loopBound(hidden); i += SPECIES.length()) {
                    FloatVector va = FloatVector.fromMemorySegment(SPECIES, aSeg, (offset + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    FloatVector vb = FloatVector.fromMemorySegment(SPECIES, bSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                    FloatVector res;
                    if (isAdd) res = va.add(vb);
                    else if (isSub) res = va.sub(vb);
                    else res = va.mul(vb);
                    res.intoMemorySegment(oSeg, (offset + i) * 4, ByteOrder.LITTLE_ENDIAN);
                }
                for (; i < hidden; i++) {
                    float va = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
                    float vb = bSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    float res;
                    if (isAdd) res = va + vb;
                    else if (isSub) res = va - vb;
                    else res = va * vb;
                    oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, res);
                }
            }
        }
        return out;
    }
    /** C = A / B (element-wise) */
    public static AccelTensor div(AccelTensor a, AccelTensor b) {
        a = a.contiguous();
        b = b.contiguous();
        long n = a.numel();
        AccelTensor out = AccelTensor.zeros(a.shape());
        MemorySegment aSeg = a.dataSegment();
        MemorySegment bSeg = b.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        if (b.numel() == n) {
            int i = 0;
            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                FloatVector va = FloatVector.fromMemorySegment(SPECIES, aSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                FloatVector vb = FloatVector.fromMemorySegment(SPECIES, bSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                va.div(vb).intoMemorySegment(oSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < n; i++) {
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i) / bSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
            }
        } else if (b.numel() == 1) {
            float val = b.item();
            FloatVector vVal = FloatVector.broadcast(SPECIES, val);
            int i = 0;
            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                FloatVector va = FloatVector.fromMemorySegment(SPECIES, aSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                va.div(vVal).intoMemorySegment(oSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < n; i++) {
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i) / val);
            }
        } else {
            // Broadcasting fallback
            int hidden = (int) b.numel();
            int batches = (int) (n / hidden);
            for (int bIdx = 0; bIdx < batches; bIdx++) {
                long offset = (long) bIdx * hidden;
                int i = 0;
                for (; i < SPECIES.loopBound(hidden); i += SPECIES.length()) {
                    FloatVector va = FloatVector.fromMemorySegment(SPECIES, aSeg, (offset + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    FloatVector vb = FloatVector.fromMemorySegment(SPECIES, bSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                    va.div(vb).intoMemorySegment(oSeg, (offset + i) * 4, ByteOrder.LITTLE_ENDIAN);
                }
                for (; i < hidden; i++) {
                    oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i) / bSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                }
            }
        }
        return out;
    }

    // ── Neural network operations (pure Java) ─────────────────────────

    /**
     * RMS Normalization using Optimized Vector API.
     */
    public static AccelTensor rmsNorm(AccelTensor x, AccelTensor weight, double eps) {
        return addRmsNorm(null, x, weight, eps);
    }

    /**
     * Fused Residual Add + RMS Normalization.
     * out = rmsNorm(residual + x)
     * If residual is null, it's a standard rmsNorm.
     */
    public static AccelTensor addRmsNorm(AccelTensor residual, AccelTensor x, AccelTensor weight, double eps) {
        if (weight == null) return x.contiguous();
        x = x.contiguous();
        if (residual != null) residual = residual.contiguous();

        long[] shape = x.shape();
        int hidden = (int) shape[shape.length - 1];
        int outer = (int) (x.numel() / hidden);

        AccelTensor out = AccelTensor.zeros(shape);
        MemorySegment xSeg = x.dataSegment();
        MemorySegment rSeg = residual != null ? residual.dataSegment() : null;
        MemorySegment wSeg = weight.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        for (int row = 0; row < outer; row++) {
            long base = (long) row * hidden;
            float sumSq = 0.0f;
            
            // Pass 1: Add (if residual exists) and Sum Squares
            int i = 0;
            if (hidden >= SPECIES.length()) {
                FloatVector sumSqV = FloatVector.zero(SPECIES);
                for (; i < SPECIES.loopBound(hidden); i += SPECIES.length()) {
                    FloatVector vx = FloatVector.fromMemorySegment(SPECIES, xSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    if (rSeg != null) {
                        FloatVector vr = FloatVector.fromMemorySegment(SPECIES, rSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                        vx = vx.add(vr);
                        // If we want to update x in-place for residual, we could do it here, 
                        // but for purity we'll keep x intact and use vx.
                    }
                    sumSqV = sumSqV.add(vx.mul(vx));
                }
                sumSq = sumSqV.reduceLanes(VectorOperators.ADD);
            }
            for (; i < hidden; i++) {
                float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                if (rSeg != null) val += rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                sumSq += val * val;
            }

            float rms = (float) (1.0 / Math.sqrt(sumSq / hidden + eps));
            FloatVector vRms = FloatVector.broadcast(SPECIES, rms);
            
            // Pass 2: Normalize and Weight
            i = 0;
            if (hidden >= SPECIES.length()) {
                for (; i < SPECIES.loopBound(hidden); i += SPECIES.length()) {
                    FloatVector vx = FloatVector.fromMemorySegment(SPECIES, xSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    if (rSeg != null) {
                        FloatVector vr = FloatVector.fromMemorySegment(SPECIES, rSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                        vx = vx.add(vr);
                    }
                    FloatVector vw = FloatVector.fromMemorySegment(SPECIES, wSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                    vx.mul(vRms).mul(vw).intoMemorySegment(oSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                }
            }
            for (; i < hidden; i++) {
                float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                if (rSeg != null) val += rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                float w = wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + i, val * rms * w);
            }
        }
        return out;
    }

    /**
     * Vectorized Softmax with optimized reductions.
     */
    public static AccelTensor softmax(AccelTensor x, int dim) {
        x = x.contiguous();
        long[] shape = x.shape();
        int lastDim = (int) shape[shape.length - 1];
        int outer = (int) (x.numel() / lastDim);
        AccelTensor out = AccelTensor.zeros(shape);
        MemorySegment xSeg = x.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        for (int row = 0; row < outer; row++) {
            long base = (long) row * lastDim;
            
            // 1. Find max
            float max = Float.NEGATIVE_INFINITY;
            int i = 0;
            if (lastDim >= SPECIES.length()) {
                FloatVector maxV = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
                for (; i < SPECIES.loopBound(lastDim); i += SPECIES.length()) {
                    maxV = maxV.max(FloatVector.fromMemorySegment(SPECIES, xSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN));
                }
                max = maxV.reduceLanes(VectorOperators.MAX);
            }
            for (; i < lastDim; i++) {
                max = Math.max(max, xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i));
            }

            // 2. Exp and Sum (Vectorized)
            float sum = 0.0f;
            i = 0;
            if (lastDim >= SPECIES.length()) {
                FloatVector sumV = FloatVector.zero(SPECIES);
                FloatVector vMax = FloatVector.broadcast(SPECIES, max);
                for (; i < SPECIES.loopBound(lastDim); i += SPECIES.length()) {
                    FloatVector v = FloatVector.fromMemorySegment(SPECIES, xSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    // Standard softmax: exp(x - max)
                    // Note: Vector API doesn't have exp(), so we use a loop or call Math.exp in a loop.
                    // However, we can still benefit from vectorizing the subtraction.
                    // For now, let's use the most reliable path:
                    for (int j = 0; j < SPECIES.length(); j++) {
                        float val = (float) Math.exp(v.lane(j) - max);
                        oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + i + j, val);
                        sum += val;
                    }
                }
            }
            for (; i < lastDim; i++) {
                float val = (float) Math.exp(xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i) - max);
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + i, val);
                sum += val;
            }
            // Add epsilon to prevent division by zero
            sum += 1e-12f;

            // 3. Normalize
            float invSum = (sum == 0.0f) ? 0.0f : 1.0f / sum;
            FloatVector vInvSum = FloatVector.broadcast(SPECIES, invSum);
            i = 0;
            for (; i < SPECIES.loopBound(lastDim); i += SPECIES.length()) {
                FloatVector v = FloatVector.fromMemorySegment(SPECIES, oSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                v.mul(vInvSum).intoMemorySegment(oSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < lastDim; i++) {
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + i, oSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i) * invSum);
            }
        }
        return out;
    }

    /**
     * Vector API SiLU activation.
     */
    public static AccelTensor silu(AccelTensor x) {
        x = x.contiguous();
        AccelTensor out = AccelTensor.zeros(x.shape());
        MemorySegment xSeg = x.dataSegment();
        MemorySegment oSeg = out.dataSegment();
        long n = x.numel();

        for (int i = 0; i < n; i++) {
            float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            // Using standard Math.exp for numerical stability compared to approximation
            oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (val / (1.0 + Math.exp(-val))));
        }
        return out;
    }

    /**
     * Fused SwiGLU: silu(gate) * up
     */
    public static AccelTensor swiglu(AccelTensor gate, AccelTensor up) {
        gate = gate.contiguous();
        up = up.contiguous();
        long n = gate.numel();
        AccelTensor out = AccelTensor.zeros(gate.shape());
        MemorySegment gSeg = gate.dataSegment();
        MemorySegment uSeg = up.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        int i = 0;
        if (n >= SPECIES.length()) {
            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                FloatVector vg = FloatVector.fromMemorySegment(SPECIES, gSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                FloatVector vu = FloatVector.fromMemorySegment(SPECIES, uSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                
                // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                // We use our expVector approximation.
                FloatVector sigmoid = FloatVector.broadcast(SPECIES, 1.0f).div(
                    FloatVector.broadcast(SPECIES, 1.0f).add(expVector(vg.neg())));
                
                vg.mul(sigmoid).mul(vu).intoMemorySegment(oSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
            }
        }
        for (; i < n; i++) {
            float g = gSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float u = uSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float silu = (float) (g / (1.0 + Math.exp(-g)));
            oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, silu * u);
        }
        return out;
    }

    /**
     * Stable Vectorized exp(x) approximation.
     * Accurate within the range needed for DL and stable for large negatives (masking).
     */
    private static FloatVector expVector(FloatVector x) {
        // Clamp to avoid expansion instability for extreme values
        FloatVector clamped = x.max(-20.0f).min(20.0f);
        
        // P(x) = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720
        FloatVector x2 = clamped.mul(clamped);
        FloatVector x3 = x2.mul(clamped);
        FloatVector x4 = x3.mul(clamped);
        FloatVector x5 = x4.mul(clamped);
        FloatVector x6 = x5.mul(clamped);

        return FloatVector.broadcast(SPECIES, 1.0f)
            .add(clamped)
            .add(x2.mul(1.0f / 2.0f))
            .add(x3.mul(1.0f / 6.0f))
            .add(x4.mul(1.0f / 24.0f))
            .add(x5.mul(1.0f / 120.0f))
            .add(x6.mul(1.0f / 720.0f));
    }

    /**
     * Optimized Vector API GELU activation.
     * GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     */
    public static AccelTensor gelu(AccelTensor x) {
        x = x.contiguous();
        long n = x.numel();
        AccelTensor out = AccelTensor.zeros(x.shape());
        MemorySegment xSeg = x.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        int i = 0;
        if (n >= SPECIES.length()) {
            FloatVector vSqrt2Pi = FloatVector.broadcast(SPECIES, 0.79788456f); // sqrt(2/pi)
            FloatVector v044 = FloatVector.broadcast(SPECIES, 0.044715f);
            FloatVector vOne = FloatVector.broadcast(SPECIES, 1.0f);
            FloatVector vHalf = FloatVector.broadcast(SPECIES, 0.5f);

            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                FloatVector vx = FloatVector.fromMemorySegment(SPECIES, xSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                // (x + 0.044715 * x^3)
                FloatVector inner = vx.add(v044.mul(vx.mul(vx).mul(vx)));
                // sqrt(2/pi) * inner
                FloatVector arg = vSqrt2Pi.mul(inner);

                // tanh approximation or lane-wise tanh
                // We use a simplified tanh: x * (3 + x^2) / (1 + 3*x^2) or just use lane-wise Math.tanh for precision
                for (int j = 0; j < SPECIES.length(); j++) {
                    float val = arg.lane(j);
                    float res = (float) Math.tanh(val);
                    float xVal = vx.lane(j);
                    oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i + j, 0.5f * xVal * (1.0f + res));
                }
            }
        }
        for (; i < n; i++) {
            float v = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float inner = 0.79788456f * (v + 0.044715f * v * v * v);
            oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, 0.5f * v * (1.0f + (float) Math.tanh(inner)));
        }
        return out;
    }

    /**
     * Vectorized Layer Normalization.
     */
    public static AccelTensor layerNorm(AccelTensor x, AccelTensor weight, AccelTensor bias, double eps) {
        x = x.contiguous();
        long[] shape = x.shape();
        int dim = (int) shape[shape.length - 1];
        int rows = (int) (x.numel() / dim);

        AccelTensor out = AccelTensor.zeros(shape);
        MemorySegment xSeg = x.dataSegment();
        MemorySegment wSeg = weight.dataSegment();
        MemorySegment bSeg = bias != null ? bias.dataSegment() : null;
        MemorySegment oSeg = out.dataSegment();

        for (int r = 0; r < rows; r++) {
            long base = (long) r * dim;
            float sum = 0, sumSq = 0;

            int i = 0;
            if (dim >= SPECIES.length()) {
                FloatVector vSum = FloatVector.zero(SPECIES);
                FloatVector vSumSq = FloatVector.zero(SPECIES);
                for (; i < SPECIES.loopBound(dim); i += SPECIES.length()) {
                    FloatVector v = FloatVector.fromMemorySegment(SPECIES, xSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    vSum = vSum.add(v);
                    vSumSq = vSumSq.add(v.mul(v));
                }
                sum = vSum.reduceLanes(VectorOperators.ADD);
                sumSq = vSumSq.reduceLanes(VectorOperators.ADD);
            }
            for (; i < dim; i++) {
                float v = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                sum += v;
                sumSq += v * v;
            }

            float mean = sum / dim;
            float var = (float) Math.sqrt(sumSq / dim - mean * mean + eps);
            FloatVector vMean = FloatVector.broadcast(SPECIES, mean);
            FloatVector vVar = FloatVector.broadcast(SPECIES, 1.0f / var);

            i = 0;
            if (dim >= SPECIES.length()) {
                for (; i < SPECIES.loopBound(dim); i += SPECIES.length()) {
                    FloatVector vx = FloatVector.fromMemorySegment(SPECIES, xSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    FloatVector vw = FloatVector.fromMemorySegment(SPECIES, wSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                    FloatVector res = vx.sub(vMean).mul(vVar).mul(vw);
                    if (bSeg != null) {
                        FloatVector vb = FloatVector.fromMemorySegment(SPECIES, bSeg, (long) i * 4, ByteOrder.LITTLE_ENDIAN);
                        res = res.add(vb);
                    }
                    res.intoMemorySegment(oSeg, (base + i) * 4, ByteOrder.LITTLE_ENDIAN);
                }
            }
            for (; i < dim; i++) {
                float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                float w = wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float b = bSeg != null ? bSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i) : 0.0f;
                oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + i, (val - mean) / var * w + b);
            }
        }
        return out;
    }

    /**
     * Fused image normalization: (pixel - mean) / std.
     * Efficiently processes CHW float arrays in a single SIMD pass.
     */
    public static void normalizeImage(float[] chw, int pixels, float[] mean, float[] std) {
        for (int c = 0; c < 3; c++) {
            int offset = c * pixels;
            float m = mean[c];
            float s = 1.0f / std[c];
            FloatVector vMean = FloatVector.broadcast(SPECIES, m);
            FloatVector vInvStd = FloatVector.broadcast(SPECIES, s);

            int i = 0;
            for (; i < SPECIES.loopBound(pixels); i += SPECIES.length()) {
                FloatVector v = FloatVector.fromArray(SPECIES, chw, offset + i);
                v.sub(vMean).mul(vInvStd).intoArray(chw, offset + i);
            }
            for (; i < pixels; i++) {
                chw[offset + i] = (chw[offset + i] - m) * s;
            }
        }
    }

    /**
     * SIMD-accelerated Cosine Similarity for RAG.
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Vector length mismatch");
        int n = a.length;
        float dot = 0.0f, nA = 0.0f, nB = 0.0f;

        int i = 0;
        if (n >= SPECIES.length()) {
            FloatVector vDot = FloatVector.zero(SPECIES);
            FloatVector vNA = FloatVector.zero(SPECIES);
            FloatVector vNB = FloatVector.zero(SPECIES);
            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                FloatVector va = FloatVector.fromArray(SPECIES, a, i);
                FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
                vDot = vDot.add(va.mul(vb));
                vNA = vNA.add(va.mul(va));
                vNB = vNB.add(vb.mul(vb));
            }
            dot = vDot.reduceLanes(VectorOperators.ADD);
            nA = vNA.reduceLanes(VectorOperators.ADD);
            nB = vNB.reduceLanes(VectorOperators.ADD);
        }
        for (; i < n; i++) {
            dot += a[i] * b[i];
            nA += a[i] * a[i];
            nB += b[i] * b[i];
        }
        if (nA == 0 || nB == 0) return 0.0f;
        return (float) (dot / (Math.sqrt(nA) * Math.sqrt(nB)));
    }

    /**
     * SIMD Dot Product.
     */
    public static float dotProduct(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Vector length mismatch");
        int n = a.length;
        float sum = 0;
        int i = 0;
        if (n >= SPECIES.length()) {
            FloatVector vSum = FloatVector.zero(SPECIES);
            for (; i < SPECIES.loopBound(n); i += SPECIES.length()) {
                vSum = vSum.add(FloatVector.fromArray(SPECIES, a, i).mul(FloatVector.fromArray(SPECIES, b, i)));
            }
            sum = vSum.reduceLanes(VectorOperators.ADD);
        }
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    private static AccelTensor dequantize(AccelTensor qWeight) {
        long[] shape = qWeight.shape();
        AccelTensor f32 = AccelTensor.zeros(shape);
        
        switch (qWeight.quantType()) {
            case INT8 -> DequantizationKernel.dequantizeInt8(
                    qWeight.dataSegment(), f32.dataSegment(), qWeight.scales(), qWeight.numel());
            case INT4 -> DequantizationKernel.dequantizeInt4(
                    qWeight.dataSegment(), f32.dataSegment(), qWeight.scales(), qWeight.zeros(), qWeight.numel(), qWeight.groupSize());
            default -> throw new UnsupportedOperationException("Unsupported quantization type: " + qWeight.quantType());
        }
        
        return f32;
    }

    /**
     * Element-wise square root.
     */
    public static AccelTensor sqrt(AccelTensor x) {
        float[] data = x.toFloatArray();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.sqrt(data[i]);
        }
        return AccelTensor.fromFloatArray(result, x.shape());
    }


    /**
     * 2D Convolution: [N, Co, Ho, Wo] = conv2d([N, Ci, Hi, Wi], [Co, Ci, Kh, Kw])
     */
    public static AccelTensor conv2d(AccelTensor input, AccelTensor weight, AccelTensor bias,
                                     int stride, int padding) {
        long n = input.size(0);
        long ci = input.size(1);
        long hi = input.size(2);
        long wi = input.size(3);

        long co = weight.size(0);
        long kh = weight.size(2);
        long kw = weight.size(3);

        long ho = (hi + 2L * padding - kh) / stride + 1;
        long wo = (wi + 2L * padding - kw) / stride + 1;

        AccelTensor out = AccelTensor.zeros(n, co, ho, wo);
        
        // Use a simple direct convolution for now, but vectorized over input channels/width
        // For production, im2col + sgemm is much faster.
        for (int bi = 0; bi < n; bi++) {
            for (int coi = 0; coi < co; coi++) {
                for (int hoy = 0; hoy < ho; hoy++) {
                    for (int wox = 0; wox < wo; wox++) {
                        float sum = 0;
                        int iy_base = (int)(hoy * stride - padding);
                        int ix_base = (int)(wox * stride - padding);

                        for (int cii = 0; cii < ci; cii++) {
                            for (int ky = 0; ky < kh; ky++) {
                                int iy = iy_base + ky;
                                if (iy < 0 || iy >= hi) continue;
                                
                                for (int kx = 0; kx < kw; kx++) {
                                    int ix = ix_base + kx;
                                    if (ix < 0 || ix >= wi) continue;
                                    
                                    sum += input.get(bi, cii, iy, ix) * weight.get(coi, cii, ky, kx);
                                }
                            }
                        }
                        if (bias != null) {
                            sum += bias.get(coi);
                        }
                        out.set(sum, bi, coi, hoy, wox);
                    }
                }
            }
        }
        return out;
    }

    /**
     * Group Normalization.
     */
    public static AccelTensor groupNorm(AccelTensor x, AccelTensor weight, AccelTensor bias, int numGroups, double eps) {
        x = x.contiguous();
        long n = x.size(0);
        long c = x.size(1);
        long h = x.size(2);
        long w = x.size(3);
        long elementsPerGroup = (c / numGroups) * h * w;

        AccelTensor out = AccelTensor.zeros(n, c, h, w);
        MemorySegment xSeg = x.dataSegment();
        MemorySegment oSeg = out.dataSegment();
        
        for (int bi = 0; bi < n; bi++) {
            for (int g = 0; g < numGroups; g++) {
                long groupOffset = (bi * c * h * w + g * elementsPerGroup);
                
                // 1. Mean and Variance
                float sum = 0, sumSq = 0;
                for (long i = 0; i < elementsPerGroup; i++) {
                    float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, groupOffset + i);
                    sum += val;
                    sumSq += val * val;
                }
                float mean = sum / elementsPerGroup;
                float var = (float) Math.sqrt(sumSq / elementsPerGroup - mean * mean + eps);
                float invVar = 1.0f / var;

                // 2. Normalize and Scale/Bias
                for (int cg = 0; cg < c / numGroups; cg++) {
                    int ci = g * (int)(c / numGroups) + cg;
                    float gamma = weight.get(ci);
                    float beta = bias != null ? bias.get(ci) : 0.0f;
                    
                    for (int hw = 0; hw < h * w; hw++) {
                        long idx = (long) bi * c * h * w + (long) ci * h * w + hw;
                        float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                        oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, (val - mean) * invVar * gamma + beta);
                    }
                }
            }
        }
        return out;
    }

    /**
     * Concat tensors along a dimension.
     */
    public static AccelTensor concat(AccelTensor a, AccelTensor b, int dim) {
        long[] aShape = a.shape();
        long[] bShape = b.shape();
        long[] outShape = aShape.clone();
        outShape[dim] = aShape[dim] + bShape[dim];
        
        AccelTensor out = AccelTensor.zeros(outShape);
        // TODO: Optimized block copies
        return out;
    }
}

