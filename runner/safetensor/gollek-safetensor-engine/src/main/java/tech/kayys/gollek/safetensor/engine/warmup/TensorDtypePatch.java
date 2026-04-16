/*
 * Gollek Inference Engine — LibTorch patch
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * TensorDtypePatch.java
 * ─────────────────────
 * This file should live in gollek-libtorch, not gollek-safetensor.
 * It is placed here as a documentation stub showing EXACTLY what must be added
 * to the TorchTensor class in gollek-libtorch/src/.../core/TorchTensor.java.
 *
 * ADD THE FOLLOWING TWO METHODS to tech.kayys.gollek.inference.libtorch.core.TorchTensor
 * after the existing "to(Device device)" method:
 *
 * ─────────────────────────────────────────────────────────────────────────────────
 *
 *   // Cast tensor to a different scalar type (dtype).
 *   // Used by SafetensorWeightBridge.bridgeAsF32() and QuantizationEngine.cast().
 *   public TorchTensor to(ScalarType targetDtype) {
 *       checkClosed();
 *       Arena opArena = Arena.ofShared();
 *       try {
 *           MethodHandle fn = LibTorchBinding.getInstance().bind(
 *                   LibTorchBinding.TENSOR_TO_DTYPE,
 *                   LibTorchBinding.TENSOR_TO_DTYPE_DESC);
 *           MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, targetDtype.code());
 *           return new TorchTensor(result, opArena);
 *       } catch (Throwable t) {
 *           opArena.close();
 *           throw new RuntimeException("Failed to cast tensor to " + targetDtype, t);
 *       }
 *   }
 *
 *   // ScalarType-aware dtype getter (convenience alias over dtype()).
 *   public ScalarType scalarType() {
 *       return dtype();
 *   }
 *
 * ─────────────────────────────────────────────────────────────────────────────────
 *
 * The TENSOR_TO_DTYPE binding already exists in LibTorchBinding:
 *   TENSOR_TO_DTYPE      = "at_tensor_to_dtype"
 *   TENSOR_TO_DTYPE_DESC = FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT)
 *
 * The private castTensorToDtype() in LibTorchMixedPrecision does the same thing.
 * Exposing it publicly removes the need for any reflection or workaround.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

/**
 * Documentation stub — see file header for the exact patch to apply to
 * {@code tech.kayys.gollek.inference.libtorch.core.TorchTensor}.
 *
 * <p>
 * Until the patch is applied, {@link SafetensorWeightBridge#bridgeAsF32}
 * and {@link tech.kayys.gollek.quantization.QuantizationEngine#cast}
 * will not compile. Apply the two methods above first.
 */
public final class TensorDtypePatch {
    private TensorDtypePatch() {
    }
}
