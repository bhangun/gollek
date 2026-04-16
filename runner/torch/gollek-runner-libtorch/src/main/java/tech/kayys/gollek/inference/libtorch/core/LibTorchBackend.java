package tech.kayys.gollek.inference.libtorch.core;

import tech.kayys.gollek.runtime.tensor.Backend;
import tech.kayys.gollek.runtime.tensor.BackendType;
import tech.kayys.gollek.runtime.tensor.DType;
import tech.kayys.gollek.runtime.tensor.Device;
import tech.kayys.gollek.runtime.tensor.ExecutionContext;
import tech.kayys.gollek.runtime.tensor.DefaultTensor;

/**
 * LibTorch backend implementation for the core runtime graph and tensor
 * abstractions.
 * <p>
 * Delegates standard tensor operations to the native PyTorch C++ API via FFM.
 * This backend provides GPU-accelerated operations via CUDA/MPS and CPU
 * fallback.
 * </p>
 *
 * @see Backend
 * @see TorchTensor
 * @since 1.0
 */
public class LibTorchBackend implements Backend {

    @Override
    public BackendType type() {
        return BackendType.LIBTORCH;
    }

    @Override
    public tech.kayys.gollek.runtime.tensor.Tensor add(
            tech.kayys.gollek.runtime.tensor.Tensor a,
            tech.kayys.gollek.runtime.tensor.Tensor b,
            ExecutionContext ctx) {
        TorchTensor ltA = requireLibTorch(a);
        TorchTensor ltB = requireLibTorch(b);
        TorchTensor result = ltA.add(ltB);
        if (ctx != null)
            ctx.track(result);
        return result;
    }

    @Override
    public tech.kayys.gollek.runtime.tensor.Tensor matmul(
            tech.kayys.gollek.runtime.tensor.Tensor a,
            tech.kayys.gollek.runtime.tensor.Tensor b,
            ExecutionContext ctx) {
        TorchTensor ltA = requireLibTorch(a);
        TorchTensor ltB = requireLibTorch(b);
        TorchTensor result = ltA.matmul(ltB);
        if (ctx != null)
            ctx.track(result);
        return result;
    }

    @Override
    public tech.kayys.gollek.runtime.tensor.Tensor relu(
            tech.kayys.gollek.runtime.tensor.Tensor a,
            ExecutionContext ctx) {
        TorchTensor ltA = requireLibTorch(a);
        TorchTensor result = ltA.relu();
        if (ctx != null)
            ctx.track(result);
        return result;
    }

    @Override
    public tech.kayys.gollek.runtime.tensor.Tensor createTensor(
            long[] shape,
            DType dtype,
            Device device,
            ExecutionContext ctx) {
        ScalarType st = mapDType(dtype);
        tech.kayys.gollek.inference.libtorch.core.Device.Type dt = mapDevice(device);

        TorchTensor result = TorchTensor.empty(shape, st,
                new tech.kayys.gollek.inference.libtorch.core.Device(dt, -1));
        if (ctx != null)
            ctx.track(result);
        return result;
    }

    /**
     * Downcasts a generic tensor to a concrete LibTorch TorchTensor.
     * <p>
     * Supports direct TorchTensor instances and unwraps DefaultTensor if needed.
     * </p>
     *
     * @param t the generic tensor
     * @return the LibTorch tensor
     * @throws IllegalArgumentException if the tensor is not a LibTorch tensor
     */
    private TorchTensor requireLibTorch(tech.kayys.gollek.runtime.tensor.Tensor t) {
        if (t instanceof TorchTensor lt) {
            return lt;
        }
        if (t instanceof DefaultTensor dt) {
            // In a fully integrated system we would wrap DefaultTensor's pointer
            // back into a transient LibTorch tensor. But for now we enforce using
            // the plugin's unified TorchTensor wrapper.
            throw new IllegalArgumentException("Expected LibTorch native tensor, got DefaultTensor");
        }
        throw new IllegalArgumentException("Unsupported tensor type for LibTorch backend: " + t.getClass().getName());
    }

    private ScalarType mapDType(DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> ScalarType.FLOAT;
            case FLOAT16 -> ScalarType.HALF;
            case BFLOAT16 -> ScalarType.BFLOAT16;
            case INT8 -> ScalarType.CHAR;
            case INT4 -> ScalarType.QUINT8; // No native INT4 in ATen, map to QInt8 or UInt8
            case QINT8 -> ScalarType.QINT8;
            case QINT4 -> ScalarType.QUINT8;
            default -> throw new IllegalArgumentException("Unsupported DType: " + dtype);
        };
    }

    private tech.kayys.gollek.inference.libtorch.core.Device.Type mapDevice(Device device) {
        return switch (device) {
            case CPU -> tech.kayys.gollek.inference.libtorch.core.Device.Type.CPU;
            case CUDA -> tech.kayys.gollek.inference.libtorch.core.Device.Type.CUDA;
            case METAL -> tech.kayys.gollek.inference.libtorch.core.Device.Type.MPS;
            case ROCM -> tech.kayys.gollek.inference.libtorch.core.Device.Type.HIP;
            case TPU -> tech.kayys.gollek.inference.libtorch.core.Device.Type.XLA;
            case NPU -> tech.kayys.gollek.inference.libtorch.core.Device.Type.CPU;
            default -> tech.kayys.gollek.inference.libtorch.core.Device.Type.CPU;
        };
    }
}
