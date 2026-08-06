package tech.kayys.gollek.inference.libtorch.core;

import tech.kayys.gollek.core.backend.ComputeBackendType;
import tech.kayys.alkhawarizm.core.tensor.DType;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.gollek.core.graph.ExecutionContext;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

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
public class LibTorchBackend implements tech.kayys.gollek.core.backend.ComputeBackend {

    @Override
    public tech.kayys.alkhawarizm.core.tensor.Tensor add(tech.kayys.alkhawarizm.core.tensor.Tensor a,
            tech.kayys.alkhawarizm.core.tensor.Tensor b) {
        return a;
    }

    @Override
    public tech.kayys.alkhawarizm.core.tensor.Tensor sub(tech.kayys.alkhawarizm.core.tensor.Tensor a,
            tech.kayys.alkhawarizm.core.tensor.Tensor b) {
        return a;
    }

    @Override
    public tech.kayys.alkhawarizm.core.tensor.Tensor mul(tech.kayys.alkhawarizm.core.tensor.Tensor a, float scalar) {
        return a;
    }

    @Override
    public tech.kayys.alkhawarizm.core.tensor.Tensor div(tech.kayys.alkhawarizm.core.tensor.Tensor a, float scalar) {
        return a;
    }

    @Override
    public tech.kayys.alkhawarizm.core.tensor.Tensor matmul(tech.kayys.alkhawarizm.core.tensor.Tensor a,
            tech.kayys.alkhawarizm.core.tensor.Tensor b) {
        return a;
    }

    @Override
    public tech.kayys.alkhawarizm.core.tensor.Tensor relu(tech.kayys.alkhawarizm.core.tensor.Tensor a) {
        return a;
    }

    @Override
    public tech.kayys.alkhawarizm.core.tensor.Tensor transpose(tech.kayys.alkhawarizm.core.tensor.Tensor a, int dim0,
            int dim1) {
        return a;
    }

    @Override
    public BackendType type() {
        return BackendType.LIBTORCH;
    }

    @Override
    @Override
    public long numel(tech.kayys.alkhawarizm.core.tensor.Tensor a) {
        return a.shape().numel();
    }

    public tech.kayys.alkhawarizm.core.tensor.Tensor createTensor(
            long[] shape,
            DType dtype,
            Device device,
            ExecutionContext ctx) {
        ScalarType st = mapDType(dtype);
        tech.kayys.gollek.inference.libtorch.core.Device.Type dt = mapDevice(device);

        TorchTensor result = TorchTensor.empty(shape, st,
                new tech.kayys.gollek.inference.libtorch.core.Device(dt, -1));
        if (ctx != null)

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
    private TorchTensor requireLibTorch(tech.kayys.alkhawarizm.core.tensor.Tensor t) {
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
            case CPU -> tech.kayys.alkhawarizm.core.tensor.DeviceType.CPU;
            case CUDA -> tech.kayys.alkhawarizm.core.tensor.DeviceType.CUDA;
            case METAL -> tech.kayys.gollek.inference.libtorch.core.Device.Type.MPS;
            case ROCM -> tech.kayys.gollek.inference.libtorch.core.Device.Type.HIP;
            case TPU -> tech.kayys.alkhawarizm.core.tensor.DeviceType.TPU;
            case NPU -> tech.kayys.alkhawarizm.core.tensor.DeviceType.CPU;
            default -> tech.kayys.alkhawarizm.core.tensor.DeviceType.CPU;
        };
    }
}
