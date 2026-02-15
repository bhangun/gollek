#include "pytorch_java_ffm.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>

// Helper function to convert void* to torch::Tensor*
inline torch::Tensor* toTensor(void* ptr) {
    return reinterpret_cast<torch::Tensor*>(ptr);
}

// Helper function to convert torch::Tensor* to void*
inline void* fromTensor(torch::Tensor* tensor) {
    return reinterpret_cast<void*>(tensor);
}

// ============================================================================
// Tensor Creation Functions
// ============================================================================

void* at_empty(int64_t* sizes, int ndim, void* options) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto tensor = new torch::Tensor(torch::empty(shape));
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_empty: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_zeros(int64_t* sizes, int ndim, void* options) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto tensor = new torch::Tensor(torch::zeros(shape));
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_zeros: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_ones(int64_t* sizes, int ndim, void* options) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto tensor = new torch::Tensor(torch::ones(shape));
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_ones: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_randn(int64_t* sizes, int ndim, void* options) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto tensor = new torch::Tensor(torch::randn(shape));
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_randn: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_rand(int64_t* sizes, int ndim, void* options) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto tensor = new torch::Tensor(torch::rand(shape));
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_rand: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_arange(double start, double end, double step, void* options) {
    try {
        auto tensor = new torch::Tensor(torch::arange(start, end, step));
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_arange: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_linspace(double start, double end, int64_t steps, void* options) {
    try {
        auto tensor = new torch::Tensor(torch::linspace(start, end, steps));
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_linspace: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_from_blob(void* data, int64_t* sizes, int ndim, void* options) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto tensor = new torch::Tensor(
            torch::from_blob(data, shape, torch::kFloat32)
        );
        return fromTensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_from_blob: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Tensor Operations
// ============================================================================

void* tensor_add(void* self, void* other, double alpha) {
    try {
        auto result = new torch::Tensor(
            toTensor(self)->add(*toTensor(other), alpha)
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_add: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_sub(void* self, void* other, double alpha) {
    try {
        auto result = new torch::Tensor(
            toTensor(self)->sub(*toTensor(other), alpha)
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_sub: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_mul(void* self, void* other) {
    try {
        auto result = new torch::Tensor(
            toTensor(self)->mul(*toTensor(other))
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_mul: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_div(void* self, void* other) {
    try {
        auto result = new torch::Tensor(
            toTensor(self)->div(*toTensor(other))
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_div: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_matmul(void* self, void* other) {
    try {
        auto result = new torch::Tensor(
            toTensor(self)->matmul(*toTensor(other))
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_matmul: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_mm(void* self, void* other) {
    try {
        auto result = new torch::Tensor(
            toTensor(self)->mm(*toTensor(other))
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_mm: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_reshape(void* self, int64_t* shape, int ndim) {
    try {
        std::vector<int64_t> new_shape(shape, shape + ndim);
        auto result = new torch::Tensor(
            toTensor(self)->reshape(new_shape)
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_reshape: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Tensor Properties
// ============================================================================

int64_t* tensor_sizes(void* self) {
    try {
        auto sizes = toTensor(self)->sizes();
        int64_t* result = new int64_t[sizes.size()];
        std::copy(sizes.begin(), sizes.end(), result);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_sizes: " << e.what() << std::endl;
        return nullptr;
    }
}

int64_t tensor_size(void* self, int dim) {
    try {
        return toTensor(self)->size(dim);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_size: " << e.what() << std::endl;
        return -1;
    }
}

int64_t tensor_dim(void* self) {
    try {
        return toTensor(self)->dim();
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_dim: " << e.what() << std::endl;
        return -1;
    }
}

int64_t tensor_numel(void* self) {
    try {
        return toTensor(self)->numel();
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_numel: " << e.what() << std::endl;
        return -1;
    }
}

void* tensor_data_ptr(void* self) {
    try {
        return toTensor(self)->data_ptr();
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_data_ptr: " << e.what() << std::endl;
        return nullptr;
    }
}

int tensor_scalar_type(void* self) {
    try {
        return static_cast<int>(toTensor(self)->scalar_type());
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_scalar_type: " << e.what() << std::endl;
        return -1;
    }
}

bool tensor_requires_grad(void* self) {
    try {
        return toTensor(self)->requires_grad();
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_requires_grad: " << e.what() << std::endl;
        return false;
    }
}

void* tensor_cuda(void* self) {
    try {
        auto result = new torch::Tensor(toTensor(self)->cuda());
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_cuda: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_cpu(void* self) {
    try {
        auto result = new torch::Tensor(toTensor(self)->cpu());
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_cpu: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_clone(void* self) {
    try {
        auto result = new torch::Tensor(toTensor(self)->clone());
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_clone: " << e.what() << std::endl;
        return nullptr;
    }
}

void* tensor_detach(void* self) {
    try {
        auto result = new torch::Tensor(toTensor(self)->detach());
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_detach: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Autograd Functions
// ============================================================================

void tensor_backward(void* self, void* gradient, bool retain_graph, bool create_graph) {
    try {
        if (gradient) {
            toTensor(self)->backward(*toTensor(gradient), retain_graph, create_graph);
        } else {
            toTensor(self)->backward({}, retain_graph, create_graph);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_backward: " << e.what() << std::endl;
    }
}

void* tensor_grad(void* self) {
    try {
        auto& grad = toTensor(self)->grad();
        if (grad.defined()) {
            return fromTensor(new torch::Tensor(grad));
        }
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_grad: " << e.what() << std::endl;
        return nullptr;
    }
}

void tensor_zero_grad(void* self) {
    try {
        auto& grad = toTensor(self)->mutable_grad();
        if (grad.defined()) {
            grad.zero_();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_zero_grad: " << e.what() << std::endl;
    }
}

void* tensor_requires_grad_(void* self, bool requires_grad) {
    try {
        auto result = new torch::Tensor(
            toTensor(self)->set_requires_grad(requires_grad)
        );
        return fromTensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in tensor_requires_grad_: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Neural Network Functions
// ============================================================================

void* nn_functional_linear(void* input, void* weight, void* bias) {
    try {
        torch::Tensor result;
        if (bias) {
            result = torch::nn::functional::linear(
                *toTensor(input), *toTensor(weight), *toTensor(bias)
            );
        } else {
            result = torch::nn::functional::linear(
                *toTensor(input), *toTensor(weight)
            );
        }
        return fromTensor(new torch::Tensor(result));
    } catch (const std::exception& e) {
        std::cerr << "Error in nn_functional_linear: " << e.what() << std::endl;
        return nullptr;
    }
}

void* nn_functional_relu(void* input, bool inplace) {
    try {
        torch::nn::functional::ReLUFuncOptions options;
        options.inplace(inplace);
        auto result = torch::nn::functional::relu(*toTensor(input), options);
        return fromTensor(new torch::Tensor(result));
    } catch (const std::exception& e) {
        std::cerr << "Error in nn_functional_relu: " << e.what() << std::endl;
        return nullptr;
    }
}

void* nn_functional_gelu(void* input) {
    try {
        auto result = torch::nn::functional::gelu(*toTensor(input));
        return fromTensor(new torch::Tensor(result));
    } catch (const std::exception& e) {
        std::cerr << "Error in nn_functional_gelu: " << e.what() << std::endl;
        return nullptr;
    }
}

void* nn_functional_softmax(void* input, int dim) {
    try {
        torch::nn::functional::SoftmaxFuncOptions options(dim);
        auto result = torch::nn::functional::softmax(*toTensor(input), options);
        return fromTensor(new torch::Tensor(result));
    } catch (const std::exception& e) {
        std::cerr << "Error in nn_functional_softmax: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// CUDA Functions
// ============================================================================

bool cuda_is_available() {
    return torch::cuda::is_available();
}

int cuda_device_count() {
    return torch::cuda::device_count();
}

void cuda_set_device(int device) {
    torch::cuda::set_device(device);
}

void cuda_synchronize() {
    torch::cuda::synchronize();
}

void cuda_empty_cache() {
    c10::cuda::CUDACachingAllocator::emptyCache();
}

// ============================================================================
// Utility Functions
// ============================================================================

void tensor_destroy(void* tensor) {
    if (tensor) {
        delete toTensor(tensor);
    }
}

void module_destroy(void* module) {
    // Implementation depends on how modules are wrapped
}

void optimizer_destroy(void* optimizer) {
    // Implementation depends on how optimizers are wrapped
}
