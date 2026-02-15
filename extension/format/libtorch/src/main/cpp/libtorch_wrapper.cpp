// libtorch_wrapper.cpp
// C API wrapper for LibTorch to be called from Java FFM
// Compile: g++ -shared -fPIC -o liblibtorch_wrapper.so libtorch_wrapper.cpp \
//          -I/path/to/libtorch/include -L/path/to/libtorch/lib \
//          -ltorch -ltorch_cpu -lc10 -std=c++17

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" {

// ============================================================================
// Tensor Creation
// ============================================================================

void* at_zeros(const int64_t* sizes, int64_t ndim, int32_t scalar_type) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto options = torch::TensorOptions()
            .dtype(static_cast<torch::ScalarType>(scalar_type));
        auto tensor = torch::zeros(shape, options);
        return new torch::Tensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_zeros: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_ones(const int64_t* sizes, int64_t ndim, int32_t scalar_type) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto options = torch::TensorOptions()
            .dtype(static_cast<torch::ScalarType>(scalar_type));
        auto tensor = torch::ones(shape, options);
        return new torch::Tensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_ones: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_rand(const int64_t* sizes, int64_t ndim, int32_t scalar_type) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto options = torch::TensorOptions()
            .dtype(static_cast<torch::ScalarType>(scalar_type));
        auto tensor = torch::rand(shape, options);
        return new torch::Tensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_rand: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_randn(const int64_t* sizes, int64_t ndim, int32_t scalar_type) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto options = torch::TensorOptions()
            .dtype(static_cast<torch::ScalarType>(scalar_type));
        auto tensor = torch::randn(shape, options);
        return new torch::Tensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_randn: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_empty(const int64_t* sizes, int64_t ndim, int32_t scalar_type) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto options = torch::TensorOptions()
            .dtype(static_cast<torch::ScalarType>(scalar_type));
        auto tensor = torch::empty(shape, options);
        return new torch::Tensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_empty: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_from_blob(void* data, const int64_t* sizes, int64_t ndim, int32_t scalar_type) {
    try {
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto options = torch::TensorOptions()
            .dtype(static_cast<torch::ScalarType>(scalar_type));
        auto tensor = torch::from_blob(data, shape, options);
        return new torch::Tensor(tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_from_blob: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Tensor Operations
// ============================================================================

void* at_add(void* tensor, void* other, double alpha) {
    try {
        auto t1 = static_cast<torch::Tensor*>(tensor);
        auto t2 = static_cast<torch::Tensor*>(other);
        auto result = t1->add(*t2, alpha);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_add: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_sub(void* tensor, void* other, double alpha) {
    try {
        auto t1 = static_cast<torch::Tensor*>(tensor);
        auto t2 = static_cast<torch::Tensor*>(other);
        auto result = t1->sub(*t2, alpha);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_sub: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_mul(void* tensor, void* other) {
    try {
        auto t1 = static_cast<torch::Tensor*>(tensor);
        auto t2 = static_cast<torch::Tensor*>(other);
        auto result = t1->mul(*t2);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_mul: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_div(void* tensor, void* other) {
    try {
        auto t1 = static_cast<torch::Tensor*>(tensor);
        auto t2 = static_cast<torch::Tensor*>(other);
        auto result = t1->div(*t2);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_div: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_matmul(void* tensor, void* other) {
    try {
        auto t1 = static_cast<torch::Tensor*>(tensor);
        auto t2 = static_cast<torch::Tensor*>(other);
        auto result = t1->matmul(*t2);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_matmul: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_mm(void* tensor, void* other) {
    try {
        auto t1 = static_cast<torch::Tensor*>(tensor);
        auto t2 = static_cast<torch::Tensor*>(other);
        auto result = t1->mm(*t2);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_mm: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

void* at_relu(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = torch::relu(*t);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_relu: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_sigmoid(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = torch::sigmoid(*t);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_sigmoid: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_tanh(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = torch::tanh(*t);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tanh: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_softmax(void* tensor, int64_t dim, int32_t dtype) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = torch::softmax(*t, dim, static_cast<torch::ScalarType>(dtype));
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_softmax: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_log_softmax(void* tensor, int64_t dim, int32_t dtype) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = torch::log_softmax(*t, dim, static_cast<torch::ScalarType>(dtype));
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_log_softmax: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Autograd Operations
// ============================================================================

void at_backward(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        t->backward();
    } catch (const std::exception& e) {
        std::cerr << "Error in at_backward: " << e.what() << std::endl;
    }
}

void* at_grad(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        if (t->grad().defined()) {
            return new torch::Tensor(t->grad());
        }
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Error in at_grad: " << e.what() << std::endl;
        return nullptr;
    }
}

void at_set_requires_grad(void* tensor, bool requires_grad) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        t->set_requires_grad(requires_grad);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_set_requires_grad: " << e.what() << std::endl;
    }
}

bool at_requires_grad(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        return t->requires_grad();
    } catch (const std::exception& e) {
        std::cerr << "Error in at_requires_grad: " << e.what() << std::endl;
        return false;
    }
}

void at_zero_grad(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        if (t->grad().defined()) {
            t->grad().zero_();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in at_zero_grad: " << e.what() << std::endl;
    }
}

// ============================================================================
// Tensor Properties
// ============================================================================

void* at_tensor_data_ptr(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        return t->data_ptr();
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_data_ptr: " << e.what() << std::endl;
        return nullptr;
    }
}

int64_t at_tensor_size(void* tensor, int64_t dim) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        return t->size(dim);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_size: " << e.what() << std::endl;
        return -1;
    }
}

int64_t at_tensor_dim(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        return t->dim();
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_dim: " << e.what() << std::endl;
        return -1;
    }
}

int64_t at_tensor_numel(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        return t->numel();
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_numel: " << e.what() << std::endl;
        return -1;
    }
}

int32_t at_tensor_dtype(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        return static_cast<int32_t>(t->scalar_type());
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_dtype: " << e.what() << std::endl;
        return -1;
    }
}

// ============================================================================
// Device Management
// ============================================================================

bool at_cuda_is_available() {
    return torch::cuda::is_available();
}

int at_cuda_device_count() {
    return torch::cuda::device_count();
}

void* at_tensor_to_device(void* tensor, int32_t device_type) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        torch::Device device(static_cast<torch::DeviceType>(device_type));
        auto result = t->to(device);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_to_device: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_tensor_to_dtype(void* tensor, int32_t scalar_type) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->to(static_cast<torch::ScalarType>(scalar_type));
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_to_dtype: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_tensor_cpu(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->cpu();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_cpu: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_tensor_cuda(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->cuda();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_cuda: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Tensor Manipulation
// ============================================================================

void* at_reshape(void* tensor, const int64_t* sizes, int64_t ndim) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto result = t->reshape(shape);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_reshape: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_view(void* tensor, const int64_t* sizes, int64_t ndim) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        std::vector<int64_t> shape(sizes, sizes + ndim);
        auto result = t->view(shape);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_view: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_transpose(void* tensor, int64_t dim0, int64_t dim1) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->transpose(dim0, dim1);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_transpose: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_permute(void* tensor, const int64_t* dims, int64_t ndim) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        std::vector<int64_t> dimensions(dims, dims + ndim);
        auto result = t->permute(dimensions);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_permute: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_squeeze(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->squeeze();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_squeeze: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_unsqueeze(void* tensor, int64_t dim) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->unsqueeze(dim);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_unsqueeze: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Memory Management
// ============================================================================

void at_tensor_delete(void* tensor) {
    try {
        if (tensor != nullptr) {
            delete static_cast<torch::Tensor*>(tensor);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_delete: " << e.what() << std::endl;
    }
}

void* at_tensor_clone(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->clone();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_clone: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_tensor_detach(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->detach();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_tensor_detach: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

void* at_sum(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->sum();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_sum: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_mean(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->mean();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_mean: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_max(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->max();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_max: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_min(void* tensor) {
    try {
        auto t = static_cast<torch::Tensor*>(tensor);
        auto result = t->min();
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_min: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Neural Network Operations
// ============================================================================

void* at_linear(void* input, void* weight, void* bias) {
    try {
        auto inp = static_cast<torch::Tensor*>(input);
        auto w = static_cast<torch::Tensor*>(weight);
        torch::Tensor* b = bias ? static_cast<torch::Tensor*>(bias) : nullptr;
        
        auto result = b ? torch::nn::functional::linear(*inp, *w, *b) 
                        : torch::nn::functional::linear(*inp, *w);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_linear: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_conv2d(void* input, void* weight, void* bias, 
                const int64_t* stride, int64_t stride_len,
                const int64_t* padding, int64_t padding_len) {
    try {
        auto inp = static_cast<torch::Tensor*>(input);
        auto w = static_cast<torch::Tensor*>(weight);
        torch::Tensor* b = bias ? static_cast<torch::Tensor*>(bias) : nullptr;
        
        std::vector<int64_t> stride_vec(stride, stride + stride_len);
        std::vector<int64_t> padding_vec(padding, padding + padding_len);
        
        auto options = torch::nn::functional::Conv2dFuncOptions()
            .stride(stride_vec)
            .padding(padding_vec);
        
        if (b) {
            options = options.bias(*b);
        }
        
        auto result = torch::nn::functional::conv2d(*inp, *w, options);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_conv2d: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_dropout(void* input, double p, bool training) {
    try {
        auto inp = static_cast<torch::Tensor*>(input);
        auto result = torch::nn::functional::dropout(*inp, 
            torch::nn::functional::DropoutFuncOptions().p(p).training(training));
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_dropout: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

void* at_mse_loss(void* input, void* target) {
    try {
        auto inp = static_cast<torch::Tensor*>(input);
        auto tgt = static_cast<torch::Tensor*>(target);
        auto result = torch::nn::functional::mse_loss(*inp, *tgt);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_mse_loss: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_cross_entropy_loss(void* input, void* target) {
    try {
        auto inp = static_cast<torch::Tensor*>(input);
        auto tgt = static_cast<torch::Tensor*>(target);
        auto result = torch::nn::functional::cross_entropy(*inp, *tgt);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_cross_entropy_loss: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_nll_loss(void* input, void* target) {
    try {
        auto inp = static_cast<torch::Tensor*>(input);
        auto tgt = static_cast<torch::Tensor*>(target);
        auto result = torch::nn::functional::nll_loss(*inp, *tgt);
        return new torch::Tensor(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in at_nll_loss: " << e.what() << std::endl;
        return nullptr;
    }
}

} // extern "C"
