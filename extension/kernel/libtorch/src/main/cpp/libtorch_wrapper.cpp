#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

extern "C" {

// ── JIT / Module ───────────────────────────────────────────────────

void* at_jit_load(const char* path) {
    try {
        auto module = new torch::jit::Module(torch::jit::load(path));
        return static_cast<void*>(module);
    } catch (const std::exception& e) {
        std::cerr << "at_jit_load error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* at_jit_module_forward(void* module_ptr, void* input_tensor_ptr) {
    try {
        auto module = static_cast<torch::jit::Module*>(module_ptr);
        auto input = *static_cast<at::Tensor*>(input_tensor_ptr);
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        auto output = module->forward(inputs).toTensor();
        return new at::Tensor(output);
    } catch (const std::exception& e) {
        std::cerr << "at_jit_module_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

void at_jit_module_free(void* module_ptr) {
    if (module_ptr) {
        delete static_cast<torch::jit::Module*>(module_ptr);
    }
}

// ── Tensor Management ─────────────────────────────────────────────

void at_free(void* tensor_ptr) {
    if (tensor_ptr) {
        delete static_cast<at::Tensor*>(tensor_ptr);
    }
}

void* at_clone(void* tensor_ptr) {
    auto tensor = static_cast<at::Tensor*>(tensor_ptr);
    return new at::Tensor(tensor->clone());
}

// ── Tensor Creation ───────────────────────────────────────────────

void* at_empty(long* sizes, long ndim, int dtype, int device) {
    std::vector<int64_t> size_vec(sizes, sizes + ndim);
    auto options = torch::TensorOptions()
        .dtype(static_cast<at::ScalarType>(dtype))
        .device(static_cast<at::DeviceType>(device));
    return new at::Tensor(torch::empty(size_vec, options));
}

void* at_zeros(long* sizes, long ndim, int dtype, int device) {
    std::vector<int64_t> size_vec(sizes, sizes + ndim);
    auto options = torch::TensorOptions()
        .dtype(static_cast<at::ScalarType>(dtype))
        .device(static_cast<at::DeviceType>(device));
    return new at::Tensor(torch::zeros(size_vec, options));
}

void* at_ones(long* sizes, long ndim, int dtype, int device) {
    std::vector<int64_t> size_vec(sizes, sizes + ndim);
    auto options = torch::TensorOptions()
        .dtype(static_cast<at::ScalarType>(dtype))
        .device(static_cast<at::DeviceType>(device));
    return new at::Tensor(torch::ones(size_vec, options));
}

void* at_randn(long* sizes, long ndim, int dtype, int device) {
    std::vector<int64_t> size_vec(sizes, sizes + ndim);
    auto options = torch::TensorOptions()
        .dtype(static_cast<at::ScalarType>(dtype))
        .device(static_cast<at::DeviceType>(device));
    return new at::Tensor(torch::randn(size_vec, options));
}

void* at_rand(long* sizes, long ndim, int dtype, int device) {
    std::vector<int64_t> size_vec(sizes, sizes + ndim);
    auto options = torch::TensorOptions()
        .dtype(static_cast<at::ScalarType>(dtype))
        .device(static_cast<at::DeviceType>(device));
    return new at::Tensor(torch::rand(size_vec, options));
}

void* at_from_blob(void* data, long* sizes, long ndim, int dtype) {
    std::vector<int64_t> size_vec(sizes, sizes + ndim);
    auto options = torch::TensorOptions().dtype(static_cast<at::ScalarType>(dtype));
    return new at::Tensor(torch::from_blob(data, size_vec, options).clone());
}

// ── Tensor Properties ─────────────────────────────────────────────

long at_dim(void* tensor_ptr) {
    return static_cast<at::Tensor*>(tensor_ptr)->dim();
}

long at_numel(void* tensor_ptr) {
    return static_cast<at::Tensor*>(tensor_ptr)->numel();
}

long at_size(void* tensor_ptr, long dim) {
    return static_cast<at::Tensor*>(tensor_ptr)->size(dim);
}

int at_scalar_type(void* tensor_ptr) {
    return static_cast<int>(static_cast<at::Tensor*>(tensor_ptr)->scalar_type());
}

void* at_data_ptr(void* tensor_ptr) {
    return static_cast<at::Tensor*>(tensor_ptr)->data_ptr();
}

int at_device_type(void* tensor_ptr) {
    return static_cast<int>(static_cast<at::Tensor*>(tensor_ptr)->device().type());
}

// ── Tensor Operations ─────────────────────────────────────────────

void* at_add(void* a, void* b) {
    return new at::Tensor(static_cast<at::Tensor*>(a)->add(*static_cast<at::Tensor*>(b)));
}

void* at_sub(void* a, void* b) {
    return new at::Tensor(static_cast<at::Tensor*>(a)->sub(*static_cast<at::Tensor*>(b)));
}

void* at_mul(void* a, void* b) {
    return new at::Tensor(static_cast<at::Tensor*>(a)->mul(*static_cast<at::Tensor*>(b)));
}

void* at_div(void* a, void* b) {
    return new at::Tensor(static_cast<at::Tensor*>(a)->div(*static_cast<at::Tensor*>(b)));
}

void* at_matmul(void* a, void* b) {
    return new at::Tensor(static_cast<at::Tensor*>(a)->matmul(*static_cast<at::Tensor*>(b)));
}

void* at_cat(void* tensors_ptr, long count, long dim) {
    auto tensors = static_cast<at::Tensor**>(tensors_ptr);
    std::vector<at::Tensor> tensor_vec;
    for (int i = 0; i < count; i++) {
        tensor_vec.push_back(*tensors[i]);
    }
    return new at::Tensor(at::cat(tensor_vec, dim));
}

void* at_reshape(void* tensor_ptr, long* sizes, long ndim) {
    std::vector<int64_t> size_vec(sizes, sizes + ndim);
    return new at::Tensor(static_cast<at::Tensor*>(tensor_ptr)->reshape(size_vec));
}

void* at_transpose(void* tensor_ptr, long dim0, long dim1) {
    return new at::Tensor(static_cast<at::Tensor*>(tensor_ptr)->transpose(dim0, dim1));
}

// ── Unary Operations ──────────────────────────────────────────────

void* at_neg(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->neg()); }
void* at_abs(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->abs()); }
void* at_sqrt(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->sqrt()); }
void* at_log(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->log()); }
void* at_exp(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->exp()); }

// ── Reductions ────────────────────────────────────────────────────

void* at_sum(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->sum()); }
void* at_mean(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->mean()); }
void* at_max(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->max()); }
void* at_min(void* a) { return new at::Tensor(static_cast<at::Tensor*>(a)->min()); }
void* at_argmax(void* a, long dim) { return new at::Tensor(static_cast<at::Tensor*>(a)->argmax(dim)); }

// ── Comparison ────────────────────────────────────────────────────

void* at_eq(void* a, void* b) { return new at::Tensor(static_cast<at::Tensor*>(a)->eq(*static_cast<at::Tensor*>(b))); }
void* at_gt(void* a, void* b) { return new at::Tensor(static_cast<at::Tensor*>(a)->gt(*static_cast<at::Tensor*>(b))); }
void* at_lt(void* a, void* b) { return new at::Tensor(static_cast<at::Tensor*>(a)->lt(*static_cast<at::Tensor*>(b))); }

// ── Autograd ──────────────────────────────────────────────────────

void at_backward(void* tensor_ptr) {
    static_cast<at::Tensor*>(tensor_ptr)->backward();
}

void* at_grad(void* tensor_ptr) {
    return new at::Tensor(static_cast<at::Tensor*>(tensor_ptr)->grad());
}

void* at_requires_grad_(void* tensor_ptr, bool requires_grad) {
    static_cast<at::Tensor*>(tensor_ptr)->requires_grad_(requires_grad);
    return tensor_ptr;
}

void* at_no_grad_guard() {
    return new torch::NoGradGuard();
}

void at_no_grad_guard_release(void* guard_ptr) {
    if (guard_ptr) {
        delete static_cast<torch::NoGradGuard*>(guard_ptr);
    }
}

// ── NN Functional ─────────────────────────────────────────────────

void* at_relu(void* a) { return new at::Tensor(torch::relu(*static_cast<at::Tensor*>(a))); }
void* at_gelu(void* a) { return new at::Tensor(torch::gelu(*static_cast<at::Tensor*>(a))); }
void* at_sigmoid(void* a) { return new at::Tensor(torch::sigmoid(*static_cast<at::Tensor*>(a))); }
void* at_tanh(void* a) { return new at::Tensor(torch::tanh(*static_cast<at::Tensor*>(a))); }

void* at_softmax(void* a, long dim) {
    return new at::Tensor(torch::softmax(*static_cast<at::Tensor*>(a), dim));
}

void* at_dropout(void* a, double p, bool train) {
    return new at::Tensor(torch::dropout(*static_cast<at::Tensor*>(a), p, train));
}

void* at_conv2d(void* input, void* weight, void* bias, 
              long stride_h, long stride_w, 
              long padding_h, long padding_w, 
              long dilation) {
    return new at::Tensor(torch::conv2d(
        *static_cast<at::Tensor*>(input),
        *static_cast<at::Tensor*>(weight),
        bias ? *static_cast<at::Tensor*>(bias) : at::Tensor(),
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation, dilation}
    ));
}

void* at_batch_norm(void* input, void* weight, void* bias,
                  void* running_mean, void* running_var,
                  bool training, double momentum, double eps) {
    return new at::Tensor(torch::batch_norm(
        *static_cast<at::Tensor*>(input),
        weight ? *static_cast<at::Tensor*>(weight) : at::Tensor(),
        bias ? *static_cast<at::Tensor*>(bias) : at::Tensor(),
        running_mean ? *static_cast<at::Tensor*>(running_mean) : at::Tensor(),
        running_var ? *static_cast<at::Tensor*>(running_var) : at::Tensor(),
        training, momentum, eps, true
    ));
}

void* at_layer_norm(void* input, long* normalized_shape, long shape_len,
                  void* weight, void* bias, double eps) {
    std::vector<int64_t> shape(normalized_shape, normalized_shape + shape_len);
    return new at::Tensor(torch::layer_norm(
        *static_cast<at::Tensor*>(input),
        shape,
        weight ? *static_cast<at::Tensor*>(weight) : at::Tensor(),
        bias ? *static_cast<at::Tensor*>(bias) : at::Tensor(),
        eps
    ));
}

void* at_embedding(void* weight, void* indices, long padding_idx, bool scale_grad_by_freq, bool sparse) {
    return new at::Tensor(torch::embedding(
        *static_cast<at::Tensor*>(weight),
        *static_cast<at::Tensor*>(indices),
        padding_idx, scale_grad_by_freq, sparse
    ));
}

void* at_index_select(void* input, long dim, void* index) {
    return new at::Tensor(static_cast<at::Tensor*>(input)->index_select(dim, *static_cast<at::Tensor*>(index)));
}

// ── Loss ──────────────────────────────────────────────────────────

void* at_mse_loss(void* input, void* target) {
    return new at::Tensor(torch::mse_loss(*static_cast<at::Tensor*>(input), *static_cast<at::Tensor*>(target)));
}

void* at_cross_entropy_loss(void* input, void* target) {
    return new at::Tensor(torch::cross_entropy_loss(*static_cast<at::Tensor*>(input), *static_cast<at::Tensor*>(target)));
}

void* at_binary_cross_entropy(void* input, void* target) {
    return new at::Tensor(torch::binary_cross_entropy(*static_cast<at::Tensor*>(input), *static_cast<at::Tensor*>(target)));
}

// ── Optimizer Hooks ───────────────────────────────────────────────

void at_sgd_step(void* param, void* grad, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
    auto p = static_cast<at::Tensor*>(param);
    auto g = static_cast<at::Tensor*>(grad);
    // Simple SGD implementation
    torch::NoGradGuard guard;
    if (weight_decay != 0) {
        g->add_(*p, weight_decay);
    }
    p->sub_(*g, lr);
}

void at_adam_step(void* param, void* grad, void* exp_avg, void* exp_avg_sq, long step, double lr, double beta1, double beta2, double eps, bool amsgrad) {
    // Basic Adam step
    torch::NoGradGuard guard;
    auto p = static_cast<at::Tensor*>(param);
    auto g = static_cast<at::Tensor*>(grad);
    auto m = static_cast<at::Tensor*>(exp_avg);
    auto v = static_cast<at::Tensor*>(exp_avg_sq);

    m->mul_(beta1).add_(*g, 1 - beta1);
    v->mul_(beta2).addcmul_(*g, *g, 1 - beta2);

    double bias_correction1 = 1 - std::pow(beta1, step);
    double bias_correction2 = 1 - std::pow(beta2, step);
    double step_size = lr / bias_correction1;

    p->addcdiv_(*m, v->sqrt().add_(eps * std::sqrt(bias_correction2)), -step_size);
}

// ── Persistence ───────────────────────────────────────────────────

void at_save(void* tensor_ptr, const char* path) {
    torch::save(*static_cast<at::Tensor*>(tensor_ptr), path);
}

void* at_load(const char* path) {
    at::Tensor tensor;
    torch::load(tensor, path);
    return new at::Tensor(tensor);
}

// ── CUDA ──────────────────────────────────────────────────────────

bool at_cuda_is_available() { return torch::cuda::is_available(); }
int at_cuda_device_count() { return torch::cuda::device_count(); }

void* at_to_device(void* tensor_ptr, int device_type, int device_index) {
    auto tensor = static_cast<at::Tensor*>(tensor_ptr);
    return new at::Tensor(tensor->to(torch::Device(static_cast<torch::DeviceType>(device_type), static_cast<int16_t>(device_index))));
}

} // extern "C"
