#ifndef PYTORCH_JAVA_FFM_H
#define PYTORCH_JAVA_FFM_H

#include <torch/torch.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Tensor Creation Functions
// ============================================================================

void* at_empty(int64_t* sizes, int ndim, void* options);
void* at_zeros(int64_t* sizes, int ndim, void* options);
void* at_ones(int64_t* sizes, int ndim, void* options);
void* at_randn(int64_t* sizes, int ndim, void* options);
void* at_rand(int64_t* sizes, int ndim, void* options);
void* at_arange(double start, double end, double step, void* options);
void* at_linspace(double start, double end, int64_t steps, void* options);
void* at_from_blob(void* data, int64_t* sizes, int ndim, void* options);

// ============================================================================
// Tensor Operations
// ============================================================================

void* tensor_add(void* self, void* other, double alpha);
void* tensor_sub(void* self, void* other, double alpha);
void* tensor_mul(void* self, void* other);
void* tensor_div(void* self, void* other);
void* tensor_matmul(void* self, void* other);
void* tensor_mm(void* self, void* other);
void* tensor_bmm(void* self, void* other);
void* tensor_reshape(void* self, int64_t* shape, int ndim);
void* tensor_view(void* self, int64_t* shape, int ndim);
void* tensor_transpose(void* self, int dim0, int dim1);
void* tensor_permute(void* self, int* dims, int ndim);
void* tensor_squeeze(void* self, int dim);
void* tensor_unsqueeze(void* self, int dim);
void* tensor_sum(void* self, int* dims, int ndim, bool keepdim);
void* tensor_mean(void* self, int* dims, int ndim, bool keepdim);
void* tensor_max(void* self, int dim, bool keepdim);
void* tensor_min(void* self, int dim, bool keepdim);

// ============================================================================
// Tensor Properties
// ============================================================================

int64_t* tensor_sizes(void* self);
int64_t tensor_size(void* self, int dim);
int64_t tensor_dim(void* self);
int64_t tensor_numel(void* self);
void* tensor_data_ptr(void* self);
int tensor_scalar_type(void* self);
void* tensor_device(void* self);
bool tensor_requires_grad(void* self);
void* tensor_to(void* self, void* device, int dtype, bool non_blocking, bool copy);
void* tensor_cuda(void* self);
void* tensor_cpu(void* self);
void* tensor_clone(void* self);
void* tensor_detach(void* self);

// ============================================================================
// Autograd Functions
// ============================================================================

void tensor_backward(void* self, void* gradient, bool retain_graph, bool create_graph);
void* tensor_grad(void* self);
void tensor_zero_grad(void* self);
void* tensor_requires_grad_(void* self, bool requires_grad);

// ============================================================================
// Neural Network Functions
// ============================================================================

void* nn_functional_linear(void* input, void* weight, void* bias);
void* nn_functional_conv1d(void* input, void* weight, void* bias, 
                           int64_t* stride, int64_t* padding, int64_t* dilation, int64_t groups);
void* nn_functional_conv2d(void* input, void* weight, void* bias, 
                           int64_t* stride, int64_t* padding, int64_t* dilation, int64_t groups);
void* nn_functional_conv3d(void* input, void* weight, void* bias, 
                           int64_t* stride, int64_t* padding, int64_t* dilation, int64_t groups);
void* nn_functional_max_pool2d(void* input, int64_t* kernel_size, int64_t* stride, 
                               int64_t* padding, int64_t* dilation, bool ceil_mode);
void* nn_functional_avg_pool2d(void* input, int64_t* kernel_size, int64_t* stride, 
                               int64_t* padding, bool ceil_mode, bool count_include_pad);
void* nn_functional_relu(void* input, bool inplace);
void* nn_functional_gelu(void* input);
void* nn_functional_dropout(void* input, double p, bool training);
void* nn_functional_batch_norm(void* input, void* running_mean, void* running_var, 
                                void* weight, void* bias, bool training, double momentum, double eps);
void* nn_functional_layer_norm(void* input, int64_t* normalized_shape, 
                                void* weight, void* bias, double eps);
void* nn_functional_softmax(void* input, int dim);
void* nn_functional_log_softmax(void* input, int dim);
void* nn_functional_cross_entropy(void* input, void* target, void* weight, int reduction);
void* nn_functional_mse_loss(void* input, void* target, int reduction);
void* nn_functional_binary_cross_entropy(void* input, void* target, void* weight, int reduction);

// ============================================================================
// Optimizer Functions
// ============================================================================

void* optim_sgd_new(void** parameters, int num_params, double lr, 
                    double momentum, double weight_decay, double dampening, bool nesterov);
void optim_sgd_step(void* optimizer);
void optim_sgd_zero_grad(void* optimizer);

void* optim_adam_new(void** parameters, int num_params, double lr, 
                     double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
void optim_adam_step(void* optimizer);
void optim_adam_zero_grad(void* optimizer);

// ============================================================================
// Module Functions
// ============================================================================

void* module_new();
void module_register_parameter(void* module, const char* name, void* tensor);
void module_register_buffer(void* module, const char* name, void* tensor);
void module_register_module(void* module, const char* name, void* submodule);
void** module_parameters(void* module);
void module_train(void* module, bool mode);
void module_eval(void* module);
void module_to(void* module, void* device, int dtype, bool non_blocking);
void module_zero_grad(void* module);

// ============================================================================
// Serialization Functions
// ============================================================================

void torch_save(void* tensor, const char* path);
void* torch_load(const char* path);
void jit_save(void* module, const char* path);
void* jit_load(const char* path);

// ============================================================================
// CUDA Functions
// ============================================================================

bool cuda_is_available();
int cuda_device_count();
void cuda_set_device(int device);
void cuda_synchronize();
void cuda_empty_cache();

// ============================================================================
// Utility Functions
// ============================================================================

void tensor_destroy(void* tensor);
void module_destroy(void* module);
void optimizer_destroy(void* optimizer);

#ifdef __cplusplus
}
#endif

#endif // PYTORCH_JAVA_FFM_H
