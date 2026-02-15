package io.github.pytorch.ffm;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Low-level FFM bindings to LibTorch C++ API
 * Uses JDK 25 Foreign Function & Memory API
 */
public class LibTorchFFM {
    
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBTORCH;
    
    // Memory layouts
    public static final AddressLayout C_POINTER = ValueLayout.ADDRESS
            .withTargetLayout(MemoryLayout.sequenceLayout(Long.MAX_VALUE, ValueLayout.JAVA_BYTE));
    
    static {
        // Load LibTorch native library
        System.loadLibrary("torch");
        System.loadLibrary("torch_cpu");
        System.loadLibrary("c10");
        
        LIBTORCH = SymbolLookup.loaderLookup();
    }
    
    // ============================================================================
    // Tensor Creation Functions
    // ============================================================================
    
    /**
     * at::Tensor at::empty({sizes}, options)
     */
    public static final MethodHandle AT_EMPTY;
    
    /**
     * at::Tensor at::zeros({sizes}, options)
     */
    public static final MethodHandle AT_ZEROS;
    
    /**
     * at::Tensor at::ones({sizes}, options)
     */
    public static final MethodHandle AT_ONES;
    
    /**
     * at::Tensor at::randn({sizes}, options)
     */
    public static final MethodHandle AT_RANDN;
    
    /**
     * at::Tensor at::rand({sizes}, options)
     */
    public static final MethodHandle AT_RAND;
    
    /**
     * at::Tensor at::arange(start, end, step, options)
     */
    public static final MethodHandle AT_ARANGE;
    
    /**
     * at::Tensor at::linspace(start, end, steps, options)
     */
    public static final MethodHandle AT_LINSPACE;
    
    /**
     * at::Tensor at::from_blob(data, sizes, strides, deleter, options)
     */
    public static final MethodHandle AT_FROM_BLOB;
    
    // ============================================================================
    // Tensor Operations
    // ============================================================================
    
    /**
     * at::Tensor at::Tensor::add(other, alpha)
     */
    public static final MethodHandle TENSOR_ADD;
    
    /**
     * at::Tensor at::Tensor::sub(other, alpha)
     */
    public static final MethodHandle TENSOR_SUB;
    
    /**
     * at::Tensor at::Tensor::mul(other)
     */
    public static final MethodHandle TENSOR_MUL;
    
    /**
     * at::Tensor at::Tensor::div(other)
     */
    public static final MethodHandle TENSOR_DIV;
    
    /**
     * at::Tensor at::Tensor::matmul(other)
     */
    public static final MethodHandle TENSOR_MATMUL;
    
    /**
     * at::Tensor at::Tensor::mm(other)
     */
    public static final MethodHandle TENSOR_MM;
    
    /**
     * at::Tensor at::Tensor::bmm(other)
     */
    public static final MethodHandle TENSOR_BMM;
    
    /**
     * at::Tensor at::Tensor::reshape(shape)
     */
    public static final MethodHandle TENSOR_RESHAPE;
    
    /**
     * at::Tensor at::Tensor::view(shape)
     */
    public static final MethodHandle TENSOR_VIEW;
    
    /**
     * at::Tensor at::Tensor::transpose(dim0, dim1)
     */
    public static final MethodHandle TENSOR_TRANSPOSE;
    
    /**
     * at::Tensor at::Tensor::permute(dims)
     */
    public static final MethodHandle TENSOR_PERMUTE;
    
    /**
     * at::Tensor at::Tensor::squeeze(dim)
     */
    public static final MethodHandle TENSOR_SQUEEZE;
    
    /**
     * at::Tensor at::Tensor::unsqueeze(dim)
     */
    public static final MethodHandle TENSOR_UNSQUEEZE;
    
    /**
     * at::Tensor at::Tensor::sum(dim, keepdim)
     */
    public static final MethodHandle TENSOR_SUM;
    
    /**
     * at::Tensor at::Tensor::mean(dim, keepdim)
     */
    public static final MethodHandle TENSOR_MEAN;
    
    /**
     * at::Tensor at::Tensor::max(dim, keepdim)
     */
    public static final MethodHandle TENSOR_MAX;
    
    /**
     * at::Tensor at::Tensor::min(dim, keepdim)
     */
    public static final MethodHandle TENSOR_MIN;
    
    // ============================================================================
    // Tensor Properties
    // ============================================================================
    
    /**
     * int64_t* at::Tensor::sizes()
     */
    public static final MethodHandle TENSOR_SIZES;
    
    /**
     * int64_t at::Tensor::size(dim)
     */
    public static final MethodHandle TENSOR_SIZE;
    
    /**
     * int64_t at::Tensor::dim()
     */
    public static final MethodHandle TENSOR_DIM;
    
    /**
     * int64_t at::Tensor::numel()
     */
    public static final MethodHandle TENSOR_NUMEL;
    
    /**
     * void* at::Tensor::data_ptr()
     */
    public static final MethodHandle TENSOR_DATA_PTR;
    
    /**
     * at::ScalarType at::Tensor::scalar_type()
     */
    public static final MethodHandle TENSOR_SCALAR_TYPE;
    
    /**
     * at::Device at::Tensor::device()
     */
    public static final MethodHandle TENSOR_DEVICE;
    
    /**
     * bool at::Tensor::requires_grad()
     */
    public static final MethodHandle TENSOR_REQUIRES_GRAD;
    
    /**
     * at::Tensor at::Tensor::to(device, dtype, non_blocking, copy)
     */
    public static final MethodHandle TENSOR_TO;
    
    /**
     * at::Tensor at::Tensor::cuda()
     */
    public static final MethodHandle TENSOR_CUDA;
    
    /**
     * at::Tensor at::Tensor::cpu()
     */
    public static final MethodHandle TENSOR_CPU;
    
    /**
     * at::Tensor at::Tensor::clone()
     */
    public static final MethodHandle TENSOR_CLONE;
    
    /**
     * at::Tensor at::Tensor::detach()
     */
    public static final MethodHandle TENSOR_DETACH;
    
    // ============================================================================
    // Autograd Functions
    // ============================================================================
    
    /**
     * void at::Tensor::backward(gradient, retain_graph, create_graph)
     */
    public static final MethodHandle TENSOR_BACKWARD;
    
    /**
     * at::Tensor at::Tensor::grad()
     */
    public static final MethodHandle TENSOR_GRAD;
    
    /**
     * void at::Tensor::zero_grad()
     */
    public static final MethodHandle TENSOR_ZERO_GRAD;
    
    /**
     * at::Tensor at::Tensor::requires_grad_(bool)
     */
    public static final MethodHandle TENSOR_REQUIRES_GRAD_;
    
    // ============================================================================
    // Neural Network Functions (torch::nn::functional)
    // ============================================================================
    
    /**
     * at::Tensor torch::nn::functional::linear(input, weight, bias)
     */
    public static final MethodHandle NN_LINEAR;
    
    /**
     * at::Tensor torch::nn::functional::conv1d(input, weight, bias, stride, padding, dilation, groups)
     */
    public static final MethodHandle NN_CONV1D;
    
    /**
     * at::Tensor torch::nn::functional::conv2d(input, weight, bias, stride, padding, dilation, groups)
     */
    public static final MethodHandle NN_CONV2D;
    
    /**
     * at::Tensor torch::nn::functional::conv3d(input, weight, bias, stride, padding, dilation, groups)
     */
    public static final MethodHandle NN_CONV3D;
    
    /**
     * at::Tensor torch::nn::functional::max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
     */
    public static final MethodHandle NN_MAX_POOL2D;
    
    /**
     * at::Tensor torch::nn::functional::avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
     */
    public static final MethodHandle NN_AVG_POOL2D;
    
    /**
     * at::Tensor torch::nn::functional::relu(input, inplace)
     */
    public static final MethodHandle NN_RELU;
    
    /**
     * at::Tensor torch::nn::functional::gelu(input)
     */
    public static final MethodHandle NN_GELU;
    
    /**
     * at::Tensor torch::nn::functional::dropout(input, p, training)
     */
    public static final MethodHandle NN_DROPOUT;
    
    /**
     * at::Tensor torch::nn::functional::batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
     */
    public static final MethodHandle NN_BATCH_NORM;
    
    /**
     * at::Tensor torch::nn::functional::layer_norm(input, normalized_shape, weight, bias, eps)
     */
    public static final MethodHandle NN_LAYER_NORM;
    
    /**
     * at::Tensor torch::nn::functional::softmax(input, dim)
     */
    public static final MethodHandle NN_SOFTMAX;
    
    /**
     * at::Tensor torch::nn::functional::log_softmax(input, dim)
     */
    public static final MethodHandle NN_LOG_SOFTMAX;
    
    /**
     * at::Tensor torch::nn::functional::cross_entropy(input, target, weight, reduction)
     */
    public static final MethodHandle NN_CROSS_ENTROPY;
    
    /**
     * at::Tensor torch::nn::functional::mse_loss(input, target, reduction)
     */
    public static final MethodHandle NN_MSE_LOSS;
    
    /**
     * at::Tensor torch::nn::functional::binary_cross_entropy(input, target, weight, reduction)
     */
    public static final MethodHandle NN_BINARY_CROSS_ENTROPY;
    
    // ============================================================================
    // Optimizer Functions
    // ============================================================================
    
    /**
     * torch::optim::SGD* torch::optim::SGD::new(parameters, lr, momentum, weight_decay, dampening, nesterov)
     */
    public static final MethodHandle OPTIM_SGD_NEW;
    
    /**
     * void torch::optim::SGD::step()
     */
    public static final MethodHandle OPTIM_SGD_STEP;
    
    /**
     * void torch::optim::SGD::zero_grad()
     */
    public static final MethodHandle OPTIM_SGD_ZERO_GRAD;
    
    /**
     * torch::optim::Adam* torch::optim::Adam::new(parameters, lr, betas, eps, weight_decay, amsgrad)
     */
    public static final MethodHandle OPTIM_ADAM_NEW;
    
    /**
     * void torch::optim::Adam::step()
     */
    public static final MethodHandle OPTIM_ADAM_STEP;
    
    /**
     * void torch::optim::Adam::zero_grad()
     */
    public static final MethodHandle OPTIM_ADAM_ZERO_GRAD;
    
    // ============================================================================
    // Module Functions
    // ============================================================================
    
    /**
     * torch::nn::Module* torch::nn::Module::new()
     */
    public static final MethodHandle MODULE_NEW;
    
    /**
     * void torch::nn::Module::register_parameter(name, tensor)
     */
    public static final MethodHandle MODULE_REGISTER_PARAMETER;
    
    /**
     * void torch::nn::Module::register_buffer(name, tensor)
     */
    public static final MethodHandle MODULE_REGISTER_BUFFER;
    
    /**
     * void torch::nn::Module::register_module(name, module)
     */
    public static final MethodHandle MODULE_REGISTER_MODULE;
    
    /**
     * std::vector<at::Tensor> torch::nn::Module::parameters()
     */
    public static final MethodHandle MODULE_PARAMETERS;
    
    /**
     * void torch::nn::Module::train(mode)
     */
    public static final MethodHandle MODULE_TRAIN;
    
    /**
     * void torch::nn::Module::eval()
     */
    public static final MethodHandle MODULE_EVAL;
    
    /**
     * void torch::nn::Module::to(device, dtype, non_blocking)
     */
    public static final MethodHandle MODULE_TO;
    
    /**
     * void torch::nn::Module::zero_grad()
     */
    public static final MethodHandle MODULE_ZERO_GRAD;
    
    // ============================================================================
    // Serialization Functions
    // ============================================================================
    
    /**
     * void torch::save(tensor, path)
     */
    public static final MethodHandle TORCH_SAVE;
    
    /**
     * at::Tensor torch::load(path)
     */
    public static final MethodHandle TORCH_LOAD;
    
    /**
     * void torch::jit::save(module, path)
     */
    public static final MethodHandle JIT_SAVE;
    
    /**
     * torch::jit::Module torch::jit::load(path)
     */
    public static final MethodHandle JIT_LOAD;
    
    // ============================================================================
    // CUDA Functions
    // ============================================================================
    
    /**
     * bool torch::cuda::is_available()
     */
    public static final MethodHandle CUDA_IS_AVAILABLE;
    
    /**
     * int torch::cuda::device_count()
     */
    public static final MethodHandle CUDA_DEVICE_COUNT;
    
    /**
     * void torch::cuda::set_device(device)
     */
    public static final MethodHandle CUDA_SET_DEVICE;
    
    /**
     * void torch::cuda::synchronize()
     */
    public static final MethodHandle CUDA_SYNCHRONIZE;
    
    /**
     * void torch::cuda::empty_cache()
     */
    public static final MethodHandle CUDA_EMPTY_CACHE;
    
    // Static initializer to bind all function handles
    static {
        try {
            // Function descriptor for common signatures
            FunctionDescriptor voidDesc = FunctionDescriptor.ofVoid();
            FunctionDescriptor pointerDesc = FunctionDescriptor.of(C_POINTER);
            FunctionDescriptor intDesc = FunctionDescriptor.of(ValueLayout.JAVA_INT);
            FunctionDescriptor longDesc = FunctionDescriptor.of(ValueLayout.JAVA_LONG);
            FunctionDescriptor boolDesc = FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN);
            
            // Tensor Creation
            AT_EMPTY = bindFunction("at_empty", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, C_POINTER));
            AT_ZEROS = bindFunction("at_zeros", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, C_POINTER));
            AT_ONES = bindFunction("at_ones", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, C_POINTER));
            AT_RANDN = bindFunction("at_randn", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, C_POINTER));
            AT_RAND = bindFunction("at_rand", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, C_POINTER));
            AT_ARANGE = bindFunction("at_arange", 
                FunctionDescriptor.of(C_POINTER, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, 
                    ValueLayout.JAVA_DOUBLE, C_POINTER));
            AT_LINSPACE = bindFunction("at_linspace", 
                FunctionDescriptor.of(C_POINTER, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, 
                    ValueLayout.JAVA_LONG, C_POINTER));
            AT_FROM_BLOB = bindFunction("at_from_blob", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT, C_POINTER));
            
            // Tensor Operations
            TENSOR_ADD = bindFunction("tensor_add", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_DOUBLE));
            TENSOR_SUB = bindFunction("tensor_sub", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_DOUBLE));
            TENSOR_MUL = bindFunction("tensor_mul", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER));
            TENSOR_DIV = bindFunction("tensor_div", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER));
            TENSOR_MATMUL = bindFunction("tensor_matmul", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER));
            TENSOR_MM = bindFunction("tensor_mm", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER));
            TENSOR_BMM = bindFunction("tensor_bmm", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER));
            TENSOR_RESHAPE = bindFunction("tensor_reshape", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            TENSOR_VIEW = bindFunction("tensor_view", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            TENSOR_TRANSPOSE = bindFunction("tensor_transpose", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            TENSOR_PERMUTE = bindFunction("tensor_permute", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            TENSOR_SQUEEZE = bindFunction("tensor_squeeze", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            TENSOR_UNSQUEEZE = bindFunction("tensor_unsqueeze", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            TENSOR_SUM = bindFunction("tensor_sum", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN));
            TENSOR_MEAN = bindFunction("tensor_mean", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN));
            TENSOR_MAX = bindFunction("tensor_max", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN));
            TENSOR_MIN = bindFunction("tensor_min", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN));
            
            // Tensor Properties
            TENSOR_SIZES = bindFunction("tensor_sizes", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            TENSOR_SIZE = bindFunction("tensor_size", 
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, C_POINTER, ValueLayout.JAVA_INT));
            TENSOR_DIM = bindFunction("tensor_dim", 
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, C_POINTER));
            TENSOR_NUMEL = bindFunction("tensor_numel", 
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, C_POINTER));
            TENSOR_DATA_PTR = bindFunction("tensor_data_ptr", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            TENSOR_SCALAR_TYPE = bindFunction("tensor_scalar_type", 
                FunctionDescriptor.of(ValueLayout.JAVA_INT, C_POINTER));
            TENSOR_DEVICE = bindFunction("tensor_device", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            TENSOR_REQUIRES_GRAD = bindFunction("tensor_requires_grad", 
                FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, C_POINTER));
            TENSOR_TO = bindFunction("tensor_to", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT, 
                    ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN));
            TENSOR_CUDA = bindFunction("tensor_cuda", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            TENSOR_CPU = bindFunction("tensor_cpu", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            TENSOR_CLONE = bindFunction("tensor_clone", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            TENSOR_DETACH = bindFunction("tensor_detach", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            
            // Autograd
            TENSOR_BACKWARD = bindFunction("tensor_backward", 
                FunctionDescriptor.ofVoid(C_POINTER, C_POINTER, ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN));
            TENSOR_GRAD = bindFunction("tensor_grad", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            TENSOR_ZERO_GRAD = bindFunction("tensor_zero_grad", 
                FunctionDescriptor.ofVoid(C_POINTER));
            TENSOR_REQUIRES_GRAD_ = bindFunction("tensor_requires_grad_", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_BOOLEAN));
            
            // Neural Network Functions
            NN_LINEAR = bindFunction("nn_functional_linear", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER));
            NN_CONV2D = bindFunction("nn_functional_conv2d", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER, C_POINTER, 
                    C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            NN_MAX_POOL2D = bindFunction("nn_functional_max_pool2d", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER, C_POINTER, 
                    C_POINTER, ValueLayout.JAVA_BOOLEAN));
            NN_AVG_POOL2D = bindFunction("nn_functional_avg_pool2d", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER, C_POINTER, 
                    ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN));
            NN_RELU = bindFunction("nn_functional_relu", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_BOOLEAN));
            NN_GELU = bindFunction("nn_functional_gelu", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            NN_DROPOUT = bindFunction("nn_functional_dropout", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_BOOLEAN));
            NN_BATCH_NORM = bindFunction("nn_functional_batch_norm", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER, C_POINTER, C_POINTER, 
                    ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE));
            NN_LAYER_NORM = bindFunction("nn_functional_layer_norm", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_DOUBLE));
            NN_SOFTMAX = bindFunction("nn_functional_softmax", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            NN_LOG_SOFTMAX = bindFunction("nn_functional_log_softmax", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            NN_CROSS_ENTROPY = bindFunction("nn_functional_cross_entropy", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            NN_MSE_LOSS = bindFunction("nn_functional_mse_loss", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            NN_BINARY_CROSS_ENTROPY = bindFunction("nn_functional_binary_cross_entropy", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER, C_POINTER, ValueLayout.JAVA_INT));
            
            // Optimizers
            OPTIM_SGD_NEW = bindFunction("optim_sgd_new", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE, 
                    ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_BOOLEAN));
            OPTIM_SGD_STEP = bindFunction("optim_sgd_step", 
                FunctionDescriptor.ofVoid(C_POINTER));
            OPTIM_SGD_ZERO_GRAD = bindFunction("optim_sgd_zero_grad", 
                FunctionDescriptor.ofVoid(C_POINTER));
            OPTIM_ADAM_NEW = bindFunction("optim_adam_new", 
                FunctionDescriptor.of(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE, 
                    ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, 
                    ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_BOOLEAN));
            OPTIM_ADAM_STEP = bindFunction("optim_adam_step", 
                FunctionDescriptor.ofVoid(C_POINTER));
            OPTIM_ADAM_ZERO_GRAD = bindFunction("optim_adam_zero_grad", 
                FunctionDescriptor.ofVoid(C_POINTER));
            
            // Module
            MODULE_NEW = bindFunction("module_new", 
                FunctionDescriptor.of(C_POINTER));
            MODULE_REGISTER_PARAMETER = bindFunction("module_register_parameter", 
                FunctionDescriptor.ofVoid(C_POINTER, C_POINTER, C_POINTER));
            MODULE_REGISTER_BUFFER = bindFunction("module_register_buffer", 
                FunctionDescriptor.ofVoid(C_POINTER, C_POINTER, C_POINTER));
            MODULE_REGISTER_MODULE = bindFunction("module_register_module", 
                FunctionDescriptor.ofVoid(C_POINTER, C_POINTER, C_POINTER));
            MODULE_PARAMETERS = bindFunction("module_parameters", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            MODULE_TRAIN = bindFunction("module_train", 
                FunctionDescriptor.ofVoid(C_POINTER, ValueLayout.JAVA_BOOLEAN));
            MODULE_EVAL = bindFunction("module_eval", 
                FunctionDescriptor.ofVoid(C_POINTER));
            MODULE_TO = bindFunction("module_to", 
                FunctionDescriptor.ofVoid(C_POINTER, C_POINTER, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN));
            MODULE_ZERO_GRAD = bindFunction("module_zero_grad", 
                FunctionDescriptor.ofVoid(C_POINTER));
            
            // Serialization
            TORCH_SAVE = bindFunction("torch_save", 
                FunctionDescriptor.ofVoid(C_POINTER, C_POINTER));
            TORCH_LOAD = bindFunction("torch_load", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            JIT_SAVE = bindFunction("jit_save", 
                FunctionDescriptor.ofVoid(C_POINTER, C_POINTER));
            JIT_LOAD = bindFunction("jit_load", 
                FunctionDescriptor.of(C_POINTER, C_POINTER));
            
            // CUDA
            CUDA_IS_AVAILABLE = bindFunction("cuda_is_available", 
                FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN));
            CUDA_DEVICE_COUNT = bindFunction("cuda_device_count", 
                FunctionDescriptor.of(ValueLayout.JAVA_INT));
            CUDA_SET_DEVICE = bindFunction("cuda_set_device", 
                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
            CUDA_SYNCHRONIZE = bindFunction("cuda_synchronize", 
                FunctionDescriptor.ofVoid());
            CUDA_EMPTY_CACHE = bindFunction("cuda_empty_cache", 
                FunctionDescriptor.ofVoid());
                
        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }
    
    private static MethodHandle bindFunction(String name, FunctionDescriptor descriptor) {
        try {
            MemorySegment symbol = LIBTORCH.find(name)
                .orElseThrow(() -> new UnsatisfiedLinkError("Symbol not found: " + name));
            return LINKER.downcallHandle(symbol, descriptor);
        } catch (Exception e) {
            System.err.println("Warning: Could not bind function: " + name);
            return null; // Return null for optional functions
        }
    }
    
    /**
     * Utility method to allocate native memory using Arena
     */
    public static MemorySegment allocate(long size, Arena arena) {
        return arena.allocate(size, 8); // 8-byte alignment
    }
    
    /**
     * Utility method to create a string in native memory
     */
    public static MemorySegment allocateString(String str, Arena arena) {
        return arena.allocateFrom(str);
    }
    
    /**
     * Utility method to read a string from native memory
     */
    public static String getString(MemorySegment segment) {
        return segment.getString(0);
    }
}
