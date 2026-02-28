package tech.kayys.gollek.inference.libtorch.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Low-level FFM binding layer to the LibTorch C++ API.
 * <p>
 * Uses JDK 25 Foreign Function &amp; Memory API to bind to native LibTorch
 * functions.
 * All bindings are lazily resolved: a missing symbol returns
 * {@code Optional.empty()}
 * instead of crashing, enabling graceful degradation when only a subset of
 * LibTorch is available.
 * <p>
 * Thread-safe singleton — acquire via {@link #getInstance()}.
 */
public final class LibTorchBinding {

        private static final Logger log = Logger.getLogger(LibTorchBinding.class);
        private static final Linker LINKER = Linker.nativeLinker();

        // Concatenation support
        public static final String TENSOR_CAT = "at_cat";
        public static final FunctionDescriptor TENSOR_CAT_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG);

        private static volatile LibTorchBinding instance;

        private final SymbolLookup lookup;
        private final ConcurrentMap<String, Optional<MethodHandle>> handleCache = new ConcurrentHashMap<>();

        // ── Tensor creation ───────────────────────────────────────────────

        /** at::empty(IntArrayRef size, TensorOptions options) → Tensor */
        public static final String TENSOR_EMPTY = "at_empty";
        public static final FunctionDescriptor TENSOR_EMPTY_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT);

        /** at::zeros(IntArrayRef size, TensorOptions options) → Tensor */
        public static final String TENSOR_ZEROS = "at_zeros";
        public static final FunctionDescriptor TENSOR_ZEROS_DESC = TENSOR_EMPTY_DESC;

        /** at::ones(IntArrayRef size, TensorOptions options) → Tensor */
        public static final String TENSOR_ONES = "at_ones";
        public static final FunctionDescriptor TENSOR_ONES_DESC = TENSOR_EMPTY_DESC;

        /** at::randn(IntArrayRef size, TensorOptions options) → Tensor */
        public static final String TENSOR_RANDN = "at_randn";
        public static final FunctionDescriptor TENSOR_RANDN_DESC = TENSOR_EMPTY_DESC;

        /** at::rand(IntArrayRef size, TensorOptions options) → Tensor */
        public static final String TENSOR_RAND = "at_rand";
        public static final FunctionDescriptor TENSOR_RAND_DESC = TENSOR_EMPTY_DESC;

        /** at::from_blob(void* data, IntArrayRef size, ScalarType dtype) → Tensor */
        public static final String TENSOR_FROM_BLOB = "at_from_blob";
        public static final FunctionDescriptor TENSOR_FROM_BLOB_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_INT);

        // ── Tensor properties ─────────────────────────────────────────────

        /** tensor.dim() → int64_t */
        public static final String TENSOR_DIM = "at_dim";
        public static final FunctionDescriptor TENSOR_DIM_DESC = FunctionDescriptor.of(
                        ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

        /** tensor.numel() → int64_t */
        public static final String TENSOR_NUMEL = "at_numel";
        public static final FunctionDescriptor TENSOR_NUMEL_DESC = TENSOR_DIM_DESC;

        /** tensor.size(int dim) → int64_t */
        public static final String TENSOR_SIZE = "at_size";
        public static final FunctionDescriptor TENSOR_SIZE_DESC = FunctionDescriptor.of(
                        ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

        /** tensor.scalar_type() → int */
        public static final String TENSOR_DTYPE = "at_scalar_type";
        public static final FunctionDescriptor TENSOR_DTYPE_DESC = FunctionDescriptor.of(
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

        /** tensor.data_ptr() → void* */
        public static final String TENSOR_DATA_PTR = "at_data_ptr";
        public static final FunctionDescriptor TENSOR_DATA_PTR_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS);

        /** tensor.device_type() → int */
        public static final String TENSOR_DEVICE_TYPE = "at_device_type";
        public static final FunctionDescriptor TENSOR_DEVICE_TYPE_DESC = FunctionDescriptor.of(
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

        // ── Tensor operations ─────────────────────────────────────────────

        /** tensor.add(other) → Tensor */
        public static final String TENSOR_ADD = "at_add";
        public static final FunctionDescriptor TENSOR_BINARY_OP_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

        public static final String TENSOR_SUB = "at_sub";
        public static final String TENSOR_MUL = "at_mul";
        public static final String TENSOR_DIV = "at_div";
        public static final String TENSOR_MATMUL = "at_matmul";

        /** tensor.reshape(IntArrayRef shape) → Tensor */
        public static final String TENSOR_RESHAPE = "at_reshape";
        public static final FunctionDescriptor TENSOR_RESHAPE_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

        /** tensor.transpose(dim0, dim1) → Tensor */
        public static final String TENSOR_TRANSPOSE = "at_transpose";
        public static final FunctionDescriptor TENSOR_TRANSPOSE_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG);

        // ── Unary operations ──────────────────────────────────────────────

        public static final String TENSOR_NEG = "at_neg";
        public static final String TENSOR_ABS = "at_abs";
        public static final String TENSOR_SQRT = "at_sqrt";
        public static final String TENSOR_LOG = "at_log";
        public static final String TENSOR_EXP = "at_exp";

        public static final FunctionDescriptor TENSOR_UNARY_OP_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS);

        // ── Reduction operations ──────────────────────────────────────────

        /** tensor.sum() → Tensor */
        public static final String TENSOR_SUM = "at_sum";
        public static final String TENSOR_MEAN = "at_mean";
        public static final String TENSOR_MAX = "at_max";
        public static final String TENSOR_MIN = "at_min";

        public static final FunctionDescriptor TENSOR_REDUCE_DESC = TENSOR_UNARY_OP_DESC;

        /** tensor.argmax(dim) → Tensor */
        public static final String TENSOR_ARGMAX = "at_argmax";
        public static final FunctionDescriptor TENSOR_ARGMAX_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

        // ── Comparison ────────────────────────────────────────────────────

        public static final String TENSOR_EQ = "at_eq";
        public static final String TENSOR_GT = "at_gt";
        public static final String TENSOR_LT = "at_lt";

        // ── Autograd ──────────────────────────────────────────────────────

        /** tensor.backward() */
        public static final String TENSOR_BACKWARD = "at_backward";
        public static final FunctionDescriptor TENSOR_BACKWARD_DESC = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);

        /** tensor.grad() → Tensor */
        public static final String TENSOR_GRAD = "at_grad";
        public static final FunctionDescriptor TENSOR_GRAD_DESC = TENSOR_UNARY_OP_DESC;

        /** tensor.requires_grad_(bool) → Tensor */
        public static final String TENSOR_REQUIRES_GRAD = "at_requires_grad_";
        public static final FunctionDescriptor TENSOR_REQUIRES_GRAD_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN);

        /** torch::no_grad guard (returns guard handle) */
        public static final String NO_GRAD_GUARD = "at_no_grad_guard";
        public static final FunctionDescriptor NO_GRAD_GUARD_DESC = FunctionDescriptor.of(ValueLayout.ADDRESS);

        public static final String NO_GRAD_GUARD_RELEASE = "at_no_grad_guard_release";
        public static final FunctionDescriptor NO_GRAD_GUARD_RELEASE_DESC = FunctionDescriptor
                        .ofVoid(ValueLayout.ADDRESS);

        // ── NN Functional ─────────────────────────────────────────────────

        /** relu(input) → Tensor */
        public static final String NN_RELU = "at_relu";
        public static final String NN_GELU = "at_gelu";
        public static final String NN_SIGMOID = "at_sigmoid";
        public static final String NN_TANH = "at_tanh";

        /** softmax(input, dim) → Tensor */
        public static final String NN_SOFTMAX = "at_softmax";
        public static final FunctionDescriptor NN_SOFTMAX_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

        /** dropout(input, p, training) → Tensor */
        public static final String NN_DROPOUT = "at_dropout";
        public static final FunctionDescriptor NN_DROPOUT_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_BOOLEAN);

        /** conv2d(input, weight, bias, ...) → Tensor */
        public static final String NN_CONV2D = "at_conv2d";
        public static final FunctionDescriptor NN_CONV2D_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_LONG);

        /** batch_norm(input, weight, bias, running_mean, running_var, ...) → Tensor */
        public static final String NN_BATCH_NORM = "at_batch_norm";
        public static final FunctionDescriptor NN_BATCH_NORM_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

        /** layer_norm(input, normalized_shape, weight, bias, eps) → Tensor */
        public static final String NN_LAYER_NORM = "at_layer_norm";
        public static final FunctionDescriptor NN_LAYER_NORM_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, // input
                        ValueLayout.ADDRESS, // normalized_shape array
                        ValueLayout.JAVA_LONG, // normalized_shape len
                        ValueLayout.ADDRESS, // weight (nullable)
                        ValueLayout.ADDRESS, // bias (nullable)
                        ValueLayout.JAVA_DOUBLE); // eps

        /** embedding(weight, indices, padding_idx, scale_grad, sparse) → Tensor */
        public static final String NN_EMBEDDING = "at_embedding";
        public static final FunctionDescriptor NN_EMBEDDING_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_LONG, ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN);

        /** index_select(input, dim, index) → Tensor */
        public static final String TENSOR_INDEX_SELECT = "at_index_select";
        public static final FunctionDescriptor TENSOR_INDEX_SELECT_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

        // ── Loss functions ────────────────────────────────────────────────

        /** mse_loss(input, target) → Tensor */
        public static final String MSE_LOSS = "at_mse_loss";
        public static final FunctionDescriptor MSE_LOSS_DESC = TENSOR_BINARY_OP_DESC;

        /** cross_entropy_loss(input, target) → Tensor */
        public static final String CROSS_ENTROPY = "at_cross_entropy_loss";
        public static final FunctionDescriptor CROSS_ENTROPY_DESC = TENSOR_BINARY_OP_DESC;

        /** binary_cross_entropy(input, target) → Tensor */
        public static final String BCE_LOSS = "at_binary_cross_entropy";
        public static final FunctionDescriptor BCE_LOSS_DESC = TENSOR_BINARY_OP_DESC;

        // ── Optimizer ─────────────────────────────────────────────────────

        /** sgd_step(param, grad, lr, momentum, dampening, weight_decay, nesterov) */
        public static final String SGD_STEP = "at_sgd_step";
        public static final FunctionDescriptor SGD_STEP_DESC = FunctionDescriptor.ofVoid(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
                        ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
                        ValueLayout.JAVA_BOOLEAN);

        /** adam_step(param, grad, exp_avg, exp_avg_sq, ...) */
        public static final String ADAM_STEP = "at_adam_step";
        public static final FunctionDescriptor ADAM_STEP_DESC = FunctionDescriptor.ofVoid(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
                        ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_BOOLEAN);

        // ── Serialization / JIT ───────────────────────────────────────────

        /** torch::jit::load(path) → Module */
        public static final String JIT_LOAD = "at_jit_load";
        public static final FunctionDescriptor JIT_LOAD_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS);

        /** module.forward(input) → Tensor */
        public static final String JIT_MODULE_FORWARD = "at_jit_module_forward";
        public static final FunctionDescriptor JIT_MODULE_FORWARD_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

        /** destroy module handle */
        public static final String JIT_MODULE_FREE = "at_jit_module_free";
        public static final FunctionDescriptor JIT_MODULE_FREE_DESC = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);

        /** apply LoRA adapter weights to a target module parameter */
        public static final String JIT_APPLY_LORA = "at_jit_apply_lora";
        public static final FunctionDescriptor JIT_APPLY_LORA_DESC = FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, // module ptr
                        ValueLayout.ADDRESS, // base_name C string
                        ValueLayout.ADDRESS, // lora_A tensor ptr
                        ValueLayout.ADDRESS, // lora_B tensor ptr
                        ValueLayout.JAVA_FLOAT); // scale

        /** torch::save(tensor, path) */
        public static final String TENSOR_SAVE = "at_save";
        public static final FunctionDescriptor TENSOR_SAVE_DESC = FunctionDescriptor.ofVoid(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS);

        /** torch::load(path) → Tensor */
        public static final String TENSOR_LOAD = "at_load";
        public static final FunctionDescriptor TENSOR_LOAD_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS);

        // ── CUDA ──────────────────────────────────────────────────────────

        /** torch::cuda::is_available() → bool */
        public static final String CUDA_IS_AVAILABLE = "at_cuda_is_available";
        public static final FunctionDescriptor CUDA_IS_AVAILABLE_DESC = FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN);

        /** torch::cuda::device_count() → int */
        public static final String CUDA_DEVICE_COUNT = "at_cuda_device_count";
        public static final FunctionDescriptor CUDA_DEVICE_COUNT_DESC = FunctionDescriptor.of(ValueLayout.JAVA_INT);

        /** Whether active CUDA device supports BF16 acceleration. */
        public static final String CUDA_IS_BF16_SUPPORTED = "at_cuda_is_bf16_supported";
        public static final FunctionDescriptor CUDA_IS_BF16_SUPPORTED_DESC = FunctionDescriptor.of(
                        ValueLayout.JAVA_BOOLEAN);

        /** CUDA device SM capability (major*10 + minor), e.g. 89 for Ada. */
        public static final String CUDA_DEVICE_SM = "at_cuda_device_sm";
        public static final FunctionDescriptor CUDA_DEVICE_SM_DESC = FunctionDescriptor.of(
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);

        /** Cast tensor to target dtype (at::ScalarType code). */
        public static final String TENSOR_TO_DTYPE = "at_tensor_to_dtype";
        public static final FunctionDescriptor TENSOR_TO_DTYPE_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

        /** tensor.to(device) → Tensor */
        public static final String TENSOR_TO_DEVICE = "at_to_device";
        public static final FunctionDescriptor TENSOR_TO_DEVICE_DESC = FunctionDescriptor.of(
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);

        // ── Memory management ─────────────────────────────────────────────

        /** Free a tensor handle */
        public static final String TENSOR_FREE = "at_free";
        public static final FunctionDescriptor TENSOR_FREE_DESC = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);

        /** tensor.clone() → Tensor */
        public static final String TENSOR_CLONE = "at_clone";
        public static final FunctionDescriptor TENSOR_CLONE_DESC = TENSOR_UNARY_OP_DESC;

        // ── Instance management ───────────────────────────────────────────

        private LibTorchBinding(SymbolLookup lookup) {
                this.lookup = lookup;
        }

        /**
         * Initialize the singleton with a specific SymbolLookup.
         *
         * @param lookup the symbol lookup for the loaded LibTorch library
         * @return the initialized binding instance
         */
        public static LibTorchBinding initialize(SymbolLookup lookup) {
                if (instance == null) {
                        synchronized (LibTorchBinding.class) {
                                if (instance == null) {
                                        instance = new LibTorchBinding(lookup);
                                        log.debug("LibTorchBinding initialized");
                                }
                        }
                }
                return instance;
        }

        /**
         * Get the singleton instance. Must be {@link #initialize(SymbolLookup)}d first.
         *
         * @throws IllegalStateException if not yet initialized
         */
        public static LibTorchBinding getInstance() {
                if (instance == null) {
                        throw new IllegalStateException(
                                        "LibTorchBinding not initialized. Call initialize() first or ensure LibTorchBeanProducer is active.");
                }
                return instance;
        }

        /**
         * Check if the binding has been initialized.
         */
        public static boolean isInitialized() {
                return instance != null;
        }

        // ── Symbol resolution ─────────────────────────────────────────────

        /**
         * Resolve a native function and return a downcall MethodHandle.
         * Returns {@code Optional.empty()} if the symbol is not found, enabling
         * graceful degradation.
         *
         * @param name       native symbol name
         * @param descriptor function descriptor
         * @return optional method handle
         */
        public Optional<MethodHandle> bindOptional(String name, FunctionDescriptor descriptor) {
                return handleCache.computeIfAbsent(name, k -> {
                        try {
                                Optional<MemorySegment> symbol = lookup.find(k);
                                if (symbol.isPresent()) {
                                        MethodHandle handle = LINKER.downcallHandle(symbol.get(), descriptor);
                                        log.debugf("Bound native function: %s", k);
                                        return Optional.of(handle);
                                }
                                log.debugf("Native function not found (optional): %s", k);
                                return Optional.empty();
                        } catch (Exception e) {
                                log.warnf(e, "Failed to bind native function: %s", k);
                                return Optional.empty();
                        }
                });
        }

        /**
         * Resolve a native function and return its MethodHandle.
         *
         * @param name       native symbol name
         * @param descriptor function descriptor
         * @return the method handle
         * @throws UnsatisfiedLinkError if the symbol is not found
         */
        public MethodHandle bind(String name, FunctionDescriptor descriptor) {
                return bindOptional(name, descriptor)
                                .orElseThrow(() -> new UnsatisfiedLinkError(
                                                "Required LibTorch symbol not found: " + name
                                                                + ". Ensure the correct LibTorch version is installed."));
        }

        /**
         * Check if a native symbol is available.
         *
         * @param name symbol name
         * @return true if the symbol can be resolved
         */
        public boolean hasSymbol(String name) {
                return lookup.find(name).isPresent();
        }

        /**
         * Get the underlying SymbolLookup.
         */
        public SymbolLookup lookup() {
                return lookup;
        }
}
