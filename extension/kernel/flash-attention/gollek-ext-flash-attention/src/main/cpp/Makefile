# Makefile for FlashAttention-3 CUDA Kernels
# Targets Nvidia Hopper architecture (sm_90) for FP8 and TMA support

LIBTORCH_PATH ?= /work/libtorch
CUDA_PATH     ?= /usr/local/cuda

NVCC := $(CUDA_PATH)/bin/nvcc

# Compilation Flags for Hopper
# -shared -Xcompiler -fPIC: Generate position independent shared library (.so) for Java FFM
# -gencode arch=compute_90,code=sm_90: Target H100/Hopper specifically
NVCC_FLAGS := -O3 -shared -Xcompiler -fPIC -std=c++17 \
              -D_GLIBCXX_USE_CXX11_ABI=1 \
              -gencode arch=compute_90,code=sm_90

INCLUDES := -I$(CUDA_PATH)/include \
            -I$(LIBTORCH_PATH)/include \
            -I$(LIBTORCH_PATH)/include/torch/csrc/api/include

LIBS := -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cuda -lc10 -lcudart

TARGET := libgollek_fa3_kernels.so
SRC := gollek_fa3_kernels.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
