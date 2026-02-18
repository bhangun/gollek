# ==============================================================================
# Gollek PagedAttention CUDA Kernel — Makefile
# ==============================================================================
#
# Targets:
#   make         — Build libgollek_kernels.so
#   make clean   — Remove build artifacts
#   make install — Copy .so to resources dir for Maven packaging
#   make test    — (Future) Run kernel unit tests
#
# Requirements:
#   - CUDA Toolkit 12.x+
#   - NVIDIA GPU with compute capability 8.0+ (A100, H100)
#   - nvcc compiler
#
# Environment:
#   CUDA_HOME — Path to CUDA toolkit (default: /usr/local/cuda)
#   LIBTORCH_HOME — Path to LibTorch C++ distribution (optional, for linking)
#
# ==============================================================================

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

# GPU architecture targets
# sm_80 = A100, sm_86 = RTX 3090, sm_89 = RTX 4090, sm_90 = H100
GPU_ARCHS := -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

# Compiler flags
NVCC_FLAGS := -shared -Xcompiler -fPIC -std=c++17 $(GPU_ARCHS)
NVCC_FLAGS += -O3  # Optimization level

# Optional: LibTorch include paths for tensor interop
ifdef LIBTORCH_HOME
	NVCC_FLAGS += -I$(LIBTORCH_HOME)/include \
	              -I$(LIBTORCH_HOME)/include/torch/csrc/api/include
	LINK_FLAGS := -L$(LIBTORCH_HOME)/lib -ltorch -lc10
else
	LINK_FLAGS :=
endif

# Source and output
SRC     := gollek_kernels.cu
OUTPUT  := libgollek_kernels.so
BUILDDIR := ../../../../target/native

.PHONY: all clean install

all: $(BUILDDIR)/$(OUTPUT)

$(BUILDDIR)/$(OUTPUT): $(SRC)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LINK_FLAGS)
	@echo "✓ Built $(OUTPUT) → $(BUILDDIR)/"

clean:
	rm -f $(BUILDDIR)/$(OUTPUT)
	@echo "✓ Cleaned"

# Install to Maven resources so it's included in the JAR
install: $(BUILDDIR)/$(OUTPUT)
	@mkdir -p ../resources/native/linux-x86_64
	cp $(BUILDDIR)/$(OUTPUT) ../resources/native/linux-x86_64/
	@echo "✓ Installed to resources/native/linux-x86_64/"

# Print build configuration
info:
	@echo "CUDA_HOME     = $(CUDA_HOME)"
	@echo "NVCC          = $(NVCC)"
	@echo "GPU_ARCHS     = $(GPU_ARCHS)"
	@echo "LIBTORCH_HOME = $(LIBTORCH_HOME)"
	@echo "OUTPUT        = $(BUILDDIR)/$(OUTPUT)"
