# Makefile for GPU-Accelerated Flight Dynamics Simulator
#
# Author: Kushal Koirala
# Target: RTX 4060 (sm_89) / RTX 30xx (sm_86) / RTX 20xx (sm_75)
#
# Usage:
#   make          - Build shared library and test/benchmark executables
#   make test     - Build and run tests
#   make bench    - Build and run benchmarks
#   make clean    - Remove build artifacts

# Compiler
NVCC := nvcc

# CUDA architecture - adjust for your GPU:
#   sm_75 - RTX 20xx (Turing)
#   sm_86 - RTX 30xx (Ampere)
#   sm_89 - RTX 40xx (Ada Lovelace) <- RTX 4060
CUDA_ARCH ?= sm_89

# Compiler flags
NVCC_FLAGS := -arch=$(CUDA_ARCH) -O3 --use_fast_math -Xcompiler -fPIC
DEBUG_FLAGS := -g -G -DDEBUG
INCLUDE_FLAGS := -I.

# Source files - consolidated into single file to avoid device function linking issues
KERNEL_SRC := flight_sim_kernels.cu

# Output names
LIB_NAME := libflightdynamics
SHARED_LIB := $(LIB_NAME).so

# Default target
all: $(SHARED_LIB) test_main benchmark

# Build shared library
$(SHARED_LIB): $(KERNEL_SRC) flight_dynamics.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -shared -o $@ $(KERNEL_SRC) -lcudart
	@echo ""
	@echo "✓ Built shared library: $@"
	@echo ""

# Build test executable
test_main: test_main.cu $(SHARED_LIB)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ test_main.cu -L. -lflightdynamics -lcudart -Xlinker -rpath -Xlinker .
	@echo "✓ Built test executable: $@"

# Build benchmark executable
benchmark: benchmark.cu $(SHARED_LIB)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -o $@ benchmark.cu -L. -lflightdynamics -lcudart -Xlinker -rpath -Xlinker .
	@echo "✓ Built benchmark executable: $@"

# Run tests
test: test_main
	@echo ""
	@echo "Running tests..."
	@echo ""
	./test_main

# Run benchmarks
bench: benchmark
	@echo ""
	@echo "Running benchmarks..."
	@echo ""
	./benchmark

# Debug build
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: clean all

# Clean
clean:
	rm -f $(SHARED_LIB) test_main benchmark *.o

# Show GPU info
gpu-info:
	@nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
	@echo ""
	@nvcc --version

# Help
help:
	@echo "GPU-Accelerated Flight Dynamics Simulator"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build library and executables (default)"
	@echo "  test      - Build and run tests"
	@echo "  bench     - Build and run benchmarks"
	@echo "  clean     - Remove build artifacts"
	@echo "  gpu-info  - Show GPU information"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH - Target GPU architecture (default: sm_89 for RTX 4060)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build for RTX 4060"
	@echo "  make CUDA_ARCH=sm_86    # Build for RTX 3080"
	@echo "  make bench              # Run benchmarks"

.PHONY: all test bench debug clean gpu-info help
