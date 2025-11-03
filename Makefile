# --- Configuration ---

# Directories
SRC_DIR = src
INC_DIR = include
OUT_NAME = my_opengl_app.exe

# CUDA Configuration
CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib/x64  # Assuming 64-bit build

# Compilers
NVCC = nvcc
CXX = g++

# Compiler/Linker Flags
# -I: Include directories
# -ccbin: Specify the host C++ compiler (g++)
NVCC_FLAGS = -m64 -std=c++17 -I$(INC_DIR) -I$(CUDA_INC)

# Linker Flags
# -lglfw3, -lopengl32: The main libraries you require
# -lcudart: CUDA Runtime library
# -Xlinker -mwindows: Suppress the console window (passed to g++ linker)
# -luser32 -lkernel32 -lgdi32: Explicitly link required Windows system libraries
MSVC_LIBS = opengl32.lib glfw3.lib cudart.lib user32.lib kernel32.lib gdi32.lib
LDFLAGS = -L$(CUDA_LIB) $(MSVC_LIBS) -Xlinker /SUBSYSTEM:WINDOWS


# Source Files
ALL_SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)

# --- Targets ---

.PHONY: all clean run

all: $(OUT_NAME)

# Single rule to compile and link all source files using NVCC
$(OUT_NAME): $(ALL_SRCS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(OUT_NAME)

run: $(OUT_NAME)
	./$(OUT_NAME)