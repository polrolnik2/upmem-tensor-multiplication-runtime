# PIM Matrix Multiplication Benchmarks

This project implements a framework for multiplying matrices using UPMEM PIM - a commercially available processing-in-memory solution. It provides all necessary primitives to execute and control the process of matrix multiplication on this device, including patterns for distributing matrices, executing operations, and gathering results.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Git LFS Setup](#git-lfs-setup)
  - [Building with Docker](#building-with-docker)
  - [Building with CMake Directly](#building-with-cmake-directly)
- [Integration Guide](#integration-guide)
- [Documentation](#documentation)

## Prerequisites

- **Git LFS**: Required for downloading the UPMEM SDK tarball
- **CMake**: Version 3.20 or higher
- **Python**: Python 3.7 or higher with `pyyaml` package
- **UPMEM SDK**: Version 2023.2.0 (included via Git LFS)
- **Docker** (optional): For containerized builds and testing

## Getting Started

### Git LFS Setup

This repository uses Git Large File Storage (LFS) to manage the UPMEM SDK tarball (`lib/upmem.tar.gz`). You must have Git LFS installed and configured before cloning or pulling the repository.

#### Install Git LFS

**macOS:**
```bash
brew install git-lfs
```

**Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
```

#### Initialize Git LFS

After installing Git LFS, initialize it:
```bash
git lfs install
```

#### Clone the Repository

If you haven't cloned yet:
```bash
git clone <repository-url>
cd pim-matmul-benchmarks
```

If you've already cloned without Git LFS:
```bash
git lfs fetch
git lfs pull
```

Verify that the UPMEM SDK was downloaded correctly:
```bash
ls -lh lib/upmem.tar.gz  # Should show actual file size (~100MB+), not a few KB
```

### Building with Docker

Docker provides a consistent build environment with all dependencies pre-configured.

#### Build the Docker Image

```bash
docker-compose build dev
```

#### Interactive Development Environment

Start an interactive shell with the project mounted:
```bash
docker-compose run --rm dev
```

Inside the container, source the environment and build:
```bash
source /opt/upmem-2023.2.0-Linux-x86_64/upmem_env.sh simulator
source /workspace/source.me
mkdir -p build
cmake -S . -B build
make -C build all
```

#### One-Step Build

Build the entire project in one command:
```bash
docker-compose run --rm build
```

#### Run Unit Tests

Execute all unit tests:
```bash
docker-compose run --rm unittest
```

#### Docker Compose Services

- **dev**: Interactive development environment with full project access
- **build**: Automated build service that compiles the project
- **unittest**: Runs all unit tests via CTest

### Building with CMake Directly

For native builds on your host system, follow these steps.

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git-lfs \
    python3 python3-pip doxygen \
    libelf-dev libnuma-dev libgomp1 \
    pkg-config gdb
```

**macOS:**
```bash
brew install cmake git-lfs python doxygen pkg-config
```

#### 3. Set Up Python Environment

```bash
# Source the project environment script
source source.me

# Or manually set up Python environment
python3 -m venv scripts/pim-matmul-env
source scripts/pim-matmul-env/bin/activate
pip install --upgrade pip
pip install pyyaml
```

#### 4. Configure and Build

```bash
# Configure the project
mkdir -p build
cmake -S . -B build

# Build all targets
make -C build all

# Or use CMake directly
cmake --build build

# Build specific targets
make -C build pim_matmul        # Build library only
make -C build tests             # Build tests only
make -C build benchmarks        # Build benchmarks only
```

#### 5. Run Tests

```bash
# Run all tests
cd build
ctest --output-on-failure -V

# Run specific test
./tests/test_matrix_create_from_2d_array_and_free
```

#### CMake Configuration Options

Customize the build with CMake options:

```bash
cmake -S . -B build \
  -DBUILD_TESTS=ON \                    # Enable/disable tests (default: ON)
  -DBUILD_BENCHMARKS=ON \                # Enable/disable benchmarks (default: ON)
  -DPARAMS_FILE=path/to/params.yaml \    # Custom params file
  -DCMAKE_BUILD_TYPE=Release             # Release or Debug (default: Debug)
```

## Integration Guide

To integrate this PIM matrix multiplication framework into your existing CMake-based project:

### Method 1: Add as Subdirectory

1. **Add the project as a subdirectory** (e.g., via git submodule or copying):
   ```bash
   git submodule add <repository-url> external/pim-matmul-benchmarks
   ```

2. **In your project's CMakeLists.txt**, add:
   ```cmake
   # Add PIM matmul subdirectory
   add_subdirectory(external/pim-matmul-benchmarks)
   
   # Link your target to the PIM library
   target_link_libraries(your_target PRIVATE pim_matmul)
   ```

3. **Ensure UPMEM SDK is configured** in your environment before running CMake:
   ```bash
   export PKG_CONFIG_PATH="/opt/upmem-2023.2.0-Linux-x86_64/share/pkgconfig:${PKG_CONFIG_PATH}"
   export PATH="/opt/upmem-2023.2.0-Linux-x86_64/bin:${PATH}"
   source /opt/upmem-2023.2.0-Linux-x86_64/upmem_env.sh simulator
   ```

### Method 2: Install and Use find_package

1. **Build and install the library**:
   ```bash
   cd pim-matmul-benchmarks
   mkdir -p build && cd build
   cmake -S .. -B . -DCMAKE_INSTALL_PREFIX=/usr/local
   make install
   ```

2. **In your project's CMakeLists.txt**:
   ```cmake
   find_package(pim_matmul REQUIRED)
   target_link_libraries(your_target PRIVATE pim_matmul)
   ```

### Using the API

Include the necessary headers in your code:

```c
#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"

// Your code here
Matrix* matrixA = matrix_create_from_row_major_array(...);
Matrix* matrixB = matrix_create_from_row_major_array(...);
// Perform PIM multiplication...
```

### Key Integration Considerations

1. **UPMEM SDK Dependency**: Ensure the UPMEM SDK is installed and environment variables are set before building your project.

2. **DPU Binary Path**: The framework automatically builds and embeds the DPU kernel binary path. If you need a custom DPU binary:
   ```bash
   cmake -DPIM_MATMUL_DPU_BINARY_PATH=/path/to/your/dpu/binary ...
   ```

3. **Runtime Parameters**: Customize runtime behavior via `defn/params.yaml` or provide your own:
   ```bash
   cmake -DPARAMS_FILE=/path/to/your/params.yaml ...
   ```

4. **Include Directories**: The library automatically exports these include directories:
   - `src/` - Core library headers
   - `common/` - Common utilities and helpers
   - `lib/simplepim/` - SimplePIM library

5. **Required Libraries**: The framework links against:
   - UPMEM DPU libraries (via `dpu-pkg-config`)
   - Math library (`-lm`)

## Documentation

- **API Documentation**: Refer to the inline documentation in header files or generate Doxygen docs:
  ```bash
  doxygen Doxyfile
  ```
  
- **Build Targets**: View available make targets:
  ```bash
  make help  # If using the project's Makefile
  ```

- **Examples**: See `benchmarks/` directory for usage examples:
  - `1gb_square_benchmark.c` - Large square matrix multiplication
  - `back_to_back_multiplication_benchmark.c` - Sequential multiplications
  - `test_from_file.c` - Loading matrices from files

## License

See [LICENSE](LICENSE) for details.
