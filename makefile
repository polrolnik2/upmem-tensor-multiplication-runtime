# Docker image name
DOCKER_IMAGE := pim-matmul-dev

help:
	@echo "Available targets:"
	@echo "  make docker-build - Build the Docker image"
	@echo "  make SimplePIM - Clone SimplePIM library if not present"
	@echo "  make build-dpu - Build DPU binaries"
	@echo "  make run-unittests - Run unittests in Docker"
	@echo "  make clean - Clean up build artifacts"
	@echo "  make docs - Generate documentation"
	@echo "  make docs-docker - Generate documentation in Docker"
	@echo "  make docs-clean - Clean documentation"
	@echo "  make docs-view - View documentation in browser"

# Build the Docker image
docker-build:
	docker build --platform linux/amd64 -t $(DOCKER_IMAGE) .

CC ?= gcc

DEBUG ?= 0
TIMER ?= 0

# Use project root from environment variable
ROOT := $(PIM_MATMUL_BENCHMARKS_ROOT)

# Helper to extract sources from YAML
DEPS_YAML := $(ROOT)/defn/dependencies.yaml
DEPS_SRCS := $(shell python3 -c "import yaml,sys; print(' '.join(yaml.safe_load(open('$(DEPS_YAML)'))['sources']))" 2>/dev/null || echo "")
INCLUDE_DIRS := $(shell python3 -c "import yaml; print(' '.join('-I'+d for d in yaml.safe_load(open('$(DEPS_YAML)'))['include_dirs']))" 2>/dev/null || echo "")

# Extract all runtime parameters as compiler flags
PARAMS_YAML := $(ROOT)/defn/params.yaml
RUNTIME_PARAM_FLAGS := $(shell python3 -c "import yaml; params=yaml.safe_load(open('$(PARAMS_YAML)')); print(' '.join([f'-D{k}={v}' for item in params.get('runtime_params', []) for k, v in item.items()]))" 2>/dev/null || echo "")

CFLAGS += $(INCLUDE_DIRS)
CFLAGS += $(RUNTIME_PARAM_FLAGS)
ifeq ($(DEBUG),1)
CFLAGS += -DDEBUG
endif
ifeq ($(TIMER),1)
CFLAGS += -DTIMER
endif

clean:
	rm -rf lib/simplepim
	rm -rf $(BIN_DIR)
	rm -rf $(PIM_MATMUL_BENCHMARKS_ROOT)/scratch/
	rm -rf $(DOCS_DIR)

BIN_DIR := bin

# Helper to extract unittest sources from YAML
UNITTEST_SRCS := $(shell python3 -c "import yaml; print(' '.join(yaml.safe_load(open('defn/unittests.yaml'))['unittest']))" 2>/dev/null || echo "")
UNITTEST_BINS := $(addprefix bin/,$(basename $(notdir $(UNITTEST_SRCS))))

# Ensure bin/ exists
bin:
	@mkdir -p bin

run-unittests: bin build-dpu
	@mkdir -p scratch; \
	set -e; \
	for t in $(UNITTEST_SRCS); do \
	  echo "Building and running $$t"; \
	  make -C tests $$t DEBUG=$(DEBUG) TIMER=$(TIMER); \
	done; \
	python3 scripts/parse_unittest_logs.py

# Build DPU binaries
build-dpu: bin
	dpu-upmem-dpurte-clang -O2 $(CFLAGS) -g -o $(ROOT)/bin/matrix_multiply_dpu $(ROOT)/src/dpu/pim_dpu_matrix_multiply.c -I src

# Documentation directories
DOCS_DIR := docs
DOCS_HTML_DIR := $(DOCS_DIR)/html
DOCS_LATEX_DIR := $(DOCS_DIR)/latex

# Source files for documentation
DOC_SOURCES := $(wildcard src/*.h) $(wildcard src/*.c) $(wildcard tests/*.h) $(wildcard tests/*.c)

# Generate documentation with Doxygen
docs: $(DOCS_HTML_DIR)/index.html

$(DOCS_HTML_DIR)/index.html: $(DOC_SOURCES) Doxyfile
	@mkdir -p $(DOCS_DIR)
	doxygen Doxyfile

# Generate Doxyfile if it doesn't exist
Doxyfile:
	doxygen -g
	@echo "Generated default Doxyfile. You may want to customize it."
	@echo "Key settings to consider:"
	@echo "  - PROJECT_NAME = \"PIM Matrix Multiplication Benchmarks\""
	@echo "  - INPUT = src tests"
	@echo "  - RECURSIVE = YES"
	@echo "  - GENERATE_HTML = YES"
	@echo "  - GENERATE_LATEX = YES"
	@echo "  - EXTRACT_ALL = YES"
	@echo "  - EXTRACT_PRIVATE = YES"
	@echo "  - EXTRACT_STATIC = YES"

# Generate documentation in Docker environment
docs-docker: docker-build
	docker run --rm --platform linux/amd64 -v $(CURDIR):/workspace $(DOCKER_IMAGE) bash -c \
		"cd /workspace && \
		apt-get update -qq && apt-get install -y -qq doxygen graphviz && \
		make docs"

# Clean documentation
docs-clean:
	rm -rf $(DOCS_DIR)

# View documentation (opens in default browser)
docs-view: docs
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(DOCS_HTML_DIR)/index.html; \
	elif command -v open > /dev/null; then \
		open $(DOCS_HTML_DIR)/index.html; \
	else \
		echo "Documentation generated at: $(DOCS_HTML_DIR)/index.html"; \
	fi

.PHONY: SimplePIM clean build-unittests run-unittests build-dpu docs docs-docker docs-clean docs-view