# Use project root from environment variable
ROOT := $(PIM_MATMUL_BENCHMARKS_ROOT)

CC ?= gcc
BIN_DIR := $(ROOT)/bin
DEPS_YAML := $(ROOT)/defn/dependencies.yaml
DEPS_SRCS := $(shell python3 -c "import yaml,sys; print(' '.join('$(ROOT)/'+s for s in yaml.safe_load(open('$(DEPS_YAML)'))['sources']))" 2>/dev/null || echo "")
INCLUDE_DIRS := $(shell python3 -c "import yaml; print(' '.join('-I$(ROOT)/'+d for d in yaml.safe_load(open('$(DEPS_YAML)'))['include_dirs']))" 2>/dev/null || echo "")

# Extract all runtime parameters as compiler flags
PARAMS_YAML := $(ROOT)/defn/params.yaml
RUNTIME_PARAM_FLAGS := $(shell python3 -c "import yaml; params=yaml.safe_load(open('$(PARAMS_YAML)')); print(' '.join([f'-D{k}={v}' for item in params.get('runtime_params', []) for k, v in item.items()]))" 2>/dev/null || echo "")

DEBUG ?= 0
TIMER ?= 0

CFLAGS += $(INCLUDE_DIRS)
CFLAGS += -Icommon/
CFLAGS += $(RUNTIME_PARAM_FLAGS)
ifeq ($(DEBUG),1)
CFLAGS += -DDEBUG 
endif
ifeq ($(TIMER),1)
CFLAGS += -DTIMER 
endif

# Get all test source files
TEST_SOURCES := $(wildcard test_*.c)
TEST_NAMES := $(basename $(notdir $(TEST_SOURCES)))

# Generate phony targets for autocomplete
.PHONY: help list-tests

help:
	@echo "Available test targets (with autocomplete):"
	@echo "  make build-<test_name>  - Build a specific test"
	@echo "  make run-<test_name>    - Run a specific test (must be built first)"
	@echo "  make <test_name>        - Build and run a specific test"
	@echo ""
	@echo "Examples:"
	@echo "  make build-test_matrix_create_from_2d_array_and_free"
	@echo "  make run-test_matrix_create_from_2d_array_and_free"
	@echo "  make test_matrix_create_from_2d_array_and_free"
	@echo ""
	@echo "To list all available tests: make list-tests"

list-tests:
	@echo "Available tests:"
	@for test in $(sort $(TEST_NAMES)); do echo "  $$test"; done

# Pattern rule: build-<test_name> - Build a specific test
build-%: 
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(ROOT)/scratch/compile_logs
	@echo "Building test: $*"
	@test_file=""; \
	if [ -f "$*.c" ]; then \
		test_file="$*.c"; \
	elif [ -f "individual/$*.c" ]; then \
		test_file="individual/$*.c"; \
	else \
		echo "Error: Test file $*.c not found"; \
		exit 1; \
	fi; \
	$(CC) $(CFLAGS) -g $$test_file $(DEPS_SRCS) -o $(BIN_DIR)/$* `dpu-pkg-config --cflags --libs dpu` -lm -ldl -lelf -lgomp -lpython3.7m  -lnuma 2>&1 | tee $(ROOT)/scratch/compile_logs/$*.log

# Pattern rule: run-<test_name> - Run a specific test
run-%:
	@mkdir -p $(ROOT)/scratch/runtime_logs
	@if [ ! -f "$(BIN_DIR)/$*" ]; then \
		echo "Error: Test binary $(BIN_DIR)/$* not found. Build it first with: make build-$*"; \
		exit 1; \
	fi
	@echo "Running test: $*"
	@$(BIN_DIR)/$* 2>&1 | tee $(ROOT)/scratch/runtime_logs/$*.log

# Pattern rule: <test_name> - Build and run a specific test
$(TEST_NAMES): %: build-% run-%
	@echo "Completed: $@"