# tests/Makefile
# Usage: make -C tests run FILE=matrix-op-unittests.c

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

all: build run

build:
	mkdir -p $(BIN_DIR); \
	$(CC) $(CFLAGS) -g $(FILE) $(DEPS_SRCS) -o $(BIN_DIR)/$$name `dpu-pkg-config --cflags --libs dpu` -lm -ldl -lelf -lpython3.7m -lnuma -lgomp 2>&1 | tee $(ROOT)/scratch/compile_logs/$$name.log; \


run:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make run FILE=yourtest.c"; \
		exit 1; \
	fi; \
	name=$$(basename $(FILE) .c); \
	mkdir -p $(ROOT)/scratch/compile_logs; \
	mkdir -p $(ROOT)/scratch/runtime_logs; \
	echo "Running $(BIN_DIR)/$$name"; \
	$(BIN_DIR)/$$name 2>&1 | tee $(ROOT)/scratch/runtime_logs/$$name.log
