# tests/Makefile
# Usage: make -C tests run FILE=matrix-op-unittests.c

# Use project root from environment variable
ROOT := $(PIM_MATMUL_BENCHMARKS_ROOT)

DOCKER_IMAGE := pim-matmul-dev

CC ?= gcc
BIN_DIR := $(ROOT)/bin
DEPS_YAML := $(ROOT)/defn/dependencies.yaml
DEPS_SRCS := $(shell python3 -c "import yaml,sys; print(' '.join('$(ROOT)/'+s for s in yaml.safe_load(open('$(DEPS_YAML)'))['sources']))" 2>/dev/null || echo "")
INCLUDE_DIRS := $(shell python3 -c "import yaml; print(' '.join('-I$(ROOT)/'+d for d in yaml.safe_load(open('$(DEPS_YAML)'))['include_dirs']))" 2>/dev/null || echo "")

# Extract all runtime parameters as compiler flags
PARAMS_YAML := $(ROOT)/defn/params.yaml
RUNTIME_PARAM_FLAGS := $(shell python3 -c "import yaml; params=yaml.safe_load(open('$(PARAMS_YAML)')); print(' '.join([f'-D{k}={v}' for item in params.get('runtime_params', []) for k, v in item.items()]))" 2>/dev/null || echo "")

CFLAGS += $(INCLUDE_DIRS)
CFLAGS += -Icommon/
CFLAGS += $(RUNTIME_PARAM_FLAGS)
CFLAGS += -DTIMER 

DOCKER ?= 0

compile:
ifeq ($(DOCKER),1)
	docker run --rm --platform linux/amd64 -v $(ROOT):/workspace $(DOCKER_IMAGE) bash -c \
		". /opt/upmem-2025.1.0-Linux-x86_64/upmem_env.sh simulator && \
		. /workspace/source.me && \
		$(CC) $(CFLAGS) -g $(FILE) $(DEPS_SRCS) -o $(BIN_DIR)/$$(basename $(FILE) .c) `dpu-pkg-config --cflags --libs dpu` -lm -ldl -lelf -lpython3.7m -lnuma -lgomp"
else
	$(CC) $(CFLAGS) -g $(FILE) $(DEPS_SRCS) -o $(BIN_DIR)/$$(basename $(FILE) .c) `dpu-pkg-config --cflags --libs dpu` -lm -ldl -lelf -lpython3.7m -lnuma -lgomp
endif

docker-run:
	docker run --rm --platform linux/amd64 -v $(ROOT):/workspace $(DOCKER_IMAGE) bash -c \
		". /opt/upmem-2025.1.0-Linux-x86_64/upmem_env.sh simulator && \
		. /workspace/source.me && \
		$(BIN_DIR)/$$(basename $(FILE) .c)"
