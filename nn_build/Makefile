CC          := g++
CFLAGS      := -ffast-math -std=c++11 -g -I/home/mur1/Projects/repos/compiler/openssl-1.1.1l/openssl-1.1.1l/include

CC_CU		:= nvcc
CFLAGS_CU	:= -use_fast_math -std=c++11 -g -G

CC_C		:= g++
CFLAGS_C	:= -ffast-math -std=c++11 -g -I/home/mur1/Projects/repos/compiler/openssl-1.1.1l/openssl-1.1.1l/include

LD          := g++ -g
LDFLAGS     := -L-I/home/mur1/Projects/repos/compiler/openssl-1.1.1l/openssl-1.1.1l/ -l:libssl.so.1.1 -l:libcrypto.so.1.1 -pthread -lcudart

SRC_DIR		:= src
BUILD_DIR	:= build

SRC_CPP		:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ_CPP     := $(patsubst src/%.cpp,build/%.o,$(SRC_CPP))

SRC_CU		:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cu))
OBJ_CU		:= $(patsubst src/%.cu,build/%.o,$(SRC_CU))

SRC_C		:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.c))
OBJ_C		:= $(patsubst src/%.c,build/%.o,$(SRC_C))

OBJ		:= $(OBJ_CPP) $(OBJ_CU) $(OBJ_C)

VPATH		:= src

define make-goal-cpp
$1/%.o: %.cpp
	$(CC) $(CFLAGS) -c $$< -o $$@
endef

define make-goal-c
$1/%.o: %.c
	$(CC_C) $(CFLAGS_C) -c $$< -o $$@
endef

define make-goal-cu
$1/%.o: %.cu
	$(CC_CU) $(CFLAGS_CU) -c $$< -o $$@
endef

all: checkdirs nn_build

nn_build: $(OBJ)
	$(LD) $^ $(LDFLAGS) -o $@

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal-cpp,$(bdir))))
$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal-cu,$(bdir))))
$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal-c,$(bdir))))
