# Vamana build for this project's copy of the algorithm.
#
# Self-contained on purpose: it does NOT include ParlayANN's bench/parallelDefsANN,
# so it does not depend on any local edit to the ParlayANN checkout. Only the
# upstream headers are used from there, unmodified.
#
#   make                                  # expects ../../ParlayANN
#   make PARLAYANN=/path/to/ParlayANN     # checkout elsewhere
#   make CXX=g++-13                       # pick a compiler
#   make ARCHFLAGS=                       # portable build, no -march=native

PARLAYANN ?= ../../ParlayANN
ALGORITHMS = $(PARLAYANN)/algorithms

# NB: use CXX, not CC -- make predefines CC=cc, so `CC ?= g++` would be ignored
# and the C driver would fail to link the C++ standard library.
CXX ?= g++

# -march/-mcpu native: x86 needs -mcx16 for the 128-bit CAS parlay uses; arm64
# has it natively and rejects the flag, and Apple clang wants -mcpu over -march.
# Override with `make ARCHFLAGS=` when building for a different machine than the
# one compiling (common on HPC login vs compute nodes).
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  ARCHFLAGS ?= -mcpu=native
else ifeq ($(UNAME_M),aarch64)
  ARCHFLAGS ?= -mcpu=native
else
  ARCHFLAGS ?= -mcx16 -march=native
endif

# jemalloc if the system has it, silently skipped otherwise.
JEMALLOC_LIBDIR := $(shell jemalloc-config --libdir 2>/dev/null)
ifneq ($(JEMALLOC_LIBDIR),)
  JEMALLOC = -L$(JEMALLOC_LIBDIR) -ljemalloc
endif

# -I . first so our index.h/neighbors.h win over ParlayANN's copies.
CPPFLAGS = -I . -I $(ALGORITHMS) -I $(PARLAYANN)/parlaylib/include
CXXFLAGS = $(ARCHFLAGS) -O3 -std=c++17 -DNDEBUG -DHOMEGROWN -DSTATS -pthread
LDFLAGS  = -pthread -ldl $(JEMALLOC)

BENCH  = neighbors
DRIVER = $(ALGORITHMS)/bench/$(BENCH)Time.C

HEADERS = index.h neighbors.h gamma_args.h

all : $(BENCH)

$(BENCH) : $(DRIVER) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -include $(BENCH).h -o $@ $(DRIVER) $(LDFLAGS)

clean :
	rm -f $(BENCH)

.PHONY : all clean
