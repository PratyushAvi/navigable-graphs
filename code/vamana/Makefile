# Vamana build for this project's copy of the algorithm.
#
# index.h/neighbors.h live here and are ours to modify; the supporting headers
# (utils/, bench/, parlaylib) still come from the ParlayANN checkout, so PARLAYANN
# points at it. Override on the command line for a checkout somewhere else:
#
#   make PARLAYANN=/path/to/ParlayANN

PARLAYANN ?= ../../ParlayANN
ALGORITHMS = $(PARLAYANN)/algorithms

include $(ALGORITHMS)/bench/parallelDefsANN

# -I . comes first so our index.h/neighbors.h win over ParlayANN's copies.
CCFLAGS += -I $(ALGORITHMS)

REQUIRE = index.h neighbors.h \
	$(ALGORITHMS)/utils/beamSearch.h $(ALGORITHMS)/utils/check_nn_recall.h \
	$(ALGORITHMS)/utils/NSGDist.h $(ALGORITHMS)/utils/parse_results.h \
	$(ALGORITHMS)/utils/graph.h $(ALGORITHMS)/utils/point_range.h \
	$(ALGORITHMS)/utils/euclidian_point.h $(ALGORITHMS)/utils/mips_point.h \
	$(ALGORITHMS)/utils/jl_point.h $(ALGORITHMS)/utils/hashset.h \
	$(ALGORITHMS)/utils/types.h

BENCH = neighbors
TIME = $(ALGORITHMS)/bench/$(BENCH)Time.C
INCLUDE = -I $(PARLAYANN)/parlaylib/include/

all : $(BENCH)

$(BENCH) : $(TIME) $(BENCH).h $(REQUIRE)
	$(CC) -DSTATS $(CFLAGS) $(INCLUDE) -include $(BENCH).h -o $(BENCH) $(TIME) $(LFLAGS)

clean :
	rm -f $(BENCH)

.PHONY : all clean
