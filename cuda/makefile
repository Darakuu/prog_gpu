TARGETS = vecsum

CPPFLAGS = -g

all: $(TARGETS)

%: %.cu
	nvcc $(CPPFLAGS) $(CUFLAGS) -o $@.out $< $(LDFLAGS) $(LDLIBS)