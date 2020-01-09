TARGETS = vecsum scan

CPPFLAGS = -g

all: $(TARGETS)

%: %.cu
	nvcc $(CPPFLAGS) $(CUFLAGS) -o $@.out $< $(LDFLAGS) $(LDLIBS)