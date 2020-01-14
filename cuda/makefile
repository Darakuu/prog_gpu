TARGETS = vecsum scan scan4

CPPFLAGS = -g

all: $(TARGETS)

%: %.cu
	nvcc $(CPPFLAGS) $(CUFLAGS) -o $@.out $< $(LDFLAGS) $(LDLIBS)