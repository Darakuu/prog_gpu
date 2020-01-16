TARGETS = vecsum scan scan4 imgcopy

CPPFLAGS = -g

all: $(TARGETS)

%: %.cu
	nvcc $(CPPFLAGS) $(CUFLAGS) -o $@.out $< $(LDFLAGS) $(LDLIBS)
