TARGETS = vecsum scan scan4 imgcopy imgcopy_ptr

CPPFLAGS = -g

all: $(TARGETS)

%: %.cu
	nvcc $(CPPFLAGS) $(CUFLAGS) -o $@.out $< $(LDFLAGS) $(LDLIBS)
