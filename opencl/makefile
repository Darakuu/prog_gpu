LDLIBS=-lOpenCL

DIR_VEC=vectors/
DIR_MAT=matrices/
DIR_IMG=images/
DIR_RED=reductions/

TARGETS = vecsum vecsum4 vecsum8 vecsum4by4 vecsum4by2
TARGETS+= vecsmooth vecsmooth_lmem vecsmooth_v4
TARGETS+= matinit transpose transpose_lmem
TARGETS+= $(DIR_IMG)transpose_img $(DIR_IMG)transpose_img4 $(DIR_IMG)imgcopy $(DIR_IMG)imgzoom
TARGETS+= $(DIR_RED)reduce2

other: 	$(addsuffix .out,$(TARGETS))
	rm -f $(TARGETS)
all: $(TARGETS)	

