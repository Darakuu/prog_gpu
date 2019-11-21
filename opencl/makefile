LDLIBS=-lOpenCL

DIR_VEC=vectors/
DIR_MAT=matrices/
DIR_IMG=images/

TARGETS = vecsum vecsum4 vecsum8 vecsum4by4 vecsum4by2
TARGETS+= vecsmooth vecsmooth_lmem vecsmooth_v4
TARGETS+= matinit transpose transpose_lmem
TARGETS+= $(DIR_IMG)transpose_img $(DIR_IMG)transpose_img4 $(DIR_IMG)imgcopy

other: 	$(addsuffix .out,$(TARGETS))
	rm -f $(TARGETS)
all: $(TARGETS)	

